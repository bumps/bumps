# from .main import setup_bumps

import itertools
from typing import Dict, List, Literal, Optional, Union, TypedDict
from datetime import datetime
import warnings
from queue import Queue
from collections import deque
from aiohttp import web
import numpy as np
import asyncio
import socketio
from pathlib import Path, PurePath
import json
from copy import deepcopy
from blinker import Signal

import mimetypes
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit, FitDriver, fit
from bumps.serialize import to_dict, from_dict
from bumps.mapper import MPMapper
from bumps.parameter import Parameter, Variable, unique
import bumps.fitproblem
import bumps.plotutil
import bumps.dream.views, bumps.dream.varplot, bumps.dream.stats
import bumps.errplot
import refl1d.errors
import refl1d.fitproblem, refl1d.probe
from refl1d.experiment import Experiment

# Register the refl1d model loader
import refl1d.fitplugin
import bumps.cli
bumps.cli.install_plugin(refl1d.fitplugin)

from .fit_thread import FitThread, EVT_FIT_COMPLETE, EVT_FIT_PROGRESS
from .profile_plot import plot_sld_profile_plotly
from .varplot import plot_vars

# can get by name and not just by id
EVT_LOG = Signal('log')
MODEL_EXT = '.json'

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTERS_BY_ID = dict([(fitter.id, fitter) for fitter in FITTERS])
print(FITTERS_BY_ID)
FITTER_DEFAULTS = {}
for fitter in FITTERS:
    FITTER_DEFAULTS[fitter.id] = {
        "name": fitter.name,
        "settings": dict(fitter.settings)
    }


routes = web.RouteTableDef()
# sio = socketio.AsyncServer(cors_allowed_origins="*", serializer='msgpack')
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
client_path = Path(__file__).parent.parent / 'client'
index_path = client_path / 'dist'
static_assets_path = index_path / 'assets'

sio.attach(app)

TopicName = Literal[
    "log",
    "update_parameters",
    "update_model",
    "model_loaded",
    "fit_active",
    "uncertainty_update",
    "convergence_update",
    "fitter_settings",
    "fitter_active",
]
topics: Dict[TopicName, "deque[Dict]"] = {
    "log": deque([]),
    "update_parameters": deque([], maxlen=1),
    "update_model": deque([], maxlen=1),
    "model_loaded": deque([], maxlen=1),
    "fit_active": deque([], maxlen=1),
    "convergence_update": deque([], maxlen=1),
    "uncertainty_update": deque([], maxlen=1),
    "fitter_settings": deque([], maxlen=1),
    "fitter_active": deque([], maxlen=1),
}
app["topics"] = topics
app["problem"] = {"fitProblem": None, "pathlist": None, "filename": None}
app["fitting"] = {
    "fit_thread": None,
    "abort": False,
    "uncertainty_state": None,
    "population": None,
    "abort_queue": Queue(),
}
problem = None


def rest_get(fn):
    """
    Add a REST (GET) route for the function, which can also be used for 
    """
    @routes.get(f"/{fn.__name__}")
    async def handler(request: web.Request):
        result = await fn(**request.query)
        return web.json_response(result)
    
    # pass the function to the next decorator unchanged...
    return fn

async def index(request):
    """Serve the client-side application."""
    # check if the locally-build site has the correct version:
    with open(client_path / 'package.json', 'r') as package_json:
        client_version = json.load(package_json)['version'].strip()

    try:
        local_version = open(index_path / 'VERSION', 'rt').read().strip()
    except FileNotFoundError:
        local_version = None

    print(index_path, local_version, client_version, local_version == client_version)
    if client_version == local_version:
        return web.FileResponse(index_path / 'index.html')
    else:
        CDN = f"https://cdn.jsdelivr.net/npm/refl1d-webview-client@{client_version}/dist"
        with open(client_path / 'index_template.txt', 'r') as index_template:
            index_html = index_template.read().format(cdn=CDN)
        return web.Response(body=index_html, content_type="text/html")
    
@sio.event
async def connect(sid, environ, data=None):
    # re-send last message for all topics
    # now that panels are retrieving topics when they load, is this still
    # needed or useful?
    for topic, contents in topics.items():
        message = contents[-1] if len(contents) > 0 else None
        if message is not None:
            await sio.emit(topic, message, to=sid)
    print("connect ", sid)

@sio.event
async def load_problem_file(sid: str, pathlist: List[str], filename: str):
    path = Path(*pathlist, filename)
    print(f'model loading: {path}')
    log_handler({"message": f'model_loading: {path}'})
    if filename.endswith(".json"):
        with open(path, "rt") as input_file:
            serialized = json.loads(input_file.read())
        problem = from_dict(serialized)
    else:
        from bumps.cli import load_model
        problem = load_model(str(path))
    app["problem"]["fitProblem"] = problem
    app["problem"]["pathlist"] = pathlist
    app["problem"]["filename"] = filename
    print(f'model loaded: {path}')
    log_handler({"message": f'model loaded: {path}'})


    model_names = [getattr(m, 'name', None) for m in list(problem.models)]
    await publish("", "model_loaded", {"pathlist": pathlist, "filename": filename, "model_names": model_names})
    await publish("", "update_model", True)
    await publish("", "update_parameters", True)

@sio.event
async def save_problem_file(sid: str, pathlist: Optional[List[str]] = None, filename: Optional[str] = None, overwrite: bool = False):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        print("Save failed: no problem loaded.")
        return
    if pathlist is None:
        pathlist = app["problem"]["pathlist"]
    if filename is None:
        filename = app["problem"]["filename"]

    if pathlist is None or filename is None:
        print("no filename and path provided to save")
        return

    path = Path(*pathlist)
    save_filename = Path(filename).stem + MODEL_EXT
    print({"path": path, "filename": save_filename})
    if not overwrite and Path.exists(path / save_filename):
        #confirmation needed:
        return True

    serialized = to_dict(fitProblem)
    with open(Path(path, save_filename), "wt") as output_file:
        output_file.write(json.dumps(serialized))

    log_handler({"message": f'Saved: {filename} at path: {path}'})

@sio.event
async def start_fit(sid: str="", fitter_id: str="", kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    mapper = MPMapper.start_mapper(fitProblem, None, cpus=0)
    monitors = []
    fitclass = FITTERS_BY_ID[fitter_id]
    driver = FitDriver(fitclass=fitclass, mapper=mapper, problem=fitProblem, monitors=monitors, **kwargs)
    x, fx = driver.fit()
    driver.show()

@sio.event
async def stop_fit(sid: str = ""):
    abort_queue: Queue = app["fitting"]["abort_queue"]
    abort_queue.put(True)

@sio.event
async def start_fit_thread(sid: str="", fitter_id: str="", options=None):
    options = {} if options is None else options
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    fitclass = FITTERS_BY_ID[fitter_id]
    if app["problem"].get("fit_thread", None) is not None:
        # warn that fit is alread running...
        print("fit already running...")
        return
    
    # TODO: better access to model parameters
    if len(fitProblem.getp()) == 0:
        raise ValueError("Problem has no fittable parameters")

    # Start a new thread worker and give fit problem to the worker.
    # Clear abort and uncertainty state
    app["fitting"]["abort"] = False
    app["fitting"]["uncertainty_state"] = None
    abort_queue: Queue = app["fitting"]["abort_queue"]
    for _ in range(abort_queue.qsize()):
        abort_queue.get_nowait()
        abort_queue.task_done()
    fit_thread = FitThread(
        abort_queue=abort_queue,
        problem=fitProblem,
        fitclass=fitclass,
        options=options,
        # Number of seconds between updates to the GUI, or 0 for no updates
        convergence_update=5,
        uncertainty_update=3600,
        )
    fit_thread.start()
    await publish("", "fit_active", to_dict(dict(fitter_id=fitter_id, options=options)))
    log_handler({"message": json.dumps(to_dict(options), indent=2), "title": f"starting fitter {fitter_id}"})

def fit_progress_handler(event: Dict):
    print("progress event message: ", event.get("message", ""))
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    message = event.get("message", None)
    if message == 'complete' or message == 'improvement':
        fitProblem.setp(event["point"])
        fitProblem.model_update()
        publish_sync("", "update_parameters")
        if message == 'complete':
            publish_sync("", "fit_active", False)
    elif message == 'convergence_update':
        app["fitting"]["population"] = event["pop"]
        publish_sync("", "convergence_update")
    elif message == 'uncertainty_update' or message == 'uncertainty_final':
        app["fitting"]["uncertainty_state"] = event["uncertainty_state"]
        publish_sync("", "uncertainty_update")

EVT_FIT_PROGRESS.connect(fit_progress_handler)

def fit_complete_handler(event):
    print("complete event: ", event.get("message", ""))
    message = event.get("message", None)
    fit_thread = app["fitting"]["fit_thread"]
    if fit_thread is not None:
        fit_thread.join(1) # 1 second timeout on join
        if fit_thread.is_alive():
            EVT_LOG.send({"message": "fit thread failed to complete"})
    app["fitting"]["fit_thread"] = None
    problem: refl1d.fitproblem.FitProblem = event["problem"]
    chisq = nice(2*event["value"]/problem.dof)
    problem.setp(event["point"])
    problem.model_update()
    publish_sync("", "update_parameters", True)
    EVT_LOG.send({"message": event["info"], "title": f"done with chisq {chisq}"})

EVT_FIT_COMPLETE.connect(fit_complete_handler)

def log_handler(message):
    publish_sync("", "log", message)

EVT_LOG.connect(log_handler)

def get_single_probe_data(theory, probe, substrate=None, surface=None, label=''):
    fresnel_calculator = probe.fresnel(substrate, surface)
    Q, FQ = probe.apply_beam(probe.calc_Q, fresnel_calculator(probe.calc_Q))
    Q, R = theory
    output: Dict[str, Union[str, np.ndarray]]
    assert isinstance(FQ, np.ndarray)
    if len(Q) != len(probe.Q):
        # Saving interpolated data
        output = dict(Q = Q, theory = R, fresnel=np.interp(Q, probe.Q, FQ))
    elif getattr(probe, 'R', None) is not None:
        output = dict(Q = probe.Q, dQ = probe.dQ, R = probe.R, dR = probe.dR, theory = R, fresnel = FQ, background=probe.background.value, intensity=probe.intensity.value)
    else:
        output = dict(Q = probe.Q, dQ = probe.dQ, theory = R, fresnel = FQ)
    output['label'] = f"{probe.label()} {label}"
    return output

def get_probe_data(theory, probe, substrate=None, surface=None):
    if isinstance(probe, refl1d.probe.PolarizedNeutronProbe):
        output = []
        for xsi, xsi_th, suffix in zip(probe.xs, theory, ('--', '-+', '+-', '++')):
            if xsi is not None:
                output.append(get_single_probe_data(xsi_th, xsi, substrate, surface, suffix))
        return output
    else:
        return [get_single_probe_data(theory, probe, substrate, surface)]

@sio.event
@rest_get
async def get_plot_data(sid: str="", view: str = 'linear'):
    # TODO: implement view-dependent return instead of doing this in JS
    # (calculate x,y,dy.dx for given view, excluding log)
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    result = []
    for model in fitProblem.models:
        assert(isinstance(model, Experiment))
        theory = model.reflectivity()
        probe = model.probe
        result.append(get_probe_data(theory, probe, model._substrate, model._surface))
        # fresnel_calculator = probe.fresnel(model._substrate, model._surface)
        # Q, FQ = probe.apply_beam(probe.calc_Q, fresnel_calculator(probe.calc_Q))
        # Q, R = theory
        # assert isinstance(FQ, np.ndarray)
        # if len(Q) != len(probe.Q):
        #     # Saving interpolated data
        #     output = dict(Q = Q, R = R, fresnel=np.interp(Q, probe.Q, FQ))
        # elif getattr(probe, 'R', None) is not None:
        #     output = dict(Q = probe.Q, dQ = probe.dQ, R = probe.R, dR = probe.dR, fresnel = FQ)
        # else:
        #     output = dict(Q = probe.Q, dQ = probe.dQ, R = R, fresnel = FQ)
        # result.append(output)
    return to_dict(result)

@sio.event
@rest_get
async def get_model(sid: str=""):
    from bumps.serialize import to_dict
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    return to_dict(fitProblem)

@sio.event
@rest_get
async def get_profile_plot(sid: str="", model_index: int=0):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return None
    models = list(fitProblem.models)
    if model_index > len(models):
        return None
    model = models[model_index]
    assert (isinstance(model, Experiment))

    # data = np.random.random((100,200))
    # x = np.arange(0, 10, 0.1)
    # y = np.random.random(x.shape) + x
    # import plotly.express as px
    # fig = px.imshow(data)
    # fig.update_yaxes(exponentformat='e',
    #                  title={'text': r"$\frac{I}{I_0}$"})
    # fig = px.line(x=x, y=y)

    fig = plot_sld_profile_plotly(model)

    # print(to_dict(fig.to_dict()))

    return to_dict(fig.to_dict())

@sio.event
@rest_get
async def get_profile_data(sid: str="", model_index: int=0):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return None
    models = list(fitProblem.models)
    if (model_index > len(models)):
        return None
    model = models[model_index]
    assert(isinstance(model, Experiment))
    output = {}
    output["ismagnetic"] = model.ismagnetic
    if model.ismagnetic:
        if not model.step_interfaces:
            z, rho, irho, rhoM, thetaM = model.magnetic_step_profile()
            output['step_profile'] = dict(z=z, rho=rho, irho=irho, rhoM=rhoM, thetaM=thetaM)
        z, rho, irho, rhoM, thetaM = model.magnetic_smooth_profile()
        output['smooth_profile'] = dict(z=z, rho=rho, irho=irho, rhoM=rhoM, thetaM=thetaM)
    else:
        if not model.step_interfaces:
            z, rho, irho = model.step_profile()
            output['step_profile'] = dict(z=z, rho=rho, irho=irho)
        z, rho, irho = model.smooth_profile()
        output['smooth_profile'] = dict(z=z, rho=rho, irho=irho)
    return to_dict(output)

@sio.event
@rest_get
async def get_convergence_plot(sid: str=""):
    # NOTE: this is slow.  Creating the figure takes around 0.15 seconds, 
    # and converting to mpld3 can take as much as 0.5 seconds.
    # Might want to replace with plotting on the client side (normalizing population takes around 1 ms)
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    population = app["fitting"]["population"]
    if population is not None and fitProblem is not None:
        import mpld3
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import time
        start_time = time.time()
        print('queueing new convergence plot...', start_time)

        normalized_pop = 2*population/fitProblem.dof
        best, pop = normalized_pop[:, 0], normalized_pop[:, 1:]
        print("time to normalize population: ", time.time() - start_time)
        fig = plt.figure()
        axes = fig.add_subplot(111)

        ni,npop = pop.shape
        iternum = np.arange(1,ni+1)
        tail = int(0.25*ni)
        c = bumps.plotutil.coordinated_colors(base=(0.4,0.8,0.2))
        if npop==5:
            axes.fill_between(iternum[tail:], pop[tail:,1], pop[tail:,3],
                                color=c['light'], label='_nolegend_')
            axes.plot(iternum[tail:],pop[tail:,2],
                        label="80% range", color=c['base'])
            axes.plot(iternum[tail:],pop[tail:,0],
                        label="_nolegend_", color=c['base'])
        axes.plot(iternum[tail:], best[tail:], label="best",
                    color=c['dark'])
        axes.set_xlabel('iteration number')
        axes.set_ylabel('chisq')
        axes.legend()
        #plt.gca().set_yscale('log')
        # fig.draw()
        print("time to render but not serialize...", time.time() - start_time)
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        # await sio.emit("profile_plot", dfig, to=sid)
        end_time = time.time()
        print("time to draw convergence plot:", end_time - start_time)
        return dfig
    else:
        return None

@sio.event
@rest_get
async def get_correlation_plot(sid: str = ""):
    uncertainty_state = app["fitting"].get("uncertainty_state", None)
    if uncertainty_state is not None:
        import mpld3
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import time
        start_time = time.time()
        print('queueing new correlation plot...', start_time)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        bumps.dream.views.plot_corrmatrix(uncertainty_state.draw(), fig=fig)
        print("time to render but not serialize...", time.time() - start_time)
        fig.canvas.draw()
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        # await sio.emit("profile_plot", dfig, to=sid)
        end_time = time.time()
        print("time to draw correlation plot:", end_time - start_time)
        return dfig
    else:
        return None

@sio.event
@rest_get
async def get_uncertainty_plot(sid: str = ""):
    uncertainty_state = app["fitting"].get("uncertainty_state", None)
    if uncertainty_state is not None:
        import time
        start_time = time.time()
        print('queueing new uncertainty plot...', start_time)
        draw = uncertainty_state.draw()
        stats = bumps.dream.stats.var_stats(draw)
        fig = plot_vars(draw, stats)
        return to_dict(fig.to_dict())
    else:
        return None

@sio.event
@rest_get
async def get_model_uncertainty_plot(sid: str = ""):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    uncertainty_state = app["fitting"].get("uncertainty_state", None)
    if uncertainty_state is not None:
        import mpld3
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import time
        start_time = time.time()
        print('queueing new model uncertainty plot...', start_time)

        fig = plt.figure()
        errs = bumps.errplot.calc_errors_from_state(fitProblem, uncertainty_state)
        print('errors calculated: ', time.time() - start_time)
        bumps.errplot.show_errors(errs, fig=fig)
        print("time to render but not serialize...", time.time() - start_time)
        fig.canvas.draw()
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        # await sio.emit("profile_plot", dfig, to=sid)
        end_time = time.time()
        print("time to draw model uncertainty plot:", end_time - start_time)
        return dfig
    else:
        return None

@sio.event
@rest_get
async def get_parameter_trace_plot(sid: str=""):
    uncertainty_state = app["fitting"].get("uncertainty_state", None)
    if uncertainty_state is not None:
        import mpld3
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import time

        start_time = time.time()
        print('queueing new parameter_trace plot...', start_time)

        fig = plt.figure()
        axes = fig.add_subplot(111)

        # begin plotting:
        var = 0
        portion = None
        draw, points, _ = uncertainty_state.chains()
        label = uncertainty_state.labels[var]
        start = int((1-portion)*len(draw)) if portion else 0
        genid = np.arange(uncertainty_state.generation-len(draw)+start, uncertainty_state.generation)+1
        axes.plot(genid*uncertainty_state.thinning,
             np.squeeze(points[start:, uncertainty_state._good_chains, var]))
        axes.set_xlabel('Generation number')
        axes.set_ylabel(label)
        fig.canvas.draw()

        print("time to render but not serialize...", time.time() - start_time)
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        # await sio.emit("profile_plot", dfig, to=sid)
        end_time = time.time()
        print("time to draw parameter_trace plot:", end_time - start_time)
        return dfig
    else:
        return None
    

@sio.event
@rest_get
async def get_parameters(sid: str = "", only_fittable: bool = False):
    fitProblem: refl1d.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return []

    all_parameters = fitProblem.model_parameters()
    if only_fittable:
        parameter_infos = params_to_list(unique(all_parameters))
        # only include params with priors:
        parameter_infos = [pi for pi in parameter_infos if pi['fittable'] and not pi['fixed']]
    else:
        parameter_infos = params_to_list(all_parameters)
        
    return to_dict(parameter_infos)

@sio.event
async def set_parameter(sid: str, parameter_id: str, property: Literal["value01", "value", "min", "max"], value: Union[float, str, bool]):
    fitProblem: bumps.fitproblem.FitProblem = app["problem"]["fitProblem"]
    if fitProblem is None:
        return

    parameter = fitProblem._parameters_by_id.get(parameter_id, None)
    if parameter is None:
        warnings.warn(f"Attempting to update parameter that doesn't exist: {parameter_id}")
        return

    if parameter.prior is None:
        warnings.warn(f"Attempting to set prior properties on parameter without priors: {parameter}")
        return

    if property == "value01":
        new_value  = parameter.prior.put01(value)
        nice_new_value = nice(new_value, digits=VALUE_PRECISION)
        parameter.clip_set(nice_new_value)
    elif property == "value":
        new_value = float(value)
        nice_new_value = nice(new_value, digits=VALUE_PRECISION)
        parameter.clip_set(nice_new_value)
    elif property == "min":
        lo = float(value)
        hi = parameter.prior.limits[1]
        parameter.range(lo, hi)
        parameter.add_prior()
    elif property == "max":
        lo = parameter.prior.limits[0]
        hi = float(value)
        parameter.range(lo, hi)
        parameter.add_prior()
    elif property == "fixed":
        if parameter.fittable:
            parameter.fixed = bool(value)
            fitProblem.model_reset()
            # print(f"setting parameter: {parameter}.fixed to {value}")
    fitProblem.model_update()
    await publish("", "update_parameters", True)
    return

@sio.event
async def publish(sid: str, topic: str, message=None):
    timestamp_str = f"{datetime.now().timestamp():.6f}"
    contents = {"message": message, "timestamp": timestamp_str}
    topics[topic].append(contents)
    await sio.emit(topic, contents)
    # print("emitted: ", topic, contents)

def publish_sync(sid: str, topic: str, message=None):
    return asyncio.run_coroutine_threadsafe(publish(sid, topic, message), app.loop)

@sio.event
@rest_get
async def get_topic_messages(sid: str="", topic: str="", max_num=None) -> List[Dict]:
    # this is a GET request in disguise -
    # emitter must handle the response in a callback,
    # as no separate response event is emitted.
    q = topics.get(topic, None)
    if q is None:
        raise ValueError(f"Topic: {topic} not defined")
    elif max_num is None:
        return list(q)
    else:
        q_length = len(q)
        start = max(q_length - max_num, 0)
        return list(itertools.islice(q, start, q_length))

@sio.event
@rest_get
async def get_dirlisting(sid: str="", pathlist: Optional[List[str]]=None):
    # GET request
    # TODO: use psutil to get disk listing as well?
    if pathlist is None:
        pathlist = []
    subfolders = []
    files = []
    for p in Path(*pathlist).iterdir():
        if p.is_dir():
            subfolders.append(p.name)
        else:
            # files.append(p.resolve().name)
            files.append(p.name)
    return dict(subfolders=subfolders, files=files)

@sio.event
@rest_get
async def get_current_pathlist(sid: str="") -> List[str]:
    return list(app['problem']['pathlist'])

@sio.event
@rest_get
async def get_fitter_defaults(sid: str=""):
    return FITTER_DEFAULTS

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

 
@sio.event
async def shutdown(sid: str=""):
    await stop_fit()
    # disconnect all clients:
    clients = list(sio.manager.rooms.get('/', {None: {}}).get(None).keys())
    for client in clients:
        await sio.disconnect(client)
    import signal
    import sys
    # signal.raise_signal(signal.SIGINT)
    await app.shutdown()
    await app.cleanup()
    print("killing...")
    signal.raise_signal(signal.SIGKILL)

if Path.exists(static_assets_path):
    app.router.add_static('/assets', static_assets_path)
app.router.add_get('/', index)

VALUE_PRECISION = 6
VALUE_FORMAT = "{{:.{:d}g}}".format(VALUE_PRECISION)

def nice(v, digits=4):
    """Fix v to a value with a given number of digits of precision"""
    from math import log10, floor
    if v == 0. or not np.isfinite(v):
        return v
    else:
        sign = v/abs(v)
        place = floor(log10(abs(v)))
        scale = 10**(place-(digits-1))
        return sign*floor(abs(v)/scale+0.5)*scale
    

class ParamInfo(TypedDict):
    id: str
    name: str
    paths: List[str]
    value_str: str
    fittable: bool
    fixed: bool
    writable: bool
    value01: Optional[float]
    min_str: Optional[str]
    max_str: Optional[str]


def params_to_list(params, lookup=None, pathlist=None, links=None) -> List[ParamInfo]:
    lookup: Dict[str, ParamInfo] = {} if lookup is None else lookup
    pathlist = [] if pathlist is None else pathlist
    if isinstance(params,dict):
        for k in sorted(params.keys()):
            params_to_list(params[k], lookup=lookup, pathlist=pathlist + [k])
    elif isinstance(params, tuple) or isinstance(params, list):
        for i, v in enumerate(params):
            params_to_list(v, lookup=lookup, pathlist=pathlist + [f"[{i:d}]"])
    elif isinstance(params, Parameter):
        path = ".".join(pathlist)
        existing = lookup.get(params.id, None)
        if existing is not None:
            existing["paths"].append(".".join(pathlist))
        else:
            value_str = VALUE_FORMAT.format(nice(params.value))
            has_prior = params.has_prior()
            new_item: ParamInfo = { 
                "id": params.id,
                "name": str(params.name),
                "paths": [path],
                "writable": type(params.slot) in [Variable, Parameter], 
                "value_str": value_str, "fittable": params.fittable, "fixed": params.fixed }
            if has_prior:
                assert(params.prior is not None)
                lo, hi = params.prior.limits
                new_item['value01'] = params.prior.get01(params.value)
                new_item['min_str'] = VALUE_FORMAT.format(nice(lo))
                new_item['max_str'] = VALUE_FORMAT.format(nice(hi))
            lookup[params.id] = new_item
    return list(lookup.values())

import argparse

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--debug', action='store_true', help='autoload modules on change')
    parser.add_argument('-x', '--headless', action='store_true', help='do not automatically load client in browser')
    parser.add_argument('--external', action='store_true', help='listen on all interfaces, including external (local connections only if not set)')
    parser.add_argument('-p', '--port', default=0, type=int, help='port on which to start the server')
    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    args = parser.parse_args()

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    # set initial path to cwd:
    app['problem']['pathlist'] = Path().absolute().parts
    app.add_routes(routes)
    hostname = 'localhost' if not args.external else '0.0.0.0'

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((hostname, args.port))
    _, port = sock.getsockname()
    if not args.headless:
        import webbrowser
        webbrowser.open_new_tab(f"http://{hostname}:{port}/")
    web.run_app(app, sock=sock)

if __name__ == '__main__':
    main()
