# from .main import setup_bumps

from dataclasses import dataclass
import itertools
import threading
import signal
import socket
from types import GeneratorType
from typing import Callable, Dict, List, Literal, Optional, Union, Tuple, TypedDict, cast
from datetime import datetime
import warnings
from queue import Queue
from collections import deque
from aiohttp import web, ClientSession
import numpy as np
import asyncio
import socketio
from pathlib import Path, PurePath
import json
from copy import deepcopy
from blinker import Signal
from uuid import uuid4

import mimetypes
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit, FitDriver, fit, nllf_scale, format_uncertainty
from bumps.serialize import to_dict as serialize, from_dict_threaded as deserialize
from bumps.mapper import MPMapper
from bumps.parameter import Parameter, Variable, unique
import bumps.fitproblem
import bumps.plotutil
import bumps.dream.views, bumps.dream.varplot, bumps.dream.stats, bumps.dream.state
import bumps.errplot

from .fit_thread import FitThread, EVT_FIT_COMPLETE, EVT_FIT_PROGRESS

### BEGIN PATCH
# patch the plotly library to disable levenshtein lookup for missing strings
# which is taking a ridiculous amount of time in make_subplots
# See: https://github.com/plotly/plotly.py/issues/4100
def disable_find_closest_string(string, strings):
    raise ValueError()

import _plotly_utils.utils
_plotly_utils.utils.find_closest_string = disable_find_closest_string
### END PATCH

from .varplot import plot_vars
from .state_hdf5_backed import SERIALIZERS, State
# from .state import State

TRACE_MEMORY = False

# can get by name and not just by id
MODEL_EXT = '.json'

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTERS_BY_ID = dict([(fitter.id, fitter) for fitter in FITTERS])
# print(FITTERS_BY_ID)
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

TopicNameType = Literal[
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

# will be replaced with file-backed state during startup, if needed:
state = State()

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
        CDN = f"https://cdn.jsdelivr.net/npm/bumps-webview-client@{client_version}/dist"
        with open(client_path / 'index_template.txt', 'r') as index_template:
            index_html = index_template.read().format(cdn=CDN)
        return web.Response(body=index_html, content_type="text/html")
    
@sio.event
async def connect(sid, environ, data=None):
    # re-send last message for all topics
    # now that panels are retrieving topics when they load, is this still
    # needed or useful?
    for topic, contents in state.topics.items():
        message = contents[-1] if len(contents) > 0 else None
        if message is not None:
            await sio.emit(topic, message, to=sid)
    print("connect ", sid)

@sio.event
async def load_problem_file(sid: str, pathlist: List[str], filename: str):    
    path = Path(*pathlist, filename)
    print(f'model loading: {path}')
    await log(f'model_loading: {path}')
    if filename.endswith(".json"):
        with open(path, "rt") as input_file:
            serialized = json.loads(input_file.read())
        problem = deserialize(serialized)
    else:
        from bumps.cli import load_model
        problem = load_model(str(path))
    assert isinstance(problem, bumps.fitproblem.FitProblem)
    # problem_state = ProblemState(problem, pathlist, filename)
    state.problem.filename = filename
    state.problem.pathlist = pathlist
    await set_problem(problem, path)
    await publish("", "model_loaded", {"pathlist": pathlist, "filename": filename})

async def set_problem(problem: bumps.fitproblem.FitProblem, path: Optional[Path] = None):
    state.problem.fitProblem = problem
    path_string = "(no path)" if path is None else str(path)
    print(f'model loaded: {path_string}')
    await log(f'model loaded: {path_string}')
    await publish("", "update_model", True)
    await publish("", "update_parameters", True)

@sio.event
async def save_problem_file(sid: str, pathlist: Optional[List[str]] = None, filename: Optional[str] = None, overwrite: bool = False):
    problem_state = state.problem
    if problem_state is None:
        print("Save failed: no problem loaded.")
        return
    if pathlist is None:
        pathlist = problem_state.pathlist
    if filename is None:
        filename = problem_state.filename

    if pathlist is None or filename is None:
        print("no filename and path provided to save")
        return

    path = Path(*pathlist)
    save_filename = Path(filename).stem + MODEL_EXT
    print({"path": path, "filename": save_filename})
    if not overwrite and Path.exists(path / save_filename):
        #confirmation needed:
        return save_filename

    serialized = serialize(problem_state.fitProblem)
    with open(Path(path, save_filename), "wt") as output_file:
        output_file.write(json.dumps(serialized))

    await log(f'Saved: {filename} at path: {path}')
    return False

@sio.event
async def start_fit(sid: str="", fitter_id: str="", kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    problem_state = state.problem
    if problem_state is None:
        await log("Error: Can't start fit if no problem loaded")
    else:
        fitProblem = problem_state.fitProblem
        mapper = MPMapper.start_mapper(fitProblem, None, cpus=0)
        monitors = []
        fitclass = FITTERS_BY_ID[fitter_id]
        driver = FitDriver(fitclass=fitclass, mapper=mapper, problem=fitProblem, monitors=monitors, **kwargs)
        x, fx = driver.fit()
        driver.show()

@sio.event
async def stop_fit(sid: str = ""):
    if state.fit_thread is not None:
        if state.fit_stopped_future is None:
            loop = asyncio.get_running_loop()
            state.fit_stopped_future = loop.create_future()
        state.abort_queue.put(True)
        await state.fit_stopped_future
        state.fit_stopped_future = None



def get_chisq(problem: bumps.fitproblem.FitProblem, nllf=None):
    nllf = problem.nllf() if nllf is None else nllf
    scale, err = nllf_scale(problem)
    chisq = format_uncertainty(scale*nllf, err)
    return chisq

def get_num_steps(fitter_id: str, num_fitparams: int, options: Optional[Dict] = None):
    options = FITTER_DEFAULTS[fitter_id] if options is None else options
    steps = options['steps']
    if fitter_id == 'dream' and steps == 0:
        print('dream: ', options)
        total_pop = options['pop'] * num_fitparams
        print('total_pop: ', total_pop)
        sample_steps = int(options['samples'] / total_pop)
        print('sample_steps: ', sample_steps)
        print('steps: ', options['burn'] + sample_steps)
        return options['burn'] + sample_steps
    else:
        return steps

@sio.event
async def start_fit_thread(sid: str="", fitter_id: str="", options=None, terminate_on_finish=False):
    options = {} if options is None else options    # session_id: str = app["active_session"]
    fitProblem = state.problem.fitProblem if state.problem is not None else None
    if fitProblem is None:
        await log("Error: Can't start fit if no problem loaded")
    else:
        fit_state = state.fitting
        fitclass = FITTERS_BY_ID[fitter_id]
        if state.fit_thread is not None:
            # warn that fit is alread running...
            print("fit already running...")
            await log("Can't start fit, a fit is already running...")
            return
        
        # TODO: better access to model parameters
        num_params = len(fitProblem.getp())
        if num_params == 0:
            raise ValueError("Problem has no fittable parameters")

        # Start a new thread worker and give fit problem to the worker.
        # Clear abort and uncertainty state
        # state.abort = False
        # state.fitting.uncertainty_state = None
        num_steps = get_num_steps(fitter_id, num_params, options)
        state.abort_queue = abort_queue = Queue()
        fit_thread = FitThread(
            abort_queue=abort_queue,
            problem=fitProblem,
            fitclass=fitclass,
            options=options,
            # session_id=session_id,
            # Number of seconds between updates to the GUI, or 0 for no updates
            convergence_update=5,
            uncertainty_update=3600,
            terminate_on_finish=terminate_on_finish,
            )
        fit_thread.start()
        state.fit_thread = fit_thread
        await sio.emit("fit_progress", {}) # clear progress
        await publish("", "fit_active", to_json_compatible_dict(dict(fitter_id=fitter_id, options=options, num_steps=num_steps)))
        await log(json.dumps(to_json_compatible_dict(options), indent=2), title = f"starting fitter {fitter_id}")

async def _fit_progress_handler(event: Dict):
    # session_id = event["session_id"]
    if TRACE_MEMORY:
        import tracemalloc
        if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("memory use:")
                for stat in top_stats[:15]:
                    print(stat)
                
    problem_state = state.problem
    fitProblem = problem_state.fitProblem if problem_state is not None else None
    if fitProblem is None:
        raise ValueError("should never happen: fit progress reported for session in which fitProblem is undefined")
    message = event.get("message", None)
    if message == 'complete' or message == 'improvement':
        fitProblem.setp(event["point"])
        fitProblem.model_update()
        await publish("", "update_parameters", True)
        if message == 'complete':
            await publish("", "fit_active", {})
    elif message == 'convergence_update':
        state.fitting.population = event["pop"]
        await publish("", "convergence_update", True)
    elif message == 'progress':
        await sio.emit("fit_progress", to_json_compatible_dict(event))
    elif message == 'uncertainty_update' or message == 'uncertainty_final':
        state.fitting.uncertainty_state = cast(bumps.dream.state.MCMCDraw, event["uncertainty_state"])
        await publish("", "uncertainty_update", True)

def fit_progress_handler(event: Dict):
    asyncio.run_coroutine_threadsafe(_fit_progress_handler(event), app.loop)


async def _fit_complete_handler(event):
    print("complete event: ", event.get("message", ""))
    message = event.get("message", None)
    fit_thread = state.fit_thread
    terminate = False
    if fit_thread is not None:
        print(fit_thread)
        terminate = fit_thread.terminate_on_finish
        fit_thread.join(1) # 1 second timeout on join
        if fit_thread.is_alive():
            await log("fit thread failed to complete")
    state.fit_thread = None
    problem: bumps.fitproblem.FitProblem = event["problem"]
    chisq = nice(2*event["value"]/problem.dof)
    problem.setp(event["point"])
    problem.model_update()
    state.problem.fitProblem = problem
    await publish("", "fit_active", {})
    await publish("", "update_parameters", True)
    await log(event["info"], title=f"done with chisq {chisq}")
    fut = state.fit_stopped_future
    if fut is not None:
        fut.set_result(True)
    if terminate:
        await shutdown()

def fit_complete_handler(event: Dict):
    asyncio.run_coroutine_threadsafe(_fit_complete_handler(event), app.loop)

async def log(message: str, title: Optional[str] = None):
    await publish("", "log", {"message": message, "title": title})

@sio.event
async def get_data_plot(sid: str=""):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    import mpld3
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    print('queueing new data plot...', start_time)
    fig = plt.figure()
    fitProblem.plot()
    dfig = mpld3.fig_to_dict(fig)
    plt.close(fig)
    # await sio.emit("profile_plot", dfig, to=sid)
    end_time = time.time()
    print("time to draw data plot:", end_time - start_time)
    return {"fig_type": "mpld3", "plotdata": dfig}
    

@sio.event
@rest_get
async def get_model(sid: str=""):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    return serialize(fitProblem)

@sio.event
@rest_get
async def get_convergence_plot(sid: str=""):
    # NOTE: this is slow.  Creating the figure takes around 0.15 seconds, 
    # and converting to mpld3 can take as much as 0.5 seconds.
    # Might want to replace with plotting on the client side (normalizing population takes around 1 ms)
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    population = state.fitting.population
    if population is not None:
        import plotly.graph_objects as go

        normalized_pop = 2*population/fitProblem.dof
        best, pop = normalized_pop[:, 0], normalized_pop[:, 1:]

        ni,npop = pop.shape
        iternum = np.arange(1,ni+1)
        tail = int(0.25*ni)

        c = bumps.plotutil.coordinated_colors(base=(0.4,0.8,0.2))
        cp = dict((k, f"rgba({','.join([str(int(cmp*255)) for cmp in cc])})") for k,cc in c.items())
        fig = go.Figure()
        if npop==5:
            fig.add_trace(go.Scatter(x=iternum[tail:], y=pop[tail:,3], mode="lines", line_width=0, showlegend=False))
            fig.add_trace(go.Scatter(x=iternum[tail:], y=pop[tail:,1], fill="tonexty", mode="lines", line_color=cp["light"], line_width=0, showlegend=False))
            fig.add_trace(go.Scatter(x=iternum[tail:], y=pop[tail:,2], name="80% range", mode="lines", line_color=cp["base"]))

        fig.add_trace(go.Scatter(x=iternum[tail:], y=best[tail:], name="best", line_color=cp["dark"], mode="lines"))
        fig.update_layout(template="simple_white", legend=dict(x=1,y=1,xanchor="right",yanchor="top"))
        fig.update_layout(title=dict(text="Convergence", xanchor="center", x=0.5))
        fig.update_xaxes(title="iteration number")
        fig.update_yaxes(title="chisq")
        return to_json_compatible_dict(fig.to_dict())
    else:
        return None

@sio.event
@rest_get
async def get_correlation_plot(sid: str = "", nbins: int=50):
    from .corrplot import Corr2d
    uncertainty_state = state.fitting.uncertainty_state

    if isinstance(uncertainty_state, bumps.dream.state.MCMCDraw):
        import time
        start_time = time.time()
        print('queueing new correlation plot...', start_time)
        draw = uncertainty_state.draw()
        c = Corr2d(draw.points.T, bins=nbins, labels=draw.labels)
        fig = c.plot()
        print("time to render but not serialize...", time.time() - start_time)
        serialized = to_json_compatible_dict(fig.to_dict())
        end_time = time.time()
        print("time to draw correlation plot:", end_time - start_time)
        return serialized
    else:
        return None

@sio.event
@rest_get
async def get_uncertainty_plot(sid: str = ""):
    uncertainty_state = state.fitting.uncertainty_state
    if uncertainty_state is not None:
        import time
        start_time = time.time()
        print('queueing new uncertainty plot...', start_time)
        draw = uncertainty_state.draw()
        stats = bumps.dream.stats.var_stats(draw)
        fig = plot_vars(draw, stats)
        return to_json_compatible_dict(fig.to_dict())
    else:
        return None

@sio.event
@rest_get
async def get_model_uncertainty_plot(sid: str = ""):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    uncertainty_state = state.fitting.uncertainty_state
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
    uncertainty_state = state.fitting.uncertainty_state
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
    if state.problem is None or state.problem.fitProblem is None:
        return []
    fitProblem = state.problem.fitProblem

    all_parameters = fitProblem.model_parameters()
    if only_fittable:
        parameter_infos = params_to_list(unique(all_parameters))
        # only include params with priors:
        parameter_infos = [pi for pi in parameter_infos if pi['fittable'] and not pi['fixed']]
    else:
        parameter_infos = params_to_list(all_parameters)
        
    return to_json_compatible_dict(parameter_infos)

@sio.event
async def set_parameter(sid: str, parameter_id: str, property: Literal["value01", "value", "min", "max"], value: Union[float, str, bool]):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem

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
            # model has been changed: setp and getp will return different values!
            await publish("", "update_model", True)
    fitProblem.model_update()
    await publish("", "update_parameters", True)
    return

@sio.event
async def publish(sid: str, topic: TopicNameType, message=None):
    timestamp_str = f"{datetime.now().timestamp():.6f}"
    contents = {"message": message, "timestamp": timestamp_str}
    # session = get_session(session_id)    
    state.topics[topic].append(contents)
    # if session_id == app["active_session"]:
    #     await sio.emit(topic, contents)
    await sio.emit(topic, contents)
    # print("emitted: ", topic, contents)


@sio.event
@rest_get
async def get_topic_messages(sid: str="", topic: Optional[TopicNameType] = None, max_num=None) -> List[Dict]:
    # this is a GET request in disguise -
    # emitter must handle the response in a callback,
    # as no separate response event is emitted.
    if topic is None:
        return []    
    topics = state.topics
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
async def get_current_pathlist(sid: str="") -> Optional[List[str]]:
    problem_state = state.problem
    pathlist = problem_state.pathlist if problem_state is not None else None
    return pathlist

@sio.event
@rest_get
async def get_fitter_defaults(sid: str=""):
    return FITTER_DEFAULTS

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

async def disconnect_all_clients():
    # disconnect all clients:
    clients = list(sio.manager.rooms.get('/', {None: {}}).get(None).keys())
    for client in clients:
        await sio.disconnect(client)

@sio.event
@rest_get
async def shutdown(sid: str=""):
    print("killing...")
    await stop_fit()
    await sio.emit("server_shutting_down")
    signal.raise_signal(signal.SIGTERM)

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


def to_json_compatible_dict(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_json_compatible_dict(v) for v in obj)
    elif isinstance(obj, GeneratorType):
        return list(to_json_compatible_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((to_json_compatible_dict(k), to_json_compatible_dict(v))
                        for k, v in obj.items())
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in ['f', 'i']:
        return obj.tolist()
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
        return to_json_compatible_dict(obj.tolist())
    elif isinstance(obj, float):
        return str(obj) if np.isinf(obj) else obj
    elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise ValueError("obj %s is not serializable" % str(obj))


class ParamInfo(TypedDict, total=False):
    id: str
    name: str
    paths: List[str]
    value_str: str
    fittable: bool
    fixed: bool
    writable: bool
    value01: float
    min_str: str
    max_str: str


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

@dataclass
class Options:
    """ provide type hints for arguments """
    filename: Optional[str] = None
    headless: bool = False
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    fit: Optional[str] = None
    start: bool = False
    store: Optional[str] = None
    exit: bool = False
    serializer: SERIALIZERS = "dill"
    trace: bool = False
    zeroconf: bool = False


def get_commandline_options(arg_defaults: Optional[Dict]=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', help='problem file to load, .py or .json (serialized) fitproblem')
    # parser.add_argument('-d', '--debug', action='store_true', help='autoload modules on change')
    parser.add_argument('-x', '--headless', action='store_true', help='do not automatically load client in browser')
    parser.add_argument('--external', action='store_true', help='listen on all interfaces, including external (local connections only if not set)')
    parser.add_argument('-p', '--port', default=0, type=int, help='port on which to start the server')
    parser.add_argument('--hub', default=None, type=str, help='api address of parent hub (only used when called as subprocess)')
    parser.add_argument('--fit', default=None, type=str, choices=list(FITTERS_BY_ID.keys()), help='fitting engine to use; see manual for details')
    parser.add_argument('--start', action='store_true', help='start fit when problem loaded')
    parser.add_argument('--store', default=None, type=str, help='backing file for state')
    parser.add_argument('--exit', action='store_true', help='end process when fit complete (fit results lost unless store is specified)')
    parser.add_argument('--serializer', default='dill', type=str, choices=["pickle", "dill", "dataclass"], help='strategy for serializing problem, will use value from store if it has already been defined')
    parser.add_argument('--trace', action='store_true', help='enable memory tracing (prints after every uncertainty update in dream)')
    parser.add_argument('--zeroconf', action='store_true', help='register with local zeroconf')
    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    if arg_defaults is not None:
        parser.set_defaults(**arg_defaults)
    args = parser.parse_args(namespace=Options())
    return args


def setup_app(index: Callable=index, static_assets_path: Path=static_assets_path, sock: Optional[socket.socket] = None, options: Options = Options()):
    if Path.exists(static_assets_path):
        app.router.add_static('/assets', static_assets_path)
    app.router.add_get('/', index)

    if options.store is not None:
        state.setup_backing(session_file_name=options.store, in_memory=False)

    if state.problem.serializer is None:
        state.problem.serializer = options.serializer

    if options.trace:
        global TRACE_MEMORY
        TRACE_MEMORY = True

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    if options.fit is not None:
        app.on_startup.append(lambda App: publish('', 'fitter_active', options.fit))

    fitter_id = options.fit
    if fitter_id is None:
        fitter_active_topic = state.topics["fitter_active"]
        if len(fitter_active_topic) > 0:
            fitter_id = fitter_active_topic[-1]["message"]
    if fitter_id is None:
        fitter_id = 'amoeba'
    fitter_settings = FITTER_DEFAULTS[fitter_id]

    # if args.steps is not None:
    #     fitter_settings["steps"] = args.steps

    if options.filename is not None:
        filepath = Path(options.filename)
        pathlist = list(filepath.parent.parts)
        filename = filepath.name
        start = options.start
        print(f"fitter for filename {filename} is {fitter_id}")
        async def startup_task(App=None):
            await load_problem_file("", pathlist, filename)
            if start:
                await start_fit_thread("", fitter_id, fitter_settings, options.exit)
        
        app.on_startup.append(startup_task)

        # app.on_startup.append()

    # app.on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    async def notice(message: str):
        print(message)
    app.on_cleanup.append(lambda App: notice("cleanup task"))
    app.on_shutdown.append(lambda App: notice("shutdown task"))
    app.on_shutdown.append(lambda App: stop_fit())
    app.on_shutdown.append(lambda App: disconnect_all_clients())
    app.on_shutdown.append(lambda App: state.cleanup())
    # set initial path to cwd:
    state.problem.pathlist = list(Path().absolute().parts)
    app.add_routes(routes)
    hostname = 'localhost' if not options.external else '0.0.0.0'

    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((hostname, options.port))
    host, port = sock.getsockname()
    state.hostname = host
    state.port = port
    if options.hub is not None:
        async def register_instance(application: web.Application):
            async with ClientSession() as client_session:
                await client_session.post(options.hub, json={"host": hostname, "port": port})
        app.on_startup.append(register_instance)
    if not options.headless:
        import webbrowser
        async def open_browser(app: web.Application):
            app.loop.call_later(0.25, lambda: webbrowser.open_new_tab(f"http://{hostname}:{port}/"))
        app.on_startup.append(open_browser)

    if options.zeroconf:
        from .zeroconf_registry import LocalRegistry
        registry = LocalRegistry()
        async def register_zeroconf(app: web.Application):
            return await registry.register(port, "Bumps")
        app.on_startup.append(register_zeroconf)
        app.on_shutdown.append(lambda App: registry.close())

    if TRACE_MEMORY:
        import tracemalloc
        tracemalloc.start()

    EVT_FIT_PROGRESS.connect(fit_progress_handler)
    EVT_FIT_COMPLETE.connect(fit_complete_handler)

    return sock

def main(options: Optional[Options] = None, sock: Optional[socket.socket] = None):
    options = get_commandline_options() if options is None else options
    asyncio.run(start_app(options, sock))

async def start_app(options: Options, sock: socket.socket):
    runsock = setup_app(options=options, sock=sock)
    await web._run_app(app, sock=runsock)

if __name__ == '__main__':
    main()
