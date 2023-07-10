import itertools
from types import GeneratorType
from typing import Any, Callable, Coroutine, Dict, List, Literal, Mapping, Optional, Sequence, Union, Tuple, TypedDict, cast
from datetime import datetime
import numbers
import warnings
from queue import Queue
import numpy as np
import asyncio
from pathlib import Path, PurePath
import json
from copy import deepcopy
import sys

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit, FitDriver, fit, nllf_scale, format_uncertainty
from bumps.mapper import MPMapper
from bumps.parameter import Parameter, Variable, unique
import bumps.cli
import bumps.fitproblem
import bumps.plotutil
import bumps.dream.views, bumps.dream.varplot, bumps.dream.stats, bumps.dream.state
import bumps.errplot

from .state_hdf5_backed import State, serialize, deserialize, SERIALIZER_EXTENSIONS
from .fit_thread import FitThread, EVT_FIT_COMPLETE, EVT_FIT_PROGRESS
from .varplot import plot_vars

REGISTRY: Dict[str, Callable] = {}
MODEL_EXT = '.json'
TRACE_MEMORY = False
CAN_THREAD = sys.platform != 'emscripten'

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTERS_BY_ID = dict([(fitter.id, fitter) for fitter in FITTERS])
# print(FITTERS_BY_ID)
FITTER_DEFAULTS = {}
for fitter in FITTERS:
    FITTER_DEFAULTS[fitter.id] = {
        "name": fitter.name,
        "settings": dict(fitter.settings)
    }

state = State()
app: Any

def register(fn: Callable):
    REGISTRY[fn.__name__] = fn
    return fn

async def emit(
    event: Any,
    data: Optional[Any] = None,
    to: Optional[Any] = None,
    room: Optional[Any] = None,
    skip_sid: Optional[Any] = None,
    namespace: Optional[Any] = None,
    callback: Optional[Any] = None,
    ignore_queue: bool = False
) -> Coroutine[Any, Any, None] :
    # to be defined when initializing server
    pass

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

@register
async def load_problem_file(pathlist: List[str], filename: str):    
    path = Path(*pathlist, filename)
    print(f'model loading: {path}')
    await log(f'model_loading: {path}')
    if filename.endswith(".json"):
        with open(path, "rt") as input_file:
            serialized = input_file.read()
        problem = deserialize(serialized, method='dataclass')
    else:
        from bumps.cli import load_model
        problem = load_model(str(path))
    assert isinstance(problem, bumps.fitproblem.FitProblem)
    # problem_state = ProblemState(problem, pathlist, filename)
    state.problem.filename = filename
    state.problem.pathlist = pathlist
    await set_problem(problem, Path(*pathlist), filename)

async def set_problem(problem: bumps.fitproblem.FitProblem, path: Optional[Path] = None, filename: str = ""):
    state.problem.fitProblem = problem
    pathlist = list(path.parts) if path is not None else []
    path_string = "(no path)" if path is None else str(path / filename)
    print(f'model loaded: {path_string}')
    await log(f'model loaded: {path_string}')
    await publish("update_model", True)
    await publish("update_parameters", True)
    await publish("model_loaded", {"pathlist": pathlist, "filename": filename})


@register
async def save_problem_file(pathlist: Optional[List[str]] = None, filename: Optional[str] = None, overwrite: bool = False):
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
        return {"filename": "", "check_overwrite": False}

    path = Path(*pathlist)
    serializer = state.problem.serializer
    extension = SERIALIZER_EXTENSIONS[serializer]
    save_filename = f"{Path(filename).stem}.{extension}"
    if not overwrite and Path.exists(path / save_filename):
        #confirmation needed:
        return {"filename": save_filename, "check_overwrite": True}

    serialized = serialize(problem_state.fitProblem, method=serializer)
    with open(Path(path, save_filename), "wb") as output_file:
        output_file.write(serialized)

    await log(f'Saved: {save_filename} at path: {path}')
    return {"filename": save_filename, "check_overwrite": False}

@register
async def export_results(export_path: Union[str, List[str]]=""):
    from concurrent.futures import ThreadPoolExecutor

    problem_state = state.problem
    if problem_state is None:
        print("Save failed: no problem loaded.")
        return

    problem = deepcopy(problem_state.fitProblem)
    # TODO: if making a temporary copy of the uncertainty state is going to cause memory
    # issues, we could try to copy and then fall back to just using the live object,
    # or we could just always use the live object, which is unlikely to be changed before
    # the export completes, anyway.
    uncertainty_state = deepcopy(state.fitting.uncertainty_state)

    if not isinstance(export_path, list):
        export_path = [export_path]
    path = Path(*export_path)
    await emit("export_started", str(path))
    if CAN_THREAD:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_export_results, path, problem, uncertainty_state)
            await asyncio.wrap_future(future)
    else:
        _export_results(path, problem, uncertainty_state)
    await emit("export_completed", str(path))


def _export_results(path: Path, problem: bumps.fitproblem.FitProblem, uncertainty_state: Optional[bumps.dream.state.MCMCDraw]):
    from bumps.util import redirect_console

    basename = problem.name if problem.name is not None else "problem"
    # Storage directory
    path.mkdir(parents=True, exist_ok=True)
    output_pathstr = str( path / basename )

    # Ask model to save its information
    problem.save(output_pathstr)

    # Save a snapshot of the model that can (hopefully) be reloaded
    serializer = state.problem.serializer
    extension = SERIALIZER_EXTENSIONS[serializer]
    save_filename = f"{output_pathstr}.{extension}"
    serialized = serialize(problem, serializer)
    with open(save_filename, "wb") as output_file:
        output_file.write(serialized)

    # Save the current state of the parameters
    with redirect_console(str(path / f"{basename}.out")):
        problem.show()

    pardata = "".join("%s %.15g\n"%(name, value) for name, value in
                        zip(problem.labels(), problem.getp()))

    open( path / f"{basename}.par",'wt').write(pardata)

    # Produce model plots
    problem.plot(figfile = output_pathstr)

    # Produce uncertainty plots
    if uncertainty_state is not None:
        with redirect_console( str(path / f"{basename}.err")):
            uncertainty_state.show(figfile = output_pathstr)
        uncertainty_state.save(output_pathstr)

@register
async def apply_parameters(pathlist: List[str], filename: str):
    path = Path(*pathlist)
    fullpath = path / filename
    try:
        bumps.cli.load_best(state.problem.fitProblem, fullpath)
        await publish("update_parameters", True)
        await log(f"applied parameters from {fullpath}")
    except Exception:
        await log(f"unable to apply parameters from {fullpath}")


@register
async def start_fit(fitter_id: str="", kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    problem_state = state.problem
    if problem_state is None:
        await log("Error: Can't start fit if no problem loaded")
    else:
        fitProblem = problem_state.fitProblem
        mapper = MPMapper.start_mapper(fitProblem, None, cpus=state.parallel)
        monitors = []
        fitclass = FITTERS_BY_ID[fitter_id]
        driver = FitDriver(fitclass=fitclass, mapper=mapper, problem=fitProblem, monitors=monitors, **kwargs)
        x, fx = driver.fit()
        driver.show()

@register
async def stop_fit():
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

def get_running_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None

@register
async def start_fit_thread(fitter_id: str="", options=None, terminate_on_finish=False):
    options = {} if options is None else options    # session_id: str = app["active_session"]
    fitProblem = state.problem.fitProblem if state.problem is not None else None
    if fitProblem is None:
        await log("Error: Can't start fit if no problem loaded")
    else:
        state.calling_loop = get_running_loop()
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
            parallel=state.parallel,
            # session_id=session_id,
            # Number of seconds between updates to the GUI, or 0 for no updates
            convergence_update=5,
            uncertainty_update=3600,
            terminate_on_finish=terminate_on_finish,
            )
        await emit("fit_progress", {}) # clear progress
        await publish("fit_active", to_json_compatible_dict(dict(fitter_id=fitter_id, options=options, num_steps=num_steps)))
        await log(json.dumps(to_json_compatible_dict(options), indent=2), title = f"starting fitter {fitter_id}")
        if CAN_THREAD:
            fit_thread.start()
        else:
            fit_thread.run()
        state.fit_thread = fit_thread

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
        await publish("update_parameters", True)
        if message == 'complete':
            await publish("fit_active", {})
    elif message == 'convergence_update':
        state.fitting.population = event["pop"]
        await publish("convergence_update", True)
    elif message == 'progress':
        await emit("fit_progress", to_json_compatible_dict(event))
    elif message == 'uncertainty_update' or message == 'uncertainty_final':
        state.fitting.uncertainty_state = cast(bumps.dream.state.MCMCDraw, event["uncertainty_state"])
        await publish("uncertainty_update", True)

async def _fit_complete_handler(event):
    print("complete event: ", event.get("message", ""))
    message = event.get("message", None)
    fit_thread = state.fit_thread
    terminate = False
    if fit_thread is not None:
        print(fit_thread)
        terminate = fit_thread.terminate_on_finish
        if CAN_THREAD:
            fit_thread.join(1) # 1 second timeout on join
            if fit_thread.is_alive():
                await log("fit thread failed to complete")
    state.fit_thread = None
    problem: bumps.fitproblem.FitProblem = event["problem"]
    chisq = nice(2*event["value"]/problem.dof)
    problem.setp(event["point"])
    problem.model_update()
    state.problem.fitProblem = problem
    await publish("fit_active", {})
    await publish("update_parameters", True)
    await log(event["info"], title=f"done with chisq {chisq}")
    fut = state.fit_stopped_future
    if fut is not None:
        fut.set_result(True)
    if terminate:
        await shutdown()

def fit_progress_handler(event: Dict):
    loop = getattr(state, 'calling_loop', None)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_fit_progress_handler(event), loop)

def fit_complete_handler(event: Dict):
    loop = getattr(state, 'calling_loop', None)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_fit_complete_handler(event), loop)

EVT_FIT_PROGRESS.connect(fit_progress_handler, weak=True)
EVT_FIT_COMPLETE.connect(fit_complete_handler, weak=True)

async def log(message: str, title: Optional[str] = None):
    await publish("log", {"message": message, "title": title})

@register
async def get_data_plot():
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
    end_time = time.time()
    print("time to draw data plot:", end_time - start_time)
    return {"fig_type": "mpld3", "plotdata": dfig}
    

@register
async def get_model():
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    serialized = serialize(fitProblem, 'dataclass') if state.problem.serializer == 'dataclass' else 'null'
    return serialized

@register
async def get_convergence_plot():
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

@register
async def get_correlation_plot(sort: bool=True, max_rows: int=8, nbins: int=50):
    from .corrplot import Corr2d
    uncertainty_state = state.fitting.uncertainty_state

    if isinstance(uncertainty_state, bumps.dream.state.MCMCDraw):
        import time
        start_time = time.time()
        print('queueing new correlation plot...', start_time)
        draw = uncertainty_state.draw()
        c = Corr2d(draw.points.T, bins=nbins, labels=draw.labels)
        fig = c.plot(sort=sort, max_rows=max_rows)
        print("time to render but not serialize...", time.time() - start_time)
        serialized = to_json_compatible_dict(fig.to_dict())
        end_time = time.time()
        print("time to draw correlation plot:", end_time - start_time)
        return serialized
    else:
        return None

@register
async def get_uncertainty_plot():
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

@register
async def get_model_uncertainty_plot():
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
        end_time = time.time()
        print("time to draw model uncertainty plot:", end_time - start_time)
        return dfig
    else:
        return None

@register
async def get_parameter_trace_plot():
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
        end_time = time.time()
        print("time to draw parameter_trace plot:", end_time - start_time)
        return dfig
    else:
        return None
    

@register
async def get_parameters(only_fittable: bool = False):
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

@register
async def set_parameter(parameter_id: str, property: Literal["value01", "value", "min", "max"], value: Union[float, str, bool]):
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
            await publish("update_model", True)
    fitProblem.model_update()
    await publish("update_parameters", True)
    return

@register
async def publish(topic: TopicNameType, message=None):
    timestamp_str = f"{datetime.now().timestamp():.6f}"
    contents = {"message": message, "timestamp": timestamp_str}
    # session = get_session(session_id)    
    state.topics[topic].append(contents)
    # if session_id == app["active_session"]:
    await emit(topic, contents)
    # print("emitted: ", topic, contents)


@register
async def get_topic_messages(topic: Optional[TopicNameType] = None, max_num=None) -> List[Dict]:
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

@register
async def get_dirlisting(pathlist: Optional[List[str]]=None):
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

@register
async def get_current_pathlist() -> Optional[List[str]]:
    problem_state = state.problem
    pathlist = problem_state.pathlist if problem_state is not None else []
    return pathlist

@register
async def get_fitter_defaults(*args):
    return FITTER_DEFAULTS

@register
async def shutdown():
    print("killing...")
    await stop_fit()
    await emit("server_shutting_down")
    shutdown_result = asyncio.gather(_shutdown(), return_exceptions=True)

async def _shutdown():
    raise SystemExit(0)

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

JSON_TYPE = Union[str, float, bool, None, Sequence['JSON_TYPE'], Mapping[str, 'JSON_TYPE']]

def to_json_compatible_dict(obj) -> JSON_TYPE:
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
    elif isinstance(obj, bool) or isinstance(obj, str) or obj is None:
        return obj
    elif isinstance(obj, numbers.Number):
        return str(obj) if np.isinf(obj) else float(obj)
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
            # add index to last item in pathlist (in-place):
            new_pathlist = pathlist.copy()
            if len(pathlist) < 1:
                new_pathlist.append("")
            new_pathlist[-1] = f"{new_pathlist[-1]}[{i:d}]"
            params_to_list(v, lookup=lookup, pathlist=new_pathlist)
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
                "tags": params.tags,
                "writable": type(params.slot) in [Variable, Parameter], 
                "value_str": value_str, "fittable": params.fittable, "fixed": params.fixed }
            if has_prior:
                assert(params.prior is not None)
                lo, hi = params.prior.limits
                new_item['value01'] = params.prior.get01(params.value)
                new_item['min_str'] = VALUE_FORMAT.format(nice(lo))
                new_item['max_str'] = VALUE_FORMAT.format(nice(hi))
            lookup[params.id] = new_item
    elif callable(getattr(params, 'parameters', None)):
        # handle Expression, Constant, etc.
        subparams = params.parameters()
        params_to_list(subparams, lookup=lookup, pathlist=pathlist)
    return list(lookup.values())
