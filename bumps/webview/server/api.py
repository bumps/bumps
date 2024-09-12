from functools import lru_cache
import itertools
from types import GeneratorType
from typing import Any, Callable, Coroutine, Dict, List, Literal, Mapping, Optional, Sequence, Union, Tuple, TypedDict, cast
from datetime import datetime
import numbers
import warnings
from threading import Event
import numpy as np
import asyncio
from pathlib import Path, PurePath
import json
from copy import deepcopy
import os
import uuid

from bumps.fitters import DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit, FitDriver, fit, nllf_scale, format_uncertainty
from bumps.mapper import MPMapper
from bumps.parameter import Parameter, Constant, Variable, unique
import bumps.cli
import bumps.fitproblem
import bumps.dream.views, bumps.dream.varplot, bumps.dream.stats, bumps.dream.state
import bumps.errplot

from .state_hdf5_backed import UNDEFINED, UNDEFINED_TYPE, State, serialize_problem, deserialize_problem, SERIALIZER_EXTENSIONS
from .fit_thread import FitThread, EVT_FIT_COMPLETE, EVT_FIT_PROGRESS
from .varplot import plot_vars
from .logger import logger, console_handler


REGISTRY: Dict[str, Callable] = {}
MODEL_EXT = '.json'
TRACE_MEMORY = False

FITTERS = (DreamFit, LevenbergMarquardtFit, SimplexFit, DEFit, MPFit, BFGSFit)
FITTERS_BY_ID = dict([(fitter.id, fitter) for fitter in FITTERS])
FITTER_DEFAULTS = {}
for fitter in FITTERS:
    FITTER_DEFAULTS[fitter.id] = {
        "name": fitter.name,
        "settings": dict(fitter.settings)
    }

state = State()

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
    "log", # log messages
]

@register
async def load_problem_file(pathlist: List[str], filename: str, autosave_previous: bool = True):
    path = Path(*pathlist, filename)
    logger.info(f'model loading: {path}')
    await log(f'model_loading: {path}')
    if filename.endswith(".json"):
        with open(path, "rt") as input_file:
            serialized = input_file.read()
        problem = deserialize_problem(serialized, method='dataclass')
    else:
        from bumps.cli import load_model
        problem = load_model(str(path))
    assert isinstance(problem, bumps.fitproblem.FitProblem)
    # problem_state = ProblemState(problem, pathlist, filename)
    try:
        _serialized_problem = serialize_problem(problem, method='dataclass')
        state.problem.serializer = 'dataclass'
    except Exception as exc:
        logger.info(f"Could not serialize problem as JSON (dataclass): {exc}, switching to dill")
        state.problem.serializer = 'dill'
    if (state.shared.autosave_history and
        autosave_previous and
        state.problem is not None and
        state.problem.fitProblem is not None):
        await save_to_history("autosaved before loading new model")
    state.shared.model_file = dict(filename=filename, pathlist=pathlist)
    state.shared.model_loaded = now_string()
    await set_problem(problem, Path(*pathlist), filename)


@register
async def set_serialized_problem(serialized, new_model: bool = False, name: Optional[str] = None):
    fitProblem = deserialize_problem(serialized, method='dataclass')
    state.problem.serializer = 'dataclass'
    await set_problem(fitProblem, new_model=new_model, name=name)
 

async def set_problem(problem: bumps.fitproblem.FitProblem, path: Optional[Path] = None, filename: str = "", new_model: bool = True, name: Optional[str] = None):
    if state.problem is None or state.problem.fitProblem is None:
        update = False
    state.problem.fitProblem = problem
    name = name if name is not None else problem.name
    if name is None:
        name = filename
    state.shared.updated_model = now_string()
    state.shared.updated_parameters = now_string()

    if new_model:
        pathlist = list(path.parts) if path is not None else []
        path_string = "(no path)" if path is None else str(path / filename)
        await log(f'model loaded: {path_string}')
        state.shared.model_file = dict(filename=filename, pathlist=pathlist)
        state.shared.model_loaded = now_string()
        if (state.shared.autosave_history and
            state.problem is not None and
            state.problem.fitProblem is not None):
            await save_to_history(f"Loaded model: {name}", keep=True)
        await add_notification(content=path_string, title="Model loaded", timeout=2000)
    state.autosave()


@register
async def get_history():
    return state.get_history()


@register
async def remove_history_item(timestamp: str):
    state.remove_history_item(timestamp)
    state.shared.updated_history = now_string()


@register
async def save_to_history(label: str, include_population: bool = False, include_uncertainty: bool = False, keep: bool = False):
    state.save_to_history(label, include_population=include_population, include_uncertainty=include_uncertainty, keep=keep)

@register
async def reload_history_item(timestamp: str):
    state.reload_history_item(timestamp)


@register
async def set_keep_history(timestamp: str, keep: bool):
    state.history.set_keep(timestamp, keep)
    state.shared.updated_history = now_string()

@register
async def update_history_label(timestamp: str, label: str):
    state.history.update_label(timestamp, label)
    state.shared.updated_history = now_string()


@register
async def save_problem_file(pathlist: Optional[List[str]] = None, filename: Optional[str] = None, overwrite: bool = False):
    problem_state = state.problem
    if problem_state is None:
        logger.warning("Save failed: no problem loaded.")
        return
    if pathlist is None:
        pathlist = problem_state.pathlist
    if filename is None:
        filename = problem_state.filename

    if pathlist is None or filename is None:
        logger.warning("no filename and path provided to save")
        return {"filename": "", "check_overwrite": False}

    path = Path(*pathlist)
    serializer = state.problem.serializer
    extension = SERIALIZER_EXTENSIONS[serializer]
    save_filename = f"{Path(filename).stem}.{extension}"
    if not overwrite and Path.exists(path / save_filename):
        #confirmation needed:
        return {"filename": save_filename, "check_overwrite": True}

    serialized = serialize_problem(problem_state.fitProblem, method=serializer)
    with open(Path(path, save_filename), "wb") as output_file:
        output_file.write(serialized)

    await log(f'Saved: {save_filename} at path: {path}')
    return {"filename": save_filename, "check_overwrite": False}

@register
async def save_session():
    state.save()

@register
async def save_session_copy(pathlist: List[str], filename: str):
    path = Path(*pathlist)
    state.write_session_file(str(path / filename))

@register
async def load_session(pathlist: List[str], filename: str, read_only: bool = False):
    path = Path(*pathlist)
    state.setup_backing(filename, pathlist, read_only=read_only)
    state.shared.updated_model = now_string()
    state.shared.updated_parameters = now_string()

@register
async def set_session_output_file(pathlist: Optional[List[str]] = None, filename: Optional[str] = None):
    if filename is None or pathlist is None:
        await state.shared.set('session_output_file', None)
    else:
        await state.shared.set('session_output_file', dict(filename=filename, pathlist=pathlist))

@register
async def get_serializer():
    output = {"serializer": "", "extension": ""}
    problem_state = state.problem
    if problem_state is not None:
        serializer = problem_state.serializer
        output["serializer"] = serializer
        output["extension"] = SERIALIZER_EXTENSIONS[serializer]
    return output

@register
async def export_results(export_path: Union[str, List[str]]=""):
    from concurrent.futures import ThreadPoolExecutor

    problem_state = state.problem
    if problem_state is None:
        logger.warning("Save failed: no problem loaded.")
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
    notification_id = await add_notification(content=f"<span>{str(path)}</span>", title="Export started", timeout=None)
    try:
        await asyncio.to_thread(_export_results, path, problem, uncertainty_state)
    finally:
        await emit("cancel_notification", notification_id)


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
    try:
        serialized = serialize_problem(problem, serializer)
        with open(save_filename, "wb") as output_file:
            output_file.write(serialized)
    except Exception as exc:
        logger.error(f"Error exporting model: {exc}")

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
async def save_parameters(pathlist: List[str], filename: str, overwrite: bool = False):
    problem_state = state.problem
    if problem_state is None:
        await log("Error: Can't save parameters if no problem loaded")
        return
    problem = problem_state.fitProblem
    path = Path(*pathlist)
    if not overwrite and Path.exists(path / filename):
        #confirmation needed:
        return {"filename": filename, "check_overwrite": True}

    pardata = "".join("%s %.15g\n" % (name, value)
                      for name, value in zip(problem.labels(), problem.getp()))
    
    with open( path / filename, "wt") as pars_file:
        pars_file.write(pardata)
    return {"filename": filename, "check_overwrite": False}

@register
async def apply_parameters(pathlist: List[str], filename: str):
    path = Path(*pathlist)
    fullpath = path / filename
    try:
        bumps.cli.load_best(state.problem.fitProblem, fullpath)
        state.shared.updated_parameters = now_string()
        await log(f"applied parameters from {fullpath}")
        await add_notification(f"applied parameters from {fullpath}", title="Parameters applied", timeout=2000)
    except Exception:
        await log(f"unable to apply parameters from {fullpath}")
        await add_notification(f"unable to apply parameters from {fullpath}", title="Error applying parameters", timeout=2000)


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
        if state.fit_thread.is_alive():
            state.fit_abort_event.set()
            loop = getattr(state, 'calling_loop', None)
            if loop is not None:
                await loop.run_in_executor(None, state.fit_complete_event.wait)
    else:
        state.shared.active_fit = {}

@register
async def get_chisq(problem: Optional[bumps.fitproblem.FitProblem]=None, nllf=None):
    problem = state.problem.fitProblem if problem is None else problem
    if problem is None:
        return ''
    nllf = problem.nllf() if nllf is None else nllf
    scale, err = nllf_scale(problem)
    chisq = format_uncertainty(scale*nllf, err)
    return chisq

def get_num_steps(fitter_id: str, num_fitparams: int, options: Optional[Dict] = None):
    options = FITTER_DEFAULTS[fitter_id] if options is None else options
    steps = options['steps']
    if fitter_id == 'dream':
        if steps == 0:
            total_pop = options['pop'] * num_fitparams
            steps = int(options['samples'] / total_pop)
        steps += options['burn']
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
            logger.warning("fit already running...")
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
        state.fit_abort_event.clear()
        state.fit_complete_event.clear()
        state.fit_uncertainty_final.clear()

        fit_thread = FitThread(
            abort_event=state.fit_abort_event,
            problem=fitProblem,
            fitclass=fitclass,
            options=options,
            parallel=state.parallel,
            # session_id=session_id,
            # Number of seconds between updates to the GUI, or 0 for no updates
            convergence_update=5,
            uncertainty_update=state.shared.autosave_session_interval,
            terminate_on_finish=terminate_on_finish,
            )
        await emit("fit_progress", {}) # clear progress
        state.shared.active_fit = to_json_compatible_dict(dict(fitter_id=fitter_id, options=options, num_steps=num_steps))
        await log(json.dumps(to_json_compatible_dict(options), indent=2), title = f"starting fitter {fitter_id}")
        state.autosave()
        fit_thread.start()
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
        state.shared.updated_parameters = now_string()
        if message == 'complete':
            state.shared.active_fit = {}
    elif message == 'convergence_update':
        state.fitting.population = event["pop"]
        state.shared.updated_convergence = now_string()
    elif message == 'progress':
        await emit("fit_progress", to_json_compatible_dict(event))
    elif message == 'uncertainty_update' or message == 'uncertainty_final':
        state.fitting.uncertainty_state = cast(bumps.dream.state.MCMCDraw, event["uncertainty_state"])
        state.shared.updated_uncertainty = now_string()
        state.autosave()
        if message == 'uncertainty_final':
            # fit is not complete until uncertainty is saved.
            state.fit_uncertainty_final.set()

async def _fit_complete_handler(event):
    message = event.get("message", None)
    fit_thread = state.fit_thread
    terminate = False
    if fit_thread is not None:
        terminate = fit_thread.terminate_on_finish
        fit_thread.join(1) # 1 second timeout on join
        if fit_thread.is_alive():
            await log("fit thread failed to complete")
    state.fit_thread = None
    state.shared.active_fit = {}
    if message == "error":
        await log(event["traceback"], title=f"fit failed with error: {event.get('error_string')}")
        logger.warning(f"fit failed with error: {event.get('error_string')}\n{event['traceback']}")
    else:
        problem: bumps.fitproblem.FitProblem = event["problem"]
        chisq = nice(2*event["value"]/problem.dof)
        problem.setp(event["point"])
        problem.model_update()
        state.problem.fitProblem = problem
        if state.shared.autosave_history:
            await save_to_history(f"fit complete: {event['fitter_id']}", include_population=True, include_uncertainty=True)
        state.shared.updated_parameters = now_string()
        await log(event["info"], title=f"done with chisq {chisq}")
        logger.info(f"fit done with chisq {chisq}")

    if fit_thread.fitclass.id == 'dream':
        # print("waiting for uncertainty to complete...")
        await asyncio.to_thread(state.fit_uncertainty_final.wait)

    state.fit_complete_event.set()

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
    topic = "log"
    contents = {"message": {"message": message, "title": title}, "timestamp": now_string()}
    # session = get_session(session_id)    
    state.topics[topic].append(contents)
    # if session_id == app["active_session"]:
    await emit(topic, contents)

@register
async def get_data_plot(model_indices: Optional[List[int]] = None):
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = deepcopy(state.problem.fitProblem)
    import mpld3
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    logger.info(f'queueing new data plot... {start_time}')
    fig = plt.figure()
    for i, model in enumerate(fitProblem.models):
        if model_indices is not None and i not in model_indices:
            continue
        model.plot()
    plt.text(0.01, 0.01,
             'chisq=%s' % fitProblem.chisq_str(),
             transform=plt.gca().transAxes)
    dfig = mpld3.fig_to_dict(fig)
    plt.close(fig)
    end_time = time.time()
    logger.info(f"time to draw data plot: {end_time - start_time}")
    return {"fig_type": "mpld3", "plotdata": dfig}

@register
async def get_model_names():
    problem = state.problem.fitProblem
    if problem is None:
        return None
    return [p.name if p.name is not None else f"model_{i}" for (i,p) in enumerate(problem.models)]

@register
async def get_model():
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    serialized = serialize_problem(fitProblem, 'dataclass') if state.problem.serializer == 'dataclass' else 'null'
    return serialized

@register
async def get_convergence_plot():
    if state.problem is None or state.problem.fitProblem is None:
        return None
    fitProblem = state.problem.fitProblem
    population = state.fitting.population
    if population is not None:
        normalized_pop = 2*population/fitProblem.dof
        best, pop = normalized_pop[:, 0], normalized_pop[:, 1:]

        ni,npop = pop.shape
        iternum = np.arange(1,ni+1)
        tail = int(0.25*ni)
        fig = {"data": [], "layout": {}} # go.Figure()
        hovertemplate = "(%{{x}}, %{{y}})<br>{label}<extra></extra>"
        x = iternum[tail:].tolist()
        if npop==5:
            # fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,4].tolist(), name="95%", mode="lines", line=dict(color="lightblue", width=1), showlegend=True, hovertemplate=hovertemplate.format(label="95%")))
            fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,3].tolist(), mode="lines", line=dict(color="lightgreen", width=0), showlegend=False, hovertemplate=hovertemplate.format(label="80%")))
            fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,1].tolist(), name="20% to 80% range", fill="tonexty", mode="lines", line=dict(color="lightgreen", width=0), hovertemplate=hovertemplate.format(label="20%")))
            fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,2].tolist(), name="population median", mode="lines", line=dict(color="green"), opacity=0.5))
            fig['data'].append(dict(type="scattergl", x=x, y=pop[tail:,0].tolist(), name="population best", mode="lines", line=dict(color="green", dash="dot")))

        fig['data'].append(dict(type="scattergl", x=x, y=best[tail:].tolist(), name="best", line=dict(color="red", width=1), mode="lines"))
        fig['layout'].update(dict(template="simple_white", legend=dict(x=1,y=1,xanchor="right",yanchor="top")))
        fig['layout'].update(dict(title=dict(text="Convergence", xanchor="center", x=0.5)))
        fig['layout'].update(dict(uirevision=1))
        fig['layout'].update(dict(xaxis=dict(title="iteration number", showline=True, showgrid=False, zeroline=False)))
        fig['layout'].update(dict(yaxis=dict(title="chisq", showline=True, showgrid=False, zeroline=False)))
        return to_json_compatible_dict(fig)
    else:
        return None

@lru_cache(maxsize=30)
def _get_correlation_plot(sort: bool=True, max_rows: int=8, nbins: int=50, vars=None, timestamp: str=""):
    from .corrplot import Corr2d
    uncertainty_state = state.fitting.uncertainty_state

    if isinstance(uncertainty_state, bumps.dream.state.MCMCDraw):
        import time
        start_time = time.time()
        logger.info(f'queueing new correlation plot... {start_time}')
        draw = uncertainty_state.draw(vars=vars)
        c = Corr2d(draw.points.T, bins=nbins, labels=draw.labels)
        fig = c.plot(sort=sort, max_rows=max_rows)
        logger.info(f"time to render but not serialize... {time.time() - start_time}")
        serialized = to_json_compatible_dict(fig.to_dict())
        end_time = time.time()
        logger.info(f"time to draw correlation plot: {end_time - start_time}")
        return serialized
    else:
        return None

@register
async def get_correlation_plot(sort: bool=True, max_rows: int=8, nbins: int=50, vars=None, timestamp: str=""):
    # need vars to be immutable (hashable) for caching based on arguments:
    vars = tuple(vars) if vars is not None else None
    result = await asyncio.to_thread(_get_correlation_plot, sort=sort, max_rows=max_rows, nbins=nbins, vars=vars, timestamp=timestamp)
    return result

@lru_cache(maxsize=30)
def _get_uncertainty_plot(timestamp: str="", cbar_colors: int=8):
    uncertainty_state = state.fitting.uncertainty_state
    if uncertainty_state is not None:
        import time
        start_time = time.time()
        logger.info(f'queueing new uncertainty plot... {start_time}')
        draw = uncertainty_state.draw()
        nbins = max(min(draw.points.shape[0]//20000, 100), 30)
        stats = bumps.dream.stats.var_stats(draw)
        fig = plot_vars(draw, stats, nbins=nbins, cbar_colors=cbar_colors)
        logger.info(f"time to draw uncertainty plot: {time.time() - start_time}")
        return to_json_compatible_dict(fig)
    else:
        return None

@register
async def get_uncertainty_plot(timestamp: str=""):
    result = await asyncio.to_thread(_get_uncertainty_plot, timestamp=timestamp, cbar_colors=8)
    return result

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
        logger.info(f'queueing new model uncertainty plot... {start_time}')

        fig = plt.figure()
        errs = bumps.errplot.calc_errors_from_state(fitProblem, uncertainty_state)
        logger.info(f'errors calculated: {time.time() - start_time}')
        bumps.errplot.show_errors(errs, fig=fig)
        logger.info(f"time to render but not serialize... {time.time() - start_time}")
        fig.canvas.draw()
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        end_time = time.time()
        logger.info(f"time to draw model uncertainty plot: {end_time - start_time}")
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
        logger.info(f'queueing new parameter_trace plot... {start_time}')

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

        logger.info(f"time to render but not serialize... {time.time() - start_time}")
        dfig = mpld3.fig_to_dict(fig)
        plt.close(fig)
        end_time = time.time()
        logger.info(f"time to draw parameter_trace plot: {end_time - start_time}")
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
            # logger.info(f"setting parameter: {parameter}.fixed to {value}")
            # model has been changed: setp and getp will return different values!
            state.shared.updated_model = now_string()
    fitProblem.model_update()
    state.shared.updated_parameters = now_string()
    return

def now_string():
    return f"{datetime.now().timestamp():.6f}"

@register
async def publish(topic: str, message: Any = None):
    timestamp_str = f"{datetime.now().timestamp():.6f}"
    contents = {"message": message, "timestamp": timestamp_str}
    # session = get_session(session_id)    
    state.topics[topic].append(contents)
    # if session_id == app["active_session"]:
    await emit(topic, contents)
    # logger.info(f"emitted: {topic} :: {contents}")

@register
async def get_shared_setting(setting: str):
    value = await state.shared.get(setting)
    return to_json_compatible_dict(value)

@register
async def set_shared_setting(setting: str, value: Any):
    await state.shared.set(setting, value)

async def notify_shared_setting(setting: str, value: Any):
    await emit(setting, to_json_compatible_dict(value))

object.__setattr__(state.shared, 'notify', notify_shared_setting)

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
    subfolders = []
    files = []
    path = Path(state.base_path) if (pathlist is None or len(pathlist) == 0) else Path(*pathlist)
    abs_path = path.absolute()
    if not abs_path.exists():
        return { "error": f"Path does not exist: {abs_path}" }
    for p in abs_path.iterdir():
        if not p.exists():
            continue
        stat = p.stat()
        mtime = stat.st_mtime
        fileinfo = {
            "name": p.name,
            "modified": mtime
        }
        if p.is_dir():
            fileinfo["size"] = len(list(p.glob('*')))
            subfolders.append(fileinfo)
        else:
            # files.append(p.resolve().name)
            fileinfo["size"] = stat.st_size
            files.append(fileinfo)
    # for Windows: list drives as well
    drives = os.listdrives() if hasattr(os, 'listdrives') else []
    return dict(drives=drives, pathlist=abs_path.parts, subfolders=subfolders, files=files)

@register
async def get_fitter_defaults(*args):
    return FITTER_DEFAULTS

@register
async def shutdown():
    logger.info("killing...")
    await stop_fit()
    state.autosave()
    await emit("server_shutting_down")
    shutdown_result = asyncio.gather(_shutdown(), return_exceptions=True)

async def add_notification(content: str, title: str = "Notification", timeout: Optional[int] = None):
    id = None
    if timeout is None:
        id = str(uuid.uuid4())
        await emit("add_notification", {"title": title, "content": content, "id": id})
    else:
        await emit("add_notification", {"title": title, "content": content, "timeout": timeout})
    return id

async def _shutdown():
    raise SystemExit(0)

VALUE_PRECISION = 6
VALUE_FORMAT = "{{:.{:d}g}}".format(VALUE_PRECISION)

def nice(v, digits=4):
    """Fix v to a value with a given number of digits of precision"""
    from math import log10, floor
    v = float(v)
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
    elif isinstance(obj, UNDEFINED_TYPE):
        return None
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
    elif isinstance(params, Parameter) or isinstance(params, Constant):
        path = ".".join(pathlist)
        existing = lookup.get(params.id, None)
        if existing is not None:
            existing["paths"].append(".".join(pathlist))
        else:
            value_str = VALUE_FORMAT.format(nice(params.value))
            if hasattr(params, "has_prior"):
                has_prior = params.has_prior()
            else:
                has_prior = False

            if hasattr(params, "slot"):
                writable = type(params.slot) in [Variable, Parameter]
            else:
                writable = False
            new_item: ParamInfo = { 
                "id": params.id,
                "name": str(params.name),
                "paths": [path],
                "tags": getattr(params, 'tags', []),
                "writable": writable,
                "value_str": value_str, "fittable": params.fittable, "fixed": params.fixed }
            if has_prior:
                assert(params.prior is not None)
                lo, hi = params.prior.limits
                new_item['value01'] = params.prior.get01(float(params.value))
                new_item['min_str'] = VALUE_FORMAT.format(nice(lo))
                new_item['max_str'] = VALUE_FORMAT.format(nice(hi))
            lookup[params.id] = new_item
    elif callable(getattr(params, 'parameters', None)):
        # handle Expression, Constant, etc.
        subparams = params.parameters()
        params_to_list(subparams, lookup=lookup, pathlist=pathlist)
    return list(lookup.values())
