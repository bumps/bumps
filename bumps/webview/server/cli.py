import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List
import warnings
import signal

from . import api
from . import persistent_settings
from .logger import logger, list_handler, console_handler
from .fit_thread import EVT_FIT_PROGRESS
from .fit_options import parse_fit_options
from .state_hdf5_backed import SERIALIZERS, UNDEFINED


# TODO: try datargs to build a parser from the typed dataclass
@dataclass
class BumpsOptions:
    """provide type hints for arguments"""

    filename: Optional[str] = None
    headless: bool = True
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    fit: Optional[str] = None
    start: bool = False
    read_store: Optional[str] = None
    write_store: Optional[str] = None
    store: Optional[str] = None
    exit: bool = False
    serializer: SERIALIZERS = "dill"
    trace: bool = False
    parallel: int = 0
    path: Optional[str] = None
    no_auto_history: bool = False
    convergence_heartbeat: bool = False
    use_persistent_path: bool = False
    fit_options: Optional[List[str]] = None
    chisq: bool = False
    version: bool = False

    # Simulate
    simulate: bool = False
    simrandom: bool = False
    shake: bool = False
    noise: float = 5
    seed: int = 0  # want seed for simulation and reproducible stochastic fits


OPTIONS_CLASS = BumpsOptions
APPLICATION_NAME = "bumps"


def get_commandline_options(arg_defaults: Optional[Dict] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        nargs="?",
        help="problem file to load, .py or .json (serialized) fitproblem",
    )
    # parser.add_argument('-d', '--debug', action='store_true', help='autoload modules on change')

    # Webserver controls
    server = parser.add_argument_group("Server controls")
    server.add_argument(
        "-x",
        "--headless",
        action="store_true",
        help="do not automatically load client in browser",
    )
    server.add_argument(
        "--external",
        action="store_true",
        help="listen on all interfaces, including external (local connections only if not set)",
    )
    server.add_argument(
        "-p",
        "--port",
        default=0,
        type=int,
        help="port on which to start the server",
    )
    server.add_argument(
        "--hub",
        default=None,
        type=str,
        help="api address of parent hub (only used when called as subprocess)",
    )

    # Fitter controls
    fitter = parser.add_argument_group("Fitting controls")
    fitter.add_argument(
        "--fit",
        default=None,
        type=str,
        choices=list(api.FITTERS_BY_ID.keys()),
        help="fitting engine to use; see manual for details",
    )
    fitter.add_argument(
        "--fit-options",
        nargs="*",
        type=str,
        help="fit options as key=value pairs",
    )
    fitter.add_argument(
        "--model-args",
        nargs="*",
        type=str,
        help="arguments to send to the model loader",
    )

    fitter.add_argument(
        "--parallel",
        default=0,
        type=int,
        help="run fit using multiprocessing for parallelism; use --parallel=0 for all cpus",
    )

    # Session file controls.
    session = parser.add_argument_group("Session file management")
    session.add_argument(
        "--store",
        default=None,
        type=str,
        help="set read_store and write_store to same file",
    )
    session.add_argument(
        "--read-store",
        default=None,
        type=str,
        help="read initial session state from file (overrides --store)",
    )
    session.add_argument(
        "--write-store",
        default=None,
        type=str,
        help="output file for session state (overrides --store)",
    )
    session.add_argument(
        "--serializer",
        default=OPTIONS_CLASS.serializer,
        type=str,
        choices=["pickle", "dill", "dataclass"],
        help="strategy for serializing problem, will use value from store if it has already been defined",
    )
    session.add_argument(
        "--no-auto-history",
        action="store_true",
        help="disable auto-appending problem state to history on load and at fit end",
    )
    session.add_argument(
        "--path",
        default=None,
        type=str,
        help="set initial path for save and load dialogs",
    )
    session.add_argument(
        "--use-persistent-path",
        action="store_true",
        help="save most recently used path to disk for persistence between sessions",
    )

    # Simulation controls.
    sim = parser.add_argument_group("Simulation")
    sim.add_argument(
        "--simulate",
        action="store_true",
        help="simulate a dataset using the initial problem parameters",
    )
    sim.add_argument(
        "--simrandom",
        action="store_true",
        help="simulate a dataset using the randome problem parameters",
    )
    sim.add_argument(
        "--shake",
        action="store_true",
        help="set random parameters before fitting",
    )
    sim.add_argument(
        "--noise",
        type=float,
        default=5.0,
        help="percent noise to add to the simulated data",
    )
    sim.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random number seed, or 0 for none",
    )

    # Program controls.
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--edit",
        action="store_true",
        help="start web interface to view and edit the problem",
    )
    misc.add_argument(
        "--start",
        action="store_true",
        help="start fit when problem loaded",
    )
    misc.add_argument(
        "--exit",
        action="store_true",
        help="end process when fit complete (fit results lost unless write_store is specified)",
    )
    misc.add_argument(
        "--convergence-heartbeat",
        action="store_true",
        help="enable convergence heartbeat for jupyter kernel (keeps kernel alive during fit)",
    )
    misc.add_argument(
        "--trace",
        action="store_true",
        help="enable memory tracing (prints after every uncertainty update in dream)",
    )
    misc.add_argument(
        "--chisq",
        action="store_true",
        help="print χ²",
    )
    misc.add_argument(
        "--export",
        type=str,
        help="Directory path for data and figure export.",
    )
    misc.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print version number",
    )

    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    namespace = OPTIONS_CLASS()
    if arg_defaults is not None:
        logger.debug(f"arg_defaults: {arg_defaults}")
        for k, v in arg_defaults.items():
            setattr(namespace, k, v)
    args = parser.parse_args(namespace=namespace)
    # print(f"{type(args)=} {args=}")
    return args


def interpret_fit_options(options: OPTIONS_CLASS = OPTIONS_CLASS()):
    # ordered list of actions to do on startup
    on_startup: List[Callable] = []

    if options.version:
        from bumps import __version__

        print(__version__)

    if options.use_persistent_path:
        api.state.base_path = persistent_settings.get_value(
            "base_path", str(Path().absolute()), application=APPLICATION_NAME
        )
    elif options.path is not None and Path(options.path).exists():
        api.state.base_path = options.path
    else:
        api.state.base_path = str(Path.cwd().absolute())

    if options.read_store is not None and options.store is not None:
        warnings.warn("read_store and store are both set; read_store will be used to initialize state")
    if options.write_store is not None and options.store is not None:
        warnings.warn("write_store and store are both set; write_store will be used to save state")

    read_store = options.read_store if options.read_store is not None else options.store
    write_store = options.write_store if options.write_store is not None else options.store

    # TODO: why is session file read immediately but model.py delayed?
    if read_store is not None:
        read_store_path = Path(read_store).absolute()
        api.state.read_session_file(str(read_store_path))
        if write_store is None:
            api.state.shared.session_output_file = dict(
                pathlist=list(read_store_path.parent.parts),
                filename=read_store_path.name,
            )
    if write_store is not None:
        write_store_path = Path(write_store).absolute()
        api.state.shared.session_output_file = dict(
            pathlist=list(write_store_path.parent.parts), filename=write_store_path.name
        )
        api.state.shared.autosave_session = True

    if api.state.problem.serializer is None or api.state.problem.serializer == "":
        api.state.problem.serializer = options.serializer

    if options.no_auto_history:
        api.state.shared.autosave_history = False

    if options.trace:
        global TRACE_MEMORY
        TRACE_MEMORY = True
        api.TRACE_MEMORY = True

    # on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    if options.fit is not None:
        on_startup.append(lambda App: api.state.shared.set("selected_fitter", options.fit))

    fitter_id = options.fit
    if fitter_id is None:
        fitter_id = api.state.shared.selected_fitter
    if fitter_id is None or fitter_id is UNDEFINED:
        fitter_id = "amoeba"
    fitter_settings = parse_fit_options(fitter_id=fitter_id, fit_options=options.fit_options)

    api.state.parallel = options.parallel

    if options.filename is not None:
        filepath = Path(options.filename).absolute()
        pathlist = list(filepath.parent.parts)
        filename = filepath.name
        logger.debug(f"fitter for filename {filename} is {fitter_id}")

        async def load_problem(App=None):
            await api.load_problem_file(pathlist, filename, args=options.model_args)

        on_startup.append(load_problem)

    # TODO: reproducibility requires using the same seed -- where is it printed?
    # TODO: store the seed with the fit results
    # TODO: make sure seed works with async and threading and mpi
    if options.seed:
        np.random.seed(options.seed)

    if options.simulate or options.simrandom:
        noise = None if options.noise <= 0.0 else options.noise

        async def simulate(App=None):
            problem = api.state.problem.fitProblem
            if options.simrandom:
                problem.randomize()
            problem.simulate_data(noise=noise)
            if options.shake:
                # Show the parameters for the simulated model, but only if
                # we are going to shake the model later.
                # TODO: how do these make it to the GUI?
                print("simulation parameters")
                print(problem.summarize())
                print("chisq", problem.chisq_str())

        on_startup.append(simulate)

    if options.shake:

        async def shake(App=None):
            problem = api.state.problem.fitProblem
            problem.randomize()

        on_startup.append(shake)

    if options.chisq:

        async def show_chisq(App=None):
            if api.state.problem is not None:
                # show_cov won't work yet because I don't have driver
                # defined and because I don't have a --cov option.
                # if opts.cov: fitdriver.show_cov()
                problem = api.state.problem.fitProblem
                print("chisq", problem.chisq_str())

        on_startup.append(show_chisq)

    elif options.start:

        async def start_fit(App=None):
            # print(f"{fitter_settings=}")
            if api.state.problem is not None:
                problem = api.state.problem.fitProblem
                await api.start_fit_thread(fitter_id, fitter_settings)
                # await api.state.fit_complete_future

        on_startup.append(start_fit)
        if options.exit:
            api.state.shutdown_on_fit_complete = True

    else:
        # signal that no fit is running at startup, even if a fit was
        # interrupted and the state was saved:
        on_startup.append(lambda App: api.state.shared.set("active_fit", {}))

    if options.export:

        async def delayed_export(App=None):
            await api.wait_for_fit_complete()
            await api.export_results(options.export)

        on_startup.append(delayed_export)

    return on_startup


async def _run_operations(on_start):
    for step in on_start:
        await step(None)
    await api.wait_for_fit_complete()


def sigint_handler(sig, frame):
    import sys
    import time

    if api.state.fit_abort_event.is_set():
        print("Second KeyboardInterrupt, exiting immediately")
        sys.exit(0)
    print("\nCaught KeyboardInterrupt, stopping fit")
    api.state.fit_abort_event.set()


def start_batch(options: Optional[OPTIONS_CLASS] = None):
    async def emit(*args, **kw):
        ...
        # print("emit", args, kw)

    api.EMITTERS["cli"] = emit
    signal.signal(signal.SIGINT, sigint_handler)
    on_start = interpret_fit_options(options)
    asyncio.run(_run_operations(on_start))
    print("completed run")


def main(options: Optional[OPTIONS_CLASS] = None):
    # TODO: where do we configure matplotlib?
    # Need to set matplotlib to a non-interactive backend because it is being used in the
    # the export thread. The next_color method calls gca() which needs to produce a blank
    # graph even when there is none (we ask for next color before making the plot).
    import matplotlib as mpl

    mpl.use("agg")
    # this entrypoint can be used to start gui, so set headless = False
    # (other contexts e.g. jupyter notebook will directly call start_app)
    logger.addHandler(console_handler)
    options = get_commandline_options(arg_defaults={"headless": False}) if options is None else options
    logger.info(options)

    if options.edit:  # gui mode
        from .webserver import start_from_cli

        start_from_cli(options)
    else:  # console mode
        api.state.console_update_interval = 1
        start_batch(options)


if __name__ == "__main__":
    main()
