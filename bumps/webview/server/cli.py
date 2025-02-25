import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List

from . import api
from . import persistent_settings
from .logger import logger, list_handler, console_handler
from .fit_thread import EVT_FIT_PROGRESS
from .fit_options import parse_fit_options
from .state_hdf5_backed import SERIALIZERS, UNDEFINED


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


OPTIONS_CLASS = BumpsOptions


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
    server.add_argument("-p", "--port", default=0, type=int, help="port on which to start the server")
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

    # Program controls.
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--start", action="store_true", help="start fit when problem loaded")
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
            await api.load_problem_file(pathlist, filename)

        on_startup.append(load_problem)

    if options.chisq:

        async def show_chisq(App=None):
            if api.state.problem is not None:
                # show_cov won't work yet because I don't have driver
                # defined and because I don't have a --cov option.
                # if opts.cov: fitdriver.show_cov()
                chisq = api.state.problem.fitProblem.chisq_str()
                print("chisq", chisq)

        on_startup.append(show_chisq)
    elif options.start:

        async def start_fit(App=None):
            if api.state.problem is not None:
                await api.start_fit_thread(fitter_id, fitter_settings, options.exit)
                import time

                time.sleep(30)

        on_startup.append(start_fit)
    else:
        # signal that no fit is running at startup, even if a fit was
        # interrupted and the state was saved:
        on_startup.append(lambda App: api.state.shared.set("active_fit", {}))

    return on_startup


async def _run_operations(on_start):
    for step in on_start:
        await step(None)


def main(options: Optional[OPTIONS_CLASS] = None):
    # this entrypoint will be used to start gui, so set headless = False
    # (other contexts e.g. jupyter notebook will directly call start_app)
    logger.addHandler(console_handler)
    options = get_commandline_options(arg_defaults={"headless": False}) if options is None else options
    logger.info(options)

    async def emit(*args, **kw):
        print("emit", args, kw)

    api.EMITTERS["cli"] = emit
    on_start = interpret_fit_options(options)
    asyncio.run(_run_operations(on_start))
    print("completed run")


if __name__ == "__main__":
    main()
