# Note: the following text appears at the end of the bumps -h command line help.
"""
Basic command line usage::

    # Run a model from show its χ² value. This is useful for debugging the model file.
    bumps model.py --chisq

    # Run a simple batch fit, appending results to a store file.
    bumps -b --store=T1.hdf model.py

    # Run a DREAM fit to explore parameter uncertainties
    bumps -b --store=T1.hdf model.py --fit=dream

    # Load and fit the last model in a session file.
    bumps -b --store=T1.hdf

Basic interactive usage::

    # Open a web browser to the bumps application. Show the initial model if any.
    bumps [model.py]

    # Open a web browser and start a fit
    bumps model.py --start

    # Watch fit progress and exit when complete
    bumps model.py --run --store=T1.hdf

There are many more options available to control the fit, particularly for
batch mode fitting, and to control the viewer. To see them type::

    bumps --help
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, List, Any
import warnings
import signal
import sys
import logging
from dataclasses import field

from bumps import __version__
from . import api
from . import persistent_settings
from . import fit_options
from .logger import logger, setup_console_logging
from .state_hdf5_backed import SERIALIZERS


# TODO: try datargs to build a parser from the typed dataclass
@dataclass
class BumpsOptions:
    """provide type hints for arguments"""

    # TODO: verify that attributes correspond to command line options
    # Note: order of attributes should match order of arguments in
    # the options processor while we are relying on manual verification.

    # Positional arguments.
    filename: Optional[str] = None
    args: Optional[List[str]] = None

    # Fitter controls.
    fit_options: Dict[str, Any] = field(default_factory=dict)

    # Session file controls.
    store: Optional[str] = None
    read_store: Optional[str] = None
    write_store: Optional[str] = None
    serializer: SERIALIZERS = "dill"
    no_auto_history: bool = False
    path: Optional[str] = None
    use_persistent_path: bool = False

    # Simulation controls.
    simulate: bool = False
    simrandom: bool = False
    shake: bool = False
    noise: float = 5
    seed: int = 0

    # Program controls.
    chisq: bool = False
    export: Optional[str] = None
    trace: bool = False
    parallel: int = 0
    mpi: bool = False

    # Webserver controls.
    webview: bool = True
    start: bool = False
    run: bool = False
    headless: bool = False
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    convergence_heartbeat: bool = False


OPTIONS_CLASS = BumpsOptions
APPLICATION_NAME = "bumps"


class DictAction(argparse.Action):
    # Note: implicit __init__ inherited from argparse.Action
    def __call__(self, parser, namespace, values, option_string=None):
        if self.type is bool:
            values = True
        # Find target dict name and key.
        key, subkey = self.dest.split(".")
        # Grab the target dict, if present.
        store = getattr(namespace, key, None)
        if store is None:
            store = {}
        store[subkey] = values
        # Store the target back in the namespace in case it was missing or None.
        setattr(namespace, key, store)


class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Help formatter customization:
    * Don't reflow text
    * Show non-trivial default values
    """

    def _get_help_string(self, action):
        help_text = action.help
        if action.default is not argparse.SUPPRESS:
            # Only show default if it's not None or False
            if action.default is not None and action.default is not False:
                help_text += f" [default: {action.default}]"
        return help_text


def get_commandline_options(arg_defaults: Optional[Dict] = None):
    # TODO: if running as a refl1d we should show refl1d instead of bumps
    parser = argparse.ArgumentParser(
        prog="bumps",
        formatter_class=HelpFormatter,
        epilog=__doc__.replace("::", ":"),
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="problem file to load, .py or .json (serialized) fitproblem",
    )
    parser.add_argument(
        "--args",
        nargs="*",
        type=str,
        help="arguments to send to the model loader on load",
    )
    # parser.add_argument('-d', '--debug', action='store_true', help='autoload modules on change')

    # Fitter controls
    fit_options.form_fit_options_associations()  # sets fitter list for each option
    fitter = parser.add_argument_group("Fitting controls")
    for name, option in fit_options.FIT_OPTIONS.items():
        # Check if the option is used by any of the active fitters. For example,
        # if parallel tempering is not available in fit_options.FITTERS then the
        # option --nT will not be available.
        if not option.fitters:
            continue
        stype = option.stype
        metavar = " " if stype is bool else name.replace("_", "-").upper()
        choices = stype if isinstance(stype, list) else None
        help = f"{option.label}  [{', '.join(option.fitters)}]\n{option.description}"
        fitter.add_argument(
            f"--{name.replace('_', '-')}",
            metavar=metavar,
            type=str if choices else stype,
            choices=choices,
            action=DictAction,
            dest=f"fit_options.{name}",
            help=help,
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
        metavar="STORE",
        default=None,
        type=str,
        help="read initial session state from file (overrides --store)",
    )
    session.add_argument(
        "--write-store",
        metavar="STORE",
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
        help="set initial path for save and load dialogs [webview only]",
    )
    session.add_argument(
        "--use-persistent-path",
        action="store_true",
        help="save most recently used path to disk for persistence between sessions [webview only]",
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
        help="simulate a dataset using the random problem parameters",
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
        "--chisq",
        action="store_true",
        help="print χ² and exit",
    )
    misc.add_argument(
        "--export",
        type=str,
        help="Directory path for data and figure export.",
    )
    # fitter.add_argument(
    #    "--pars",
    #    default=None,
    #    type=str,
    #    help="Start a fit from a previously saved result."
    # )
    misc.add_argument(
        "--parallel",
        default=0,
        type=int,
        help="run fit using multiprocessing for parallelism; use --parallel=0 for all cpus",
    )
    misc.add_argument(
        "--mpi",
        action="store_true",
        help="Use MPI to distribute work across a cluster",
    )
    misc.add_argument(
        "--trace",
        action="store_true",
        help="enable memory tracing (prints after every uncertainty update in dream)",
    )
    misc.add_argument(
        "--loglevel",
        type=str,
        choices=["debug", "info", "warn", "error", "critical"],
    )
    misc.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbose output (-v for info, -vv for debug)",
    )
    # TODO: show version numbers for both refl1d and bumps?
    misc.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Webserver controls
    server = parser.add_argument_group("Webview server controls")
    server.add_argument(
        "--webview",
        action="store_true",
        dest="webview",
        help="run bumps fit with the webview server (this is the default)",
    )
    server.add_argument(
        "-b",
        "--no-webview",
        action="store_false",
        dest="webview",
        help="run bumps fit in batch mode without the webview server",
    )
    server.add_argument(
        "-s",
        "--start",
        action="store_true",
        help="start fit immediately, leaving it open when done",
    )
    server.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="run the fit to completion",
    )
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
    server.add_argument(
        "--convergence-heartbeat",
        action="store_true",
        help="enable convergence heartbeat for jupyter kernel (keeps kernel alive during fit)",
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
    on_complete: List[Callable] = []

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
        # TODO: Why are we splitting path into parts?
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

    fitopts, errors = fit_options.update_options(options.fit_options)
    if errors:
        warnings.warn("\n".join(errors))

    # on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    fitter_id = fitopts["fit"]
    on_startup.append(lambda App: api.state.shared.set("selected_fitter", fitter_id))
    # TODO: send commandline options to the webview interface

    api.state.parallel = options.parallel

    if options.filename is not None:
        filepath = Path(options.filename).absolute()
        pathlist = list(filepath.parent.parts)
        filename = filepath.name
        logger.debug(f"fitter for filename {filename} is {fitter_id}")

        async def load_problem(App=None):
            await api.load_problem_file(pathlist, filename, args=options.args)

        on_startup.append(load_problem)

    # TODO: make sure --seed on command line produces the same result each time
    # TODO: store the seed with the fit results and print on console monitor
    # TODO: need separate seed for simulate and for fit
    # TODO: make sure seed works with async and threading and mpi
    # Reproducibility requires a known seed and a guaranteed order of execution.
    # So long as the startup items loop over awaits rather than using asyncio gather
    # this should be guaranteed.
    if options.seed:
        # raise NotImplementedError("--seed not yet supported")
        import numpy as np

        np.random.seed(options.seed)

    if options.simulate or options.simrandom:
        noise = None if options.noise <= 0.0 else options.noise

        async def simulate(App=None):
            if api.state.problem is None or api.state.problem.fitProblem is None:
                return
            problem = api.state.problem.fitProblem
            if options.simrandom:
                problem.randomize()
            problem.simulate_data(noise=noise)
            # TODO: How do these make it to the GUI?
            print("simulation parameters")
            print(problem.summarize())
            print("chisq", problem.chisq_str())

        on_startup.append(simulate)

    if options.shake:
        on_startup.append(lambda App=None: api.shake_parameters())

    need_mapper = not options.chisq
    if need_mapper:
        from bumps.mapper import MPIMapper, MPMapper, SerialMapper, using_mpi

        async def start_mapper(App=None):
            # TODO: When running Only load problem from root on MPI?
            # MPI is assumed
            # You can still run non-picklable problems on MPI using batch mode
            # since each work loads its own copy of the problem.
            if api.state.problem is not None:
                problem = api.state.problem.fitProblem
            else:
                problem = None
            if using_mpi() or options.mpi:
                # print("Starting with MPI mapper")
                mapper = MPIMapper
            elif options.parallel == 1:
                mapper = SerialMapper
            else:
                mapper = MPMapper
            mapper.start_worker(problem)
            # if mapper == MPIMapper: print(f"{MPIMapper.rank}: got beyond start worker.")
            api.state.mapper = mapper

        on_startup.append(start_mapper)

    webview = options.webview
    autostart = not webview or options.start or options.run
    autostop = not webview or options.run

    if options.chisq:

        async def show_chisq(App=None):
            if api.state.problem is not None:
                # show_cov won't work yet because I don't have driver
                # defined and because I don't have a --cov option.
                # if opts.cov: fitdriver.show_cov()
                problem = api.state.problem.fitProblem
                print("chisq", problem.chisq_str())

        on_startup.append(show_chisq)

    elif autostart:  # if batch mode then start the fit

        async def start_fit(App=None):
            # print(f"{fitter_settings=}")
            if api.state.problem is not None:
                await api.start_fit_thread(fitopts)

        on_startup.append(start_fit)
        api.state.console_update_interval = 0 if webview else 1

        if write_store is None and autostop:
            # TODO: can we specify problem.path in the model file?
            # TODO: can we default the session file name to model.hdf?
            raise RuntimeError("Need to add '--store=path' to the command line.")

        # TODO: if not autostop maybe --export after fit only or after every fit
        if options.export and autostop:
            # print("adding completion lambda")
            on_complete.append(lambda App=None: api.export_results(options.export))

        # TODO: cleaner handling of autostop
        if webview and autostop:  # trigger shutdown on fit complete [webview only]
            # print("setting shutdown True")
            api.state.shutdown_on_fit_complete = True

    else:
        # signal that no fit is running at startup, even if a fit was
        # interrupted and the state was saved:
        on_startup.append(lambda App=None: api.state.shared.set("active_fit", {}))

    # TODO: maybe warn when --export option is ignored
    # if options.export and not autostop:
    #     ...

    return on_startup, on_complete


async def _run_operations(on_startup, on_complete):
    for step in on_startup:
        await step(None)
    # print("waiting for fit complete")
    await api.wait_for_fit_complete()
    # print("running complete actions")
    for step in on_complete:
        await step(None)
    # if api.state.mapper is not None:
    #    api.mapper.stop_mapper()
    #    api.mapper = None
    # print("shutting down")
    await api.shutdown()
    # print("exiting")


def sigint_handler(sig, frame):
    """
    Support user interrupt of the fit in batch mode.

    The first Ctrl-C triggers a graceful shutdown. The second Ctrl-C should abort
    immediately.
    """
    # The first Ctrl-C will set fit_abort_event, and the second Ctrl-C will see that
    # the event is set and abort immediately.
    # Scenarios:
    # 1) Ctrl-C before the fit starts. The abort flag starts clear, so we set it.
    # If another Ctrl-C is received before the start_fit_thread is called then it
    # will still be set and we will abort. Otherwise, start_fit_thread will clear
    # the flag before starting the fit and the first Ctrl-C is ignored.
    # 2) Ctrl-C during fit. start_fit_thread clears the abort flag before starting
    # so we set it. If another Ctrl-C is received, either during the fit or after
    # the fit during save and cleanup, the flag will be set and we will abort.
    # 3) Ctrl-C after fit.  The abort flag will still be clear so we set it. The
    # subsequent Ctrl-C will abort.
    # Note: the same behaviour should work if we attach this signal handler
    # when starting webview. If stop fit is triggered by the GUI, then like the
    # first Ctrl-S it will set the abort event. Futher stop requests do nothing,
    # but Ctrl-C in the terminal will exit the program.
    if api.state.fit_abort_event.is_set():
        print("Program is not stopping. Aborting.")
        sys.exit(0)
    # print("\nCaught KeyboardInterrupt, stopping fit")
    api.state.fit_abort_event.set()


def run_batch_fit(options: Optional[OPTIONS_CLASS] = None):
    # TODO: use GUI notifications instead of console monitor for console output
    # TODO: provide info such as steps k of n and chisq with emit notifications
    # async def emit_to_console(*args, **kw):
    #    ...
    #    print("console", args, kw)
    # api.EMITTERS["console"] = emit_to_console

    signal.signal(signal.SIGINT, sigint_handler)
    on_startup, on_complete = interpret_fit_options(options)
    asyncio.run(_run_operations(on_startup, on_complete))
    # print("completed run")


def main_webview():
    """Entry point for the interactive interface"""
    return main(webview=True)


def main(options: Optional[OPTIONS_CLASS] = None, webview: bool = False):
    # TODO: where do we configure matplotlib?
    # Need to set matplotlib to a non-interactive backend because it is being used in the
    # the export thread. The next_color method calls gca() which needs to produce a blank
    # graph even when there is none (we ask for next color before making the plot).
    import matplotlib as mpl
    from bumps.mapper import using_mpi
    from .webserver import start_from_cli

    mpl.use("agg")
    if options is None:
        options = get_commandline_options()

    # TODO: cleaner way to isolate MPI?
    # TODO: cleaner handling of worker exit.
    # TODO: allow mpi fits from a jupyter slurm allocation spanning multiple nodes.
    # Only want one aiohttp server running so we need to know up front whether
    # the process we are running is the controller or one of the workers.
    # In order to support unpickleable problems in MPI we need to load the model
    # independently for each worker. Unfortunately the problem loader is in an
    # asynchronous function so we need the full async processing loop to load
    # the problem before we start the worker loop. The current hack is to call
    # sys.exit() after the worker loop finishes so that the remaining on startup
    # and on complete actions are skipped. We could instead pass an is_worker flag
    # into the options processor so that there are no tasks to skip.
    if options.mpi or using_mpi():
        # ** Warning **: importing MPI from mpi4py calls MPI_Init() which triggers
        # network traffic. Only import it when you know you are using MPI calls.
        from mpi4py import MPI

        is_controller = MPI.COMM_WORLD.rank == 0
    else:
        is_controller = True

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    setup_console_logging(levels[min(options.verbose, len(levels) - 1)])
    # from .logger import capture_warnings
    # capture_warnings(monkeypatch=True)
    logger.info(options)

    no_view = options.chisq
    webview = options.webview and not no_view
    if webview and is_controller:  # gui mode
        start_from_cli(options)
    else:  # console mode
        run_batch_fit(options)


if __name__ == "__main__":
    main()
