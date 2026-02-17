# Note: the following text appears at the end of the bumps -h command line help.
"""
Basic command line usage::

    # Run a model from show its χ² value. This is useful for debugging the model file.
    bumps model.py --chisq

    # Run a simple batch fit on model.py, appending results to a store file.
    bumps --batch --session=T1.hdf model.py

    # Run a DREAM fit on model.py to explore parameter uncertainties
    bumps --batch --session=T1.hdf model.py --fit=dream

    # Load and resume the last fit in a session file. The model.py file is ignored.
    bumps --batch --session=T1.hdf [model.py] --resume

Basic interactive usage::

    # Open a web browser to the bumps application. Show the initial model if any.
    bumps [model.py]

    # Open a web browser and start a fit
    bumps model.py --start

    # Watch fit progress and exit when complete
    bumps model.py --run --session=T1.hdf

There are many more options available to control the fit, particularly for
batch mode fitting. To see them type::

    bumps --help
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, Optional, List, Any
import warnings
import signal
import sys
from dataclasses import field
import hashlib
# from textwrap import dedent

from bumps import __version__ as bumps_version
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitproblem import FitProblem, load_problem
from . import api
from . import persistent_settings
from . import fit_options
from .logger import logger, setup_console_logging, LOGLEVEL
from .state_hdf5_backed import SERIALIZERS, FitResult

# Ick! PREFERRED_PORT is here rather than in webserver so that it can
# be used in the command line help without a circular import.
PREFERRED_PORT = 5148  # "SLAB"


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
    resume: bool = False
    show_cov: bool = False
    show_err: bool = False
    show_entropy: Optional[str] = None

    # Session file controls.
    session: Optional[str] = None
    read_session: Optional[str] = None
    write_session: Optional[str] = None
    serializer: SERIALIZERS = "dataclass"
    auto_history: bool = True
    path: Optional[str] = None
    use_persistent_path: bool = False
    reload_export: Optional[str] = None
    pars: Optional[str] = None

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
    mpi: Optional[bool] = None

    # Webserver controls.
    mode: str = "edit"
    """One of "batch", "edit", "start" or "run" depending on -b, -s and -r options"""
    headless: bool = False
    external: bool = False
    port: int = 0
    hub: Optional[str] = None
    convergence_heartbeat: bool = False


class DictAction(argparse.Action):
    """
    Gather argparse command line options into a dict entry.

    Given *dest="group.key"* in the parser argument definition, add *group* to the
    returned namespace and set *group["key"]=value*.

    There is special handling of bool types, converting them to True/False values.
    """

    # Note: implicit __init__ inherited from argparse.Action
    def __call__(self, parser, namespace, values, option_string=None):
        if self.type is bool:
            values = not option_string.startswith("--no-")
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
                help_text += f" (default: {action.default})"
        return help_text


def _branding():
    """Return a string with version and system information."""
    output = f"{'=' * 55}\n"
    output += f"{api.state.app_name}\t\t{api.state.app_version}\n"
    if api.state.app_name != "bumps":
        output += f"bumps\t\t{bumps_version}\n"
    output += "Python\t\t" + ".".join(map(str, sys.version_info[:3])) + "\n"
    output += f"Platform\t{sys.platform}\n"
    output += f"{'=' * 55}\n"
    return output


def get_commandline_options(arg_defaults: Optional[Dict] = None):
    """Parse bumps command line options."""
    # TODO: if running as a refl1d we should show prog=refl1d instead of prog=bumps
    # TODO: allow --pars from session file
    # TODO: missing options from pre-1.0
    """

    # Wait for someone to ask for the following
    --overwrite                    [new version extends session file]
        if store already exists, replace it
    --checkpoint=0                 [verify we have checkpointing in batch mode]
        save fit state every n hours, or 0 for no checkpoints
    --resynth=0
        run resynthesis error analysis for n generations
    --time_model
        run the model --steps times in order to estimate total run time.
    --profile
        run the python profiler on the model; use --steps to run multiple
        models for better statistics
    --stepmon
        show details for each step in .log file

    # Won't implement
    --plot=linear|log|residuals    [plugin specific]
        type of plot to display
    --view=linear|log              [plugin specific]
        one of the predefined problem views; reflectometry also has fresnel,
        logfresnel, q4 and residuals
    --staj                         [plugin specific. Can plugins extend argparse?]
        output staj file when done [Refl1D only]

    # Superceded
    -m/-c/-p command               [we are shipping a python environment]
        run the python interpreter with bumps on the path:
            m: command is a module such as bumps.cli, run as __main__
            c: command is a python one-line command
            p: command is the name of a python script
    -i                             [our python environment can install ipython with pip]
        start the interactive interpreter
    --noshow                       [use --export to produce plots]
        semi-batch; send output to console but don't show plots after fit
    --preview                      [use webview instead]
        display model but do not perform a fitting operation
    --edit                         [default]
        start the gui
    --batch                        [current version doesn't save .mon]
        batch mode; save output in .mon file and don't show plots after fit
    """
    prog = api.state.app_name
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=HelpFormatter,
        epilog=__doc__.replace("::", ":").replace("bumps", prog),
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="problem file to load, .py or .json (serialized) fitproblem",
    )
    # TODO: Don't need --args specifier since this is just the extras after model.py
    parser.add_argument(
        "--args",
        nargs="*",
        type=str,
        help="arguments to send to the model loader on load",
    )
    # parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help='autoload modules on change')

    # Fitter controls
    fit_options.form_fit_options_associations()  # sets fitter list for each option
    fitter = parser.add_argument_group("Fitting options")
    for name, option in fit_options.FIT_OPTIONS.items():
        stype = option.stype
        metavar = "" if stype is bool else name.replace("_", "-").upper()
        choices = stype if isinstance(stype, list) else None
        if name == "fit":
            choices = FIT_AVAILABLE_IDS
        # For the trim option allow both --trim and --no-trim. The DictAction class
        # checks for leading "--no-" in the option string when assigning to bool.
        flagname = name.replace("_", "-")
        flags = (f"--{flagname}", f"--no-{flagname}") if stype is bool else (f"--{flagname}",)
        fitter.add_argument(
            *flags,
            action=DictAction,
            dest=f"fit_options.{name}",
            type=str if choices else stype,
            metavar=metavar,
            nargs=0 if stype is bool else None,
            choices=choices,
            # Don't show parameters that don't appear in the visible optimizers.
            # For example, don't show --nT which is only available when --fit=pt.
            help=option.build_help() if option.fitters else argparse.SUPPRESS,
        )

    # Fit outputs
    output = parser.add_argument_group("Fitting outputs")
    output.add_argument(
        "--err",
        action="store_true",
        dest="show_err",
        help="Show uncertainty from covariance",
    )
    output.add_argument(
        "--cov",
        action="store_true",
        dest="show_cov",
        help="Show covariance matrix on output",
    )
    output.add_argument(
        "--entropy",
        dest="show_entropy",
        type=str,  # TODO: type is str|None
        default=None,
        const="gmm",
        nargs="?",
        choices=["gmm", "mvn", "wnn", "llf"],
        help=dedent("""\
        Compute entropy from uncertainty distribution [dream only]:
            mvn: Fit to a single multivariate normal
            gmm: Fit to a Gaussian mixture model
            wnn: Weighted Kozeachenko-Leonenko nearest neighbour (Berrettt 2016)
            llf: Use local density to estimate loglikelihood factor (Kramer 2010)
        """),
    )

    # Session file controls.
    session = parser.add_argument_group("Session file management")
    session.add_argument(
        "--session",
        metavar="SESSION",
        default=None,
        type=str,
        help="set read/write session to same file",
    )
    session.add_argument(
        "--read-session",
        metavar="SESSION",
        default=None,
        type=str,
        help="read initial session state from file (overrides --session)",
    )
    session.add_argument(
        "--write-session",
        metavar="SESSION",
        default=None,
        type=str,
        help="output file for session state (overrides --session)",
    )
    session.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        help=dedent("""\
            Resume the most recent fit from the saved session file. [dream, de]
            Note that this loads the model from the session and ignores any model
            file specified on the command line.
            """),
    )
    session.add_argument(
        "--serializer",
        default=BumpsOptions.serializer,
        type=str,
        choices=["pickle", "cloudpickle", "dill", "dataclass"],
        help="strategy for serializing problem, will use value from session if it has already been defined",
    )
    session.add_argument(
        "--auto-history",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="auto-append problem state to history on load and at the end of the fit",
    )
    session.add_argument(
        "--path",
        default=None,
        type=str,
        help="set initial path for save and load dialogs (webview only)",
    )
    session.add_argument(
        "--use-persistent-path",
        action=argparse.BooleanOptionalAction,
        help="save most recently used path to disk for persistence between sessions (webview only)",
    )
    session.add_argument(
        "--reload-export",
        type=str,
        help="reload a bumps 0.x store directory as if it were a session file",
    )

    # Simulation controls.
    sim = parser.add_argument_group("Simulation")
    sim.add_argument(
        "--simulate",
        action=argparse.BooleanOptionalAction,
        help="simulate a dataset using the initial problem parameters",
    )
    sim.add_argument(
        "--simrandom",
        action=argparse.BooleanOptionalAction,
        help="simulate a dataset using the random problem parameters",
    )
    sim.add_argument(
        "--shake",
        action=argparse.BooleanOptionalAction,
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
        help="directory path for data and figure export.",
    )
    misc.add_argument("--pars", default=None, type=str, help="start a fit from an exported result.")
    misc.add_argument(
        "--parallel",
        default=0,
        type=int,
        help="run fit using multiprocessing for parallelism; use --parallel=0 for all cpus",
    )
    misc.add_argument(
        "--threads",
        action="store_true",
        help=argparse.SUPPRESS,
        # help="run fit using multithreading for parallelism",
    )
    misc.add_argument(
        "--mpi",
        action=argparse.BooleanOptionalAction,
        help="Use MPI for parallelization (only needed if we fail to detect MPI correctly)",
    )
    misc.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        help="enable memory tracing (prints after every uncertainty update in dream)",
    )
    misc.add_argument(
        "--loglevel",
        type=str,
        choices=list(LOGLEVEL.keys()),
        help="display logging to console",
        default="warn",
    )
    # Show version numbers for both bumps and child program
    misc.add_argument(
        "-V",
        "--version",
        action="version",
        version=_branding(),
    )

    # TODO: restructure so that -b -s -r --webview override each other
    # Maybe a runmode enum: 0=batch 1=edit 2=start 3=run
    # Webserver controls
    server = parser.add_argument_group("Webview server controls")
    server.add_argument(
        "--edit",
        "--webview",
        action="store_const",
        const="edit",
        dest="mode",
        help="run bumps fit with the webview server (this is the default)",
    )
    server.add_argument(
        "-b",
        "--batch",
        action="store_const",
        const="batch",
        dest="mode",
        help="run bumps fit in batch mode without the webview server",
    )
    server.add_argument(
        "-s",
        "--start",
        action="store_const",
        const="start",
        dest="mode",
        help="start fit immediately, leaving it open when done",
    )
    server.add_argument(
        "-r",
        "--run",
        action="store_const",
        const="run",
        dest="mode",
        help="run the fit to completion",
    )
    server.add_argument(
        "-x",
        "--headless",
        action=argparse.BooleanOptionalAction,
        help="do not automatically load client in browser",
    )
    server.add_argument(
        "--external",
        action=argparse.BooleanOptionalAction,
        help="listen on all interfaces, including external (local connections only if not set)",
    )
    server.add_argument(
        "-p",
        "--port",
        default=0,
        type=int,
        help=f"web server port; use --port=0 to first try {PREFERRED_PORT} then fall back to a random port",
    )
    server.add_argument(
        "--hub",
        default=None,
        type=str,
        # Don't show jupyter-only parameters to the user.
        # help="api address of parent hub (only used when called as subprocess)",
        help=argparse.SUPPRESS,
    )
    server.add_argument(
        "--convergence-heartbeat",
        action=argparse.BooleanOptionalAction,
        # Don't show jupyter-only parameters to the user.
        # help="enable convergence heartbeat for jupyter kernel (keeps kernel alive during fit)",
        help=argparse.SUPPRESS,
    )

    # parser.add_argument('-c', '--config-file', type=str, help='path to JSON configuration to load')
    namespace = BumpsOptions()
    if arg_defaults is not None:
        logger.debug(f"arg_defaults: {arg_defaults}")
        for k, v in arg_defaults.items():
            setattr(namespace, k, v)
    args = parser.parse_args(namespace=namespace)
    # print(f"Parse output: {args=}")
    return args


def interpret_fit_options(options: BumpsOptions):
    # ordered list of actions to do on startup
    on_startup: List[Callable] = []
    on_complete: List[Callable] = []

    if options.use_persistent_path:
        api.state.base_path = persistent_settings.get_value(
            "base_path", str(Path().absolute()), application=api.state.app_name
        )
    elif options.path is not None and Path(options.path).exists():
        api.state.base_path = options.path
    else:
        api.state.base_path = str(Path.cwd().absolute())

    if options.read_session is not None and options.session is not None:
        warnings.warn("read_session and session are both set; read_session will be used to initialize state")
    if options.write_session is not None and options.session is not None:
        warnings.warn("write_session and session are both set; write_session will be used to save state")

    read_session = options.read_session if options.read_session is not None else options.session
    write_session = options.write_session if options.write_session is not None else options.session

    # The logic for setting the current fit problem is complex.
    # (1) bumps [PATH/MODEL.py] --reload-export=PATH[/MODEL.par] [--session=SESSION.h5] [--resume]
    #     Load the results of a pre-1.0 bumps fit. Use PATH/MODEL.py if specified, otherwise
    #     use ./MODEL.py in the current directory, or ./GLOB.par with PATH/*.par expands only
    #     to PATH/GLOB.par. This is useful for exploring the results of an old DREAM run, and
    #     possibly resuming the fit. This loads the model from its original location
    #     because the model script probably refers to data files using relative paths, and these
    #     aren't included in the export directory.
    # (2) bumps PATH/MODEL.py --session=SESSION.h5
    #     Load from PATH/MODEL.py even if SESSION.h5 is available. This is useful for a workflow
    #     where you tweak the model script then rerun the fit, collecting all the results in
    #     SESSION.h5.
    # (3) bumps PATH/MODEL.py --session=SESSION.h5 --resume
    #     Load from SESSION.h5 even if PATH/MODEL.py is available. This is useful for resuming
    #     the fit where it left off, even if the model constraints were adjusted in webview
    #     before fitting.
    # (4) bumps --session=SESSION.h5
    #     Load from SESSION.h5. This is useful for a workflow primarily driven by webview, but
    #     done over multiple sessions.
    #
    # More succinctly:
    #
    #     Use --reload-export if specified.
    #     Otherwise use MODEL.py if specified except on resume.
    #     Otherwise use SESSION.h5.

    # TODO: why is session file read immediately but model.py delayed?
    # If a session file exists load the problem.
    if read_session is not None:
        read_session_path = Path(read_session).absolute()
        if read_session_path.exists():
            api.state.read_session_file(str(read_session_path))
            if write_session is None:
                api.state.shared.session_output_file = dict(
                    pathlist=list(read_session_path.parent.parts),
                    filename=read_session_path.name,
                )

    if write_session is not None:
        write_session_path = Path(write_session).absolute()
        # TODO: Why are we splitting path into parts?
        api.state.shared.session_output_file = dict(
            pathlist=list(write_session_path.parent.parts), filename=write_session_path.name
        )
        api.state.shared.autosave_session = True

    if api.state.problem.serializer is None or api.state.problem.serializer == "":
        api.state.problem.serializer = options.serializer

    if not options.auto_history:
        api.state.shared.autosave_history = False

    if options.trace:
        global TRACE_MEMORY
        TRACE_MEMORY = True
        api.TRACE_MEMORY = True

    fitopts, errors = fit_options.check_options(options.fit_options)
    if errors:
        warnings.warn("\n".join(errors))
    # TODO: leave fitter_id in fitopts
    # TODO: use dict rather than list of pairs for fitopts
    fitter_id = fitopts.pop("fit")

    # Add secret fitter to the GUI (pt and scipy.leastsq as of this writing)
    # TODO: FitOptions.vue needs to update fit_names on changeActiveFitter
    if fitter_id not in api.state.shared.fitter_settings:
        fitter = fit_options.lookup_fitter(fitter_id)
        api.state.shared.fitter_settings[fitter_id] = {"name": fitter.name, "settings": dict(fitter.settings)}

    # Show selected fit options in the GUI
    api.state.shared.selected_fitter = fitter_id
    api.state.shared.fitter_settings[fitter_id]["settings"].update(fitopts)
    api.state.parallel = options.parallel

    # TODO: How do we resume if the model is saved as a pickle and can't be restored?
    # on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    # resume only works when you have an identical problem, so we don't load from file
    # if we are resuming the fit.
    if options.reload_export:

        async def load_export_to_state(App=None):
            problem, fit = reload_export(options.reload_export, modelfile=options.filename, args=options.args)
            path = Path(problem.path)
            await api.set_problem(problem, path.parent, path.name, fit=fit)

        on_startup.append(load_export_to_state)

    # on_startup.append(lambda App: publish('', 'local_file_path', Path().absolute().parts))
    # resume only works when you have an identical problem, so we don't load from file
    # if you are resuming the fit.
    elif options.filename is not None and not options.resume:

        async def load_problem_to_state(App=None):
            path = Path(options.filename).absolute()
            logger.debug(f"fitter for filename {path.name} is {fitter_id}")
            problem = load_problem(path, args=options.args)
            await api.set_problem(problem, path.parent, path.name)

        on_startup.append(load_problem_to_state)

    # TODO: allow pars to be loaded from a session file.
    if options.pars is not None:
        filepath = Path(options.pars).absolute()
        pars_pathlist = list(filepath.parent.parts)
        pars_filename = filepath.name
        on_startup.append(lambda App: api.apply_parameters(pars_pathlist, pars_filename))

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
        from bumps.mapper import MPIMapper, MPMapper, SerialMapper, ThreadPoolMapper, using_mpi

        async def start_mapper(App=None):
            # print(f"{api.state.rank}start mapper")
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
            elif options.threads:
                mapper = ThreadPoolMapper
            else:
                mapper = MPMapper
            mapper.start_worker(problem)
            # if mapper == MPIMapper: print(f"{MPIMapper.rank}: got beyond start worker.")
            api.state.mapper = mapper

        on_startup.append(start_mapper)

    webview = options.mode != "batch"
    autostart = not webview or options.mode in ("start", "run") or options.resume
    autostop = not webview or options.mode == "run"
    # print(f"{options.mode=} {webview=} {autostart=} {autostop=}")

    if options.chisq:

        async def show_chisq(App=None):
            if api.state.problem is not None:
                # show_cov won't work yet because I don't have driver
                # defined and because I don't have a --cov option.
                # if opts.cov: fitdriver.show_cov()
                problem = api.state.problem.fitProblem
                # print(problem.summarize())
                print("chisq", problem.chisq_str())

        on_startup.append(show_chisq)

    elif autostart:  # if batch mode then start the fit

        async def start_fit(App=None):
            # print(f"{api.state.rank}start fit")
            # print(f"{fitter_settings=}")
            if api.state.problem is not None:
                await api.start_fit_thread(fitter_id, fitopts, resume=options.resume)

        # TODO: This is duplicating code from bumps/fitters.py. Move it and share it.
        def _show_results(show_err: bool = True, show_cov: bool = False, show_entropy: str | None = None):
            import numpy as np
            from bumps.lsqerror import stderr
            from bumps.dream.entropy import cov_entropy
            from bumps.formatnum import format_uncertainty
            from bumps.dream.stats import var_stats, format_vars

            if not (show_err or show_cov or show_entropy):
                return

            problem, state = api.state.problem.fitProblem, api.state.fitting.fit_state
            x = problem.getp()
            # TODO: Should we show the estimates from derivatives even when running dream?
            if state is None:
                cov = problem.cov(x)
                vstats = None
                dx = stderr(cov)
                S, dS = cov_entropy(cov), 0
            else:
                draw = state.draw()
                # TODO: if there are derived variables then problem.show() doesn't work
                # TODO: sample from whole population, not just the end, to reduce autocorrelation
                if show_cov:
                    cov = np.cov(draw.points[-1000:].T)
                vstats = var_stats(draw)
                # dx = np.array([(v.p68[1] - v.p68[0]) / 2 for v in vstats])
                if show_entropy:
                    entropy_method = show_entropy
                    S, dS = state.entropy(method=entropy_method)

            if show_cov:
                problem.show_cov(x, cov)
            if show_err:
                if state is None:
                    problem.show_err(x, dx)
                else:
                    print(format_vars(vstats))
            if show_entropy:
                print(f"Entropy: {format_uncertainty(S, dS)} bits")

        async def show_results(App=None):
            _show_results(options.show_err, options.show_cov, options.show_entropy)

        on_startup.append(start_fit)
        api.state.console_update_interval = 0 if webview else 1

        if not options.export and write_session is None and autostop:
            # TODO: can we specify problem.path in the model file?
            # TODO: can we default the session file name to model.hdf?
            raise RuntimeError("Include '--session=output.h5' on the command line to save the fit.")

        # TODO: if not autostop maybe --export after fit only or after every fit
        if options.export and autostop:
            # print("adding completion lambda")
            on_complete.append(lambda App=None: api.export_results(options.export))

        if autostop:
            on_complete.append(show_results)

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


def reload_export(
    path: Path | str,
    modelfile: Path | str | None = None,
    args: list[str] | None = None,
) -> tuple[FitProblem, FitResult]:
    """
    Reload a bumps export directory.

    *path* is the path to the directory, or to a <model>.par file within that
    directory. Use the <model>.par file if you have multiple models exported to
    the same path.

    If *modelfile* is provided then use it, otherwise use <model>.py in the
    current directory. That means you can change to the directory containing
    your model then run bumps with --reload-export=path without having to list
    <model>.py on the command line. This is handy if you have several variations
    saved to different filenames stored along with your data.

    sys.argv is set to *args* before loading the model.
    """
    from bumps.fitproblem import load_problem
    from bumps.cli import load_best
    from .state_hdf5_backed import SERIALIZER_EXTENSIONS

    # Find the .par file in the export directory
    path = Path(path)
    if path.is_file():
        parfile = path
        if parfile.suffix != ".par":
            raise ValueError(f"Reload export needs path or path/model.par, not {path}")
        path = path.parent  # set path to the directory containing *.par
    else:
        # path is already a directory; look for a par file within it
        pars_glob = list(path.glob("*.par"))
        if len(pars_glob) == 0:
            raise ValueError(f"Reload export {path}/*.par does not exist")
        if len(pars_glob) > 1:
            raise ValueError(f"More than one .par file. Use {path}/model.par in reload export.")
        parfile = pars_glob[0]

    # Look for a pickle in the parfile directory and try loading that
    problem = None
    for serializer, suffix in SERIALIZER_EXTENSIONS.items():
        picklefile = parfile.with_suffix(f".{suffix}")
        if picklefile.exists():
            try:
                problem = load_problem(picklefile)
                # Note: no need to load_best because serializer contained the parameters
            except Exception as exc:
                logger.warn(f"Failed to deserialize {picklefile}; look for model file")
            break

    # Pickle failed. Try loading modelfile
    if problem is None:
        # If modelfile is not given look for one of the serializers in the current directory
        if modelfile is None:
            # Look for model.py file in the parent directory of the export directory
            modelfile = path.parent / parfile.with_suffix(".py").name
        else:
            modelfile = Path(modelfile)

        saved_model = path / modelfile.name
        if not saved_model.exists():
            raise ValueError(f"Model '{modelfile.name}' does not exist in '{path}'")
        if not modelfile.exists():
            raise ValueError(f"Model '{modelfile}' does not exist.")
        if filehash(saved_model) != filehash(modelfile):
            raise ValueError(f"Model file has been modified. Copy {saved_model} into {modelfile.parent}")

        # Load the model script and the fitted values.
        problem = load_problem(modelfile, model_options=args)
        load_best(problem, parfile)

    # Load the MCMC files
    fit_result = load_fit_result(parfile)

    return problem, fit_result


def load_fit_result(parfile: Path | str) -> FitResult:
    from bumps.dream.state import load_state

    # Reload DREAM state if it exists. Note that labels come from the parfile.
    try:
        fit_state = load_state(str(parfile.parent / parfile.stem))
        fit_state.mark_outliers()
        fit_state.portion = fit_state.trim_portion()
    except Exception as exc:
        # no fit state, but that's okay
        logger.warning(f"Could not load DREAM state: {exc}")
        fit_state = None

    # TODO: might be able to get method and options from the start of the .mon file
    if fit_state is not None:
        # Reconstruct dream options for pop and steps are set correctly.
        # Use negative pop for fixed number of chains independent of the number of parameters
        Nchains, Nvar, Nsteps = fit_state.Npop, fit_state.Nvar, fit_state.Nthin
        pop = Nchains / Nvar if Nchains % Nvar == 0 else -Nchains
        samples = Nchains * Nsteps
        method = "dream"
        options = dict(pop=pop, steps=0, samples=samples)
    else:
        method = "amoeba"  # arbitrary method
        options = {}

    # TODO: reload convergence data when it is available
    if fit_state is not None:
        convergence = build_convergence_from_fit_state(fit_state)
    else:
        convergence = []

    return FitResult(method=method, options=options, convergence=convergence, fit_state=fit_state)


def filehash(filename):
    with open(filename, "rb") as fd:
        # TODO: simplify to this when python min version is 3.11
        # return hashlib.file_digest(fd, "md5").hexdigest()
        chunk_size = 2**16
        hash = hashlib.md5()
        while chunk := fd.read(chunk_size):
            hash.update(chunk)
        return hash.hexdigest()


def build_convergence_from_fit_state(fit_state):
    """
    Build a pseudo-convergence array from a dream state object.

    "pseudo" because it doesn't include burn or thinning.

    Also, the best value seen during burn may be lower than the best value seen
    at the start of the buffer, our estimate of the best so far is an over
    estimate. It will look like the best is improving even though it is not.
    This is better than assuming the best occurred before the buffer started.
    """
    import numpy as np

    if fit_state is None:
        return []

    draws, point, logp = fit_state.chains()
    p = np.sort(abs(logp), axis=1)
    best = np.minimum.accumulate(p[:, 0])
    QI, Qmid = int(0.2 * logp.shape[1]), int(0.5 * logp.shape[1])
    quantiles = np.vstack((best, p[:, 0], p[:, QI], p[:, Qmid], p[:, -(QI + 1)], p[:, -1]))
    return quantiles.T


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


def run_batch_fit(options: BumpsOptions):
    # TODO: use GUI notifications instead of console monitor for console output
    # TODO: provide info such as steps k of n and chisq with emit notifications
    # async def emit_to_console(*args, **kw):
    #    ...
    #    print("console", args, kw)
    # api.EMITTERS["console"] = emit_to_console

    # Monkeypatch shutdown so it doesn't raise system exit
    signal.signal(signal.SIGINT, sigint_handler)
    on_startup, on_complete = interpret_fit_options(options)
    asyncio.run(_run_operations(on_startup, on_complete))
    # print("completed run")


def plugin_main(name: str, client: Path, version: str = ""):
    api.state.app_name = name
    api.state.app_version = version
    api.state.client_path = client
    main()


def main(options: Optional[BumpsOptions] = None):
    # TODO: where do we configure matplotlib?
    # Need to set matplotlib to a non-interactive backend because it is being used in the
    # the export thread. The next_color method calls gca() which needs to produce a blank
    # graph even when there is none (we ask for next color before making the plot).
    from bumps.mapper import using_mpi
    from .webserver import start_from_cli

    if options is None:
        options = get_commandline_options()

    logger.setLevel(LOGLEVEL[options.loglevel])
    setup_console_logging(options.loglevel)
    # from .logger import capture_warnings
    # capture_warnings(monkeypatch=True)
    logger.info(options)

    info_only = options.chisq
    webview = options.mode != "batch" and not info_only

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
    # Don't use MPI autodetect when --mpi/--no-mpi is given on the command line.
    if (options.mpi or (options.mpi is None and using_mpi())) and not info_only:
        # ** Warning **: importing MPI from mpi4py calls MPI_Init() which triggers
        # network traffic. Only import it when you know you are using MPI calls.
        from mpi4py import MPI

        is_controller = MPI.COMM_WORLD.rank == 0
        # api.state.rank = f"{MPI.COMM_WORLD.rank:3d}: "
    else:
        is_controller = True
        # api.state.rank = ""

    if webview and is_controller:  # gui mode
        print(_branding())
        start_from_cli(options)
    else:  # console mode
        run_batch_fit(options)


if __name__ == "__main__":
    main()
