"""
Bumps command line interface.

The functions in this module are used by the bumps command to implement
the command line interface.  Bumps plugin models can use them to create
stand alone applications with a similar interface.  For example, the
Refl1D application uses the following::

    from . import fitplugin
    import bumps.cli
    bumps.cli.set_mplconfig(appdatadir='Refl1D')
    bumps.cli.install_plugin(fitplugin)
    bumps.cli.main()

After completing a set of fits on related systems, a post-analysis script
can use :func:`load_model` to load the problem definition and
:func:`load_best` to load the best value  found in the fit.  This can
be used for example in experiment design, where you look at the expected
parameter uncertainty when fitting simulated data from a range of experimental
systems.
"""

__all__ = [
    "main",
    "install_plugin",
    "set_mplconfig",
    "config_matplotlib",
    "load_model",
    "preview",
    "load_best",
    "save_best",
    "resynth",
]

import sys
import os
import re
import warnings
import traceback

import shutil

import numpy as np
# np.seterr(all="raise")

from .fitters import FitDriver, StepMonitor, ConsoleMonitor, CheckpointMonitor, nllf_scale
from .mapper import MPMapper, MPIMapper, SerialMapper
from . import util
from . import initpop
from . import __version__
from . import plugin

from .util import pushdir


def install_plugin(p):
    """
    Replace symbols in :mod:`bumps.plugin` with application specific
    methods.
    """
    for symbol in plugin.__all__:
        if hasattr(p, symbol):
            setattr(plugin, symbol, getattr(p, symbol))


def load_model(path, model_options=None):
    """
    Load a model file.

    *path* contains the path to the model file.

    *model_options* are any additional arguments to the model.  The sys.argv
    variable will be set such that *sys.argv[1:] == model_options*.
    """
    from .fitproblem import load_problem

    # Change to the target path before loading model so that data files
    # can be given as relative paths in the model file.  Add the directory
    # to the python path (at the end) so that imports work as expected.
    directory, filename = os.path.split(path)
    with pushdir(directory):
        # Try a specialized model loader
        problem = plugin.load_model(filename)
        if problem is None:
            # print "loading",filename,"from",directory
            # TODO: eliminate pickle!!
            if filename.endswith("pickle"):
                try:
                    import dill as pickle
                except ImportError:
                    import pickle
                # First see if it is a pickle
                with open(filename, "rb") as fd:
                    problem = pickle.load(fd)
            else:
                # Then see if it is a python model script
                problem = load_problem(filename, options=model_options)

    # Guard against the user changing parameters after defining the problem.
    problem.model_reset()
    problem.path = os.path.abspath(path)
    if not hasattr(problem, "title"):
        problem.title = filename
    problem.name, _ = os.path.splitext(filename)
    problem.options = model_options
    return problem


def preview(problem, view=None):
    """
    Show the problem plots and parameters.
    """
    import pylab

    problem.show()
    problem.plot(view=view)
    pylab.show()


def save_best(fitdriver, problem, best, view=None):
    """
    Save the fit data, including parameter values, uncertainties and plots.

    *fitdriver* is the fitter that was used to drive the fit.

    *problem* is a FitProblem instance.

    *best* is the parameter set to save.
    """
    # Make sure the problem contains the best value
    # TODO: avoid recalculating if problem is already at best.
    problem.setp(best)
    # print "remembering best"
    pardata = "".join("%s %.15g\n" % (name, value) for name, value in zip(problem.labels(), problem.getp()))
    open(problem.output_path + ".par", "wt").write(pardata)

    fitdriver.save(problem.output_path)
    with util.redirect_console(problem.output_path + ".err"):
        fitdriver.show()
        fitdriver.plot(output_path=problem.output_path, view=view)
    fitdriver.show()
    # print "plotting"


PARS_PATTERN = re.compile(r"^(?P<label>.*) (?P<value>[^ ]*)\n$")


def load_best(problem, path):
    """
    Reload individual parameter values from a saved .par file.

    If the label does not exist in the file, use the value from the model
    as the default value. Ignore labels that do not exist in the model. In
    that way we can load parameters from an old fit with minimal fuss, even
    as we add, delete and move parameters in the model. If any parameters
    are missing, set *problem.undefined* to the a boolean index of the
    undefined parameters.

    There is an interaction with --init=eps and the par file. If any parameters
    are missing from the par file they will be randomized across the
    entire parameter range using the equivalent of --init=lhs. That means
    you can drop a # at the beginning of the line in the .par file
    and that parameter will be shuffled on restart, with the remaining
    parameters starting near the initial value.
    """
    # WARNING: Labels are not unique! Need to track multiple instances of
    # the same label.
    if not os.path.isfile(path):
        path = os.path.join(path, problem.name + ".par")
    if not os.path.isfile(path):
        raise ValueError("Parameter file %s does not exist." % path)
    labels = problem.labels()
    targets = {label: [] for label in labels}
    with open(path, "rt") as fid:
        for line in fid:
            m = PARS_PATTERN.match(line)
            label, value = m.group("label"), float(m.group("value"))
            # Accumulate values for labels only if they appear in the model.
            if label in targets:
                targets[label].append(value)

    # Populate model with named parameters in the order they occur in the
    # parameter file. Identify the missing parameters if any, adding them
    # to the the problem definition as an optional "undefined" attribute with
    # one bit for each parameter. This ugly hack is to support a previous
    # ugly hack in which undefined parameters are initialized with LHS but
    # defined parameters are initialized with eps, cov or random.
    # TODO: find a better way to "free" parameters on --pars.
    # TODO: find a way to "free" parameters on --resume.
    values, undefined = [], []
    for label, default_value in zip(labels, problem.getp()):
        remaining = targets[label]
        is_empty = not remaining
        # popping the next value from remaining modifies targets[label]
        values.append(default_value if is_empty else remaining.pop(0))
        undefined.append(is_empty)
    problem.setp(np.asarray(values))
    if any(undefined):
        problem.undefined = np.asarray(undefined)


# CRUFT
recall_best = load_best


def store_overwrite_query_gui(path):
    """
    Ask if store path should be overwritten.

    Use this in a call to :func:`make_store` from a graphical user interface.
    """
    import wx

    msg = path + " already exists. Press 'yes' to overwrite, or 'No' to abort and restart with newpath"
    msg_dlg = wx.MessageDialog(None, msg, "Overwrite Directory", wx.YES_NO | wx.ICON_QUESTION)
    retCode = msg_dlg.ShowModal()
    msg_dlg.Destroy()
    if retCode != wx.ID_YES:
        raise RuntimeError("Could not create path")


def store_overwrite_query(path):
    """
    Ask if store path should be overwritten.

    Use this in a call to :func:`make_store` from a command line interface.
    """
    print(path, "already exists.")
    print("Press 'y' to overwrite, or 'n' to abort and restart with --overwrite, --resume, or --store=newpath")
    ans = input("Overwrite [y/n]? ")
    if ans not in ("y", "Y", "yes"):
        sys.exit(1)


def make_store(problem, opts, exists_handler):
    """
    Create the store directory and populate it with the model definition file.
    """
    # Determine if command line override
    if opts.store:
        problem.store = opts.store
    if getattr(problem, "store", None) is None:
        raise RuntimeError("Need to specify '--store=path' on command line or problem.store='path' in definition file.")
    problem.output_path = os.path.join(problem.store, problem.name)

    # Check if already exists
    store_exists = os.path.exists(problem.output_path + ".par")
    if not opts.overwrite and opts.resume is None and store_exists:
        if opts.batch:
            print(
                problem.store + " already exists.  Restart with --overwrite, --resume, or --store=newpath",
                file=sys.stderr,
            )
            sys.exit(1)
        exists_handler(problem.output_path)

    # Create it and copy model
    if not os.path.exists(problem.store):
        os.mkdir(problem.store)
    shutil.copy2(problem.path, problem.store)


def run_profiler(problem, steps):
    """
    Model execution profiler.

    Run the program with "--profiler --steps=N" to generate a function
    profile chart breaking down the cost of evaluating N models.
    """
    # Here is the findings from one profiling session::
    #   23 ms total
    #    6 ms rendering model
    #    8 ms abeles
    #    4 ms convolution
    #    1 ms setting parameters and computing nllf
    from .util import profile

    p = initpop.random_init(int(steps), initial=None, bounds=None, use_point=False, problem=problem)
    # Note: map is an iterator in python 3
    profile(lambda *args: list(map(*args)), problem.nllf, p)


def run_timer(mapper, problem, steps):
    """
    Model execution timer.

    Run the program with "--timer --steps=N" to determine the average
    run time of the model.  If --parallel is included, then the model
    will be run in parallel on separate cores.
    """
    import time

    T0 = time.time()
    steps = int(steps)
    p = initpop.generate(problem, init="random", pop=-steps, use_point=False)
    if p.shape == (0,):
        # No fitting parameters --- generate an empty population
        p = np.empty((steps, 0))
    mapper(p)
    print("time per model eval: %g ms" % (1000 * (time.time() - T0) / steps,))


def start_remote_fit(problem, options, queue, notify):
    """
    Queue remote fit.
    """
    from jobqueue.client import connect

    try:
        from dill import dumps as dill_dumps

        dumps = lambda obj: dill_dumps(obj, recurse=True)
    except ImportError:
        from pickle import dumps

    data = dict(package="bumps", version=__version__, problem=dumps(problem), options=dumps(options))
    request = dict(
        service="fitter",
        version=__version__,  # fitter service version
        notify=notify,
        name=problem.title,
        data=data,
    )

    server = connect(queue)
    job = server.submit(request)
    return job


# ==== Main ====


def initial_model(opts):
    """
    Load and initialize the model.

    *opts* are the processed command line options.

    If --pars is in opts, then load the parameters from a .par file.

    If --simulate is in opts, then generate random data from the model.

    If --simrandom is in opts, then generate random data from a random model.

    If --shake is in opts, then use a random initial state for the fit.
    """
    if opts.seed is not None:
        np.random.seed(opts.seed)

    if opts.args:
        problem = load_model(opts.args[0], opts.args[1:])
        if opts.pars is not None:
            load_best(problem, opts.pars)
        if opts.simrandom:
            problem.randomize()
        if opts.simulate or opts.simrandom:
            noise = None if opts.noise == "data" else float(opts.noise)
            problem.simulate_data(noise=noise)
            print("simulation parameters")
            print(problem.summarize())
            print("chisq at simulation", problem.chisq_str())
        if opts.shake:
            problem.randomize()
    else:
        problem = None
    return problem


def resynth(fitdriver, problem, mapper, opts):
    """
    Generate maximum likelihood fits to resynthesized data sets.

    *fitdriver* is a :class:`bumps.fitters.FitDriver` object with a fitter
    already chosen.

    *problem* is a :func:`bumps.fitproblem.FitProblem` object.  It should
    be initialized with optimal values for the parameters.

    *mapper* is one of the available :mod:`bumps.mapper` classes.

    *opts* is a :class:`bumps.options.BumpsOpts` object representing the command
    line parameters.
    """
    make_store(problem, opts, exists_handler=store_overwrite_query)
    fid = open(problem.output_path + ".rsy", "at")
    fitdriver.mapper = mapper.start_mapper(problem, opts.args)
    for i in range(opts.resynth):
        problem.resynth_data()
        best, fbest = fitdriver.fit()
        scale, err = nllf_scale(problem)
        print("step %d chisq %g" % (i, scale * fbest))
        fid.write("%.15g " % (scale * fbest))
        fid.write(" ".join("%.15g" % v for v in best))
        fid.write("\n")
    problem.restore_data()
    fid.close()


def set_mplconfig(appdatadir):
    r"""
    Point the matplotlib config dir to %LOCALAPPDATA%\{appdatadir}\mplconfig.
    """
    if hasattr(sys, "frozen"):
        if os.name == "nt":
            mplconfigdir = os.path.join(os.environ["LOCALAPPDATA"], appdatadir, "mplconfig")
        elif sys.platform == "darwin":
            mplconfigdir = os.path.join(os.path.expanduser("~/Library/Caches"), appdatadir, "mplconfig")
        else:
            return  # do nothing on linux
        mplconfigdir = os.environ.setdefault("MPLCONFIGDIR", mplconfigdir)
        if not os.path.exists(mplconfigdir):
            os.makedirs(mplconfigdir)


def config_matplotlib(backend=None):
    """
    Setup matplotlib to use a particular backend.

    The backend should be 'WXAgg' for interactive use, or 'Agg' for batch.
    This distinction allows us to run in environments such as cluster computers
    which do not have wx installed on the compute nodes.

    This function must be called before any imports to pylab.  To allow
    this, modules should not import pylab at the module level, but instead
    import it for each function/method that uses it.  Exceptions can be made
    for modules which are completely dedicated to plotting, but these modules
    should never be imported at the module level.
    """
    import matplotlib as mpl

    # When running from a frozen environment created by py2exe, we will not
    # have a range of backends available, and must set the default to WXAgg.
    # With a full matplotlib distribution we can use whatever the user prefers.
    if hasattr(sys, "frozen"):
        if "MPLCONFIGDIR" not in os.environ:
            raise RuntimeError(r"MPLCONFIGDIR should be set to e.g., %LOCALAPPDATA%\YourApp\mplconfig")
        if backend is None:
            backend = "WXAgg"

    ## CRUFT: check that backend is valid, trying alternates if an import fails
    # if backend is None:
    #    backend = os.environ.get('MPLBACKEND', mpl.rcParams['backend'])
    # import importlib
    # for name in (backend, 'MacOSX', 'Qt5Agg', 'Qt4Agg', 'Gtk3Agg', 'TkAgg', 'WXAgg'):
    #    path = 'matplotlib.backends.backend_' + name.lower()
    #    try:
    #        importlib.import_module(path)
    #        backend = name
    #        break
    #    except ImportError:
    #        backend = None

    # Specify the backend to use for plotting and import backend dependent
    # classes.  This must be done before importing pyplot to have an
    # effect.  If no backend is given, let pyplot use the default.
    if backend is not None:
        mpl.use(backend)

    # Disable interactive mode so that plots are only updated on show() or
    # draw(). The interactive function must be called before importing pyplot,
    # otherwise it will have no effect.
    mpl.interactive(False)

    # configure the plot style
    line_width = 1
    pad = 2
    font_family = "Arial" if os.name == "nt" else "sans-serif"
    font_size = 12
    plot_style = {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": line_width,
        "axes.linewidth": line_width,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": line_width,
        "ytick.major.width": line_width,
        "xtick.minor.width": line_width,
        "ytick.minor.width": line_width,
        "xtick.major.pad": pad,
        "ytick.major.pad": pad,
        "xtick.top": True,
        "ytick.right": True,
        "font.size": font_size,
        "font.family": font_family,
        "svg.fonttype": "none",
        "savefig.dpi": 100,
    }
    mpl.rcParams.update(plot_style)


def beep():
    """
    Audio signal that fit is complete.
    """
    if sys.platform == "win32":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass
    else:
        print("\a", file=sys.__stdout__)


def run_command(c):
    """
    Run an arbitrary python command.
    """
    exec(c, globals())


def setup_logging():
    """Start logger"""
    import logging

    logging.basicConfig(level=logging.INFO)


# From http://stackoverflow.com/questions/22373927/get-traceback-of-warnings
# answered by mgab (2014-03-13)
# edited by Gareth Rees (2015-11-28)
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """
    Alternate warning printer which shows a traceback with the warning.

    To use, set *warnings.showwarning = warn_with_traceback*.
    """
    traceback.print_stack()
    log = file if hasattr(file, "write") else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main():
    """
    Run the bumps program with the command line interface.

    Input parameters are taken from sys.argv.
    """
    from . import options

    # add full traceback to warnings
    # warnings.showwarning = warn_with_traceback

    if len(sys.argv) == 1:
        sys.argv.append("-?")

    # run command with bumps in the environment
    if sys.argv[1] == "-m":
        import runpy

        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__")
        sys.exit(0)
    elif sys.argv[1] == "-p":
        import runpy

        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        sys.exit()
    elif sys.argv[1] == "-c":
        run_command(sys.argv[2])
        sys.exit()
    elif sys.argv[1] == "-i":
        sys.argv = ["ipython", "--pylab"]
        from IPython import start_ipython

        sys.exit(start_ipython())

    opts = options.getopts()
    setup_logging()

    if opts.edit:
        from .gui.gui_app import main as gui

        gui()
        return

    # Set up the matplotlib backend to minimize the wx/gui dependency.
    # If no GUI specified and not editing, then use the default mpl
    # backend for the python version.
    if opts.batch or opts.remote or opts.noshow:  # no interactivity
        config_matplotlib(backend="agg")
    else:  # let preview use default graphs
        config_matplotlib()

    problem = initial_model(opts)
    if problem is None:
        print("\n!!! Model file missing from command line --- abort !!!.", file=sys.stderr)
        sys.exit(1)

    if opts.mpi:
        MPIMapper.start_worker(problem)
        mapper = MPIMapper
    elif opts.parallel != "" or opts.worker:
        if opts.transport == "mp":
            mapper = MPMapper
        else:
            raise ValueError("unknown mapper")
    else:
        mapper = SerialMapper
    if opts.worker:
        mapper.start_worker(problem)
        return

    if np.isfinite(float(opts.time)):
        import time

        start_time = time.time()
        stop_time = start_time + float(opts.time) * 3600
        abort_test = lambda: time.time() >= stop_time
    else:
        abort_test = lambda: False

    fitdriver = FitDriver(
        opts.fit_config.selected_fitter, problem=problem, abort_test=abort_test, **opts.fit_config.selected_values
    )

    # Start fitter within the domain so that constraints are valid
    clipped = fitdriver.clip()
    if clipped:
        print("Start value clipped to range for parameter", ", ".join(clipped))

    if opts.time_model:
        run_timer(mapper.start_mapper(problem, opts.args), problem, steps=int(opts.steps))
    elif opts.profile:
        run_profiler(problem, steps=int(opts.steps))
    elif opts.chisq:
        if opts.cov:
            fitdriver.show_cov()
        print("chisq", problem.chisq_str())
        # import pprint; pprint.pprint(problem.to_dict(), indent=2, width=272)
    elif opts.preview:
        if opts.cov:
            fitdriver.show_cov()
        preview(problem, view=opts.view)
    elif opts.resynth > 0:
        resynth(fitdriver, problem, mapper, opts)

    elif opts.remote:
        # Check that problem runs before submitting it remotely
        # TODO: this may fail if problem requires remote resources such as GPU
        print("initial chisq:", problem.chisq_str())
        job = start_remote_fit(problem, opts, queue=opts.queue, notify=opts.notify)
        print("remote job:", job["id"])

    else:
        # Show command line arguments and initial model
        print("#", " ".join(sys.argv), "--seed=%d" % opts.seed)
        problem.show()

        # Check that there are parameters to be fitted.
        if not len(problem.getp()):
            print("\n!!! No parameters selected for fitting---abort !!!\n", file=sys.stderr)
            sys.exit(1)

        # Run the fit
        if opts.resume == "-":
            opts.resume = opts.store if os.path.exists(opts.store) else None
        if opts.resume:
            resume_path = os.path.join(opts.resume, problem.name)
        else:
            resume_path = None

        make_store(problem, opts, exists_handler=store_overwrite_query)

        # Redirect sys.stdout to capture progress
        if opts.batch:
            sys.stdout = open(problem.output_path + ".mon", "w")

        # TODO: fix techical debt with checkpoint monitor implementation
        # * The current checkpoint implementation is self-referential:
        #     checkpoint = lambda: save_best(fitdriver, ...)
        #     fitdriver.monitors = [..., CheckpointMonitor(checkpoint), ...]
        #   It is done this way because the checkpoint monitor needs the fitter
        #   so it can ask it to save state, but the fitter needs the list of
        #   monitors, including the checkpoint monitor, before it is run.
        # * Figures are cumulative, with each checkpoint adding a new set
        # * Figures are slow! Can they go into a separate thread?  Can we
        #   have the problem cache the best value?
        checkpoint_time = float(opts.checkpoint) * 3600

        def checkpoint(history):
            problem = fitdriver.problem
            ## Use the following to save only the fitter state
            fitdriver.fitter.save(problem.output_path)
            ## Use the following to save the fitter state plus all other
            ## plots and other output files.  This won't work yet since
            ## plots are generated sequentially, with each checkpoint producing
            ## a completely new set of plots.
            # best = history.point[0]
            # save_best(fitdriver, problem, best, view=opts.view)

        monitors = [ConsoleMonitor(problem)]
        if checkpoint_time > 0 and np.isfinite(checkpoint_time):
            mon = CheckpointMonitor(checkpoint, progress=checkpoint_time)
            monitors.append(mon)
        if opts.stepmon:
            fid = open(problem.output_path + ".log", "w")
            mon = StepMonitor(problem, fid, fields=["step", "value"])
            monitors.append(mon)
        fitdriver.monitors = monitors

        # import time; t0=time.clock()
        cpus = int(opts.parallel) if opts.parallel != "" else 0
        fitdriver.mapper = mapper.start_mapper(problem, opts.args, cpus=cpus)
        best, fbest = fitdriver.fit(resume=resume_path)
        # print("time=%g"%(time.clock()-t0),file=sys.__stdout__)
        # Note: keep this in sync with the checkpoint function above
        save_best(fitdriver, problem, best, view=opts.view)
        if opts.err or opts.cov:
            fitdriver.show_err()
        if opts.cov:
            fitdriver.show_cov()
        if opts.entropy:
            fitdriver.show_entropy(opts.entropy)
        mapper.stop_mapper()

        # If in batch mode then explicitly close the monitor file on completion
        if opts.batch:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

        # Display the plots
        if not opts.batch and not opts.mpi and not opts.noshow:
            beep()
            import pylab

            pylab.show()


# Allow  "$python -m bumps.cli args" calling pattern
if __name__ == "__main__":
    main()
