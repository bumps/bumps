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
from __future__ import with_statement, print_function

__all__ = ["main", "install_plugin", "set_mplconfig", "config_matplotlib",
           "load_model", "preview", "load_best", "save_best", "resynth"]

import sys
import os
import re

import shutil
try:
    import dill as pickle
except:
    import pickle

import numpy as np
# np.seterr(all="raise")

from . import fitters
from .fitters import FIT_OPTIONS, FitDriver, StepMonitor, ConsoleMonitor, nllf_scale
from .mapper import MPMapper, AMQPMapper, MPIMapper, SerialMapper
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
    # can be given as relative paths in the model file.  This should also
    # allow imports as expected from the model file.
    directory, filename = os.path.split(path)
    with pushdir(directory):
        # Try a specialized model loader
        problem = plugin.load_model(filename)
        if problem is None:
            # print "loading",filename,"from",directory
            if filename.endswith('pickle'):
                # First see if it is a pickle
                problem = pickle.load(open(filename, 'rb'))
            else:
                # Then see if it is a python model script
                problem = load_problem(filename, options=model_options)

    # Guard against the user changing parameters after defining the problem.
    problem.model_reset()
    problem.path = os.path.abspath(path)
    if not hasattr(problem, 'title'):
        problem.title = filename
    problem.name, _ = os.path.splitext(filename)
    problem.options = model_options
    return problem


def preview(problem):
    """
    Show the problem plots and parameters.
    """
    import pylab
    problem.show()
    problem.plot()
    pylab.show()


def save_best(fitdriver, problem, best):
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
    pardata = "".join("%s %.15g\n" % (name, value)
                      for name, value in zip(problem.labels(), problem.getp()))
    open(problem.output_path + ".par", 'wt').write(pardata)

    fitdriver.save(problem.output_path)
    with util.redirect_console(problem.output_path + ".err"):
        fitdriver.show()
        fitdriver.plot(problem.output_path)
    fitdriver.show()
    # print "plotting"


PARS_PATTERN = re.compile(r"^(?P<label>.*) (?P<value>[^ ]*)\n$")
def load_best(problem, path):
    """
    Load parameter values from a file.
    """
    labels, values = [], []
    with open(path, 'rt') as fid:
        for line in fid:
            m = PARS_PATTERN.match(line)
            labels.append(m.group('label'))
            values.append(float(m.group('value')))
    assert labels == problem.labels()
    problem.setp(np.asarray(values))
#CRUFT
recall_best = load_best


def store_overwrite_query_gui(path):
    """
    Ask if store path should be overwritten.

    Use this in a call to :func:`make_store` from a graphical user interface.
    """
    import wx
    msg_dlg = wx.MessageDialog(None, path + " already exists. Press 'yes' to overwrite, or 'No' to abort and restart with newpath", 'Overwrite Directory',
                               wx.YES_NO | wx.ICON_QUESTION)
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
    print(
        "Press 'y' to overwrite, or 'n' to abort and restart with --store=newpath")
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
    problem.output_path = os.path.join(problem.store, problem.name)

    # Check if already exists
    if not opts.overwrite and os.path.exists(problem.output_path + '.out'):
        if opts.batch:
            print(
                problem.store + " already exists.  Use --overwrite to replace.", file=sys.stderr)
            sys.exit(1)
        exists_handler(problem.output_path)

    # Create it and copy model
    try:
        os.mkdir(problem.store)
    except:
        pass
    shutil.copy2(problem.path, problem.store)

    # Redirect sys.stdout to capture progress
    if opts.batch:
        sys.stdout = open(problem.output_path + ".mon", "w")


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
    p = initpop.random_init(int(steps), None, problem)
    profile(map, problem.nllf, p)


def run_timer(mapper, problem, steps):
    """
    Model execution timer.

    Run the program with "--timer --steps=N" to determine the average
    run time of the model.  If --parallel is included, then the model
    will be run in parallel on separate cores.
    """
    import time
    T0 = time.time()
    p = initpop.random_init(int(steps), None, problem)
    mapper(p)
    print("time per model eval: %g ms" % (1000 * (time.time() - T0) / steps,))


def start_remote_fit(problem, options, queue, notify):
    """
    Queue remote fit.
    """
    from jobqueue.client import connect

    data = dict(package='bumps',
                version=__version__,
                problem=pickle.dumps(problem),
                options=pickle.dumps(options))
    request = dict(service='fitter',
                   version=__version__,  # fitter service version
                   notify=notify,
                   name=problem.title,
                   data=data)

    server = connect(queue)
    job = server.submit(request)
    return job


# ==== option parser ====

class ParseOpts:
    """
    Options parser.

    Subclass should define *MINARGS*, *FLAGS*, *VALUES* and *USAGE*.

    *MINARGS* is the minimum number of positional arguments.

    *FLAGS* is a set of arguments that may be present or absent.

    *VALUES* is a set of arguments that take values.  Value checking
    can be done in the setter for each argument in the set.  Default
    values should be set in the corresponding object attribute.

    *USAGE* is the help string to display for option "help".

    The constructor will invoke the command line parser, leaving the
    values set by the command line as attribute values.   Flag options
    will be True or False.
    """
    MINARGS = 0
    FLAGS = set()
    VALUES = set()
    USAGE = ""

    def __init__(self, args):
        self._parse(args)

    def _parse(self, args):
        flagargs = [v
                    for v in sys.argv[1:]
                    if v.startswith('--') and not '=' in v]
        flags = set(v[2:] for v in flagargs)
        if 'help' in flags or '-h' in sys.argv[1:] or '-?' in sys.argv[1:]:
            print(self.USAGE)
            sys.exit()
        unknown = flags - self.FLAGS
        if any(unknown):
            raise ValueError("Unknown options --%s.  Use -? for help."
                             % ", --".join(unknown))
        for f in self.FLAGS:
            setattr(self, f, (f in flags))

        valueargs = [v
                     for v in sys.argv[1:]
                     if v.startswith('--') and '=' in v]
        for f in valueargs:
            idx = f.find('=')
            name = f[2:idx]
            value = f[idx + 1:]
            if name not in self.VALUES:
                raise ValueError(
                    "Unknown option --%s. Use -? for help." % name)
            setattr(self, name, value)

        positionargs = [v for v in sys.argv[1:] if not v.startswith('-')]
        self.args = positionargs


class BumpsOpts(ParseOpts):
    """
    Option parser for bumps.
    """
    MINARGS = 1
    FLAGS = set(("preview", "chisq", "profiler", "timer",
                 "simulate", "simrandom", "shake",
                 "worker", "batch", "overwrite", "parallel", "stepmon",
                 "cov", "remote", "staj", "edit", "mpi", "keep_best",
                 # passed in when app is a frozen image
                 "multiprocessing-fork",
                 # passed when not running bumps, but instead using a
                 # bundled application as a python distribution with domain
                 # specific models pre-defined.
                 "i",
                 ))
    VALUES = set(("plot", "store", "resume", "fit", "noise", "seed", "pars",
                  "resynth", "transport", "notify", "queue", "time",
                  "m", "c", "p",
                  ))
    # Add in parameters from the fitters
    VALUES |= set(fitters.FitOptions.FIELDS.keys())
    pars = None
    notify = ""
    queue = "http://reflectometry.org/queue"
    resynth = "0"
    noise = "5"
    starts = "1"
    seed = ""
    time = "inf"
    PLOTTERS = "linear", "log", "residuals"
    USAGE = """\
Usage: bumps [options] modelfile [modelargs]

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.

Options:

    --preview
        display model but do not perform a fitting operation
    --pars=filename
        initial parameter values; fit results are saved as <modelname>.par
    --plot=log      [%(plotter)s]
        type of plot to display
    --simulate
        simulate a dataset using the initial problem parameters
    --simrandom
        simulate a dataset using random problem parameters
    --shake
        set random parameters before fitting
    --noise=5%%
        percent noise to add to the simulated data
    --seed=integer
        random number seed
    --cov
        compute the covariance matrix for the model when done
    --staj
        output staj file when done
    --edit
        start the gui

    --store=path
        output directory for plots and models
    --overwrite
        if store already exists, replace it
    --resume=path    [dream]
        resume a fit from previous stored state
    --parallel
        run fit using multiprocessing for parallelism
    --mpi
        run fit using MPI for parallelism (use command "mpirun -n cpus ...")
    --batch
        batch mode; don't show plots after fit
    --remote
        queue fit to run on remote server
    --notify=user@email
        remote fit notification
    --queue=http://reflectometry.org
        remote job queue
    --time=inf
        run for a maximum number of hours

    --fit=amoeba    [%(fitter)s]
        fitting engine to use; see manual for details
    --steps=400    [%(fitter)s]
        number of fit iterations after any burn-in time
    --xtol=1e-4     [de, amoeba]
        minimum population diameter
    --ftol=1e-4     [de, amoeba]
        minimum population flatness
    --pop=10        [dream, de, rl, ps]
        population size
    --burn=100      [dream, pt]
        number of burn-in iterations before accumulating stats
    --thin=1        [dream]
        number of fit iterations between steps
    --entropy=no    [dream]
        compute entropy from MCMC chain
    --nT=25
    --Tmin=0.1
    --Tmax=10       [pt]
        temperatures vector; use a higher maximum temperature and a larger
        nT if your fit is getting stuck in local minima
    --CR=0.9        [de, rl, pt]
        crossover ratio for population mixing
    --starts=1      [%(fitter)s]
        number of times to run the fit from random starting points.
    --keep_best
        when running with multiple starts, restart from a point near the
        last minimum rather than using a completely random starting point.
    --init=eps      [dream]
        population initialization method:
          eps:    ball around initial parameter set
          lhs:    latin hypercube sampling
          cov:    normally distributed according to covariance matrix
          random: uniformly distributed within parameter ranges
    --stepmon
        show details for each step
    --resynth=0
        run resynthesis error analysis for n generations

    --timer
        run the model --steps times in order to estimate total run time.
    --profiler
        run the python profiler on the model; use --steps to run multiple
        models for better statistics
    --chisq
        print the model description and chisq value and exit
    -m/-c/-p command
        run the python interpreter with bumps on the path:
            m: command is a module such as bumps.cli, run as __main__
            c: command is a python one-line command
            p: command is the name of a python script
    -i
        start the interactive interpreter
    -?/-h/--help
        display this help
""" % {'fitter': '|'.join(sorted(FIT_OPTIONS.keys())),
       'plotter': '|'.join(PLOTTERS),
       }

#    --transport=mp  {amqp|mp|mpi}
#        use amqp/multiprocessing/mpi for parallel evaluation

    _plot = 'log'

    def _set_plot(self, value):
        if value not in set(self.PLOTTERS):
            raise ValueError("unknown plot type %s; use %s"
                             % (value, "|".join(self.PLOTTERS)))
        self._plot = value
    plot = property(fget=lambda self: self._plot, fset=_set_plot)
    store = None
    resume = None
    _fitter = fitters.FIT_DEFAULT

    def _set_fitter(self, value):
        if value not in set(FIT_OPTIONS.keys()):
            raise ValueError("unknown fitter %s; use %s"
                             % (value, "|".join(sorted(FIT_OPTIONS.keys()))))
        self._fitter = value
    fit = property(fget=lambda self: self._fitter, fset=_set_fitter)
    TRANSPORTS = 'amqp', 'mp', 'mpi', 'celery'
    _transport = 'mp'

    def _set_transport(self, value):
        if value not in self.TRANSPORTS:
            raise ValueError("unknown transport %s; use %s"
                             % (value, "|".join(self.TRANSPORTS)))
        self._transport = value
    transport = property(
        fget=lambda self: self._transport, fset=_set_transport)
    meshsteps = 40


def getopts():
    """
    Process command line options.

    Option values will be stored as attributes in the returned object.
    """
    opts = BumpsOpts(sys.argv)
    opts.resynth = int(opts.resynth)
    opts.seed = int(opts.seed) if opts.seed != "" else None
    fitters.FIT_DEFAULT = opts.fit
    FIT_OPTIONS[opts.fit].set_from_cli(opts)
    return opts

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
        problem = load_model(opts.args[0],opts.args[1:])
        if opts.pars is not None:
            load_best(problem, opts.pars)
        if opts.simrandom:
            problem.randomize()
        if opts.simulate or opts.simrandom:
            noise = None if opts.noise == "data" else float(opts.noise)
            problem.simulate_data(noise=noise)
            print("simulation parameters")
            print(problem.summarize())
            print("chisq at simulation", problem.chisq())
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

    *opts* is a :class:`bumps.cli.BumpsOpts` object representing the command
    line parameters.
    """
    make_store(problem, opts, exists_handler=store_overwrite_query)
    fid = open(problem.output_path + ".rsy", 'at')
    fitdriver.mapper = mapper.start_mapper(problem, opts.args)
    for i in range(opts.resynth):
        problem.resynth_data()
        best, fbest = fitdriver.fit()
        scale, err = nllf_scale(problem)
        print("step %d chisq %g" % (i, scale * fbest))
        fid.write('%.15g ' % (scale * fbest))
        fid.write(' '.join('%.15g' % v for v in best))
        fid.write('\n')
    problem.restore_data()
    fid.close()


def set_mplconfig(appdatadir):
    r"""
    Point the matplotlib config dir to %LOCALAPPDATA%\{appdatadir}\mplconfig.
    """
    import os
    import sys
    if hasattr(sys, 'frozen'):
        mplconfigdir = os.path.join(
            os.environ['LOCALAPPDATA'], appdatadir, 'mplconfig')
        mplconfigdir = os.environ.setdefault('MPLCONFIGDIR', mplconfigdir)
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

    # When running from a frozen environment created by py2exe, we will not
    # have a range of backends available, and must set the default to WXAgg.
    # With a full matplotlib distribution we can use whatever the user prefers.
    if hasattr(sys, 'frozen'):
        if 'MPLCONFIGDIR' not in os.environ:
            raise RuntimeError(
                "MPLCONFIGDIR should be set to e.g., %LOCALAPPDATA%\YourApp\mplconfig")
        if backend is None:
            backend = 'WXAgg'

    import matplotlib

    # Specify the backend to use for plotting and import backend dependent
    # classes. Note that this must be done before importing pyplot to have an
    # effect.  If no backend is given, let pyplot use the default.
    if backend is not None:
        matplotlib.use(backend)

    # Disable interactive mode so that plots are only updated on show() or
    # draw(). Note that the interactive function must be called before
    # selecting a backend or importing pyplot, otherwise it will have no
    # effect.

    matplotlib.interactive(False)


def beep():
    """
    Audio signal that fit is complete.
    """
    if sys.platform == "win32":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    else:
        print("\a", file=sys.__stdout__)


def run_command(c):
    """
    Run an arbitrary python command.
    """
    exec(c, globals())


def main():
    """
    Run the bumps program with the command line interface.

    Input parameters are taken from sys.argv.
    """
    if len(sys.argv) == 1:
        sys.argv.append("-?")
        print("\nNo modelfile parameter was specified.\n")

    # run command with bumps in the environment
    if sys.argv[1] == '-m':
        import runpy
        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__")
        sys.exit()
    elif sys.argv[1] == '-p':
        import runpy
        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        sys.exit()
    elif sys.argv[1] == '-c':
        run_command(sys.argv[2])
        sys.exit()
    elif sys.argv[1] == '-i':
        sys.argv = ["ipython", "--pylab"]
        from IPython import start_ipython
        sys.exit(start_ipython())

    opts = getopts()

    if opts.edit:
        from .gui.gui_app import main as gui
        gui()
        return

    # Set up the matplotlib backend to minimize the wx/gui dependency.
    # If no GUI specified and not editing, then use the default mpl
    # backend for the python version.
    if opts.batch or opts.remote:  # no interactivity
        config_matplotlib(backend='Agg')
    else:  # let preview use default graphs
        config_matplotlib()

    problem = initial_model(opts)

    # TODO: AMQP mapper as implemented requires workers started up with
    # the particular problem; need to be able to transport the problem
    # to the worker instead.  Until that happens, the GUI shouldn't use
    # the AMQP mapper.
    if opts.mpi:
        MPIMapper.start_worker(problem)
        mapper = MPIMapper
    elif opts.parallel or opts.worker:
        if opts.transport == 'amqp':
            mapper = AMQPMapper
        elif opts.transport == 'mp':
            mapper = MPMapper
        elif opts.transport == 'celery':
            mapper = CeleryMapper
    else:
        mapper = SerialMapper
    if opts.worker:
        mapper.start_worker(problem)
        return

    fitopts = FIT_OPTIONS[opts.fit]
    if np.isfinite(float(opts.time)):
        import time
        start_time = time.clock()
        stop_time = start_time + float(opts.time)*3600
        abort_test=lambda: time.clock() >= stop_time
    else:
        abort_test=lambda: False
    fitdriver = FitDriver(
        fitopts.fitclass, problem=problem, abort_test=abort_test, **fitopts.options)

    if opts.timer:
        run_timer(mapper.start_mapper(problem, opts.args),
                  problem, steps=int(opts.steps))
    elif opts.profiler:
        run_profiler(problem, steps=int(opts.steps))
    elif opts.chisq:
        if opts.cov:
            print(problem.cov())
        print("chisq", problem.chisq_str())
    elif opts.preview:
        if opts.cov:
            print(problem.cov())
        preview(problem)
    elif opts.resynth > 0:
        resynth(fitdriver, problem, mapper, opts)

    elif opts.remote:

        # Check that problem runs before submitting it remotely
        chisq = problem()
        print("initial chisq:", chisq)
        job = start_remote_fit(problem, opts,
                               queue=opts.queue, notify=opts.notify)
        print("remote job:", job['id'])

    else:
        if opts.resume:
            resume_path = os.path.join(opts.resume, problem.name)
        else:
            resume_path = None

        make_store(problem, opts, exists_handler=store_overwrite_query)

        # Show command line arguments and initial model
        print("#", " ".join(sys.argv))
        problem.show()
        if opts.stepmon:
            fid = open(problem.output_path + '.log', 'w')
            fitdriver.monitors = [ConsoleMonitor(problem),
                                  StepMonitor(problem, fid, fields=['step', 'value'])]

        #import time; t0=time.clock()
        fitdriver.mapper = mapper.start_mapper(problem, opts.args)
        best, fbest = fitdriver.fit(resume=resume_path)
        # print("time=%g"%(time.clock()-t0),file=sys.__stdout__)
        save_best(fitdriver, problem, best)
        if opts.cov:
            print(problem.cov())
        mapper.stop_mapper(fitdriver.mapper)
        beep()
        if not opts.batch and not opts.mpi:
            import pylab
            pylab.show()


# Allow  "$python -m bumps.cli args" calling pattern
if __name__ == "__main__":
    main()
