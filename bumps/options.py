"""
Option parser for bumps command line
"""

import sys

import numpy as np

from .fitters import FITTERS, FIT_AVAILABLE_IDS, FIT_ACTIVE_IDS, FIT_DEFAULT_ID


# TODO: replace with standard argparse module
class ParseOpts(object):
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
    #: Value to use if a value flag is is present without '='.  This is
    #: different from the default value if the flag is not present, which
    #: is the default value set in the calling class.
    IMPLICIT_VALUES = {}
    USAGE = ""

    def __init__(self, args):
        if self.VALUES & self.FLAGS:
            raise TypeError("option used as both a flag and a value: %s" % ",".join(self.VALUES & self.FLAGS))
        self._parse(args)

    def _parse(self, args):
        if any(v in args for v in ("-?", "-h", "-help")):
            print(self.USAGE)
            sys.exit()

        # Drop the "bumps" arg from the beginning of the list.
        args = args[1:]

        # Fill in implicit values. We need to do this to support something
        # like "bumps ... --parallel=2 ... --parallel", which should have the
        # later (implicit) parameter take precedence over the earlier
        # parameters.
        args = [
            (arg + "=" + str(self.IMPLICIT_VALUES[arg[2:]]) if arg[2:] in self.IMPLICIT_VALUES else arg) for arg in args
        ]

        # Parse options.
        # Given tuples [..., (a, 1), ..., (a, 2), ...], then dict(tuples)
        # will use the later value for the key rather than the earlier
        # value, which is what we want for the command line interpreter.
        position_args = [v for v in sys.argv[1:] if not v.startswith("--")]
        flag_args = [
            v[2:]  # convert --flag => flag
            for v in args
            if v.startswith("--") and not "=" in v
        ]
        value_args = dict(
            v[2:].split("=", 1)  # convert --flag=value => (flag, value)
            for v in args
            if v.startswith("--") and "=" in v
        )

        # Check that options are valid.
        # TODO: move type checking from FitConfig.set_from_cli to here.
        flags, values = set(flag_args), set(value_args.keys())
        unknown = (flags | values) - (self.FLAGS | self.VALUES)
        unexpected_value = flags - self.FLAGS
        blank_values = set(k for k, v in value_args.items() if k in self.VALUES and v == "")
        missing_value = (values - self.VALUES) | blank_values
        errors = []
        if any(unknown):
            errors.append("Unknown options --%s." % ", --".join(unknown))
        if any(unexpected_value):
            errors.append("Unexpected value for --%s." % ", --".join(sorted(unexpected_value)))
        if any(missing_value):
            errors.append("Missing value for --%s." % ", --".join(sorted(missing_value)))
        if errors:
            message = " ".join(errors + ["Use -? for help."])
            raise ValueError(message)

        # Set the values into the fields.
        for option in self.FLAGS:
            setattr(self, option, (option in flags))

        for option, value in value_args.items():
            setattr(self, option, value)

        self.args = position_args


# === Fitter option parsing ===


class ChoiceList(object):
    def __init__(self, *choices):
        self.choices = choices

    def __call__(self, value):
        if not value in self.choices:
            raise ValueError('invalid option "%s": use %s' % (value, "|".join(self.choices)))
        else:
            return value


def yesno(value):
    if value.lower() in ("true", "yes", "on", "1"):
        return True
    elif value.lower() in ("false", "no", "off", "0"):
        return False
    raise ValueError('invalid option "%s": use yes|no')


def parse_int(value):
    float_value = float(value)
    if int(float_value) != float_value:
        raise ValueError("integer expected")
    return int(float_value)


FIT_FIELDS = dict(
    steps=("Steps", parse_int),
    samples=("Samples", parse_int),
    xtol=("x tolerance", float),
    ftol=("f(x) tolerance", float),
    alpha=("Convergence", float),
    stop=("Stopping criteria", str),
    thin=("Thinning", parse_int),
    burn=("Burn-in steps", parse_int),
    pop=("Population", float),
    init=("Initializer", ChoiceList("eps", "lhs", "cov", "random")),
    CR=("Crossover ratio", float),
    F=("Scale", float),
    nT=("# Temperatures", parse_int),
    Tmin=("Min temperature", float),
    Tmax=("Max temperature", float),
    radius=("Simplex radius", float),
    # TODO: convert --trim into a boolean flag and update docs
    trim=("Burn-in trim", yesno),
    outliers=("Outliers", ChoiceList("none", "iqr", "grubbs", "mahal")),
    starts=("Starts", parse_int),
    jump=("Jump radius", float),
)

# Make sure all settings are parseable
for fit in FITTERS:
    assert all(opt in FIT_FIELDS for opt, _ in fit.settings), "Fitter %s contains unknown settings %s" % (
        fit.id,
        ", ".join(opt for opt, _ in sorted(fit.settings) if opt not in FIT_FIELDS),
    )
del fit


class FitConfig(object):
    """
    Fit settings configuration object.

    The command line parser will define a FitConfig object which contains
    the fitter that was given on the command line and all its options.  For
    embedded bumps, which does not use the bumps command line parser, a
    new FitConfig object can be created with its own selected options.

    **Attributes**

    *ids = [id, id, ...]* is a list available fitters in "preferred" order.
    Depending on usage, you may want to sort them, or alternatively, sort
    by long name with *[id for _,id in sorted((v,k) for k,v in self.names]*

    *fitters = {id: fitclasss}* maps ids to fitters.

    *names* = {id: name}* maps ids to long names

    *settings = {id: [(option, default), ...]}* maps ids to default settings.
    The order of the settings is the preferred order to present the settings
    to the user in a GUI dialog for example.

    *values = {id: {option: value, ...}}* maps ids to the settings for
    each fitter.  Note that in the GUI, different fitters may have there
    settings recorded and preserved even when not selected.

    *active_ids = [id, id, ...]* is the list of fitters to show the user in
    a GUI dialog for example.  The other fitters should still be available from
    the command line.

    *default_id = id* is the fitter to use by default.

    *selected_id = id* is the fitter that was selected, either by command line
    or by GUI.

    *selected_values = {option: value}* returns the settings for the current
    fitter.

    *selected_name = name* returns the name of the selected fitter.

    *selected_fitter = FitClass* returns the class of the selected fitter.

    """

    def __init__(self, default=FIT_DEFAULT_ID, active=FIT_ACTIVE_IDS):
        # Keep a private copy of the configure settings rather than modifying
        # the global defaults
        self.ids = [fit.id for fit in FITTERS]
        # FITTERS is a list of FitBase classes
        # Each class has:
        #     fit.id: the short name used on the command line
        #     fit.name: the long name used in the GUI
        #     fit.settings: available options: [(key,default value), ...]
        self.fitters = dict((fit.id, fit) for fit in FITTERS)
        self.names = dict((fit.id, fit.name) for fit in FITTERS)
        self.settings = dict((fit.id, fit.settings) for fit in FITTERS)
        self.values = dict((fit.id, dict(fit.settings)) for fit in FITTERS)
        if not all(k in self.ids for k in active):
            raise ValueError("Some active fitters are not available")
        if default not in active:
            raise ValueError("default fitter is not active")
        self.active_ids = active
        self.default_id = default
        self.selected_id = default

    def set_from_cli(self, opts):
        """
        Use the BumpsOpts command line parser values to set the selected
        fitter and its configuration options.
        """
        fitter = opts.fit
        self.selected_id = fitter
        # Convert supplied options to the correct types and save them in value
        for field, reset_value in self.settings[fitter]:
            value = getattr(opts, field, None)
            parse = FIT_FIELDS[field][1]
            if value is not None:
                try:
                    self.values[fitter][field] = parse(value)
                except Exception as exc:
                    raise ValueError("error in --%s: %s" % (field, str(exc)))
                    # print("options=%s"%(str(self.options)))

    @property
    def selected_values(self):
        return self.values[self.selected_id]

    @property
    def selected_name(self):
        return self.names[self.selected_id]

    @property
    def selected_fitter(self):
        return self.fitters[self.selected_id]


#: FitConfig singleton for the common case in which only one config is needed.
#: There may be other use cases, such as saving the fit config along with the
#: rest of the state so that on resume the fit options are restored, but in that
#: case the application will not be using the singleton.
FIT_CONFIG = FitConfig()


# === Bumps options parsing ===
class BumpsOpts(ParseOpts):
    """
    Option parser for bumps.
    """

    MINARGS = 1
    # TODO: document all options in USAGE and doc/guide/options.rst
    # TODO: remove application-specific options like --staj
    FLAGS = set(
        (
            "preview",
            "chisq",
            "profile",
            "time_model",
            "simulate",
            "simrandom",
            "shake",
            "worker",  # internal, so not documented
            "multiprocessing-fork",  # passed in when app is a frozen image
            "remote",  # not active, so not documented
            "batch",
            "noshow",
            "overwrite",
            "stepmon",
            "err",
            "cov",
            "edit",
            "mpi",
            "staj",
            # passed when not running bumps, but instead using a
            # bundled application as a python distribution with domain
            # specific models pre-defined.
            "i",
        )
    )
    VALUES = set(
        (
            "plot",
            "store",
            "resume",
            "entropy",
            "fit",
            "noise",
            "seed",
            "pars",
            "resynth",
            "time",
            "checkpoint",
            "m",
            "c",
            "p",
            "parallel",
            "view",
            "trim",
            "near_best",
            "alpha",
            "outliers",
            # The following options are for remote fitting via the
            # fitting service, but this is not currently active.
            "transport",
            "notify",
            "queue",
        )
    )
    # Add in parameters from the fitters
    VALUES |= set(FIT_FIELDS.keys())
    # --parallel is equivalent to --parallel=0
    IMPLICIT_VALUES = {
        "parallel": "0",
        "entropy": "llf",
        "resume": "-",
    }
    pars = None
    notify = ""
    queue = None
    resynth = "0"
    noise = "5"
    starts = "1"
    seed = ""
    time = "inf"
    checkpoint = "0"
    parallel = ""
    entropy = None
    trim = "true"
    view = None
    alpha = 0.0
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
    --pars=filename or store path
        initial parameter values; fit results are saved as path/<modelname>.par
    --plot=log      [%(plotter)s]
        type of plot to display
    --trim=true
        trim any remaining burn before displaying plots [dream only]
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
    --err
        show uncertainty estimate from curvature at the minimum
    --cov
        show the covariance matrix for the model when done
    --entropy=gmm|mvn|wnn|llf
        compute entropy on posterior distribution [dream only]
    --staj
        output staj file when done [Refl1D only]
    --edit
        start the gui
    --view=linear|log
        one of the predefined problem views; reflectometry also has fresnel,
        logfresnel, q4 and residuals

    --store=path
        output directory for plots and models
    --overwrite
        if store already exists, replace it
    --resume=path    [dream]
        resume a fit from previous stored state; if path is '-' then use the
        path given by --store, if it exists
    --parallel=n
        run fit using multiprocessing for parallelism; use --parallel=0 for all cpus
    --mpi
        run fit using MPI for parallelism (use command "mpirun -n cpus ...")
    --batch
        batch mode; save output in .mon file and don't show plots after fit
    --noshow
        semi-batch; send output to console but don't show plots after fit
    --time=inf
        run for a maximum number of hours
    --checkpoint=0
        save fit state every n hours, or 0 for no checkpoints

    --fit=amoeba    [%(fitter)s]
        fitting engine to use; see manual for details
    --steps=0       [%(fitter)s]
        number of fit iterations after any burn-in time; use samples if steps=0
    --samples=1e4   [dream]
        set steps=samples/(pop*#pars) so the target number of samples is drawn
    --xtol=1e-4     [de, amoeba]
        minimum population diameter
    --ftol=1e-4     [de, amoeba]
        minimum population flatness
    --alpha=0.0     [dream]
        p-level for rejecting convergence; with fewer samples use a stricter
        stopping condition, such as --alpha=0.01 --samples=20000
    --outliers=none [dream]
        name of test used for removing outlier chains every N samples:
          none:   no outlier removal
          iqr:    use interquartile range on likelihood
          grubbs: use t-test on likelihood
          mahal:  use distance from parameter values on the best chain
    --pop=10        [dream, de, rl, ps]
        population size is pop times number of fitted parameters; if pop is
        negative, then set population size to -pop.
    --burn=100      [dream, pt]
        number of burn-in iterations before accumulating stats
    --thin=1        [dream]
        number of fit iterations between steps
    --nT=25
    --Tmin=0.1
    --Tmax=10       [pt]
        temperatures vector; use a higher maximum temperature and a larger
        nT if your fit is getting stuck in local minima
    --CR=0.9        [de, rl, pt]
        crossover ratio for population mixing
    --starts=1      [newton, rl, amoeba]
        number of times to run the fit from random starting points.
    --near_best
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

    --time_model
        run the model --steps times in order to estimate total run time.
    --profile
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
""" % {
        "fitter": "|".join(sorted(FIT_AVAILABLE_IDS)),
        "plotter": "|".join(PLOTTERS),
    }

    #    --remote
    #        queue fit to run on remote server
    #    --notify=user@email
    #        remote fit notification
    #    --queue=http://reflectometry.org
    #        remote job queue
    #    --transport=mp  {amqp|mp|mpi}
    #        use amqp/multiprocessing/mpi for parallel evaluation

    _plot = "log"

    def _set_plot(self, value):
        if value not in set(self.PLOTTERS):
            raise ValueError("unknown plot type %s; use %s" % (value, "|".join(self.PLOTTERS)))
        self._plot = value

    plot = property(fget=lambda self: self._plot, fset=_set_plot)
    store = None
    resume = None
    _fit = FIT_DEFAULT_ID

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        if value not in FIT_AVAILABLE_IDS:
            raise ValueError("unknown fitter %s; use %s" % (value, "|".join(sorted(FIT_AVAILABLE_IDS))))
        self._fit = value

    fit_config = FIT_CONFIG
    TRANSPORTS = "amqp", "mp", "mpi", "celery"
    _transport = "mp"

    def _set_transport(self, value):
        if value not in self.TRANSPORTS:
            raise ValueError("unknown transport %s; use %s" % (value, "|".join(self.TRANSPORTS)))
        self._transport = value

    transport = property(fget=lambda self: self._transport, fset=_set_transport)
    meshsteps = 40


def getopts():
    """
    Process command line options.

    Option values will be stored as attributes in the returned object.
    """
    opts = BumpsOpts(sys.argv)
    opts.resynth = int(opts.resynth)
    # Set a random seed if none is given; want to know the seed so we can
    # reproduce the run.  The seed needs to be saved to the monitor file.
    opts.seed = int(opts.seed) if opts.seed else np.random.randint(1000000)
    opts.fit_config.set_from_cli(opts)
    return opts
