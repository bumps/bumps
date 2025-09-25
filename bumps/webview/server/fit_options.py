from typing import Dict, List, Tuple, Any, Callable, Optional
from textwrap import dedent
from argparse import ArgumentTypeError
from dataclasses import dataclass, asdict

from bumps import fitters
from bumps.fitters import FIT_AVAILABLE_IDS


def get_fitter_defaults(fitters):
    """
    Determines the default values for each setting of each fitter.

    This comes from the Fitter.settings attribute in the fitters defined by bumps.fitter,
    with an additional ("time", 0.0) setting implicit to all fitters.
    """
    defaults = {f.id: dict(name=f.name, settings=dict(f.settings)) for f in fitters}
    # Add an implicit time=0 for the max time on each fitter.
    for k, v in defaults.items():
        v["settings"]["time"] = 0.0
    return defaults


FITTERS = (
    fitters.SimplexFit,
    fitters.DreamFit,
    fitters.DEFit,
    fitters.MPFit,
    fitters.BFGSFit,
)
"""Fitters visible to the user. This may be a subset of bumps.fitters.FITTERS"""
DEFAULT_FITTER_ID = fitters.SimplexFit.id
"""Default fitter if none specified"""
FITTER_DEFAULTS = get_fitter_defaults(FITTERS)
"""Fitter name and default settings for the visible fitters. This list will be amended if a hidden fitter is specified on the command line."""
FIT_OPTIONS: Dict[str, "Setting"] = {}
"""Options available to the fitters."""


@dataclass(init=False)
class Setting:
    name: str
    label: str
    description: str
    stype: Callable
    fitters: List[str]
    defaults: List[Any]

    def __init__(self, name: str, label: str, stype: type, description: str):
        self.name = name
        self.label = label
        self.description = dedent(description)
        self.stype = stype
        self.fitters = []
        self.defaults = []
        FIT_OPTIONS[name] = self

    def build_help(self) -> str:
        if self.name == "time":
            # Fit option processed by the fit driver, so common across all fitters
            fitters = ["all fitters"]
        elif self.name == "fit":
            # Not a fit option. Handled specially
            fitters = self.fitters
        else:
            fitters = [f"{fitter}={FITTER_DEFAULTS[fitter]['settings'][self.name]}" for fitter in self.fitters]
        return f"{self.label}  [{', '.join(fitters)}]\n{self.description}"


@dataclass(frozen=True)
class Range:
    """
    A floating point range which is used as an argparse type converter and validator.
    """

    min: float
    max: float

    # def __init__(self, min, max): self.min, self.max = min, max

    def __call__(self, v):
        v = float(v)
        if not self.min <= v <= self.max:
            raise ArgumentTypeError(f"{v} not in [{self.min:g}, {self.max:g}]")
        return v

    def __repr__(self):
        return f"float[{self.min:g},{self.max:g}]"


## Options available for the various fitters
Setting("fit", "optimizer name", [], f"Fitting engine to use (default: {DEFAULT_FITTER_ID})")

# Stopping conditions
Setting(
    "steps",
    "number of steps",
    int,
    """\
    Stop when iteration = steps. In Dream, when --steps=0 the number of steps is
    calculated using (samples / chains). Here chains = pop * pars if pop > 0
    or -pop if pop < 0.""",
)
Setting("xtol", "x tolerance", Range(0, 1), "Stop when population diameter < xtol relative to range.")
Setting("ftol", "f(x) tolerance", float, "Stop when variation in log likelihood < ftol.")
Setting(
    "alpha",
    "convergence criteria",
    Range(0, 0.1),
    """\
    Stop when probability that population is varying is less than alpha. This
    uses a Kolmogorov-Smirnov test on the log-likelihood values to see if the
    distribution at the start of the saved samples matches the distribution at
    the end. Alpha must be 0.1 or less.""",
)
Setting("time", "Max time (hours)", float, "Maximum number of hours to run the fit, or zero for no maximum.")

# Initializers
Setting(
    "init",
    "initializer",
    list("eps lhs cov random".split()),
    """\
    Population initialization method
        eps:    ball around initial parameter set
        lhs:    latin hypercube sampling
        cov:    normally istributed according to covariance matrix
        random: uniformly distributed within parameter range""",
)
Setting(
    "pop",
    "population",
    float,
    """\
    Population size is pop times number of fitted parameters. If pop is negative
    then set population size to -pop independent of fit parameters.""",
)
Setting("burn", "Burn-in steps", int, "Estimated number generations before convergence")
Setting("samples", "Samples", int, "Number of samples to draw = pop*pars*steps.")
Setting(
    "thin",
    "thinning factor",
    int,
    """\
    Number of iterations between samples; use a large number here if you find
    your problem is "stuck", with minimal change from step to step in the
    parameter trace.""",
)
Setting(
    "outliers",
    "outliers test",
    list("none iqr grubbs mahal".split()),
    """\
    Remove outlier Markov chains every n steps using the selected algorithm.
        none:   no outlier removal during fit
        iqr:    use interquartile range on likelihood
        grubbs: use t-test on likelihood
        mahal:  use distance from parameter values on the best chain
    At the end of the fit the iqr algorithm is used to remove any remaining
    outlier chains from the statistical results and plots.""",
)

# Post processing
Setting(
    "trim",
    "burn-in trim",
    bool,
    """\
    After fitting, trim samples from early in the Markov chains before it converged.""",
)

# Parallel tempering
Setting("nT", "number of temperatures", int, "Number of temperatures in the parallel tempering ladder")
Setting("Tmin", "min temperature", float, "Lowest temperature in the temperture ladder")
Setting("Tmax", "max temperature", float, "Highest temperature in the temperature ladder")

# Differential evolution
Setting("CR", "Crossover ratio", Range(0, 1), "Proportion of parameters updated in crossover step")
Setting("F", "Scale", float, "Step-size scaling on difference vector")
# TODO: DE accepts --stop=expr for bumps.mystic.stop.parse_condition(expr)
# Settings("stop", "Stopping condition", str, "Generalized stopping condition expression")

# Amoeba
Setting(
    "radius",
    "simplex radius",
    Range(0, 0.5),
    """\
    Radius around the starting point for the initial simplex. Values are in (0, 0.5],
    representing the portion of the total range of the parameter being initialized.""",
)

# Stochastic global minimization
Setting(
    "starts",
    "Auto restarts",
    int,
    """\
    Number of times to restart the amoeba fit. After each start the fitter jumps
    to a new starting position determined by the --jump option.""",
)
Setting(
    "jump",
    "jump radius",
    Range(0, 0.5),
    """\
    When running with multiple starts, we jump to a new start position for each
    restart. Jump values are in [0, 0.5], representing the portion of the total
    range of each parameter. A value of zero uses a random starting point in the
    range.""",
)


def lookup_fitter(fitter_id: str):
    # Checking the complete list of fitters, not the restricted list for webview
    for fitter in fitters.FITTERS:
        if fitter.id == fitter_id:
            return fitter
    raise ValueError(f"Unknown fitter '{fitter_id}'")


def form_fit_options_associations():
    """
    Builds the association list between settings and the optimizers which use them.

    Rerun after changes to fit_options.FITTERS
    """
    # Clear out old associations
    del FIT_OPTIONS["fit"].stype[:]
    for settings in FIT_OPTIONS.values():
        del settings.fitters[:]
        del settings.defaults[:]

    # Define the new associations
    for fitter in FITTERS:
        fitter_id = fitter.id

        # The --fit option is pressent in every fitter. Note that stype for --fit is
        # a list of strings representing available fitters, so build up that list as well.
        # Similarly --fit has the same default value for each fitter.
        settings = FIT_OPTIONS["fit"]
        settings.fitters.append(fitter_id)  # produces doc string for --fit [amoeba, dream, ...]
        settings.stype.append(fitter_id)  # does option checking for --fit=fitter
        settings.defaults.append(DEFAULT_FITTER_ID)

        # The --time option is present in every fitter, with default = 0.
        settings = FIT_OPTIONS["time"]
        settings.fitters.append(fitter_id)
        settings.defaults.append(0.0)

        # Cycle through the default options for the current fitter, and for each, add
        # the fitter and its default value to the respective lists.
        for key, value in fitter.settings:
            if key not in FIT_OPTIONS:
                raise TypeError(f"Missing type and description for fit option --{key} used by {fitter_id}")
            setting = FIT_OPTIONS[key]
            setting.fitters.append(fitter_id)
            setting.defaults.append(value)


def check_options(options: Dict[str, Any], fitter_id: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Check if the set of options is consistent for the fitter.

    Returns an updated options dictionary and a list of warnings covering unknown options and bad types.
    """
    # Note: this code is called with options set in a jupyter notebook so make
    # make sure it is robust against bad inputs.
    errors = []
    unknown = []

    # Use the default fitter if none specified.
    if not fitter_id:
        fitter_id = options.get("fit", DEFAULT_FITTER_ID)

    # Check that the fitter is one of the valid fitters. In this case we are using
    # all the fitters, not just the ones visible in the user interface so that we
    # can support deprecated and experimental fitters.
    # available = set(fitter.id for fitter in FITTERS)
    available = FIT_AVAILABLE_IDS
    # print(available)
    if fitter_id not in available:
        errors.append(f"Fitter {fitter_id} not in {', '.join(available)}. Using {DEFAULT_FITTER_ID} instead.")
        fitter_id = DEFAULT_FITTER_ID

    # TODO: default from state.share.fitter_settings instead of Fitter.settings?
    fitter: fitters.FitBase = lookup_fitter(fitter_id)
    defaults: Dict[str, Any] = dict(fitter.settings)
    # print(f"defaults for {fitter_id}: {defaults}")

    # Scan all options, correcting any errors.
    # Note: time is processed by FitDriver so it is active in all the fitters
    new_options = {"fit": fitter_id, "time": 0.0, **defaults}
    for key, value in options.items():
        # We have already checked the fit=value option above.
        if key in "fit":
            continue

        # Collect invalid options for later reporting
        if key not in new_options:
            # Skip unrecognized options.
            unknown.append(f"{key}={value}")
            continue

        # Promote int values to floats if the option expects a float
        stype = FIT_OPTIONS[key].stype
        if isinstance(value, int) and (stype is float or isinstance(stype, Range)):
            value = float(value)

        # Check the value type
        if isinstance(stype, list):  # enumeration
            if value not in stype:
                # Default to first item in an enum if the enum is recognized.
                errors.append(f"Setting {key} to {stype[0]} since {key}={value} is not in {{{'|'.join(stype)}}}")
                value = stype[0]
        elif isinstance(stype, Range):
            if not isinstance(value, float):
                # Skip values of the wrong type.
                errors.append(f"Skipping {key}={value} since it is not a number.")
                continue
            if not stype.min <= value <= stype.max:
                # Clip values to be in range.
                errors.append(f"Clipping {key}={value} to [{stype.min:g}, {stype.max:g}]")
                value = min(stype.max, max(stype.min, value))
        elif not isinstance(value, stype):
            # Skip values of the wrong type
            errors.append(f"Skipping {key}={value} since it is not type {stype.__name__}")
            continue
        new_options[key] = value

    # Report all invalid options as a single error on the error list.
    if unknown:
        errors = [f"Unused fit options in {fitter_id}: {' '.join(unknown)}", *errors]
    return new_options, errors


def _json_compatible_setting(s: Setting):
    output = asdict(s)
    stype = output["stype"]
    if stype is int:
        output["stype"] = "integer"
    elif stype is float:
        output["stype"] = "float"
    elif stype is bool:
        output["stype"] = "boolean"
    return output


def get_fit_fields():
    fit_fields = dict([(k, _json_compatible_setting(v)) for k, v in FIT_OPTIONS.items()])
    return fit_fields
