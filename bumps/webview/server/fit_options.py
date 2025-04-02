from typing import Dict, List, Tuple, Optional, Any, Callable
from textwrap import dedent
from argparse import ArgumentTypeError
from dataclasses import dataclass

from bumps import fitters
from bumps.fitters import FIT_AVAILABLE_IDS

FITTERS = (
    fitters.SimplexFit,
    fitters.DreamFit,
    fitters.DEFit,
    fitters.MPFit,
    fitters.BFGSFit,
)
DEFAULT_FITTER_ID = fitters.SimplexFit.id


def lookup_fitter(fitter_id: str):
    # Checking the complete list of fitters, not the restricted list for webview
    for fitter in fitters.FITTERS:
        if fitter.id == fitter_id:
            return fitter
    raise ValueError(f"Unknown fitter '{fitter_id}'")


FIT_OPTIONS: Dict[str, "Setting"] = {}


@dataclass(init=False)
class Setting:
    name: str
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
Setting("fit", "Optimizer", [], "Fitting engine to use.")

# Stopping conditions
Setting("steps", "Steps", int, "Stop when iteration = steps.")
Setting("xtol", "x tolerance", Range(0, 1), "Stop when population diameter < xtol relative to range.")
Setting("ftol", "f(x) tolerance", float, "Stop when variation in log likelihood < ftol.")
Setting("alpha", "Convergence", Range(0, 0.1), "Stop when probability that population is varying < alpha.")

# Initializers
Setting(
    "init",
    "Initializer",
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
    "Population",
    float,
    """\
    Population size is pop times number of fitted parameters. If pop is
    negative then set population size to -pop independent of fit parameters.""",
)
Setting("burn", "Burn-in steps", int, "Estimated number generations before convergence")
Setting("samples", "Samples", int, "Number of samples to draw = pop*pars*steps.")
Setting(
    "thin",
    "Thinning factor",
    int,
    """\
    Number of iterations between samples; use a large number here
    if you find your problem is "stuck", with minimal change from
    step to step in the parameter trace.""",
)
Setting(
    "outliers",
    "Outliers test",
    list("none iqr grubbs mahal".split()),
    """\
    Remove outlier Markov chains every n steps using the selected algorithm.
        none:   no outlier removal
        iqr:    use interquartile range on likelihood
        grubbs: use t-test on likelihood
        mahal:  use distance from parameter values on the best chain""",
)

# Post processing
Setting(
    "trim",
    "Burn-in trim",
    bool,
    """\
    After fitting, trim samples from early in the Markov chains before it converged.""",
)

# Parallel tempering
Setting("nT", "# Temperatures", int, "Number of temperatures in the parallel tempering ladder")
Setting("Tmin", "Min temperature", float, "Lowest temperature in the temperture ladder")
Setting("Tmax", "Max temperature", float, "Highest temperature in the temperature ladder")

# Differential evolution
Setting("CR", "Crossover ratio", Range(0, 1), "Proportion of parameters updated in crossover step")
Setting("F", "Scale", float, "Step-size scaling on difference vector")
# TODO: DE accepts --stop=expr for bumps.mystic.stop.parse_condition(expr)
# Settings("stop", "Stopping condition", str, "Generalized stopping condition expression")

# Amoeba
Setting(
    "radius",
    "Simplex radius",
    Range(0, 0.5),
    """\
    Radius around the starting point for the initial simplex. Values are in (0, 0.5],
    representing the portion of the total range of the parameter being initialized.""",
)

# Stochastic global minimization
Setting("starts", "Auto restarts", int, "Number of times to restart the amoeba fit.")
Setting(
    "near_best",
    "Near best",
    bool,
    """\
    When running with multiple starts, restart from a random point near the
    best minimum rather than using a completely random starting point.""",
)


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
        settings = FIT_OPTIONS["fit"]
        settings.fitters.append(fitter_id)  # produces doc string for --fit [amoeba, dream, ...]
        settings.stype.append(fitter_id)  # does option checking for --fit=fitter
        settings.defaults.append(DEFAULT_FITTER_ID)
        for key, value in fitter.settings:
            if key not in FIT_OPTIONS:
                raise TypeError(f"Missing type and description for fit option --{key} used by {fitter_id}")
            setting = FIT_OPTIONS[key]
            setting.fitters.append(fitter_id)
            setting.defaults.append(value)


def update_options(options: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Check if the set of options is consistent for the fitter.

    Returns an updated options dictionary and a list of errors covering unknown options and bad types.
    """
    # Note: this code is called with options set in a jupyter notebook so make
    # make sure it is robust against bad inputs.
    errors = []
    unknown = []
    fitter_id = options.get("fit", DEFAULT_FITTER_ID)
    # available = set(fitter.id for fitter in FITTERS)
    available = FIT_AVAILABLE_IDS
    if fitter_id not in available:
        errors.append(f"Fitter {fitter_id} not in {', '.join(available)}. Using {DEFAULT_FITTER_ID} instead.")
        fitter_id = DEFAULT_FITTER_ID
    fitter = lookup_fitter(fitter_id)
    defaults = dict(fitter.settings)
    # print(f"defaults for {fitter_id}: {defaults}")
    new_options = {"fit": fitter_id}
    for key, value in options.items():
        if key == "fit":
            # Already added.
            continue
        if key not in defaults:
            # Skip unrecognized options
            unknown.append(f"{key}={value}")
            continue
        stype = FIT_OPTIONS[key].stype
        if isinstance(stype, list):  # enumeration
            if value not in stype:
                # Default to first item in an enum if the enum is recognized.
                errors.append(f"Expected {key}={value} to be in {{{'|'.join(stype)}}}. Using {stype[0]}")
                value = stype[0]
        elif isinstance(stype, Range):
            if not isinstance(value, float):
                # Skip values of the wrong type.
                errors.append(f"Expected {key}={value} to be float. Ignored.")
                continue
            if not stype.min <= value <= stype.max:
                # Clip values to be in range.
                errors.append(f"Clipping {key}={value} to [{stype.min:g}, {stype.max:g}]")
                value = min(stype.max, max(stype.min, value))
        elif not isinstance(value, stype):
            # Skip values of the wrong type
            errors.append(f"Expected {key}={value} to have type {stype}. Ignored")
            continue
        new_options[key] = value
    if unknown:
        # Format the skipped options nicely and add to the error list
        errors = [f"Unused fit options in {fitter_id}: {' '.join(unknown)}", *errors]
    return new_options, errors


# TODO: remove unused parse_fit_options
# *** Deprecated ***
def parse_fit_options(fitter_id: str, fit_options: Optional[List[str]] = None) -> Dict:
    FITTER_DEFAULTS = {}
    for fitter in FITTERS:
        FITTER_DEFAULTS[fitter.id] = {
            "name": fitter.name,
            "settings": dict(fitter.settings),
        }
    if fitter_id not in FITTER_DEFAULTS:
        raise ValueError(f"invalid fitter: {fitter_id}")
    fitter_settings: Dict = FITTER_DEFAULTS[fitter_id]["settings"]
    if fit_options is not None:
        # fit options is a list of strings of the form "key=value"
        for option_str in fit_options:
            parts = option_str.split("=")
            if len(parts) != 2:
                raise ValueError(f"invalid fit option: {option_str}, must be of form 'key=value'")
            key, value = parts
            if key not in fitter_settings:
                raise ValueError(
                    f"invalid fit option: '{key}' for fitter '{fitter_id}'; valid options are: {list(fitter_settings.keys())}"
                )

            setting = FIT_OPTIONS[key]
            stype = setting.stype
            if stype is int:
                value = int(value)
            elif stype is float:
                value = float(value)
            elif isinstance(stype, Range):
                value = float(value)
                if not stype.min <= value <= stype.max:
                    raise ValueError(f"invalid value for {key}: {value} not in [{stype.min:g}, {stype.max:g}]")
            elif stype is str:
                pass
            elif stype is bool:
                if value.lower() in ["true", "1", "yes", "on"]:
                    value = True
                elif value.lower() in ["false", "0", "no", "off"]:
                    value = False
                else:
                    raise ValueError(f"invalid value for {key}: '{value}'; valid options are yes and no")
            elif isinstance(stype, list):
                if value not in stype:
                    raise ValueError(f"invalid value for {key}: '{value}'; valid options are: {stype}")
            else:
                raise ValueError(f"invalid type: {stype}")

            fitter_settings[key] = value

    return fitter_settings
