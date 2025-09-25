"""
Manage optimizer settings.

Each optimizer in :mod:`bumps.fitters` has its own settings attribute, giving the list of
settings available to the optimizer and the default value if the setting was not provided.

Settings for the same type of control should have the same value type and name, though the
defaults can differ between the optimizers. This module defines the metadata for the
various settings, each one having a setting name, a display label, an input type,
and a long description. These are used to construct the command line help and UI interaction.
If a fitter needs a completely new setting type it will have to be added here.

Each setting used by at least one active fitter adds a `--name` entry to the command line
parser, with a list of fitters and default setting values for that fitter. The
:func:`form_fit_options_associations` walks the available functions and forms this list.

Various setting types are available:

* *float*, *int* and *str* are simple types with no restrictions.

* *bool* defines a boolean setting, with an additional `--no-name entry` on the command line
   to indicate false values. Boolean settings can default to True or False.

* :class:`Range` restricts the setting to a valid setting range, with values clipped
   to the range during :func:`check_options` processing.

* *list* settings define an enumerated set of available string values.

"""

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
    """
    Represents a setting for a fitter.

    Attributes:
        name (str): Name of the setting.

        label (str): Label for the setting.

        description (str): Description of the setting.

        stype (Callable): Type of the setting.

        fitters (List[str]): List of fitters that use this setting.

        defaults (List[Any]): Default values for the setting across different fitters.
    """

    name: str
    label: str
    description: str
    stype: Callable
    fitters: List[str]
    defaults: List[Any]

    def __init__(self, name: str, label: str, stype: type, description: str):
        """
        Initializes a Setting instance.

        Args:
            name (str): Name of the setting.

            label (str): Label for the setting.

            stype (type): Type of the setting.

            description (str): Description of the setting.
        """
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

    Attributes:
        min (float): Minimum value of the range.

        max (float): Maximum value of the range.
    """

    min: float
    max: float

    def __call__(self, v):
        """
        Validates if a value is within the range.

        Args:
            v: Value to be validated.

        Returns:
            float: The validated value.

        Raises:
            ArgumentTypeError: If the value is not within the range.
        """
        v = float(v)
        if not self.min <= v <= self.max:
            raise ArgumentTypeError(f"{v} not in [{self.min:g}, {self.max:g}]")
        return v

    def __repr__(self):
        """
        Returns a string representation of the range.
        """
        return f"float[{self.min:g},{self.max:g}]"


## Options available for the various fitters
Setting("fit", "Optimizer", [], "Fitting engine to use.")

# Stopping conditions
Setting(
    "steps",
    "Steps",
    int,
    """\
    Stop when iteration = steps.
    In Dream, the number of steps is inferred from --samples as samples / (pop * pars)
    if --steps is zero, otherwise it uses the value of --steps.""",
)
Setting("xtol", "x tolerance", Range(0, 1), "Stop when population diameter < xtol relative to range.")
Setting("ftol", "f(x) tolerance", float, "Stop when variation in log likelihood < ftol.")
Setting(
    "alpha",
    "Convergence",
    Range(0, 0.1),
    """\
    Stop when probability that population is varying < alpha or use default
    zero for no convergence test.
    (Note that while p-values vary from 0 to 1, values for alpha > 0.1 result in
    an unstable check for convergence and are therefore disallowed)""",
)
Setting("time", "Max time", float, "Maximum number of hours to run the fit, or zero for no maximum.")

# Initializers
Setting(
    "init",
    "Initializer",
    list("eps lhs cov random".split()),
    """\
    Population initialization method
        eps:    ball around initial parameter set
        lhs:    latin hypercube sampling
        cov:    normally distributed according to covariance matrix
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
    "jump",
    "Jump radius",
    Range(0, 0.5),
    """\
    When running with multiple starts, what size of jump to take between restarts.
    Values are in [0, 0.5], representing the portion of the total range of each parameter.
    A value of zero uses a random starting point in the range.
    """,
)


def lookup_fitter(fitter_id: str):
    """
    Looks up a fitter by its ID.

    Args:
        fitter_id (str): ID of the fitter to look up.

    Returns:
        Fitter: The fitter instance corresponding to the given ID.

    Raises:
        ValueError: If the fitter ID is unknown.
    """
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


def check_options(options: Dict[str, Any], fitter_id: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Checks if the set of options is consistent for the fitter.

    Args:
        options (Dict[str, Any]): Options to be checked.

        fitter_id (Optional[str]): ID of the fitter (default=None).

    Returns:
        Tuple[Dict[str, Any], List[str]]: A tuple containing the updated options
        dictionary and a list of warnings.
    """
    # Note: this code is called with options set in a jupyter notebook so make
    # make sure it is robust against bad inputs.
    errors = []
    unknown = []
    if not fitter_id:
        fitter_id = options.get("fit", DEFAULT_FITTER_ID)
    # available = set(fitter.id for fitter in FITTERS)
    # Check against all available fitters, not just the ones visibile in the interface
    available = FIT_AVAILABLE_IDS
    if fitter_id not in available:
        errors.append(f"Fitter {fitter_id} not in {', '.join(available)}. Using {DEFAULT_FITTER_ID} instead.")
        fitter_id = DEFAULT_FITTER_ID
    # TODO: default from state.share.fitter_settings instead of Fitter.settings?
    fitter = lookup_fitter(fitter_id)
    defaults = dict(fitter.settings)
    # print(f"defaults for {fitter_id}: {defaults}")
    # Note: time is not one of the fit options but it is ever present.
    new_options = {"fit": fitter_id, "time": 0.0, **defaults}
    for key, value in options.items():
        if key == "fit":
            # Already added.
            continue
        if key not in defaults and key != "time":
            # Skip unrecognized options
            unknown.append(f"{key}={value}")
            continue
        stype = float if key == "time" else FIT_OPTIONS[key].stype
        if (stype is float or isinstance(stype, Range)) and isinstance(value, int):
            value = float(value)  # type promotion from int to float
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
            errors.append(f"Expected {key}={value} to have type {stype.__name__}. Ignored")
            continue
        new_options[key] = value
    if unknown:
        # Format the skipped options nicely and add to the error list
        errors = [f"Unused fit options in {fitter_id}: {' '.join(unknown)}", *errors]
    return new_options, errors


def _json_compatible_setting(s: Setting):
    """
    Converts a Setting instance to a JSON-compatible dictionary.

    Args:
        s (Setting): Setting instance to be converted.

    Returns:
        dict: JSON-compatible dictionary representation of the Setting.
    """
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
    """
    Returns a dictionary of fit fields.

    Returns:
        dict: Dictionary of fit fields where each key is a setting name and each value is a JSON-compatible Setting representation.
    """
    fit_fields = dict([(k, _json_compatible_setting(v)) for k, v in FIT_OPTIONS.items()])
    return fit_fields
