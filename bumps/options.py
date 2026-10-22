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
parser, with a list of fitters and default setting values for that fitter.

Various setting types are available:

* *float*, *int* and *str* are simple types with no restrictions.

* *bool* defines a boolean setting, with an additional `--no-name entry` on the command line
   to indicate false values. Boolean settings can default to True or False.

* The *Range* type restricts the setting to a valid setting range, with values clipped
   to the range during :func:`check_options` processing.

* *list* settings define an enumerated set of available string values.
"""

__all__ = [
    "check_options",
    "get_fitter_defaults",
    "get_fit_fields",
    "lookup_fitter",
    "Setting",
    # CRUFT: old fit options data structures are still used by sasview
    "FIT_CONFIG",
    "FIT_FIELDS",
    "ChoiceList",
]

from typing import Dict, List, Tuple, Any, Callable, Optional
from textwrap import dedent
from argparse import ArgumentTypeError, SUPPRESS
from dataclasses import dataclass, asdict

from bumps.fitters import FitBase, FITTERS, FIT_DEFAULT_ID, FIT_AVAILABLE_IDS, FIT_ACTIVE_IDS

# TODO: unify handling of fit options in jupyter, sasview, script, webview and bumps cli.
# Current fitter is now in shared state and fit options use Setting, but SasView and the wx GUI
# still use options.FIT_CONFIG and options.FIT_FIELDS. We can't remove the old option handler
# until SasView and webview are updated to a unified interface.

# CRUFT: SasView is still using FIT_CONFIG, FIT_FIELDS and ChoiceList
# Note: symbols are not imported directly, but instead assigned within the module so we can override
# the sphinx documentation.
from bumps.gui.old_options import FIT_CONFIG as _FIT_CONFIG, FIT_FIELDS as _FIT_FIELDS, ChoiceList as _ChoiceList

FIT_CONFIG = _FIT_CONFIG
"""*** DEPRECATED ***"""

FIT_FIELDS = _FIT_FIELDS
"""*** DEPRECATED ***"""

ChoiceList = _ChoiceList
"""*** DEPRECATED ***"""


# TODO: If --fit=pt is specified then we need the pt options in the defaults
# The problem is that the defaults are set in api.py before the command line is processed.
# Options are to start with all and strip the inactive or start with the active and add extras
def get_fitter_defaults(active_only=True) -> dict[str, dict[str, Any]]:
    """
    Determines the default values for each setting of each fitter.

    This comes from the Fitter.settings attribute in the fitters defined by bumps.fitter,
    with an additional ("time", 0.0) setting implicit to all fitters.

    Returns {fitter_id: {setting: value, ...}, ...}
    """
    fitters = [f for f in FITTERS if f.id in FIT_ACTIVE_IDS] if active_only else FITTERS
    defaults = {f.id: dict(name=f.name, settings=dict(f.settings)) for f in fitters}
    # Add an implicit time=0 for the max time on each fitter.
    for k, v in defaults.items():
        v["settings"]["time"] = 0.0
    return defaults


# TODO: move the setting registry into the Setting class
_Setting_registry: dict[str, "Setting"] = {}
# _registry: dict[str, "Setting"] = field(default_factory=dict, repr=False)


@dataclass(init=False)
class Setting:
    """
    Represents a setting for a fitter.

    Attributes:
        name (str): Name of the setting.

        label (str): Label for the setting.

        description (str): Description of the setting.

        stype (Callable): Type of the setting.
    """

    name: str
    label: str
    description: str
    stype: Callable

    def __init__(self, name: str, label: str, stype: type, description: str):
        """
        Initializes a Setting instance.

        Args:
            name (str): Name of the setting.

            label (str): Label for the setting.

            stype (type): Type of the setting.

            description (str): Description of the setting.
        """
        # These attributes are associated with each setting (name, label, description, type)
        self.name = name
        self.label = label
        self.description = flow(dedent(description))
        self.stype = stype

        # Register all settings. Make sure each is only declared once.
        if name in _Setting_registry:
            raise RuntimeError(f'Setting("{name}", ...) already defined')
        _Setting_registry[name] = self

    @classmethod
    def items(cls):
        """
        Iterate over the registered *(name, Setting)* pairs.
        """
        return _Setting_registry.items()

    # Note: don't use __class_item__ since that will look like a type annotation
    @classmethod
    def get(cls, name):
        """
        Retrieve the setting for *name*.
        """
        return _Setting_registry[name]

    def get_defaults(self, active_only=True):
        """
        Retrieve the defaults for setting *name* for all fitters that use it.

        If *active_only*, only include the active fitters, not anything experimental
        or deprecated.
        """
        defaults = [
            (fitter.id, value)
            for fitter in FITTERS
            for setting, value in fitter.settings
            if setting == self.name and (fitter.id in FIT_ACTIVE_IDS or not active_only)
        ]
        return defaults

    def help(self, active_only=True) -> str:
        """
        Help string for the --settings option in the argument parser, or SUPPRESS if the
        setting is only available for experimental or deprecated optimizers.
        """
        if self.name == "time":
            # Produces the label for the --time option. This option is processed by
            # the fit driver, so it is common across all fitters
            #     Max time (hours)  [all fitters]
            fitters = ["all fitters"]
        elif self.name == "fit":
            # Produces the list of active fitters for the --fit options:
            #     Optimizer name  [amoeba, de, dream, newton, lm]
            fitters = FIT_ACTIVE_IDS
        else:
            # Produces the label for the setting with per-fitter default values.
            # For --xtol this is:
            #     x tolerance  [amoeba=1e-06, de=1e-06, newton=1e-12, lm=1e-10]
            # Don't show parameters that don't appear in the visible optimizers.
            # For example, don't show --nT which is only available when --fit=pt.
            defaults = self.get_defaults(active_only=active_only)
            if not defaults:
                return SUPPRESS
            fitters = [f"{fitter}={value}" for fitter, value in defaults]
        return f"{self.label}  [{', '.join(fitters)}]\n{self.description}"


def flow(text: str, proportional=True) -> str:
    """Flow paragraphs in text, replacing line breaks in the paragraph with spaces.

    Line breaks are preserved within indented text.

    Paragraphs are separated by blank lines.

    When operating on a multiline string, first call textwrap.dedent to remove the common
    indent level.

    If proportional, remove extra blanks within a line because we are displaying the text in a
    proportional rather than fixed width font.
    """
    paragraphs = []
    current = []
    for line in text.split("\n"):
        line = line.rstrip()
        if proportional:
            indent = len(line) - len(line.lstrip())
            line = f"{line[:indent]}{' '.join(line[indent:].split())}"
        if not line or line[0] in " \t":
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if line:
                paragraphs.append(line)
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n".join(paragraphs)


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
Setting("fit", "Optimizer name", FIT_AVAILABLE_IDS, f"Fitting engine to use (default: {FIT_DEFAULT_ID})")

# Stopping conditions
Setting(
    "steps",
    "Number of steps",
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
    "Convergence criteria",
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
    "Population initializer",
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
    "Population size",
    float,
    """\
    Population size is pop times number of fitted parameters. If pop is negative
    then set population size to -pop independent of fit parameters.""",
)
Setting("burn", "Burn-in steps", int, "Estimated number generations before convergence")
Setting("samples", "Samples", int, "Number of samples to draw = pop*pars*steps.")
Setting(
    "thin",
    "Thinning factor",
    int,
    """\
    Number of iterations between samples; use a large number here if you find
    your problem is "stuck", with minimal change from step to step in the
    parameter trace.""",
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
Setting("nT", "Number of temperatures", int, "Number of temperatures in the parallel tempering ladder")
Setting("Tmin", "Min temperature", float, "Lowest temperature in the temperature ladder")
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
    "Jump radius",
    Range(0, 0.5),
    """\
    When running with multiple starts, we jump to a new start position for each
    restart. Jump values are in [0, 0.5], representing the portion of the total
    range of each parameter. A value of zero uses a random starting point in the
    range.""",
)


def lookup_fitter(fitter_id: str) -> FitBase:
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
    for fitter in FITTERS:
        if fitter.id == fitter_id:
            return fitter
    raise ValueError(f"Unknown fitter '{fitter_id}'")


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

    default_id = FIT_DEFAULT_ID
    # available = set(fitter.id for fitter in FITTERS)
    available_ids = FIT_AVAILABLE_IDS

    # Use the default fitter if none specified.
    if not fitter_id:
        fitter_id = options.get("fit", default_id)

    # Check that the fitter is one of the valid fitters. In this case we are using
    # all the fitters, not just the ones visible in the user interface so that we
    # can support deprecated and experimental fitters.
    # print(available)
    if fitter_id not in available_ids:
        errors.append(f"Fitter {fitter_id} not in {', '.join(available_ids)}. Using {default_id} instead.")
        fitter_id = default_id

    # TODO: default from state.share.fitter_settings instead of Fitter.settings?
    fitter: FitBase = lookup_fitter(fitter_id)
    defaults: Dict[str, Any] = dict(fitter.settings)
    # print(f"defaults for {fitter_id}: {defaults}")

    # Scan all options, correcting any errors.
    # Note: time is processed by FitDriver so it is active in all the fitters
    new_options = {"fit": fitter_id, "time": 0.0, **defaults}
    for key, value in options.items():
        # We have already checked the fit=value option above.
        if key == "fit":
            continue

        # Collect invalid options for later reporting
        if key not in new_options:
            # Skip unrecognized options.
            unknown.append(f"{key}={value}")
            continue

        # Promote int values to floats if the option expects a float
        setting = Setting.get(key)
        stype = setting.stype
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
    fit_fields = dict([(k, _json_compatible_setting(v)) for k, v in Setting.items()])
    return fit_fields
