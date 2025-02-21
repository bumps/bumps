# This program is public domain
# Author: Paul Kienzle
"""
Fitting parameter objects.

Parameters are a big part of the interface between the model and the fitting
engine.  By saving and retrieving values and ranges from the parameter, the
fitting engine does not need to be aware of the structure of the model.

Users can also perform calculations with parameters, tying together different
parts of the model, or different models.
"""

# __all__ = [ 'Parameter']
import operator
import sys
import builtins
from dataclasses import dataclass, field, InitVar
from functools import reduce
import warnings
from copy import copy
import uuid
from functools import wraps
from enum import Enum

from typing import Type, TypeVar, Optional, Any, Union, Dict, Callable, Tuple, List, Sequence
from .util import Literal

import numpy as np
from numpy import inf, isinf, isfinite

from . import bounds as mbounds
from . import pmath
from .util import field_desc, schema_config

BoundsType = mbounds.BoundsType

ValueType = Union["Expression", "Parameter", "Calculation", float]

# TODO: avoid evaluation of subexpressions if parameters do not change.
# This is especially important if the subexpression invokes an expensive
# calculation via a parameterized function.  This will require a restructuring
# of the parameter claas.  The park-1.3 solution is viable: given a parameter
# set, figure out which order the expressions need to be evaluated by
# building up a dependency graph.  With a little care, we can check which
# parameters have actually changed since the last calculation update, and
# restrict the dependency graph to just them.
# TODO: support full aliasing, so that floating point model attributes can
# be aliased to a parameter.  The same technique as subexpressions applies:
# when the parameter is changed, the model will be updated and will need
# to be re-evaluated.


# TODO: maybe move this to util?
def to_dict(p):
    if hasattr(p, "to_dict"):
        return p.to_dict()
    elif isinstance(p, (tuple, list)):
        return [to_dict(v) for v in p]
    elif isinstance(p, dict):
        return {k: to_dict(v) for k, v in p.items()}
    elif isinstance(p, (bool, str, float, int, type(None))):
        return p
    elif isinstance(p, np.ndarray):
        # TODO: what about inf, nan and object arrays?
        return p.tolist()
    elif False and callable(p):
        # TODO: consider including functions and arbitrary values
        import base64
        import dill

        encoding = base64.encodebytes(dill.dumps(p)).decode("ascii")
        return {"type": "dill", "value": str(p), "encoding": encoding}
        ## To recovert the function
        # if allow_unsafe_code:
        #     encoding = item['encoding']
        #     p = dill.loads(base64.decodebytes(encoding).encode('ascii'))
    else:
        # print(f"converting type {type(p)} to str")
        return str(p)


@dataclass(init=False)
class Uniform:
    """Uniform distribution with hard boundaries"""


@dataclass(init=False)
class Normal:
    """Normal distribution (Gaussian)"""

    std: float = field_desc("standard deviation (1-sigma)")
    mean: float = field_desc("center of the distribution")

    def __init__(self, std: float, mean: float):
        self.std = std
        self.mean = mean


# Leave out of schema for now.
# TODO: determine if this is used by anyone
# @dataclass(init=False)
class UniformSoftBounded:
    """Uniform distribution with error-function PDF on boundaries"""

    std: float = field_desc("width of the edge distribution")


DistributionType = Union[Uniform, Normal]  # , UniformSoftBounded]


class OperatorMixin:
    """
    The set of operations that can be performed on parameter-like objects
    Parameter, Constant, Expression.

    These include: +, -, *, /, //, **, abs, float, int

    Also, numpy math functions: sin, cos, tan, ...

    Much like abs(obj) => obj.__abs__(), np.sin(obj) => obj.sin()
    """

    # float(value) is special: it returns the current value rather than
    # becoming part of the parameter expression.
    value: float

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        # Note: __bool__ must return true or false, so we can't handle
        # lazy constraint expressions like not a, a or b, a and b.
        raise TypeError("use (p != 0) to test against zero")

    ...  # operators and functions will be filled in later


class ValueProtocol(OperatorMixin):
    """
    Values can be combined to form expressions
    Provide a suite of operators for creating parameter expressions.
    """

    fittable: bool = False
    fixed: bool = True
    value: float

    # TODO: Do values have names? Or do the names belong to the model parameter?
    # name: str
    # TODO: are priors on the parameter or on the value?
    # bounds: Optional[BoundsType] = None
    def parameters(self) -> List["Parameter"]:
        # default implementation:
        return []


@dataclass(init=False)
class Calculation(ValueProtocol):  # the name Function is taken (though deprecated)
    """
    A Parameter with a model-specific, calculated value.
    The function used to calculate this value should be well-documented in the
    description field, e.g.
    Stack.thickness: description = "a sum of the thicknesses of all layers in the stack"

    """

    description: str
    _function: Callable[[], float]  # added by the model; not serialized

    def __init__(self, description: str = ""):
        self.description = description

    @property
    def value(self):
        return self._function()

    def __float__(self):
        return self.value

    def set_function(self, function):
        self._function = function


class SupportsPrior:
    prior: Optional[BoundsType]
    limits: Tuple[float, float]
    distribution: DistributionType
    bounds: BoundsType

    def reset_prior(self):
        self.prior = None

    def has_prior(self):
        return (
            self.prior is not None
            and not isinstance(self.prior, mbounds.Unbounded)
            and self.prior.limits != (-np.inf, np.inf)
        )

    def add_prior(
        self,
        distribution: Optional[DistributionType] = None,
        bounds: Optional[BoundsType] = None,
        limits: Optional[Tuple[float, float]] = None,
    ):
        # use self values if they are found:
        if distribution is None and self.distribution is not None:
            distribution = self.distribution
        if bounds is None and self.bounds is not None:
            bounds = self.bounds
        if limits is None:
            if self.limits is not None:
                limits = self.limits
            else:
                limits = (-inf, inf)

        if bounds is not None:
            # get the intersection of the limits here.
            limits = (np.clip(limits[0], *bounds), np.clip(limits[1], *bounds))
        if isinstance(distribution, Normal):
            if limits == (-inf, inf):
                prior = mbounds.BoundedNormal(mean=distribution.mean, std=distribution.std, limits=limits)
            else:
                prior = mbounds.Normal(mean=distribution.mean, std=distribution.std)
        elif isinstance(distribution, UniformSoftBounded):
            lo, hi = limits
            prior = mbounds.SoftBounded(lo=lo, hi=hi, std=distribution.std)
        elif isinstance(distribution, Uniform):
            lo, hi = limits
            if isinf(lo) and isinf(hi):
                prior = mbounds.Unbounded()
            elif isinf(lo):
                prior = mbounds.BoundedAbove(hi)
            elif isinf(hi):
                prior = mbounds.BoundedBelow(lo)
            else:
                prior = mbounds.Bounded(lo, hi)
        else:
            raise ValueError("no distribution found matching %s" % (str(distribution)))

        self.prior = prior


@schema_config()
@dataclass(init=False)
class Parameter(ValueProtocol, SupportsPrior):
    """
    A parameter is a container for a symbolic value.

    Parameters have a prior probability, as set by a bounds constraint:

        import numpy as np
        from scipy.stats.distributions import lognorm
        from bumps.parameter import Parameter

        p = Parameter(3)
        p.pmp(10)               # 3 +/- 10% uniform
        p.pmp(-5,10)            # 3 in [2.85, 3.30] uniform
        p.pm(2)                 # 3 +/- 2 uniform
        p.pm(-1,2)              # 3 in [2,5] uniform
        p.range(0,5)            # 3 in [0,5] uniform
        p.dev(2)                # 3 +/- 2 gaussian
        p.soft_range(2,5,2)     # 3 in [2,5] uniform with gauss wings
        p.dev(2, limits=(0,6))  # 3 +/- 2 truncated gaussian
        p.pdf(lognorm(3, 1))    # lognormal centered on 3, width 1.

    Parameters have hard limits on the possible values, dictated by the model.
    These bounds apply in addition to any other bounds.

    Parameters can be constrained to be equal to another parameter or
    parameter expression:

        a, b = Parameter(3), Parameter(4)
        p = Parameter(limits=(6, 10))
        p.equals(a+b)
        assert p.nllf() == 0.   # within the bounds
        a.value = 20
        assert np.isinf(p.nllf()) # out of bounds

    Constraints on the computed value follow from the constraints on the
    underlying parameters in addition to any hard limits on the parameter
    value given by the model.

    **Inputs**

    *value* can be a constant, a variable, an expression or a link to
    another parameter.

    *bounds* are user-supplied limits on the parameter value within the model.
    If bounds are supplied then the parameter defaults to fittable.

    *distribution* is one of Uniform, Normal or UniformSoftBounded classes

    *fixed* is True if the parameter is fixed, even if bounds are supplied.

    *name* is the label associated with the parameter in plots. The names
    need not be unique, but it will be confusing if there are duplicates.
    The name will usually correspond to the role of the parameter in the
    model. For models with sequences (e.g., layer numbers), try using a
    layer name (e.g., based on the material in the layer) rather than a layer
    number for parameters in that layer. This will make it easier for the
    user to associate the parameters displayed at the end of the the fit
    with the layer in the model. Also, when exploring the space of models,
    the parameter names will be preserved even if a new layer is introduced
    before the existing layers. That will allow the parameters from the
    previous fit to be easily used as a seed for the fit to the new model.

    *id* must be a unique identifier associated with the parameter. This
    is used to link parameters on save and reload.

    *limits* are hard limits on the parameter value within the model. Separate
    from the prior distribution on a random variable provided by the user,
    the hard limits are restrictions on the value imposed by the model.
    For example, the thickness of a layer must be zero or more.

    Any additional keyword arguments are preserved as properties of the
    parameter. For example, *tip* and *units* for decorating an input form
    in the GUI:

         p = Parameter(10, name="width", units="cm", tip="Width of sample")

    """

    # Parameters may be dependent on other parameters, and the
    # fit engine will need to access them.
    # prior: Optional[BoundsType]
    id: str = field(metadata={"format": "uuid"})
    name: Optional[str] = field(default=None, init=False)
    fixed: bool = True
    slot: Union["Variable", ValueType]
    limits: Tuple[Union[float, Literal["-inf"]], Union[float, Literal["inf"]]] = (-inf, inf)
    bounds: Optional[Tuple[Union[float, Literal["-inf"]], Union[float, Literal["inf"]]]] = None
    distribution: DistributionType = field(default_factory=Uniform)
    discrete: bool = False
    tags: List[str] = field(default_factory=list)

    _fixed: bool

    def parameters(self):
        pars = [self]
        if hasattr(self.slot, "parameters"):
            pars += self.slot.parameters()
        return pars

    def pmp(self, plus, minus=None, limits=None):
        """
        Allow the parameter to vary as value +/- percent.

        pmp(*percent*) -> [value*(1-percent/100), value*(1+percent/100)]

        pmp(*plus*, *minus*) -> [value*(1+minus/100), value*(1+plus/100)]

        In the *plus/minus* form, one of the numbers should be plus and the
        other minus, but it doesn't matter which.

        If *limits* are provided, bound the end points of the range to lie
        within the limits.

        The resulting range is converted to "nice" numbers.
        """
        bounds = mbounds.pmp(self.value, plus, minus, limits=limits)
        self.bounds = bounds
        self.fixed = False
        return self

    def pm(self, plus, minus=None, limits=None):
        """
        Allow the parameter to vary as value +/- delta.

        pm(*delta*) -> [value-delta, value+delta]

        pm(*plus*, *minus*) -> [value+minus, value+plus]

        In the *plus/minus* form, one of the numbers should be plus and the
        other minus, but it doesn't matter which.

        If *limits* are provided, bound the end points of the range to lie
        within the limits.

        The resulting range is converted to "nice" numbers.
        """
        bounds = mbounds.pm(self.value, plus, minus, limits=limits)
        self.bounds = bounds
        self.fixed = False
        return self

    def dev(self, std, mean=None, limits=None, sigma=None, mu=None):
        """
        Allow the parameter to vary according to a normal distribution, with
        deviations from the mean added to the overall cost function for the
        model.

        If *mean* is None, then it defaults to the current parameter value.

        If *limits* are provide, then use a truncated normal distribution.

        Note: *sigma* and *mu* have been replaced by *std* and *mean*, but
        are left in for backward compatibility.
        """
        if sigma is not None or mu is not None:
            # CRUFT: remove sigma and mu parameters
            warnings.warn(DeprecationWarning("use std,mean instead of mu,sigma in Parameter.dev"))
            if sigma is not None:
                std = sigma
            if mu is not None:
                mean = mu
        if mean is None:
            mean = self.value  # Note: value is an attribute of the derived class
        self.bounds = limits if limits is not None else (-inf, inf)
        self.distribution = Normal(mean=mean, std=std)
        self.fixed = False
        return self

    # def pdf(self, dist):
    #     """
    #     Allow the parameter to vary according to any continuous scipy.stats
    #     distribution.
    #     """
    #     # TODO: have to make some kind of registry for distributions?
    #     # this will not work in new system of setting priors in model_reset.
    #     self._set_bounds((-inf, inf))
    #     self.distribution = dist
    #     return self

    def range(self, low, high):
        """
        Allow the parameter to vary within the given range.
        """
        self.bounds = (low, high)
        self.distribution = Uniform()
        self.fixed = False
        return self

    def soft_range(self, low, high, std):
        """
        Allow the parameter to vary within the given range, or with Gaussian
        probability, stray from the range.
        """
        self.bounds = (low, high)
        self.distribution = UniformSoftBounded(std=std)
        self.fixed = False
        return self

    # Delegate to slots
    @property
    def value(self):
        return int(self.slot) if self.discrete else float(self.slot)

    @value.setter
    def value(self, update):
        self.slot.value = round(update) if self.discrete else update

    @property
    def fittable(self):
        return isinstance(self.slot, Variable)

    @property
    def fixed(self):
        return not self.fittable or self._fixed

    @fixed.setter
    def fixed(self, state):
        # Can't set fixed to false if the parameter is not fittable
        if self.fittable:
            self._fixed = state
        elif not state:
            raise TypeError(f"value in {self.name} is not fittable")

    ## Use the following if bounds are on the value rather than the parameter
    # @property
    # def bounds(self):
    #    return getattr(self.slot, 'bounds', None)
    # @bounds.setter
    # def bounds(self, b):
    #    if not hasattr(self.slot, 'bounds'):
    #        raise TypeError(f"{self.name} is not fittable so bounds can't be set")
    #    if self.slot.fittable:
    #        self.slot.fixed = (b is None)
    #    self.slot.bounds = b

    # Functional form of parameter value access
    def __call__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def nllf(self) -> float:
        """
        Return -log(P) for the current parameter value.
        """
        value = self.value
        if not (self.limits[0] <= value <= self.limits[1]):
            # quick short-circuit if not meeting own limits:
            return np.inf
        else:
            logp = self.prior.nllf(value)
            if hasattr(self.slot, "nllf"):
                logp += self.slot.nllf()
            return logp

    def residual(self) -> float:
        """
        Return the z score equivalent for the current parameter value.

        That is, the given the value of the parameter in the underlying
        distribution, find the equivalent value in the standard normal.
        For a gaussian, this is the z score, in which you subtract the
        mean and divide by the standard deviation to get the number of
        sigmas away from the mean.  For other distributions, you need to
        compute the cdf of value in the parameter distribution and invert
        it using the ppf from the standard normal distribution.
        """
        return 0.0 if self.prior is None else self.prior.residual(self.value)

    def valid(self):
        """
        Return true if the parameter is within the valid range.
        """
        return not isinf(self.nllf())

    def format(self):
        """
        Format the parameter, value and range as a string.
        """
        return "%s=%g in %s" % (self, self.value, self.prior)

    def __str__(self):
        name = self.name if self.name is not None else "?"
        return name

    def __repr__(self):
        return "Parameter(%s)" % self

    # TODO: deprecate
    @classmethod
    def default(cls: type, value: Union[float, Tuple[float, float], ValueType], **kw) -> "Parameter":
        """
        Create a new parameter with the *value* and *kw* attributes. If value
        is already a parameter or expression, set it to that value.
        """
        # Need to constrain the parameter to fit within fixed limits and
        # to receive a name if a name has not already been provided.
        if isinstance(value, ValueProtocol):
            return value
        else:
            return cls(value, **kw)

    def set(self, value):
        """
        Set a new value for the parameter, ignoring the bounds.
        """
        self.slot.value = value

    def clip_set(self, value):
        """
        Set a new value for the parameter, clipping it to the bounds.
        """
        low, high = self.prior.limits
        self.slot.value = builtins.min(builtins.max(value, low), high)

    def __init__(
        self,
        value: Optional[Union[float, Tuple[float, float]]] = None,
        slot: Optional[Union["Variable", ValueType]] = None,
        # bounds: Optional[Union[BoundsType, Tuple[float, float]]]=None,
        fixed: Optional[bool] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        limits: Optional[Tuple[Union[float, Literal[None, "-inf"]], Union[float, Literal[None, "inf"]]]] = None,
        bounds: Optional[Tuple[Union[float, Literal["-inf"]], Union[float, Literal["inf"]]]] = None,
        distribution: DistributionType = Uniform(),
        discrete: bool = False,
        tags: Optional[List[str]] = None,
        **kw,
    ):
        # Check if we are started with value=range or bounds=range; if we
        # are given bounds, then assume this is a fitted parameter, otherwise
        # the parameter defaults to fixed; if value is not set, use the
        # midpoint of the range.
        if bounds is None:
            try:
                # Note: throws TypeError if not a sequence (which we want to
                # fall through to the remainder of the function), or ValueError
                # if the sequence is the wrong length (which we want to fail).
                lo, hi = value
                warnings.warn(DeprecationWarning("parameters can no longer be initialized with a fit range"))
                bounds = lo, hi
                value = None
            except TypeError:
                pass
        if fixed is None:
            fixed = bounds is None
        if slot is None:
            if value is None:
                value = float(bounds[0]) if bounds is not None else 0  # ? what else to do here?
            if isinstance(value, (float, int)):
                value = round(value) if discrete else value
                slot = Variable(value)
            elif isinstance(value, ValueProtocol):
                slot = value
            else:
                raise TypeError("value %s: %s cannot be converted to Variable" % (str(name), str(value)))
        assert isinstance(slot, (float, Variable, Expression, Parameter, Constant, Calculation))

        self.slot = slot
        self.name = name
        self.id = id if id is not None else str(uuid.uuid4())
        self.tags = tags if tags is not None else []
        if limits is None:
            limits = (-np.inf, np.inf)
        self.limits = (
            (-np.inf if limits[0] is None else float(limits[0])),
            (np.inf if limits[1] is None else float(limits[1])),
        )
        if bounds is not None:
            bounds = (
                (-np.inf if bounds[0] is None else float(bounds[0])),
                (np.inf if bounds[1] is None else float(bounds[1])),
            )
        self.bounds = bounds
        self.distribution = distribution
        # Note: fixed is True unless fixed=False or bounds=bounds were given
        # as function arguments. Note that _set_bounds() will always set the
        # fixed to False, so we need to reset it after calling _set_bounds().
        self.fixed = fixed
        self.discrete = discrete

        # Store whatever values the user needs to associate with the parameter.
        # For example, models can set units and tool tips so the user interface
        # has something to work with.
        for k, v in kw.items():
            setattr(self, k, v)
        self.prior = None  # to be filled by model_reset

    def randomize(self, rng=None):
        """
        Set a random value for the parameter.
        """
        self.value = self.prior.random(rng if rng is not None else mbounds.RNG)

    def feasible(self):
        """
        Value is within the limits defined by the model
        """
        return self.prior.limits[0] <= self.value <= self.prior.limits[1]

    def equals(self, expression: ValueType):
        """
        Set a parameter equal to another parameter or expression.

        If *expression=None* then free the parameter by giving it is own
        slot with value equal to the present value of the expression, and
        its bounds.
        """
        if isinstance(self.slot, Calculation):
            raise TypeError("parameter is calculated by the model and cannot be changed")
        elif expression is self:
            # don't make a circular reference to self.
            warnings.warn(f"{self} tried to make circular reference to self...")
            pass
        else:
            self.slot = expression

    def unlink(self):
        if isinstance(self.slot, Calculation):
            raise TypeError("parameter is calculated by the model and cannot be changed")
        # Replace the slot with a new variable initialized to the only variable value
        self.slot = Variable(self.value)

    def add_tag(self, tag: str):
        if not tag in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: Optional[str] = None):
        if tag is None:
            self.tags = []
        else:
            self.tags = [t for t in self.tags if not t == tag]

    def __copy__(self):
        """copy will only be called when a new instance is desired, with a different id"""
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.id = str(uuid.uuid4())
        return obj


def tag_all(parameter_tree, tag, remove=False):
    if isinstance(parameter_tree, dict):
        tag_all([item for item in parameter_tree.values()], tag, remove=remove)
    elif hasattr(parameter_tree, "add_tag"):
        if remove:
            parameter_tree.remove_tag(tag)
        else:
            parameter_tree.add_tag(tag)
    elif hasattr(parameter_tree, "parameters"):
        tag_all(parameter_tree.parameters(), tag, remove=remove)
    elif hasattr(parameter_tree, "__iter__"):
        for item in parameter_tree:
            tag_all(item, tag, remove=remove)
    else:
        warnings.warn(f"parameter tree should have only list, object and Parameter items: {parameter_tree}")


def untag_all(parameter_tree, tag: Optional[str] = None):
    tag_all(parameter_tree, tag, remove=True)


@dataclass
class Variable(ValueProtocol):
    """
    Saved state for a random variable in the model.
    """

    value: float

    def parameters(self):
        return []


@schema_config()
@dataclass(init=True, frozen=True, eq=False)
class Constant(ValueProtocol):  # type: ignore
    """
    Saved state for an unmodifiable value.

    A constant is like a fixed parameter. You can't change it's value, set
    it equal to another parameter, or assign a prior distribution.
    """

    value: float
    name: Optional[str] = None
    id: str = field(metadata={"format": "uuid"}, default_factory=lambda: str(uuid.uuid4()))

    fittable = False  # class property fixed across all objects
    fixed = True  # class property fixed across all objects

    def parameters(self):
        return [self]

    def __str__(self):
        return self.name


# ==== Arithmetic operators ===
class Operators(str, Enum):
    """Operators that can be used to construct Expressions"""

    # operators including abs() are defined in _build_operator_mixin()
    # functions are defined in numpy or in UserFunction (for min/max)

    # unary operator
    neg = "neg"
    pos = "pos"
    # binary operator
    add = "add"
    sub = "sub"
    mul = "mul"
    truediv = "truediv"
    floordiv = "floordiv"
    pow = "pow"
    # unary functional
    # float = "float"  => float makes values concrete
    # int = "int"  => values must be float; use floor, trunc, ceil, round
    abs = "abs"

    # unary functions
    exp = "exp"
    expm1 = "expm1"
    log = "log"
    log10 = "log10"
    log1p = "log1p"
    sqrt = "sqrt"
    degrees = "degrees"
    radians = "radians"
    sin = "sin"
    cos = "cos"
    tan = "tan"
    arcsin = "arcsin"
    arccos = "arccos"
    arctan = "arctan"
    sinh = "sinh"
    cosh = "cosh"
    tanh = "tanh"
    arcsinh = "arcsinh"
    arccosh = "arccosh"
    arctanh = "arctanh"
    ceil = "ceil"
    floor = "floor"
    trunc = "trunc"
    rint = "rint"
    round = "round"  # round(a) => rint(a)
    # binary functions
    arctan2 = "arctan2"

    # n-ary
    min = "min"  # from builtins
    max = "max"  # from builtins
    # TODO: support sum(seq) and prod(seq) for tuple and list


# Precedence for the python operators as given in manual. Numbers start
# from one at the bottom of the table. The value itself is "highest" precedence
# with a value of zero.
# https://docs.python.org/3/reference/expressions.html#operator-precedence
VALUE_PRECEDENCE = 0
CALL_PRECEDENCE = 2
OPERATOR_PRECEDENCE = {
    "pow": 4,
    "pos": 5,
    "neg": 5,
    "mul": 6,
    "truediv": 6,
    "floordiv": 6,
    "add": 7,
    "sub": 7,
    "gt": 12,
    "lt": 12,
    "ge": 12,
    "le": 12,
    "eq": 12,
    "ne": 12,
}
OPERATOR_STRING = {
    "pow": "**",
    "pos": "+",
    "neg": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "add": "+",
    "sub": "-",
    "gt": ">",
    "lt": "<",
    "ge": ">=",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
}


def _lookup_operator(op_name):
    if not hasattr(Operators, op_name) and op_name not in UserFunctionRegistry:
        raise ValueError(f"function {op_name} is not available")
    fn = None
    # Check plugins first so we can override lookups in operator and numpy.
    # This is needed for min/max.
    if fn is None:  # plugin functions from UserFunctionRegistry
        fn = UserFunctionRegistry.get(op_name, None)
    if fn is None:
        fn = getattr(operator, op_name, None)  # operators from operators
    if fn is None:  # math functions from numpy
        fn = getattr(np, op_name, None)
    if fn is None:
        raise RuntimeError(f"should not be here: {op_name} not found")
    return fn


def _precedence(obj: Any) -> int:
    """
    Return operator precedence according to the python parsing hierarchy.

    Lower values are higher precedence. Values start at 0 for constants and
    variables, and go up from there. Not all operators are covered.
    """
    if isinstance(obj, Expression):
        return OPERATOR_PRECEDENCE.get(obj.op.name, CALL_PRECEDENCE)
    return VALUE_PRECEDENCE


@dataclass(init=False)
class Expression(ValueProtocol):
    """
    Parameter expression
    """

    fittable = False
    fixed = True

    op: Union[Operators, "UserFunction"]  # Enumerated str type {function_name: display_name}
    args: Sequence[ValueType]
    _fn: Callable[..., float]  # _fn(float, float, ...) -> float

    def __init__(self, op: Union[str, Operators, "UserFunction"], args):
        op = op if (isinstance(op, Operators) or isinstance(op, UserFunction)) else getattr(Operators, op)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "_fn", _lookup_operator(op.name))
        object.__setattr__(self, "args", args)

    def parameters(self):
        # Walk expression tree combining parameters from each subexpression
        return sum((v.parameters() for v in self.args if hasattr(v, "parameters")), [])

    @property
    def value(self):
        return self._fn(*(float(arg) for arg in self.args))

    @property
    def name(self):
        return str(self)

    def __str__(self):
        prec = _precedence(self)
        vals = [str(v) for v in self.args]
        if self.op.name in ("pos", "neg"):
            # +- a with parens as needed
            a = f"({vals[0]})" if prec < _precedence(self.args[0]) else vals[0]
            return f"{OPERATOR_STRING[self.op.name]}{a}"
        elif self.op.name in ("add", "sub", "mul", "div", "truediv", "pow"):
            # a +-*/** b with parens as needed
            a = f"({vals[0]})" if prec < _precedence(self.args[0]) else vals[0]
            b = f"({vals[1]})" if prec < _precedence(self.args[1]) else vals[1]
            return f"{a} {OPERATOR_STRING[self.op.name]} {b}"
        else:
            # f(a, b, ...) with no parens needed
            return f"{self.op.name}({', '.join(v for v in vals)})"


def _make_unary_op(op_name: str):
    op = getattr(Operators, op_name)
    # Note: self is Parameter or Expression
    fn = lambda self: Expression(op, (self,))
    setattr(OperatorMixin, f"__{op_name}__", fn)


def _make_binary_op(op_name: str):
    op = getattr(Operators, op_name)

    def fn(self, other):
        return Expression(op, (self, other))

    setattr(OperatorMixin, f"__{op_name}__", fn)

    def rfn(self, other):
        return Expression(op, (other, self))

    setattr(OperatorMixin, f"__r{op_name}__", rfn)


def _make_math_fn(fn_name: str):
    op = getattr(Operators, fn_name)

    def fn(*args):  # first of args is self
        if any([isinstance(arg, ValueProtocol) for arg in args]):
            return Expression(op, args)
        else:
            # then all the args are floats: just return a float!
            realized_fn = _lookup_operator(op.name)
            return realized_fn(*args)

    # define sin, etc., in the parameter and expression so that np.sin(a)
    # will resolve to Expression('sin', tuple(a)), etc.
    setattr(OperatorMixin, fn_name, fn)
    # The np.sin(a) trick only works for a limited set of functions
    # defined by numpy itself. For arbitrary user defined functions
    # we add them to the bumps.pmath namespace so the user can find them.
    setattr(pmath, fn_name, fn)


def _build_operator_mixin():
    unary_op = set(("pos", "neg", "abs"))
    binary_op = set(("add", "sub", "mul", "floordiv", "truediv", "pow"))
    math_fn = set(v.name for v in Operators) - unary_op - binary_op
    for op_name in unary_op:
        _make_unary_op(op_name)
    for op_name in binary_op:
        _make_binary_op(op_name)
    # By adding the math functions to the mixin, calling np.sin(parameter) or
    # np.sin(expression) will return the generated expression for the object.
    for fn_name in math_fn:
        _make_math_fn(fn_name)


_build_operator_mixin()

UserFunctionRegistry: Dict[str, Callable[..., float]] = {}


# TODO: allow schema validation on user-defined functions
@dataclass(init=False)
class UserFunction:
    """
    User-defined functions.

    This is a helper class for the @function decorator, which treats the
    operator as one of the possible expression operators.

    These won't be properly serialized/deserialized through the JSON schema
    unless the function is registered in advance. The schema will not include
    these functions as possible values even if registered, so a schema
    validator may fail on one of these functions.
    """

    name: str

    # A function registry to remember the code associated with the name.
    # This is a class attribute, so it is initialized with an empty dict().
    # Ignore complaints from lint.
    # TODO: use pmath as our registry of available functions.
    def __init__(self, fn: Callable):
        name = fn.__name__
        if name in UserFunctionRegistry:
            raise TypeError(f"Function {name} already registered in bumps.")
        UserFunctionRegistry[name] = fn
        self.name = name


def function(fn: Callable):
    """
    Convert a function into a delayed evaluator.

    The value of the function is computed from the values of the parameters
    at the time that the function value is requested rather than when the
    function is created.
    """
    name = fn.__name__
    op = UserFunction(fn)

    def wrapped(*args: "ValueType"):
        return Expression(op, args)

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__ if fn.__name__.endswith("d") else f"{fn.__name__}(Parameter)"
    # Add the symbol to pmath
    setattr(pmath, name, wrapped)
    pmath.__all__.append(name)
    return wrapped


# min/max
min = function(builtins.min)
max = function(builtins.max)


# Trig functions defined in degrees rather than radians.
@function
def cosd(v):
    """Return the cosine of x (measured in in degrees)."""
    return np.cos(np.radians(v))


@function
def sind(v):
    """Return the sine of x (measured in in degrees)."""
    return np.sin(np.radians(v))


@function
def tand(v):
    """Return the tangent of x (measured in in degrees)."""
    return np.tan(np.radians(v))


@function
def arccosd(v):
    """Return the arc cosine (measured in in degrees) of x."""
    return np.degrees(np.arccos(v))


@function
def arcsind(v):
    """Return the arc sine (measured in in degrees) of x."""
    return np.degrees(np.arcsin(v))


@function
def arctand(v):
    """Return the arc tangent (measured in in degrees) of x."""
    return np.degrees(np.arctan(v))


@function
def arctan2d(dy, dx):
    """Return the arc tangent (measured in in degrees) of y/x.
    Unlike atan(y/x), the signs of both x and y are considered."""
    return np.degrees(np.arctan2(dy, dx))


# Aliases for arcsin, etc., both here in bumps.parameters and in bumps.pmath.
pmath.asin = asin = pmath.arcsin
pmath.acos = acos = pmath.arccos
pmath.atan = atan = pmath.arctan
pmath.atan2 = atan2 = pmath.arctan2

pmath.asind = asind = arcsind
pmath.acosd = acosd = arccosd
pmath.atand = atand = arctand
pmath.atan2d = atan2d = arctan2d

pmath.asinh = asinh = pmath.arcsinh
pmath.acosh = acosh = pmath.arccosh
pmath.atanh = atanh = pmath.arctanh

pmath.__all__.extend(
    (
        "asin",
        "acos",
        "atan",
        "atan2",
        "asind",
        "acosd",
        "atand",
        "atan2d",
        "asinh",
        "acosh",
        "atanh",
    )
)

# restate these for export, now that they're all defined:
ValueType = Union[Parameter, Expression, Calculation, float]


@dataclass(init=False)
class ParameterSet:
    """
    A parameter that depends on the model.
    """

    names: Optional[List[str]]
    reference: Parameter
    parameterlist: Optional[List[Parameter]]

    def __init__(
        self, reference: Parameter, names: Optional[List[str]] = None, parameterlist: Optional[List[Parameter]] = None
    ):
        """
        Create a parameter set, with one parameter for each model name.

        *names* is the list of model names.

        *reference* is the underlying :class:`parameter.Parameter` that will
        be set when the model is selected.

        *parameters* will be created, with one parameter per model.
        """
        names = names if names is not None else []
        self.names = names
        self.reference = reference
        # TODO: explain better why parameters are using np.array
        # Force numpy semantics on slice operations by using an array
        # of objects rather than a list of objects
        if parameterlist is not None:
            # we are being reinitialized with parameters
            self.parameters = np.array(parameterlist)
        else:
            self.parameters = np.array([copy(reference) for _ in names])
        # print self.reference, self.parameters
        for p, n in zip(self.parameters, names):
            p.name = " ".join((n, p.name))

        # N.B. if the reference parameter is not referenced anywhere in the models,
        # it will no longer show up in FitProblem.parameters
        # self.__class__.parameterlist = property(self._get_parameterlist) #lambda self: self.parameters.tolist())

    @property
    def parameterlist(self) -> List[Parameter]:
        return self.parameters.tolist()

    def to_dict(self):
        return {
            "type": "ParameterSet",
            "names": self.names,
            "reference": to_dict(self.reference),
            # Note: parameters are stored in a numpy array
            "parameters": to_dict(self.parameters.tolist()),
        }

    # Make the parameter set act like a list
    def __getitem__(self, i):
        """
        Return the underlying parameter for the model index.  Index can
        either be an integer or a model name.  It can also be a slice,
        in which case a new parameter set is returned.
        """
        # Try looking up the free variable by model name rather than model
        # index. If this fails, assume index is a model index.
        try:
            i = self.names.index(i)
        except ValueError:
            pass
        if isinstance(i, slice):
            obj = copy(self)
            obj.names = self.names[i]
            obj.reference = self.reference
            obj.parameters = self.parameters[i]
            return obj
        return self.parameters[i]

    def __setitem__(self, i, v):
        """
        Set the underlying parameter for the model index.  Index can
        either be an integer or a model name.  It can also be a slice,
        in which case all underlying parameters are set, either to the
        same value if *v* is a single parameter, otherwise *v* must have
        the same length as the slice.
        """
        try:
            i = self.names.index(i)
        except ValueError:
            pass
        self.parameters[i] = v

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def set_model(self, index):
        """
        Set the underlying model parameter to the value of the nth model.
        """
        self.reference.value = self.parameters[index].value

    def get_model(self, index):
        """
        Get the reference and underlying model parameter for the nth model.
        """
        return (id(self.reference), self.parameters[index])

    @property
    def values(self):
        return [p.value for p in self.parameters]

    @values.setter
    def values(self, values):
        for p, v in zip(self.parameters, values):
            p.value = v

    def range(self, *args, **kw):
        """
        Like :meth:`Parameter.range`, but applied to all models.
        """
        for p in self.parameters:
            p.range(*args, **kw)

    def pm(self, *args, **kw):
        """
        Like :meth:`Parameter.pm`, but applied to all models.
        """
        for p in self.parameters:
            p.pm(*args, **kw)

    def pmp(self, *args, **kw):
        """
        Like :meth:`Parameter.pmp`, but applied to all models.
        """
        for p in self.parameters:
            p.pmp(*args, **kw)


class Reference(Parameter):
    """
    Create an adaptor so that a model attribute can be treated as if it
    were a parameter.  This allows only direct access, wherein the
    storage for the parameter value is provided by the underlying model.

    Indirect access, wherein the storage is provided by the parameter, cannot
    be supported since the parameter has no way to detect that the model
    is asking for the value of the attribute.  This means that model
    attributes cannot be assigned to parameter expressions without some
    trigger to update the values of the attributes in the model.

    NOTE: this class can not be serialized with a dataclass schema
    TODO: can sasmodels just use Parameter directly?
    """

    def __init__(self, obj, attr, **kw):
        self.obj = obj
        self.attr = attr
        kw.setdefault("name", ".".join([obj.__class__.__name__, attr]))
        Parameter.__init__(self, **kw)

    @property
    def value(self):
        return getattr(self.obj, self.attr)

    @value.setter
    def value(self, value):
        setattr(self.obj, self.attr, value)


@dataclass(init=False)
class FreeVariables(object):
    """
    A collection of parameter sets for a group of models.

    *names* is the set of model names.

    The parameters themselves are specified as key=value pairs, with key
    being the attribute name which is used to retrieve the parameter set
    and value being a :class:`Parameter` containing the parameter that is
    shared between the models.

    In order to evaluate the log likelihood of all models simultaneously,
    the fitting program will need to call set_model with the model index
    for each model in turn in order to substitute the values from the free
    variables into the model.  This allows us to share a common sample
    across multiple data sets, with each dataset having its own values for
    some of the sample parameters.  The alternative is to copy the entire
    sample structure, sharing references to common parameters and creating
    new parameters for each model for the free parameters.  Setting up
    these copies was inconvenient.
    """

    names: List[str]
    parametersets: Dict[str, ParameterSet]

    def __init__(self, names=None, parametersets=None, **kw):
        if names is None:
            raise TypeError("FreeVariables needs name=[model1, model2, ...]")
        self.names = names
        if parametersets is not None:
            # assume that we are initializing with a dict of
            # fully initialized ParameterSet objects
            self.parametersets = parametersets
        else:
            # we are initializing with kw = Dict[key, (list of Parameters)]
            # Create slots to hold the free variables
            self.parametersets = dict((k, ParameterSet(v, names=names)) for k, v in kw.items())

    # Shouldn't need explicit __getstate__/__setstate__ but mpi4py pickle
    # chokes without it.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getattr__(self, k):
        """
        Return the parameter set for the given free parameter.
        """
        try:
            return self.parametersets[k]
        except KeyError:
            raise AttributeError("FreeVariables has no attribute %r" % k)

    def parameters(self):
        """
        Return the set of free variables for all the models.
        """
        return dict((k, v.parameters) for k, v in self.parametersets.items())

    def to_dict(self):
        return {"type": type(self).__name__, "names": self.names, "parameters": to_dict(self.parametersets)}

    def set_model(self, i):
        """
        Set the reference parameters for model *i*.
        """
        for p in self.parametersets.values():
            p.set_model(i)

    def get_model(self, i):
        """
        Get the parameters for model *i* as {reference: substitution}
        """
        return dict(p.get_model(i) for p in self.parametersets.values())


def flatten(s):
    if isinstance(s, (tuple, list, np.ndarray)):
        return reduce(lambda a, b: a + flatten(b), s, [])
    elif isinstance(s, set):
        raise TypeError("parameter flattening cannot order sets")
    elif isinstance(s, dict):
        return reduce(lambda a, b: a + flatten(s[b]), sorted(s.keys()), [])
    elif isinstance(s, ValueProtocol):
        return [s]
    elif s is None:
        return []
    else:
        raise TypeError("don't understand type %s for %r" % (type(s), s))


def format(p, indent=0, freevars=None, field=None):
    """
    Format parameter set for printing.

    Note that this only says how the parameters are arranged, not how they
    relate to each other.
    """
    freevars = {} if freevars is None else freevars
    p = freevars.get(id(p), p)
    if isinstance(p, dict) and p != {}:
        res = []
        for k in sorted(p.keys()):
            if k.startswith("_"):
                continue
            s = format(p[k], indent + 2, field=k, freevars=freevars)
            label = " " * indent + "." + k
            if s.endswith("\n"):
                res.append(label + "\n" + s)
            else:
                res.append(label + " = " + s + "\n")
        if "_index" in p:
            res.append(format(p["_index"], indent, freevars=freevars))
        return "".join(res)

    elif isinstance(p, (list, tuple, np.ndarray)) and len(p):
        res = []
        for k, v in enumerate(p):
            s = format(v, indent + 2, freevars=freevars)
            label = " " * indent + "[%d]" % k
            if s.endswith("\n"):
                res.append(label + "\n" + s)
            else:
                res.append(label + " = " + s + "\n")
        return "".join(res)

    elif isinstance(p, Parameter):
        s = ""
        if str(p) != field:
            s += str(p) + " = "
        s += "%g" % p.value
        if not p.fixed:
            if p.prior is not None:
                bounds = p.prior.limits
            elif p.bounds is not None:
                bounds = p.bounds
            else:
                bounds = p.limits
            s += " in [%g,%g]" % tuple(bounds)
        return s

    elif isinstance(p, Parameter):
        return "%s = %g" % (str(p), p.value)

    else:
        return str(p)


def summarize(pars, sorted=False):
    """
    Return a stylized list of parameter names and values with range bars
    suitable for printing.

    If sorted, then print the parameters sorted alphabetically by name.
    """
    output = []
    if sorted:
        pars = sorted(pars, key=lambda x: x.name)
    for p in pars:
        if not isfinite(p.value):
            bar = ["*invalid* "]
        else:
            position = int(p.prior.get01(p.value) * 9.999999999)
            bar = ["."] * 10
            if position < 0:
                bar[0] = "<"
            elif position > 9:
                bar[9] = ">"
            else:
                bar[position] = "|"
        output.append("%40s %s %10g in %s" % (p.name, "".join(bar), p.value, p.bounds))
    return "\n".join(output)


def unique(s) -> List[Parameter]:
    """
    Return the unique set of parameters

    The ordering is stable.  The same parameters/dependencies will always
    return the same ordering, with the first occurrence first.
    """
    # Walk structures such as dicts and lists
    pars = flatten(s)
    # print "====== flattened"
    # print "\n".join("%s:%s"%(id(p),p) for p in pars)
    # Also walk parameter expressions
    pars = pars + flatten([p.parameters() for p in pars])
    # print "====== extended"
    # print "\n".join("%s:%s"%(id(p),p) for p in pars)

    # TODO: implement n log n rather than n^2 uniqueness algorithm
    # problem is that the sorting has to be unique across a pickle.
    result = []
    for p in pars:
        if not any(p is q for q in result):
            result.append(p)

    # print "====== unique"
    # print "\n".join("%s:%s"%(id(p),p) for p in result)
    # Return the complete set of parameters
    return result


def fittable(s):
    """
    Return the list of fittable parameters in no paraticular order.

    Note that some fittable parameters may be fixed during the fit.
    """
    return [p for p in unique(s) if p.fittable]


def varying(s: List[Parameter]) -> List[Parameter]:
    """
    Return the list of fitted parameters in the model.

    This is the set of parameters that will vary during the fit.
    """
    return [p for p in unique(s) if not p.fixed]


def _has_prior(p: Parameter) -> bool:
    prior = getattr(p, "prior", None)
    limits = getattr(prior, "limits", (-np.inf, np.inf))
    return prior is not None and not isinstance(prior, mbounds.Unbounded) and limits != (-np.inf, np.inf)


def priors(s: List[Parameter]) -> List[Parameter]:
    """
    Return the list of parameters (fitted or computed) that have prior
    probabilities associated with them. This includes all varying parameters,
    plus expressions (including simple links), but ignoring constants and
    fixed parameters whose probabilities won't change the fits.
    """
    return [p for p in unique(s) if _has_prior(p)]


def randomize(s: List[Parameter]):
    """
    Set random values to the parameters in the parameter set, with
    values chosen according to the bounds.
    """
    for p in s:
        p.value = p.prior.random(1)[0]


def current(s: List[Parameter]):
    return [p.value for p in s]


# ========= trash ===================


def copy_linked(has_parameters, free_names=None):
    """
    make a copy of an object with parameters
     - then link all the parameters, except
     - those with names matching "free_names"
    """
    assert callable(getattr(has_parameters, "parameters", None)) == True
    from copy import deepcopy

    copied = deepcopy(has_parameters)
    free_names = [] if free_names is None else free_names
    original_pars = unique(has_parameters.parameters())
    copied_pars = unique(copied.parameters())
    for op, cp in zip(original_pars, copied_pars):
        if not op.name in free_names:
            cp.slot = op.slot
        else:
            cp.id = str(uuid.uuid4())
    return copied


# ==== Comparison operators ===
class Comparisons(Enum):
    """comparison operators"""

    gt = ">"
    ge = ">="
    le = "<="
    lt = "<"
    # eq = '=='
    # ne = '!='


@dataclass(init=False)
class Constraint:
    """Express inequality constraints between model elements"""

    fixed = True

    op: Comparisons
    a: ValueType
    b: ValueType

    def __init__(self, a, b, op):
        import operator

        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        op_name = str(Comparisons(op).name)
        object.__setattr__(self, "compare", getattr(operator, op_name.lower()))
        object.__setattr__(self, "op", op)

    # TODO: is this really necessary?  What is the reason for this trap?
    # It seems like being able to cast with bool(Constraint) would be
    # useful in some circumstances, like doing max(List[Parameter]), which
    # currently fails.
    def __bool__(self):
        raise TypeError("failed bool")

    __nonzero__ = __bool__

    def __float__(self):
        """return a float value that can be differentiated"""
        return 0.0 if self.satisfied else abs(float(self.a) - float(self.b))

    def __str__(self):
        return "(%s %s %s)" % (self.a, self.op, self.b)

    @property
    def satisfied(self):
        return self.compare(float(self.a), float(self.b))


def _make_constraint(op_str: str) -> Callable[..., Constraint]:
    return lambda self, other: Constraint(self, other, op_str)


def _build_constraints_mixin():
    for comp_item in Comparisons:
        op_name = comp_item.name
        op_str = comp_item.value
        setattr(OperatorMixin, f"__{op_name}__", _make_constraint(op_str))


_build_constraints_mixin()


class Alias(object):
    """
    Parameter alias.

    Rather than modifying a model to contain a parameter slot,
    allow the parameter to exist outside the model. The resulting
    parameter will have the full parameter semantics, including
    the ability to replace a fixed value with a parameter expression.

    """

    def __init__(self, obj, attr, p=None, name=None):
        self.obj = obj
        self.attr = attr
        if name is None:
            name = ".".join([obj.__class__.__name__, attr])
        self.p = Parameter.default(p, name=name)

    def update(self):
        setattr(self.obj, self.attr, self.p.value)

    def parameters(self):
        return self.p.parameters()

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "p": to_dict(self.p),
            # TODO: can't json pickle arbitrary objects
            "obj": to_dict(self.obj),
            "attr": self.attr,
        }


def substitute(a):
    """
    Return structure a with values substituted for all parameters.

    The function traverses lists, tuples and dicts recursively.  Things
    which are not parameters are returned directly.
    """
    if isinstance(a, ValueProtocol):
        return float(a.value)
    elif isinstance(a, tuple):
        return tuple(substitute(v) for v in a)
    elif isinstance(a, list):
        return [substitute(v) for v in a]
    elif isinstance(a, dict):
        return dict((k, substitute(v)) for k, v in a.items())
    elif isinstance(a, np.ndarray):
        return np.array([substitute(v) for v in a])
    else:
        return a


class Function(ValueProtocol):
    """
    **DEPRECATED**

    Delayed function evaluator.

    f.value evaluates the function with the values of the
    parameter arguments at the time f.value is referenced rather
    than when the function was invoked.
    """

    __slots__ = ["op", "args", "kw"]
    op: Callable[..., float]
    args: Optional[Any]
    kw: Dict[Any, Any]

    def __init__(self, op, *args, **kw):
        warnings.warn("Function no longer supported", DeprecationWarning, stacklevel=1)
        self.name = kw.pop("name", None)
        self.op, self.args, self.kw = op, args, kw
        self._parameters = self._find_parameters()

    def _find_parameters(self):
        # Figure out which arguments to the function are parameters
        # deps = [p for p in self.args if isinstance(p, ValueProtocol)]
        args = [arg for arg in self.args if isinstance(arg, ValueProtocol)]
        kw = dict((name, arg) for name, arg in self.kw.items() if isinstance(arg, ValueProtocol))
        deps = flatten((args, kw))
        # Find out which other parameters these parameters depend on.
        res = []
        for p in deps:
            res.extend(p.parameters())
        return res

    def parameters(self):
        return self._parameters

    def _value(self):
        # Expand args and kw, replacing instances of parameters
        # with their values
        return self.op(*substitute(self.args), **substitute(self.kw))

    value = property(_value)

    def to_dict(self):
        return {
            "type": "Function",
            "name": self.name,
            # TODO: function not stored properly in json
            "op": to_dict(self.op),
            "args": to_dict(self.args),
            "kw": to_dict(self.kw),
        }

    def __getstate__(self):
        return self.name, self.op, self.args, self.kw

    def __setstate__(self, state):
        self.name, self.op, self.args, self.kw = state
        self._parameters = self._find_parameters()

    def __str__(self):
        if self.name is not None:
            name = self.name
        else:
            args = [str(v) for v in self.args]
            kw = [str(k) + "=" + str(v) for k, v in self.kw.items()]
            name = self.op.__name__ + "(" + ", ".join(args + kw) + ")"
        return name
        # return "%s:%g" % (name, self.value)


# ===== Tests ====


def test_operator():
    a = Parameter(1, name="a")
    b = Parameter(2, name="b")
    c = Parameter(3, name="c")
    C = Constant(5, name="C")

    assert a.fixed

    # Check strings
    assert str(a + b) == "a + b"
    assert (a + b).name == "a + b"
    assert str(-a) == "-a"
    assert (-a).value == -a.value
    assert str(a + b * c) == "a + b * c"
    assert str((a + b) * c) == "(a + b) * c"
    assert str(np.sin(a + b) * c) == "sin(a + b) * c"
    assert str(a + C) == "a + C"
    assert str(a + C + 3) == "a + C + 3"
    assert str(3 + a + C) == "3 + a + C"
    assert str(a.sin()) == "sin(a)"
    assert str(atan2(a, b)) == "arctan2(a, b)"
    # float(expr) evaluates the expression; it doesn't build an expr with float.

    # Check parameters
    assert (a + b).parameters() == [a, b]
    assert (np.sin(a + b) * c).parameters() == [a, b, c]

    # Check values
    a.value = 3
    assert (a + b).value == 5.0
    assert float(a + b) == a.value + b.value
    assert a.sin().value == np.sin(a.value)
    assert (3 + a + C).value == 3 + 3 + 5
    assert np.sin(a + b).value == np.sin(a.value + b.value)
    assert atan2(a, b).value == atan2(a.value, b.value)

    # Make sure that evaluation is lazy. Capture the expression with one
    # set of values for the parameters, update them with a new set of values,
    # then check if the result is what you get when you call the function
    # directly on those new values.
    scope = locals()  # record the currently available parameter handles

    def capture_test(expr, result, **kw):
        # print("checking", expr, "for", kw, "yields", result)
        saved = {k: scope[k].value for k in kw}
        for k, v in kw.items():
            scope[k].value = float(v)
        try:
            assert expr.value == result, f"for {expr} expected {result} but got {expr.value}"
        finally:
            for k, v in saved.items():
                scope[k].value = v

    capture_test(np.sin(a + b), np.sin(0.5 + 3), a=0.5, b=3)
    capture_test(np.arctan2(a, b), atan2(0.5, 3), a=0.5, b=3)
    capture_test(np.round(a), np.round(-0.6), a=-0.6)
    capture_test(min(a, b), builtins.min(-0.6, 3), a=-0.6, b=3)
    capture_test(min(a, b, -2), builtins.min(-0.6, 3, -2), a=-0.6, b=3)
    capture_test(abs(a), 2.5, a=-2.5)

    # Check that symbols are defined in pmath
    capture_test(pmath.sind(a), np.sin(np.radians(25)), a=25)
    assert "sind" in pmath.__all__

    # TODO: can we evaluate an expression for an entire population at once?

    # Check slots
    limited = Parameter(3, name="limited", limits=[0.5, 1.5], bounds=[0, 1])
    limited.add_prior()
    assert np.isinf(limited.nllf())
    assert np.isinf(limited.nllf())
    limited.value = 0.6
    assert limited.nllf() == 0.0
    limited.value = 0.2
    assert np.isinf(limited.nllf())

    limited.equals(a + b)
    assert limited.value == (a + b).value

    assert np.isinf(limited.nllf())
    a.value = b.value = 0.1
    assert np.isinf(limited.nllf())
    a.value = b.value = 0.3
    assert limited.nllf() == 0.0
    try:
        limited.value = 5
        failed = True
    except Exception:
        # TODO: define which error improper assignment should raise
        # Currently this raises an attribute error on limited.slot.value
        failed = False
    if failed:
        raise RuntimeError("failed to raise error when assigning value to expression")

    # Check parameter list operations
    s = [a, limited]
    assert unique(s) == [a, limited, b]
    assert fittable(s) == [a, b]
    assert varying(s) == []
    b.range(0, 3)
    assert not b.fixed
    assert varying(s) == [b]
    assert current(s) == [a.value, limited.value]

    # Check normal deviation
    mu, sigma = 3, 2
    b.dev(sigma, mean=mu)
    b.value = 4
    b.add_prior()
    nllf_target = 0.5 * ((b.value - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2) / 2
    assert abs(b.nllf() - nllf_target) / nllf_target < 1e-12

    # Check conditions
    a.value, b.value = 3, 4
    capture = a < b
    assert isinstance(capture, Constraint)
    assert capture.satisfied
    a.value, b.value = 4, 3
    assert not capture.satisfied

    scope = locals()

    def raises(condition_str, exception):
        try:
            eval(condition_str, locals=scope)
        except exception:
            pass
        else:
            raise AssertionError(f"{condition_str} does not raise {exception}")

    raises("a < b < c", TypeError)
    raises("a < b and b < c", TypeError)
    raises("a < b or b < c", TypeError)
    raises("not (a < b)", TypeError)
    raises("not a", TypeError)
    raises("a and b", TypeError)
    raises("a or b", TypeError)


if __name__ == "__main__":
    test_operator()
