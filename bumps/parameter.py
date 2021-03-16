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
#__all__ = [ 'Parameter']
import operator
import sys
import builtins
from functools import reduce
import warnings
from copy import copy
import math
from functools import wraps
from enum import Enum

from .util import field, field_desc, schema, has_schema, Type, TypeVar, Optional, Any, Union, Dict, Callable, Literal, Tuple, List, Literal

import numpy as np
from numpy import inf, isinf, isfinite

from . import bounds as mbounds
bounds_classes = [c for c in mbounds.Bounds.__subclasses__() if has_schema(c)]
BoundsType = Union[tuple(bounds_classes)]

T = TypeVar('T')

FORWARD_PARAMETER_TYPES = Union['Parameter', 'Expression', 'UnaryExpression', 'Constant']

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
    if hasattr(p, 'to_dict'):
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
        encoding = base64.encodebytes(dill.dumps(p)).decode('ascii')
        return {'type': 'dill', 'value': str(p), 'encoding': encoding}
        ## To recovert the function
        # if allow_unsafe_code:
        #     encoding = item['encoding']
        #     p = dill.loads(base64.decodebytes(encoding).encode('ascii'))
    else:
        #print(f"converting type {type(p)} to str")
        return str(p)


class BaseParameter:
    """
    Root of the parameter class, defining arithmetic on parameters
    """
    # Parameters are fixed unless told otherwise
    fixed: bool
    fittable: bool
    discrete: bool
    bounds: BoundsType
    id: int
    #_bounds =  mbounds.Unbounded()
    name: Optional[str]
    value: float # value is an attribute of the derived class

    # Parameters may be dependent on other parameters, and the
    # fit engine will need to access them.
    def parameters(self):
        return [self]

    def __init__(self, value):
        """ set the value for the parameter """

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
        self.set_bounds(mbounds.Bounded(*bounds))
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
        self.set_bounds(mbounds.Bounded(*bounds))
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
        if limits is None:
            self.set_bounds(mbounds.Normal(mean, std))
        else:
            self.set_bounds(mbounds.BoundedNormal(mean, std, limits))
        return self

    def pdf(self, dist):
        """
        Allow the parameter to vary according to any continuous scipy.stats
        distribution.
        """
        self.set_bounds(mbounds.Distribution(dist))
        return self

    def range(self, low, high):
        """
        Allow the parameter to vary within the given range.
        """
        self.set_bounds(mbounds.init_bounds((low, high)))
        return self

    def soft_range(self, low, high, std):
        """
        Allow the parameter to vary within the given range, or with Gaussian
        probability, stray from the range.
        """
        self.set_bounds(mbounds.SoftBounded(low, high, std))
        return self

    def set_bounds(self, b):
        # print "setting bounds for",self
        if self.fittable:
            self.fixed = (b is None)
        self.bounds = b

    # @property
    # def bounds(self):
    #     """Fit bounds"""
    #     # print "getting bounds for",self,self._bounds
    #     return self._bounds

    # @bounds.setter
    # def bounds(self, b):
    #     # print "setting bounds for",self
    #     if self.fittable:
    #         self.fixed = (b is None)
    #     self._bounds = b

    # Functional form of parameter value access
    def __call__(self):
        return self.value

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __float__(self):
        return float(self.value)


    def nllf(self):
        """
        Return -log(P) for the current parameter value.
        """
        return self.bounds.nllf(self.value)

    def residual(self):
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
        return self.bounds.residual(self.value)

    def valid(self):
        """
        Return true if the parameter is within the valid range.
        """
        return not isinf(self.nllf())

    def format(self):
        """
        Format the parameter, value and range as a string.
        """
        return "%s=%g in %s" % (self, self.value, self.bounds)

    def __str__(self):
        name = self.name if self.name is not None else '?'
        return name

    def __repr__(self):
        return "Parameter(%s)" % self

    def to_dicto(self):
        """
        Return a dict represention of the object.
        """
        # When reconstructing a model from json we will need to tie parameters
        # together that were tied before. This can be done by managing a
        # cache of allocated parameters indexed by id, and pulling from that
        # cache on recontruction if the id already exists, otherwise create
        # a new entry. Conveniently, this will handle free variable references
        # in parameter sets as well. Note that the entire parameter description
        # will be repeated each time it occurs, but there should be few
        # enough of these that it isn't a problem.
        # TODO: use id that is stable from session to session.
        # TODO: have mechanism for clearing cache between save/load.
        return dict(
            type=type(self).__name__,
            id=id(self), # Warning: this will be different every session
            name=self.name,
            value=self.value,
            fixed=self.fixed,
            fittable=self.fittable,
            bounds=to_dict(self.bounds),
            )


@schema(classname="Constant")
class ConstantSchema:
    """
    An unmodifiable value.
    """
    name: Optional[str]
    value: float
    id: Optional[int]

class Constant(ConstantSchema, BaseParameter): # type: ignore
    
    fittable = False
    fixed = True

    def __init__(self, value, name=None, id=None):
        self._value = value
        self.name = name
        self.id = id if id is not None else builtins.id(self)

    @property
    def value(self):
        return self._value
    # to_dict() can inherit from BaseParameter


@schema()
class Parameter(BaseParameter):
    """
    A parameter is a symbolic value.

    It can be fixed or it can vary within bounds.

    p = Parameter(3).pmp(10)    # 3 +/- 10%
    p = Parameter(3).pmp(-5,10) # 3 in [2.85,3.3] rounded to 2 digits
    p = Parameter(3).pm(2)      # 3 +/- 2
    p = Parameter(3).pm(-1,2)   # 3 in [2,5]
    p = Parameter(3).range(0,5) # 3 in [0,5]

    It has hard limits on the possible values, and a range that should live
    within those hard limits.  The value should lie within the range for
    it to be valid.  Some algorithms may drive the value outside the range
    in order to satisfy soft It has a value which should lie within the range.

    Other properties can decorate the parameter, such as tip for tool tip
    and units for units.
    """
    fixed: bool = field(default=True, init=False)
    fittable: bool = field(default=True, init=False)
    discrete: bool = field(default=False, init=False)
    bounds: Union[tuple(bounds_classes)] = field(default=mbounds.Unbounded(), init=False)
    id: int = field(default=0, init=False)
    #_bounds =  mbounds.Unbounded()
    name: Optional[str] = field(default=None, init=False)
    value: float # value is an attribute of the derived class
    fittable = True
    schema_description = """
    A parameter is a symbolic value, that can be fixed or vary within bounds
    """

    @classmethod
    def default(cls: Type[FORWARD_PARAMETER_TYPES], value: Union[float, FORWARD_PARAMETER_TYPES], **kw) -> FORWARD_PARAMETER_TYPES :
        """
        Create a new parameter with the *value* and *kw* attributes, or return
        the existing parameter if *value* is already a parameter.

        The attributes are the same as those for Parameter, or whatever
        subclass *cls* of Parameter is being created.
        """
        # Need to constrain the parameter to fit within fixed limits and
        # to receive a name if a name has not already been provided.
        if isinstance(value, BaseParameter):
            return value
        else:
            return cls(value, **kw)

    def set(self, value):
        """
        Set a new value for the parameter, ignoring the bounds.
        """
        self.value = value

    def clip_set(self, value):
        """
        Set a new value for the parameter, clipping it to the bounds.
        """
        low, high = self.bounds.limits
        self.value = min(max(value, low), high)

    def __init__(self, value: float, bounds: Optional[Union[BoundsType, Tuple[float, float]]]=None, fixed=None, name=None, id=None, **kw):
        # UI nicities:
        # 1. check if we are started with value=range or bounds=range; if we
        # are given bounds, then assume this is a fitted parameter, otherwise
        # the parameter defaults to fixed; if value is not set, use the
        # midpoint of the range.
        if bounds is None:
            try:
                lo, hi = value
                warnings.warn(DeprecationWarning("parameters can no longer be initialized with a fit range"))
                bounds = lo, hi
                value = None
            except TypeError:
                pass
        if fixed is None:
            fixed = (bounds is None)
        bounds = mbounds.init_bounds(bounds)
        if value is None:
            value = bounds.start_value()

        # Store whatever values the user needs to associate with the parameter
        # Models should set units and tool tips so the user interface has
        # something to work with.
        limits = kw.pop('limits', (-inf, inf))
        for k, v in kw.items():
            setattr(self, k, v)

        # Initialize bounds, with limits clipped to the hard limits for the
        # parameter
        def clip(x, a, b):
            return min(max(x, a), b)
        self.set_bounds(bounds)
        self.bounds.limits = (clip(self.bounds.limits[0], *limits),
                              clip(self.bounds.limits[1], *limits))
        self.value = value
        self.fixed = fixed
        self.name = name
        self.id = id if id is not None else builtins.id(self)

    def randomize(self, rng=None):
        """
        Set a random value for the parameter.
        """
        self.value = self.bounds.random(rng if rng is not None else mbounds.RNG)

    def feasible(self):
        """
        Value is within the limits defined by the model
        """
        return self.bounds.limits[0] <= self.value <= self.bounds.limits[1]

    # to_dict() can inherit from BaseParameter


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
    """

    def __init__(self, obj, attr, **kw):
        self.obj = obj
        self.attr = attr
        kw.setdefault('name', ".".join([obj.__class__.__name__, attr]))
        Parameter.__init__(self, **kw)

    @property
    def value(self):
        return getattr(self.obj, self.attr)

    @value.setter
    def value(self, value):
        setattr(self.obj, self.attr, value)

    def to_dict(self):
        ret = Parameter.to_dict(self)
        ret["attr"] = self.attr
        # TODO: another impossibility---an arbitrary python object
        # Clearly we need a (safe??) json pickler to handle the full
        # complexity of an arbitrary model.
        ret["obj"] = to_dict(self.obj)
        return ret


@schema(classname="ParameterSet")
class ParameterSetSchema:
    """
    A parameter that depends on the model.
    """
    names: Optional[List[str]]
    reference: Parameter
    parameterList: Optional[List[Parameter]]

class ParameterSet(ParameterSetSchema):

    def __init__(self,
            reference: Parameter,
            names: Optional[List[str]] = None,
            parameterlist: Optional[List[Parameter]] = None
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
        # Reference is no longer directly fittable
        self.reference.fittable = False
        #self.__class__.parameterlist = property(self._get_parameterlist) #lambda self: self.parameters.tolist())

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


@schema(init=False)
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
            self.parametersets = dict((k, ParameterSet(v, names=names))
                                   for k, v in kw.items())

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
            raise AttributeError('FreeVariables has no attribute %r' % k)

    def parameters(self):
        """
        Return the set of free variables for all the models.
        """
        return dict((k, v.parameters) for k, v in self.parametersets.items())

    def to_dict(self):
        return {
            'type': type(self).__name__,
            'names': self.names,
            'parameters': to_dict(self.parametersets)
        }

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

# Current implementation computes values on the fly, so you only
# need to plug the values into the parameters and the parameters
# are automatically updated.
#
# This will not work well for wrapped models.  In those cases you
# want to do a number of optimizations, such as only updating the
#

# not including Function in typing, because it is not 
# easily serializable

# ==== Arithmetic operators ===
class OPERATORS(str, Enum):
    """all allowed binary operators"""

    add = "+"
    sub = "-"
    mul = "*"
    truediv = "/"
    div = "/" # alias for truediv
    pow = "**"

OPERATORS_ALLOWED = set(item.value for item in OPERATORS)
OPERATOR_PRECEDENCE = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "**": 3
}

@schema(init=False)
class Expression(BaseParameter):
    """
    Parameter expression (binary)
    """
    fixed = True
    fittable = False
    discrete = False
    bounds = mbounds.Unbounded()
    name = None

    a: Union[FORWARD_PARAMETER_TYPES, float]
    b: Union[FORWARD_PARAMETER_TYPES, float]
    op: OPERATORS
   
    def __init__(self, a, b, op):
        import operator
        if not op in OPERATORS_ALLOWED:
            raise ValueError("Operator %s is not in allowed operators: %s" % (op, str(OPERATORS_ALLOWED)))
        self.a, self.b = a,b
        self.op = op
        op_name = str(OPERATORS(op).name)
        self.operator = getattr(operator, op_name.lower())
        pars = []
        if isinstance(a,BaseParameter): pars += a.parameters()
        if isinstance(b,BaseParameter): pars += b.parameters()
        self._parameters = pars
        self.name = str(self)

    def parameters(self):
        return self._parameters

    @property
    def value(self):
        return self.operator(float(self.a), float(self.b))
    @property
    def dvalue(self):
        return float(self.a)
    def __str__(self):
        op_prec = OPERATOR_PRECEDENCE[self.op]
        a_str = str(self.a)
        b_str = str(self.b)
        if isinstance(self.a, Expression) and OPERATOR_PRECEDENCE[self.a.op] == op_prec:
            a_str = a_str[1:-1]
        if isinstance(self.b, Expression) and OPERATOR_PRECEDENCE[self.b.op] == op_prec:
            b_str = b_str[1:-1]
        return "(%s %s %s)" % (a_str, self.op, b_str)


def make_operator(op_str: str) -> Callable[..., Expression]:
    def o(self, other):
        return Expression(self, other, op_str)
    return o
def make_roperator(op_str: str) -> Callable[..., Expression]:
    def o(self, other):
        return Expression(other, self, op_str)
    return o

for o_item in OPERATORS:
    op_name = o_item.name
    op_str = o_item.value

    setattr(BaseParameter, '__{op_name}__'.format(op_name=op_name), make_operator(op_str))
    # set right versions, too:
    setattr(BaseParameter, '__r{op_name}__'.format(op_name=op_name), make_roperator(op_str))


class UNARY_OPERATIONS(Enum):
    """all allowed unary ops"""

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
    asin = "asin"
    acos = "acos"
    atan = "atan"
    sinh = "sinh"
    cosh = "cosh"
    tanh = "tanh"
    atanh = "atanh"
    ceil = "ceil"
    floor = "floor"
    trunc = "trunc"
    abs = "abs"

UNARY_OPERATORS = ["abs"]
UNARY_ALLOWED = set(item.value for item in UNARY_OPERATIONS)


@schema(init=False)
class UnaryExpression(BaseParameter):
    """
    Parameter unary operator
    """
    fixed = True
    fittable = False
    discrete = False
    bounds = mbounds.Unbounded()
    name = None

    a: Union[FORWARD_PARAMETER_TYPES, float]
    op: UNARY_OPERATIONS
   
    def __init__(self, a, op):
        import operator
        if not op in UNARY_ALLOWED:
            raise ValueError("Operation %s is not in allowed unary operations: %s" % (op, str(UNARY_ALLOWED)))
        self.a = a
        op_name = str(UNARY_OPERATIONS(op).name)
        if op_name in UNARY_OPERATORS:
            self.operation = getattr(operator, op_name.lower())
        else:
            self.operation = getattr(math, op_name.lower())
        self.op = op
        pars = []
        if isinstance(a,BaseParameter): pars += a.parameters()
        self._parameters = pars
        self.name = str(self)
    def parameters(self):
        return self._parameters

    @property
    def value(self):
        return self.operation(float(self.a))
    @property
    def dvalue(self):
        return float(self.a)
    def __str__(self):
        return "%s(%s)" % (self.op, self.a)

def unary(operation):
    """
    Convert a unary function into a delayed evaluator.

    The value of the function is computed from the values of the parameters
    at the time that the function value is requested rather than when the
    function is created.
    """
    # Note: @functools.wraps(op) does not work with numpy ufuncs
    # Note: @decorator does not work with builtins like abs
    def unary_generator(*args, **kw):
        return UnaryExpression(*args, op=operation.__name__)
    unary_generator.__name__ = operation.__name__
    unary_generator.__doc__ = operation.__doc__
    return unary_generator

#_abs = unary(abs)
# Numpy trick: math functions from numpy delegate to the math function of
# the class if that function exists as a class attribute.
for u_item in UNARY_OPERATIONS:
    op = u_item.value
    if op in UNARY_OPERATORS:
        setattr(BaseParameter,'__{op}__'.format(op=op), unary(getattr(operator, op)))
    else:
        setattr(BaseParameter, op, unary(getattr(math, op)))

def substitute(a):
    """
    Return structure a with values substituted for all parameters.

    The function traverses lists, tuples and dicts recursively.  Things
    which are not parameters are returned directly.
    """
    if isinstance(a, BaseParameter):
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


class Function(BaseParameter):
    """
    Delayed function evaluator.

    f.value evaluates the function with the values of the
    parameter arguments at the time f.value is referenced rather
    than when the function was invoked.
    """
    __slots__ = ['op', 'args', 'kw']
    op: Callable[..., float]
    args: Optional[Any]
    kw: Dict[Any, Any]

    def __init__(self, op, *args, **kw):
        self.name = kw.pop('name', None)
        self.op, self.args, self.kw = op, args, kw
        self._parameters = self._find_parameters()

    def _find_parameters(self):
        # Figure out which arguments to the function are parameters
        #deps = [p for p in self.args if isinstance(p,BaseParameter)]
        args = [arg for arg in self.args if isinstance(arg, BaseParameter)]
        kw = dict((name, arg) for name, arg in self.kw.items()
                  if isinstance(arg, BaseParameter))
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
        #return "%s:%g" % (name, self.value)


def function(op):
    """
    Convert a function into a delayed evaluator.

    The value of the function is computed from the values of the parameters
    at the time that the function value is requested rather than when the
    function is created.
    """
    # Note: @functools.wraps(op) does not work with numpy ufuncs
    # Note: @decorator does not work with builtins like abs
    def function_generator(*args, **kw):
        return Function(op, *args, **kw)
    function_generator.__name__ = op.__name__
    function_generator.__doc__ = op.__doc__
    return function_generator


def boxed_function(f):
    box = function(f)
    @wraps(f)
    def wrapped(*args, **kw):
        if any(isinstance(v, BaseParameter) for v in args):
            return box(*args, **kw)
        else:
            return f(*args, **kw)
    return wrapped

# arctan2 is special since either argument can be a parameter
arctan2 = boxed_function(math.atan2)

# Trig functions defined in degrees rather than radians
@boxed_function
def cosd(v):
    """Return the cosine of x (measured in in degrees)."""
    return math.cos(math.radians(v))

@boxed_function
def sind(v):
    """Return the sine of x (measured in in degrees)."""
    return math.sin(math.radians(v))

@boxed_function
def tand(v):
    """Return the tangent of x (measured in in degrees)."""
    return math.tan(math.radians(v))

@boxed_function
def acosd(v):
    """Return the arc cosine (measured in in degrees) of x."""
    return math.degrees(math.acos(v))
arccosd = acosd

@boxed_function
def asind(v):
    """Return the arc sine (measured in in degrees) of x."""
    return math.degrees(math.asin(v))
arcsind = asind

@boxed_function
def atand(v):
    """Return the arc tangent (measured in in degrees) of x."""
    return math.degrees(math.atan(v))
arctand = atand

@boxed_function
def atan2d(dy, dx):
    """Return the arc tangent (measured in in degrees) of y/x.
    Unlike atan(y/x), the signs of both x and y are considered."""
    return math.degrees(math.atan2(dy, dx))
arctan2d = atan2d



def flatten(s):
    if isinstance(s, (tuple, list, np.ndarray)):
        return reduce(lambda a, b: a + flatten(b), s, [])
    elif isinstance(s, set):
        raise TypeError("parameter flattening cannot order sets")
    elif isinstance(s, dict):
        return reduce(lambda a, b: a + flatten(s[b]), sorted(s.keys()), [])
    elif isinstance(s, BaseParameter):
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
            if k.startswith('_'):
                continue
            s = format(p[k], indent + 2, field=k, freevars=freevars)
            label = " " * indent + "." + k
            if s.endswith('\n'):
                res.append(label + "\n" + s)
            else:
                res.append(label + " = " + s + '\n')
        if '_index' in p:
            res .append(format(p['_index'], indent, freevars=freevars))
        return "".join(res)

    elif isinstance(p, (list, tuple, np.ndarray)) and len(p):
        res = []
        for k, v in enumerate(p):
            s = format(v, indent + 2, freevars=freevars)
            label = " " * indent + "[%d]" % k
            if s.endswith('\n'):
                res.append(label + '\n' + s)
            else:
                res.append(label + ' = ' + s + '\n')
        return "".join(res)

    elif isinstance(p, Parameter):
        s = ""
        if str(p) != field:
            s += str(p) + " = "
        s += "%g" % p.value
        if not p.fixed:
            s += " in [%g,%g]" %  tuple(p.bounds.limits)
        return s

    elif isinstance(p, BaseParameter):
        return "%s = %g" % (str(p), p.value)

    else:
        return "None"


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
            position = int(p.bounds.get01(p.value) * 9.999999999)
            bar = ['.'] * 10
            if position < 0:
                bar[0] = '<'
            elif position > 9:
                bar[9] = '>'
            else:
                bar[position] = '|'
        output.append("%40s %s %10g in %s" %
                      (p.name, "".join(bar), p.value, p.bounds))
    return "\n".join(output)


def unique(s):
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
    return [p for p in unique(s) if not p.fittable]


def varying(s):
    """
    Return the list of fitted parameters in the model.

    This is the set of parameters that will vary during the fit.
    """
    return [p for p in unique(s) if not p.fixed]


def randomize(s):
    """
    Set random values to the parameters in the parameter set, with
    values chosen according to the bounds.
    """
    for p in s:
        p.value = p.bounds.random(1)[0]


def current(s):
    return [p.value for p in s]

# ========= trash ===================

# this is a bit tricksy, because it's pretending to *be* an int
# the fact that it has attributes that do stuff doesn't interfere
# with its essential int-ness.
class IntegerProperty(int):
    backing_name: str = "_value"

    def __new__(cls, backing_name="_value"):
        i = int.__new__(cls, 0)
        i.backing_name = backing_name
        return i
    def __get__(self, obj, owner=None) -> int:
        if obj is None:
            return self
        else:
            return getattr(obj, self.backing_name)
    def __set__(self, obj, value: Union[float, int]):
        setattr(obj, self.backing_name, int(value))
    def __repr__(self):
        return object.__repr__(self)


@schema()
class IntegerParameter(Parameter):
    value: int
    discrete: Literal[True] = True
    _value: int

    #value = property(_get_value, _set_value)
    value = IntegerProperty("_value")


# ==== Comparison operators ===
class COMPARISONS(Enum):
    """comparison operators"""

    GT = '>'
    GE = '>='
    LE = '<='
    LT = '<'
    EQ = '=='
    NE = '!='


@schema()
class Constraint:

    a: Union[Parameter, UnaryExpression, Expression, float]
    b: Union[Parameter, UnaryExpression, Expression, float]
    op: COMPARISONS

    def __init__(self, a, b, op):
        import operator
        self.a, self.b = a, b
        op_name = str(COMPARISONS(op).name)
        self.compare = getattr(operator, op_name.lower())
        self.op = op

    def __bool__(self):
        return self.compare(float(self.a), float(self.b))
    __nonzero__ = __bool__
    def __str__(self):
        return "(%s %s %s)" %(self.a, self.op, self.b)


def make_constraint(op_str: str) -> Callable[..., Constraint]:
    def o(self, other):
        return Constraint(self, other, op_str)
    return o

for comp_item in COMPARISONS:
    op_name = comp_item.name
    op_str = comp_item.value

    setattr(BaseParameter, '__{op_name}__'.format(op_name=op_name.lower()), make_constraint(op_str))

class Alias(object):
    """
    Parameter alias.

    Rather than modifying a model to contain a parameter slot,
    allow the parameter to exist outside the model. The resulting
    parameter will have the full parameter semantics, including
    the ability to replace a fixed value with a parameter expression.

    **Deprecated** :class:`Reference` does this better.
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
            'type': type(self).__name__,
            'p': to_dict(self.p),
            # TODO: can't json pickle arbitrary objects
            'obj': to_dict(self.obj),
            'attr': self.attr,
        }

#restate these for export, now that they're all defined:
PARAMETER_TYPES = Union[Parameter, Expression, UnaryExpression, Constant]

def test_operator():
    a = Parameter(1, name='a')
    b = Parameter(2, name='b')
    a_b = a + b
    a.value = 3
    assert a_b.value == 5.
    assert a_b.name == '(a + b)'