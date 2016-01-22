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
from six.moves import reduce
import warnings
from copy import copy

from numpy import inf, isinf, isfinite

from . import bounds as mbounds

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


class BaseParameter(object):
    """
    Root of the parameter class, defining arithmetic on parameters
    """

    # Parameters are fixed unless told otherwise
    fixed = True
    fittable = False
    discrete = False
    _bounds = mbounds.Unbounded()
    name = None

    # Parameters may be dependent on other parameters, and the
    # fit engine will need to access them.
    def parameters(self):
        return [self]

    def pmp(self, *args):
        """
        Allow the parameter to vary as value +/- percent.

        pmp(*percent*) -> [value*(1-percent/100), value*(1+percent/100)]

        pmp(*plus*, *minus*) -> [value*(1+minus/100), value*(1+plus/100)]

        In the *plus/minus* form, one of the numbers should be plus and the
        other minus, but it doesn't matter which.

        The resulting range is converted to "nice" numbers.
        """
        self.bounds = mbounds.Bounded(*mbounds.pmp(self.value, *args))
        return self

    def pm(self, *args):
        """
        Allow the parameter to vary as value +/- delta.

        pm(*delta*) -> [value-delta, value+delta]

        pm(*plus*, *minus*) -> [value+minus, value+plus]

        In the *plus/minus* form, one of the numbers should be plus and the
        other minus, but it doesn't matter which.

        The resulting range is converted to "nice" numbers.
        """
        self.bounds = mbounds.Bounded(*mbounds.pm(self.value, *args))
        return self

    def dev(self, std, mean=0, limits=None, sigma=None, mu=None):
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
            if sigma is not None: std = sigma
            if mu is not None: mean = mu
        if mean is None:
            mean = self.value # Note: value is an attribute of the derived class
        if limits is None:
            self.bounds = mbounds.Normal(mean, std)
        else:
            self.bounds = mbounds.BoundedNormal(mean, std, limits)
        return self

    def pdf(self, dist):
        """
        Allow the parameter to vary according to any continuous scipy.stats
        distribution.
        """
        self.bounds = mbounds.Distribution(dist)
        return self

    def range(self, low, high):
        """
        Allow the parameter to vary within the given range.
        """
        self.bounds = mbounds.init_bounds((low, high))
        return self

    def soft_range(self, low, high, std):
        """
        Allow the parameter to vary within the given range, or with Gaussian
        probability, stray from the range.
        """
        self.bounds = mbounds.SoftBounded(low, high, std)

    @property
    def bounds(self):
        """Fit bounds"""
        # print "getting bounds for",self,self._bounds
        return self._bounds

    @bounds.setter
    def bounds(self, b):
        # print "setting bounds for",self
        if self.fittable:
            self.fixed = (b is None)
        self._bounds = b

    # Functional form of parameter value access
    def __call__(self):
        return self.value

    # Parameter algebra: express relationships between parameters
    def __gt__(self, other):
        return ConstraintGT(self, other)

    def __ge__(self, other):
        return ConstraintGE(self, other)
    def __le__(self, other):
        return ConstraintLE(self, other)

    def __lt__(self, other):
        return ConstraintLT(self, other)

    # def __eq__(self, other):
    #     return ConstraintEQ(self, other)

    # def __ne__(self, other):
    #     return ConstraintNE(self, other)

    def __add__(self, other):
        return OperatorAdd(self, other)

    def __sub__(self, other):
        return OperatorSub(self, other)

    def __mul__(self, other):
        return OperatorMul(self, other)

    def __div__(self, other):
        return OperatorDiv(self, other)

    def __pow__(self, other):
        return OperatorPow(self, other)

    def __radd__(self, other):
        return OperatorAdd(other, self)

    def __rsub__(self, other):
        return OperatorSub(other, self)

    def __rmul__(self, other):
        return OperatorMul(other, self)

    def __rdiv__(self, other):
        return OperatorDiv(other, self)

    def __rpow__(self, other):
        return OperatorPow(other, self)

    def __abs__(self):
        return _abs(self)

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __float__(self):
        return float(self.value)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

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


class Constant(BaseParameter):
    """
    An unmodifiable value.
    """
    fittable = False
    fixed = True

    @property
    def value(self):
        return self._value

    def __init__(self, value, name=None):
        self._value = value
        self.name = name


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
    fittable = True

    @classmethod
    def default(cls, value, **kw):
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

    def __init__(self, value=None, bounds=None, fixed=None, name=None, **kw):
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
        limits = kw.get('limits', (-inf, inf))
        for k, v in kw.items():
            setattr(self, k, v)

        # Initialize bounds, with limits clipped to the hard limits for the
        # parameter
        def clip(x, a, b):
            return min(max(x, a), b)
        self.bounds = bounds
        self.bounds.limits = (clip(self.bounds.limits[0], *limits),
                              clip(self.bounds.limits[1], *limits))
        self.value = value
        self.fixed = fixed
        self.name = name

    def randomize(self, rng=None):
        """
        Set a random value for the parameter.
        """
        self.value = self.bounds.rand(rng if rng is not None else mbounds.RNG)

    def feasible(self):
        """
        Value is within the limits defined by the model
        """
        return self.limits[0] <= self.value <= self.limits[1]


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


class ParameterSet(object):
    """
    A parameter that depends on the model.
    """

    def __init__(self, reference, names=None):
        """
        Create a parameter set, with one parameter for each model name.

        *names* is the list of model names.

        *reference* is the underlying :class:`parameter.Parameter` that will
        be set when the model is selected.

        *parameters* will be created, with one parameter per model.
        """
        self.names = names
        self.reference = reference
        self.parameters = [copy(reference) for _ in names]
        # print self.reference, self.parameters
        for p, n in zip(self.parameters, names):
            p.name = " ".join((n, p.name))
        # Reference is no longer directly fittable
        self.reference.fittable = False

    # Make the parameter set act like a list
    def __getitem__(self, i):
        """
        Return the underlying parameter for the model index.  Index can
        either be an integer or a model name.
        """
        # Try looking up the free variable by model name rather than model
        # index. If this fails, assume index is a model index.
        try:
            i = self.names.index(i)
        except ValueError:
            pass
        return self.parameters[i]

    def __setitem__(self, i, v):
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

    @property
    def values(self):
        return [p.value for p in self.parameters]

    @values.setter
    def values(self, values):
        for p, v in zip(self.parameters, values):
            p.value = v

    def range(self, *args, **kw):
        """
        Like :meth:`parameter.Parameter.range`, but applied to all models.
        """
        for p in self.parameters:
            p.range(*args, **kw)

    def pm(self, *args, **kw):
        """
        Like :meth:`parameter.Parameter.pm`, but applied to all models.
        """
        for p in self.parameters:
            p.pm(*args, **kw)

    def pmp(self, *args, **kw):
        """
        Like :meth:`parameter.Parameter.pmp`, but applied to all models.
        """
        for p in self.parameters:
            p.pmp(*args, **kw)


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

    def __init__(self, names=None, **kw):
        if names is None:
            raise TypeError("FreeVariables needs name=[model1, model2, ...]")
        self.names = names

        # Create slots to hold the free variables
        self._parametersets = dict((k, ParameterSet(v, names=names))
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
            return self._parametersets[k]
        except KeyError:
            raise AttributeError('FreeVariables has no attribute %r' % k)

    def parameters(self):
        """
        Return the set of free variables for all the models.
        """
        return dict((k, v.parameters) for k, v in self._parametersets.items())

    def set_model(self, i):
        """
        Set the reference parameters for model *i*.
        """
        for p in self._parametersets.values():
            p.set_model(i)


# Current implementation computes values on the fly, so you only
# need to plug the values into the parameters and the parameters
# are automatically updated.
#
# This will not work well for wrapped models.  In those cases you
# want to do a number of optimizations, such as only updating the
#

# ==== Comparison operators ===
class Constraint:
    """
    Abstract base class for constraints.
    """

    def __bool__(self):
        """
        Returns True if the condition is satisfied
        """
        raise NotImplementedError
    __nonzero__ = __bool__

    def __str__(self):
        """
        Text description of the constraint
        """
        raise NotImplementedError


def _gen_constraint(name, op):
    """
    Generate a comparison function from a comparison operator.
    """
    return '''\
class Constraint%(name)s(Constraint):
    """
    Constraint operator %(op)s
    """
    def __init__(self, a, b):
        self.a, self.b = a,b
    def __bool__(self):
        return float(self.a) %(op)s float(self.b)
    __nonzero__ = __bool__
    def __str__(self):
        return "(%%s %(op)s %%s)"%%(self.a,self.b)
''' % dict(name=name, op=op)

exec(_gen_constraint('GT', '>'))
exec(_gen_constraint('GE', '>='))
exec(_gen_constraint('LE', '<='))
exec(_gen_constraint('LT', '<'))
exec(_gen_constraint('EQ', '=='))
exec(_gen_constraint('NE', '!='))


# ==== Arithmetic operators ===
def _gen_binop(name, op):
    """
    Generate a comparison function from a comparison operator.
    """
    return '''\
class Operator%(name)s(BaseParameter):
    """
    Parameter operator %(op)s
    """
    def __init__(self, a, b):
        self.a, self.b = a,b
        pars = []
        if isinstance(a,BaseParameter): pars += a.parameters()
        if isinstance(b,BaseParameter): pars += b.parameters()
        self._parameters = pars
        self.name = str(self)
    def parameters(self):
        return self._parameters
    @property
    def value(self):
        return float(self.a) %(op)s float(self.b)
    @property
    def dvalue(self):
        return float(self.a)
    def __str__(self):
        return "(%%s %(op)s %%s)"%%(self.a,self.b)
''' % dict(name=name, op=op)

exec(_gen_binop('Add', '+'))
exec(_gen_binop('Sub', '-'))
exec(_gen_binop('Mul', '*'))
exec(_gen_binop('Div', '/'))
exec(_gen_binop('Pow', '**'))


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

    def __init__(self, op, *args, **kw):
        self.name = kw.pop('name', None)
        self.op, self.args, self.kw = op, args, kw

    def parameters(self):
        # Figure out which arguments to the function are parameters
        #deps = [p for p in self.args if isinstance(p,BaseParameter)]
        deps = flatten((self.args, self.kw))
        # Find out which other parameters these parameters depend on.
        res = []
        for p in deps:
            res.extend(p.parameters())
        return res

    def _value(self):
        # Expand args and kw, replacing instances of parameters
        # with their values
        return self.op(*substitute(self.args), **substitute(self.kw))
    value = property(_value)

    def __getstate__(self):
        return self.name, self.op, self.args, self.kw

    def __setstate__(self, state):
        self.name, self.op, self.args, self.kw = state

    def __str__(self):
        if self.name is not None:
            name = self.name
        else:
            args = [str(v) for v in self.args]
            kw = [str(k) + "=" + str(v) for k, v in self.kw.items()]
            name = self.op.__name__ + "(" + ", ".join(args + kw) + ")"
        return "%s:%g" % (name, self.value)


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
_abs = function(abs)


def flatten(s):
    if isinstance(s, (tuple, list)):
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


def format(p, indent=0):
    """
    Format parameter set for printing.

    Note that this only says how the parameters are arranged, not how they
    relate to each other.
    """
    if isinstance(p, dict) and p != {}:
        res = []
        for k in sorted(p.keys()):
            if k.startswith('_'):
                continue
            s = format(p[k], indent + 2)
            label = " " * indent + "." + k
            if s.endswith('\n'):
                res.append(label + "\n" + s)
            else:
                res.append(label + " = " + s + '\n')
        if '_index' in p:
            res .append(format(p['_index'], indent))
        return "".join(res)
    elif isinstance(p, list) and p != []:
        res = []
        for k, v in enumerate(p):
            s = format(v, indent + 2)
            label = " " * indent + "[%d]" % k
            if s.endswith('\n'):
                res.append(label + '\n' + s)
            else:
                res.append(label + ' = ' + s + '\n')
        return "".join(res)
    # elif isinstance(p, tuple) and p != ():
    #    return "".join(format(v, indent) for v in p)

    elif isinstance(p, Parameter):
        if p.fixed:
            bounds = ""
        else:
            bounds = ", bounds=(%g,%g)" %  p.bounds.limits
        return "Parameter(%g, name='%s'%s)" % (p.value, str(p), bounds)
    elif isinstance(p, BaseParameter):
        return str(p)
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
        pars = sorted(pars, cmp=lambda x, y: cmp(x.name, y.name))
    for p in pars:
        if not isfinite(p.value):
            bar = "*invalid* "
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


class IntegerParameter(Parameter):
    discrete = True

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = int(value)
    value = property(_get_value, _set_value)


class Alias(object):
    """
    Parameter alias.

    Rather than modifying a model to contain a parameter slot,
    allow the parameter to exist outside the model. The resulting
    parameter will have the full parameter semantics, including
    the ability to replace a fixed value with a parameter expression.

    # TODO: how is this any different from Reference above?
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
