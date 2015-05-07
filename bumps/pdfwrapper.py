"""
Build a bumps model from a function.

The :class:`PDF` class uses introspection to convert a negative log
likelihood function nllf(m1,m2,...) into a :class:`bumps.fitproblem.Fitness`
class that has fittable parameters m1, m2, ....

There is no attempt to manage data or uncertainties, except that an
additional plot function can be provided to display the current value
of the function in whatever way is meaningful.

The note regarding user defined functions in :mod:`bumps.curve` apply
here as well.
"""
import inspect

import numpy as np

from .parameter import Parameter
from .fitproblem import Fitness


class PDF(object):
    """
    Build a model from a function.

    This model can be fitted with any of the bumps optimizers.

    *fn* is a function that returns the negative log likelihood of seeing
    its input parameters.

    The fittable parameters are derived from the parameter names in the
    function definition, with *name* prepended to each parameter.

    The optional *plot* function takes the same arguments as *fn*, with an
    additional *view* argument which may be set from the bumps command
    line.  If provide, it should provide a visual indication of the
    function value and uncertainty on the current matplotlib.pyplot figure.

    Additional keyword arguments are treated as the initial values for
    the parameters, or initial ranges if par=(min,max).  Otherwise, the
    default is taken from the function definition (if the function uses
    par=value to define the parameter) or is set to zero if no default is
    given in the function.
    """
    def __init__(self, fn, name="", plot=None, **kw):
        # Make every name a parameter; initialize the parameters
        # with the default value if function is defined with keyword
        # initializers; override the initializers with any keyword
        # arguments specified in the fit function constructor.
        pnames, vararg, varkw, pvalues = inspect.getargspec(fn)
        if vararg or varkw:
            raise TypeError(
                "Function cannot have *args or **kwargs in declaration")
        # Parameters default to zero
        init = dict((p, 0) for p in pnames)
        # If the function provides default values, use those
        if pvalues:
            init.update(zip(pnames[-len(pvalues):], pvalues))
        # Regardless, use any values specified in the constructor, but first
        # check that they exist as function parameters.
        invalid = set(kw.keys()) - set(pnames)
        if invalid:
            raise TypeError("Invalid initializers: %s" %
                            ", ".join(sorted(invalid)))
        init.update(kw)

        # Build parameters out of ranges and initial values
        pars = dict((p, Parameter.default(init[p], name=name + p))
                    for p in pnames)

        # Make parameters accessible as model attributes
        for k, v in pars.items():
            if hasattr(self, k):
                raise TypeError("Parameter cannot be named %s" % k)
            setattr(self, k, v)

        # Remember the function, parameters, and number of parameters
        self._function = fn
        self._pnames = pnames
        self._plot = plot

    def parameters(self):
        return dict((p, getattr(self, p)) for p in self._pnames)
    parameters.__doc__ = Fitness.parameters.__doc__

    def nllf(self):
        kw = dict((p, getattr(self, p).value) for p in self._pnames)
        return self._function(**kw)
    nllf.__doc__ = Fitness.__call__.__doc__

    def chisq(self):
        return self.nllf()
    #chisq.__doc__ = Fitness.chisq.__doc__

    def chisq_str(self):
        return "%g" % self.chisq()
    #chisq_str.__doc__ = Fitness.chisq_str.__doc__

    __call__ = chisq

    def plot(self, view=None):
        if self._plot:
            kw = dict((p, getattr(self, p).value) for p in self._pnames)
            self._plot(view=view, **kw)
    plot.__doc__ = Fitness.plot.__doc__

    def numpoints(self):
        return len(self._pnames) + 1
    numpoints.__doc__ = Fitness.numpoints.__doc__

    def residuals(self):
        return np.array([self()])
    residuals.__doc__ = Fitness.residuals.__doc__


class DirectPDF(object):
    """
    Build model from probability density function *f(p)*.

    Vector *p0* of length *n* defines the initial value.

    *bounds* defines limiting values for *p* as
    *[(p1_low, p1_high), (p2_low, p2_high), ...]*.  If all parameters are
    have the same bounds, use *bounds=np.tile([low,high],[n,1])*.

    Unlike :class:`PDF`, no parameter objects are defined for the elements
    of *p*, so all are fitting parameters.
    """
    def __init__(self, f, p0, bounds=None):
        self.f = f
        self.n = len(p0)
        self.p = np.asarray(p0, 'd')
        if bounds is not None:
            self._bounds = np.asarray(bounds, 'd')
        else:
            self._bounds = np.tile((-np.inf, np.inf), (self.n, 1)).T

    def model_reset(self):
        pass

    def chisq(self):
        return self.nllf()

    def chisq_str(self):
        return "%g" % self.chisq()
    __call__ = chisq

    def nllf(self, pvec=None):
        if pvec is not None:
            self.setp(pvec)
        return self.f(self.p)

    def setp(self, p):
        self.p = p

    def getp(self):
        return self.p

    def show(self):
        print("[nllf=%g]" % self.nllf())
        print(self.summarize())

    def summarize(self):
        return str(self.getp())

    def labels(self):
        return ["P%d" % i for i in range(self.n)]

    def randomize(self):
        # TODO: doesn't respect bounds
        self.p = np.random.rand(self.n)

    def bounds(self):
        return self._bounds

    def plot(self, p=None, fignum=None, figfile=None):
        pass
        #def __deepcopy__(self, memo): return self
