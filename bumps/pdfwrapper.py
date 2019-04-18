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
from __future__ import print_function

import inspect

import numpy as np

from .parameter import Parameter
from .fitproblem import Fitness
from .bounds import init_bounds


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
    has_residuals = False  # Don't have true residuals

    def __init__(self, fn, name="", plot=None, dof=1, **kw):
        self.dof = dof
        # Make every name a parameter; initialize the parameters
        # with the default value if function is defined with keyword
        # initializers; override the initializers with any keyword
        # arguments specified in the fit function constructor.
        labels, vararg, varkw, values = inspect.getargspec(fn)
        if vararg or varkw:
            raise TypeError(
                "Function cannot have *args or **kwargs in declaration")
        # Parameters default to zero
        init = dict((p, 0.) for p in labels)
        # If the function provides default values, use those
        if values:
            init.update(zip(labels[-len(values):], values))
        # Regardless, use any values specified in the constructor, but first
        # check that they exist as function parameters.
        invalid = set(kw.keys()) - set(labels)
        if invalid:
            raise TypeError("Invalid initializers: %s" %
                            ", ".join(sorted(invalid)))
        init.update(kw)

        # Build parameters out of ranges and initial values
        pars = dict((p, Parameter.default(init[p], name=name + p))
                    for p in labels)

        # Make parameters accessible as model attributes
        for k, v in pars.items():
            if hasattr(self, k):
                raise TypeError("Parameter cannot be named %s" % k)
            setattr(self, k, v)

        # Remember the function, parameters, and number of parameters
        self._function = fn
        self._labels = labels
        self._plot = plot

    def parameters(self):
        return dict((p, getattr(self, p)) for p in self._labels)
    parameters.__doc__ = Fitness.parameters.__doc__

    def nllf(self):
        kw = dict((p, getattr(self, p).value) for p in self._labels)
        return self._function(**kw)
    nllf.__doc__ = Fitness.__call__.__doc__

    def chisq(self):
        return self.nllf()/self.dof
    #chisq.__doc__ = Fitness.chisq.__doc__

    def chisq_str(self):
        return "%g" % self.chisq()
    #chisq_str.__doc__ = Fitness.chisq_str.__doc__

    __call__ = chisq

    def plot(self, view=None):
        if self._plot:
            kw = dict((p, getattr(self, p).value) for p in self._labels)
            self._plot(view=view, **kw)
    plot.__doc__ = Fitness.plot.__doc__

    def numpoints(self):
        return len(self._labels) + 1
    numpoints.__doc__ = Fitness.numpoints.__doc__

    def residuals(self):
        return np.array([self.chisq()])
    residuals.__doc__ = Fitness.residuals.__doc__


class VectorPDF(object):
    """
    Build a model from a function.

    This model can be fitted with any of the bumps optimizers.

    *fn* is a function that returns the negative log likelihood of seeing
    its input parameters.

    Vector *p* of length *n* defines the initial value. Unlike :class:`PDF`,
    *VectorPDF* operates on a parameter vector *p* rather than individual
    parameters *p1*, *p2*, etc.  Default parameter values *p* must be
    provided in order to determine the number of parameters.

    *labels* are the names of the individual parameters.  If not present,
    the name for parameter *k* defaults to *pk*.  Each label is prefixed by
    *name*.

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
    has_residuals = False  # Don't have true residuals

    def __init__(self, fn, p, name="", plot=None, dof=1, labels=None, **kw):
        self.dof = dof
        if labels is None:
            labels = ["p"+str(k) for k, _ in enumerate(p)]
        init = dict(zip(labels, p))
        init.update(kw)

        # Build parameters out of ranges and initial values
        pars = dict((k, Parameter.default(init[k], name=name + k))
                    for k in labels)

        # Make parameters accessible as model attributes
        for k, v in pars.items():
            if hasattr(self, k):
                raise TypeError("Parameter cannot be named %s" % k)
            setattr(self, k, v)

        # Remember the function, parameters, and number of parameters
        self._function = fn
        self._labels = labels
        self._plot = plot

    def parameters(self):
        return dict((k, getattr(self, k)) for k in self._labels)
    parameters.__doc__ = Fitness.parameters.__doc__

    def nllf(self):
        pvec = np.array([getattr(self, k).value for k in self._labels])
        return self._function(pvec)
    nllf.__doc__ = Fitness.__call__.__doc__

    def chisq(self):
        return self.nllf()/self.dof
    #chisq.__doc__ = Fitness.chisq.__doc__

    def chisq_str(self):
        return "%g" % self.chisq()
    #chisq_str.__doc__ = Fitness.chisq_str.__doc__

    __call__ = chisq

    def plot(self, view=None):
        if self._plot:
            values = np.array([getattr(self, k).value for k in self._labels])
            self._plot(values, view=view)
    plot.__doc__ = Fitness.plot.__doc__

    def numpoints(self):
        return len(self._labels) + 1
    numpoints.__doc__ = Fitness.numpoints.__doc__

    def residuals(self):
        return np.array([self.chisq()])
    residuals.__doc__ = Fitness.residuals.__doc__



class DirectProblem(object):
    """
    Build model from negative log likelihood function *f(p)*.

    Vector *p* of length *n* defines the initial value.

    *bounds* defines limiting values for *p* as
    *[(p1_low, p1_high), (p2_low, p2_high), ...]*.  If all parameters are
    have the same bounds, use *bounds=np.tile([low,high],[n,1])*.

    Unlike :class:`PDF`, no parameter objects are defined for the elements
    of *p*, so all are fitting parameters.
    """
    has_residuals = False  # Don't have true residuals

    def __init__(self, f, p0, bounds=None, dof=1, labels=None, plot=None):
        self.f = f
        self.n = len(p0)
        self.p = np.asarray(p0, 'd')
        self.dof = dof
        if bounds is not None:
            self._bounds = np.asarray(bounds, 'd')
        else:
            self._bounds = np.tile((-np.inf, np.inf), (self.n, 1)).T

        self._labels = labels if labels else ["p%d" % i for i, _ in enumerate(p0)]
        self._plot = plot
        self.model_reset()

    def nllf(self, pvec=None):
        # Nllf is the primary interface from the fitters.  We are going to
        # make it as cheap as possible by not having to marshall values
        # through parameter boxes.
        return self.f(pvec) if pvec is not None else self.f(self.p)

    def model_reset(self):
        self._parameters = [Parameter(value=self.p[k],
                                      bounds=self._bounds[:, k],
                                      labels=self._labels[k])
                            for k in range(len(self.p))]

    def model_update(self):
        self.p = np.array([p.value for p in self._parameters])

    def model_parameters(self):
        return self._parameters

    def chisq(self):
        return self.nllf()/self.dof

    def chisq_str(self):
        return "%g" % self.chisq()
    __call__ = chisq

    def setp(self, p):
        # Note: setp is called
        self.p = p
        for parameter, value in zip(self._parameters, self.p):
            parameter.value = value

    def getp(self):
        return self.p

    def show(self):
        print("[nllf=%g]" % self.nllf())
        print(self.summarize())

    def summarize(self):
        return "\n".join("%40s %g"%(name, value)
                         for name, value in zip(self._labels, self.getp()))

    def labels(self):
        return self._labels

    def randomize(self, n=None):
        bounds = [init_bounds(b) for b in self._bounds.T]
        if n is not None:
            return np.array([b.random(n) for b in bounds]).T
        else:
            # Need to go through setp when updating model.
            self.setp([b.random(1)[0] for b in bounds])

    def bounds(self):
        return self._bounds

    def plot(self, p=None, fignum=None, figfile=None, view=None):
        if p is not None:
            self.setp(p)
        if self._plot:
            values = np.array([getattr(self, p).value for p in self._labels])
            self._plot(values, view=view)
    plot.__doc__ = Fitness.plot.__doc__

