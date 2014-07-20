"""
Build a bumps model from a function f and data x,y,dy.
"""
__all__ = ["Curve", "PoissonCurve"]

import inspect

import numpy as np
from numpy import log, pi, sqrt

from .parameter import Parameter


class Curve(object):

    """
    Build a model from a function and data.

    This model can be fitted with any of the bumps optimizers.

    The function *fn(x,p1,p2,...)* should return the expected value y for 
    each point x given the parameters p1, p2, ... .  Gaussian uncertainty
    dy must be specified for each point.  If measurements are drawn from
    some other uncertainty distribution, then subclass Curve and replace
    nllf with the correct probability given the residuals.  See the
    implementation of :class:`PoissonCurve` for an example.

    The fittable parameters are derived from the function definition, with
    the *name* prepended to each parameter if a name is given.

    Additional keyword arguments are treated as the initial values for
    the parameters, or initial ranges if par=(min,max).  Otherwise, the
    default is taken from the function definition (if the function uses
    par=value to define the parameter) or is set to zero if no default is
    given in the function.
    """

    def __init__(self, fn, x, y, dy=None, name="", **fnkw):
        self.x, self.y = np.asarray(x), np.asarray(y)
        if dy is None:
            self.dy = 1
        else:
            self.dy = np.asarray(dy)
            if (self.dy <= 0).any():
                raise ValueError("measurement uncertainty must be positive")

        self.fn = fn

        # Make every name a parameter; initialize the parameters
        # with the default value if function is defined with keyword
        # initializers; override the initializers with any keyword
        # arguments specified in the fit function constructor.
        pnames, vararg, varkw, pvalues = inspect.getargspec(fn)
        if vararg or varkw:
            raise TypeError(
                "Function cannot have *args or **kwargs in declaration")

        # TODO: need "self" handling for passed methods
        # assume the first argument is x
        pnames = pnames[1:]

        # Parameters default to zero
        init = dict((p, 0) for p in pnames)
        # If the function provides default values, use those
        if pvalues:
            # ignore default value for "x" parameter
            if len(pvalues) > len(pnames):
                pvalues = pvalues[1:]
            init.update(zip(pnames[-len(pvalues):], pvalues))
        # Regardless, use any values specified in the constructor, but first
        # check that they exist as function parameters.
        invalid = set(fnkw.keys()) - set(pnames)
        if invalid:
            raise TypeError("Invalid initializers: %s" %
                            ", ".join(sorted(invalid)))
        init.update(fnkw)

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
        self._cached_theory = None

    def update(self):
        self._cached_theory = None

    def parameters(self):
        return dict((p, getattr(self, p)) for p in self._pnames)

    def numpoints(self):
        return np.prod(self.y.shape)

    def theory(self, x=None):
        if self._cached_theory is None:
            if x is None:
                x = self.x
            kw = dict((p, getattr(self, p).value) for p in self._pnames)
            self._cached_theory = self._function(x, **kw)
        return self._cached_theory

    def residuals(self):
        return (self.theory() - self.y) / self.dy

    def nllf(self):
        r = self.residuals()
        return 0.5 * np.sum(r ** 2)

    def save(self, basename):
        data = np.vstack((self.x, self.y, self.dy, self.theory()))
        np.savetxt(basename + '.dat', data.T)

    def plot(self, view=None):
        import pylab
        pylab.errorbar(self.x, self.y, yerr=self.dy, fmt='.')
        pylab.plot(self.x, self.theory(), '-', hold=True)


_LOGFACTORIAL = np.array([log(np.prod(np.arange(1., k + 1)))
                             for k in range(21)])


def logfactorial(n):
    """Compute the log factorial for each element of an array"""
    result = np.empty(n.shape, dtype='double')
    idx = (n <= 20)
    result[idx] = _LOGFACTORIAL[np.asarray(n[idx], 'int32')]
    n = n[~idx]
    result[~idx] = n * \
        log(n) - n + log(n * (1 + 4 * n * (1 + 2 * n))) / 6 + log(pi) / 2
    return result


class PoissonCurve(Curve):

    r"""
    Model the measurement of a Poisson process.

    The nllf is calculated using Poisson probabilities, but the curve itself
    is displayed using the approximation that $\sigma_y \approx \sqrt(y)$.

    See :class:`Curve` for details.
    """

    def __init__(self, fn, x, y, name="", **fnkw):
        Curve.__init__(self, fn, x, y, sqrt(y), name=name, **fnkw)
        self._logfacty = np.sum(logfactorial(self.y))

    def nllf(self):
        theory = self.theory()
        if (theory <= 0).any():
            return 1e308
        return -sum(self.y * log(theory) - theory) + self._logfacty
