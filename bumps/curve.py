"""
Build a bumps model from a function and data.

Example
-------

Given a function *sin_model* which computes a sine wave at times *t*::

    from numpy import sin
    def sin_model(t, freq, phase):
        return sin(2*pi*(freq*t + phase))

and given data *(y,dy)* measured at times *t*, we can define the fit
problem as follows::

    from bumps.names import *
    M = Curve(sin_model, t, y, dy, freq=20)

The *freq* and *phase* keywords are optional initial values for the model
parameters which otherwise default to zero.  The model parameters can be
accessed as attributes on the model to set fit range::

    M.freq.range(2, 100)
    M.phase.range(0, 1)

As usual, you can initialize or assign parameter expressions to the the
parameters if you want to tie parameters together within or between models.

Note: there is sometimes difficulty getting bumps to recognize the function
during fits, which can be addressed by putting the definition in a separate
file on the python path.  With the windows binary distribution of bumps,
this can be done in the problem definition file with the following code::

    import os
    from bumps.names import *
    sys.path.insert(0, os.getcwd())

The model function can then be imported from the external module as usual::

    from sin_model import sin_model
"""
__all__ = ["Curve", "PoissonCurve", "plot_err"]

import inspect

import numpy as np
from numpy import log, pi, sqrt

from .parameter import Parameter


class Curve(object):
    r"""
    Model a measurement with a user defined function.

    The function *fn(x,p1,p2,...)* should return the expected value *y* for
    each point *x* given the parameters *p1*, *p2*, etc.  *dy* is the uncertainty
    for each measured value *y*.  If not specified, it defaults to 1.
    Initial values for the parameters can be set as *p=value* arguments to *Curve*.
    If no value is set, then the initial value will be taken from the default
    value given in the definition of *fn*, or set to 0 if the parameter is not
    defined with an initial value.  Arbitrary non-fittable data can be passed
    to the function as parameters, but only if the parameter is given a default
    value of *None* in the function definition, and has the initial value set
    as an argument to *Curve*.  Defining *state=dict(key=value, ...)* before
    *Curve*, and calling *Curve* as *Curve(..., \*\*state)* works pretty well.

    *Curve* takes two special keyword arguments: *name* and *plot*.
    *name* is added to each parameter name when the parameter is defined.
    The filename for the data is a good choice, since this allows you to keep
    the parameters straight when fitting multiple datasets simultaneously.

    Plotting defaults to a 1-D plot with error bars for the data, and a line
    for the function value.  You can assign your own plot function with
    the *plot* keyword.  The function should be defined as *plot(x,y,dy,fy,\*\*kw)*.
    The keyword arguments will be filled with the values of the parameters
    used to compute *fy*.  It will be easiest to list the parameters you
    need to make your plot as positional arguments after *x,y,dy,fy* in the
    plot function declaration.  For example, *plot(x,y,dy,fy,p3,\*\*kw)*
    will make the value of parameter *p3* available as a variable in your
    function.  The special keyword *view* will be a string containing
    *linear*, *log*, *logx* or *loglog*.

    The data uncertainty is assumed to follow a gaussian distribution.
    If measurements draw from some other uncertainty distribution, then
    subclass Curve and replace nllf with the correct probability given the
    residuals.  See the implementation of :class:`PoissonCurve` for an example.
    """
    def __init__(self, fn, x, y, dy=None, name="", plot=None, **fnkw):
        self.x, self.y = np.asarray(x), np.asarray(y)
        if dy is None:
            self.dy = 1
        else:
            self.dy = np.asarray(dy)
            if (self.dy <= 0).any():
                raise ValueError("measurement uncertainty must be positive")

        self.fn = fn
        self.name = name # if name else fn.__name__ + " "

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
        # Non-fittable parameters need to be sent in as None
        state_vars = set(p for p, v in init.items() if v is None)
        # Regardless, use any values specified in the constructor, but first
        # check that they exist as function parameters.
        invalid = set(fnkw.keys()) - set(pnames)
        if invalid:
            raise TypeError("Invalid initializers: %s" %
                            ", ".join(sorted(invalid)))
        init.update(fnkw)

        # Build parameters out of ranges and initial values
        # maybe:  name=(p+name if name.startswith('_') else name+p)
        pars = dict((p, Parameter.default(init[p], name=name + p))
                    for p in pnames if p not in state_vars)

        # Make parameters accessible as model attributes
        for k, v in pars.items():
            if hasattr(self, k):
                raise TypeError("Parameter cannot be named %s" % k)
            setattr(self, k, v)

        # Remember the function, parameters, and number of parameters
        self._function = fn
        self._pnames = [p for p in pnames if p not in state_vars]
        self._cached_theory = None
        self._plot = plot if plot is not None else plot_err
        self._state = dict((p, v) for p, v in init.items() if p in state_vars)

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
            kw.update(self._state)
            self._cached_theory = self._function(x, **kw)
        return self._cached_theory

    def simulate_data(self, noise=None):
        theory = self.theory()
        if noise is not None:
            if noise == 'data':
                pass
            elif noise < 0:
                self.dy = -theory*noise*0.01
            else:
                self.dy = noise
        self.y = theory + np.random.randn(*theory.shape)*self.dy

    def residuals(self):
        return (self.theory() - self.y) / self.dy

    def nllf(self):
        r = self.residuals()
        return 0.5 * np.sum(r ** 2)

    def save(self, basename):
        # TODO: need header line with state vars as json
        # TODO: need to support nD x,y,dy
        data = np.vstack((self.x, self.y, self.dy, self.theory()))
        np.savetxt(basename + '.dat', data.T)

    def plot(self, view=None):
        import pylab
        kw = dict((p, getattr(self, p).value) for p in self._pnames)
        kw.update(self._state)
        #print "kw_plot",kw
        if view == 'residual':
            plot_resid(self.x, self.residuals())
        else:
            plot_ratio = 4
            h = pylab.subplot2grid((plot_ratio, 1), (0, 0), rowspan=plot_ratio-1)
            self._plot(self.x, self.y, self.dy, self.theory(), view=view, **kw)
            for tick_label in pylab.gca().get_xticklabels():
                tick_label.set_visible(False)
            #pylab.gca().xaxis.set_visible(False)
            #pylab.gca().spines['bottom'].set_visible(False)
            #pylab.gca().set_xticks([])
            pylab.subplot2grid((plot_ratio, 1), (plot_ratio-1, 0), sharex=h)
            plot_resid(self.x, self.residuals())

def plot_resid(x, resid):
    import pylab
    pylab.plot(x, resid, '.')
    pylab.gca().locator_params(axis='y', tight=True, nbins=4)
    pylab.axhline(y=1, ls='dotted')
    pylab.axhline(y=-1, ls='dotted')
    pylab.ylabel("Residuals")

def plot_err(x, y, dy, fy, view=None, **kw):
    """
    Plot data *y* and error *dy* against *x*.

    *view* is one of linear, log, logx or loglog.
    """
    import pylab
    pylab.errorbar(x, y, yerr=dy, fmt='.')
    pylab.plot(x, fy, '-')
    if view == 'log':
        pylab.xscale('linear')
        pylab.yscale('log')
    elif view == 'logx':
        pylab.xscale('log')
        pylab.yscale('linear')
    elif view == 'loglog':
        pylab.xscale('log')
        pylab.yscale('log')
    else: # view == 'linear'
        pylab.xscale('linear')
        pylab.yscale('linear')

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
    Model a measurement with Poisson uncertainty.

    The nllf is calculated using Poisson probabilities, but the curve itself
    is displayed using the approximation that $\sigma_y \approx \sqrt(y)$.

    See :class:`Curve` for details.
    """
    def __init__(self, fn, x, y, name="", **fnkw):
        dy = sqrt(y) + (y == 0) if y is not None else None
        Curve.__init__(self, fn, x, y, dy, name=name, **fnkw)
        self._logfacty = logfactorial(y) if y is not None else None
        self._logfactysum = np.sum(self._logfacty)

    ## Assume gaussian residuals for now
    #def residuals(self):
    #    # TODO: provide individual probabilities as residuals
    #    # or perhaps the square roots --- whatever gives a better feel for
    #    # which points are driving the fit
    #    theory = self.theory()
    #    return np.sqrt(self.y * log(theory) - theory - self._logfacty)

    def nllf(self):
        theory = self.theory()
        if (theory <= 0).any():
            return 1e308
        return -sum(self.y * log(theory) - theory) + self._logfactysum

    def simulate_data(self, noise=None):
        theory = self.theory()
        self.y = np.random.poisson(theory)
        self.dy = sqrt(self.y) + (self.y == 0)
        self._logfacty = logfactorial(self.y)
        self._logfactysum = np.sum(self._logfacty)

    def save(self, basename):
        # TODO: need header line with state vars as json
        # TODO: need to support nD x,y,dy
        data = np.vstack((self.x, self.y, self.theory()))
        np.savetxt(basename + '.dat', data.T)
