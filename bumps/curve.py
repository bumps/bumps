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
import warnings

import numpy as np
from numpy import log, pi, sqrt

from .parameter import Parameter

def _parse_pars(fn, init=None, skip=0, name=""):
    """
    Extract parameter names from function definition.

    *fn* is the function definition.  This could be declared as
    *fn(p1, p2, p3, ...)* where *p1*, etc. are the fittable parameters.

    *init* is a dictionary of initial values for the parameters,
    overriding any default values.  If called from a constructor with
    **kwargs representing unknown named arguments, use *init=kwargs*.

    *skip* is the number of parameters to skip.  This will be *skip=0*
    for a function which defines the log likelihood directly or one
    that returns a set of residuals. For parameterized curves such as
    *fn(x, p1, p2, ...)* use *skip=1*.  For surfaces with
    *fn(x, y, p1, p2, ...)* use *skip=2*.

    *name* is added to each parameter name to differentiate it from other
    parameters in the same fit.

    A default value in the function definition such as *pk=value* will
    be set as the default value for the parameter.  If the default is
    *pk=None* then the parameter will be non-fittable, and instead set
    through *init*.
    """
    pnames, vararg, varkw, pvalues = inspect.getargspec(fn)
    if vararg or varkw:
        raise TypeError(
            "Function %r cannot have *args or **kwargs in declaration"
            % fn.__name__)

    # TODO: need "self" handling for passed methods
    # Skip the first argument if it is x or maybe skip x, y.
    pnames = pnames[skip:]

    # Parameters default to zero
    defaults = dict((p, 0) for p in pnames)

    # If the function provides default values, use those.
    if pvalues:
        # Ignore default value for "x" parameter.
        if len(pvalues) > len(pnames):
            pvalues = pvalues[-len(pnames):]
        defaults.update(zip(pnames[-len(pvalues):], pvalues))

    # Non-fittable parameters need to be sent in as None
    state_vars = set(p for p, v in defaults.items() if v is None)

    # Regardless, use any values specified in the constructor, but first
    # check that they exist as function parameters.
    invalid = set(init.keys()) - set(pnames)
    if invalid:
        raise TypeError("Invalid initializers: %s" %
                        ", ".join(sorted(invalid)))
    defaults.update(init)

    # Build parameters out of ranges and initial values
    # maybe:  name=(p+name if name.startswith('_') else name+p)
    pars = dict((p, Parameter.default(defaults[p], name=name + p))
                for p in pnames if p not in state_vars)

    state = dict((p, v) for p, v in defaults.items() if p in state_vars)

    #print("pars", pars)
    #print("state", state)
    return pars, state

def _assign_pars(obj, pars):
    # Make parameters accessible as model attributes
    for k, v in pars.items():
        if hasattr(obj, k):
            raise TypeError("Parameter cannot be named %s" % k)
        setattr(obj, k, v)


class Curve(object):
    r"""
    Model a measurement with a user defined function.

    The function *fn(x,p1,p2,...)* should return the expected value *y* for
    each point *x* given the parameters *p1*, *p2*, etc.  *dy* is the
    uncertainty for each measured value *y*.  If not specified, it defaults
    to 1. Multi-valued functions, which return multiple *y* values for each
    *x* value, should have *x* as a vector of length *n* and *y*, *dy* as
    arrays of size *[n, k]*.

    Initial values for the parameters can be set as *p=value* arguments to
    *Curve*. If no value is set, then the initial value will be taken from
    the default value given in the definition of *fn*, or set to 0 if the
    parameter is not defined with an initial value.  Arbitrary non-fittable
    data can be passed to the function as parameters, but only if the
    parameter is given a default value of *None* in the function definition,
    and has the initial value set as an argument to *Curve*.  Defining
    *state=dict(key=value, ...)* before *Curve*, and calling *Curve* as
    *Curve(..., \*\*state)* works pretty well.

    *Curve* takes the following special keyword arguments:

    * *name* is added to each parameter name when the parameter is defined.
      The filename for the data is a good choice, since this allows you to keep
      the parameters straight when fitting multiple datasets simultaneously.

    * *plot* is an alternative plotting function. The function should be
      defined as *plot(x,y,dy,fy,\*\*kw)*. The keyword arguments will be
      filled with the values of the parameters used to compute *fy*.  It
      will be easiest to list the parameters you need to make your plot
      as positional arguments after *x,y,dy,fy* in the plot function
      declaration.  For example, *plot(x,y,dy,fy,p3,\*\*kw)* will make the
      value of parameter *p3* available as a variable in your function.  The
      special keyword *view* will be a string containing *linear*, *log*,
      *logx*, or *loglog*.  If only showing the residuals, the string
      will be *residual*.

    * *plot_x* is an array giving the sample points to use when plotting
      the theory function, if different from the *x* values at which the
      function is sampled.  Use this to draw a smooth curve between the
      fitted points.  This value is ignored if you provide your own plot
      function.

    * *labels* are the axis labels for the plot.  This should include
      units in parentheses. If the function is multi-valued then
      use *['x axis', 'y axis', 'line 1', 'line 2', ...]*.

    The data uncertainty is assumed to follow a gaussian distribution.
    If measurements draw from some other uncertainty distribution, then
    subclass Curve and replace nllf with the correct probability given the
    residuals.  See the implementation of :class:`PoissonCurve` for an example.
    """
    def __init__(self, fn, x, y, dy=None, name="", labels=None,
                 plot=None, plot_x=None, **kwargs):
        self.x, self.y = np.asarray(x), np.asarray(y)
        if dy is None:
            self.dy = 1
        else:
            self.dy = np.asarray(dy)
            if (self.dy <= 0).any():
                raise ValueError("measurement uncertainty must be positive")

        if len(self.x.shape) == 1 and len(self.y.shape) > 1:
            num_curves = self.y.shape[0]
        else:
            num_curves = 1
        self._num_curves = num_curves  # use same value everywhere

        # interpret labels parameter
        if labels is None:
            labels = ['x', 'y']
        elif len(labels) < 2 or len(labels) != num_curves+2:
            if num_curves > 1:
                lines = "line1, ..., line%d"%num_curves
            else:
                lines = "line"
            raise TypeError("labels should be [x, y, %s]"%lines)

        if len(labels) == 2:
            if num_curves > 1:
                line_labels = ['y%d'%k for k in range(num_curves)]
            else:
                line_labels = [labels[1]]
            labels = list(labels) + line_labels
        self.labels = labels


        # TODO: self.fn is a duplicate of self._function below. Deprecated?
        self.fn = fn
        self.name = name # if name else fn.__name__ + " "
        self.plot_x = plot_x

        pars, state = _parse_pars(fn, init=kwargs, skip=1)

        # Make parameters accessible as model attributes
        _assign_pars(self, pars)
        #_assign_pars(state, self)  # ... and state variables as well

        # Remember the function, parameters, and number of parameters
        # Note: we are remembering the parameter names and not the
        # parameters themselves so that the caller can tie parameters
        # together using model1.par = model2.par.  Otherwise we would
        # need to override __setattr__ to intercept assignment to the
        # parameter attributes and redirect them to the a _pars dictionary.
        # ... and similarly for state if we decide to make them attributes.
        self._function = fn
        self._pnames = list(sorted(pars.keys()))
        self._state = state
        self._plot = plot
        self._cached_theory = None

    def update(self):
        self._cached_theory = None

    def parameters(self):
        return dict((p, getattr(self, p)) for p in self._pnames)

    def numpoints(self):
        return np.prod(self.y.shape)

    def theory(self, x=None):
        # Use cache if x is None, otherwise compute theory with x.
        if x is None:
            if self._cached_theory is None:
                self._cached_theory = self._compute_theory(self.x)
            return self._cached_theory
        return self._compute_theory(x)

    def _compute_theory(self, x):
        kw = self._fetch_pars()
        return self._function(x, **kw)

    def _fetch_pars(self):
        kw = dict((p, getattr(self, p).value) for p in self._pnames)
        kw.update(self._state)
        return kw

    def simulate_data(self, noise=None):
        theory = self.theory()
        if noise is not None:
            if noise == 'data':
                pass
            elif noise < 0:
                self.dy = -0.01*noise*theory
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
        if len(self.x.shape) > 1:
            warnings.warn("Save not supported for nD x values")
            return

        theory = self.theory()
        if self._num_curves > 1:
            # Multivalued y, dy for single valued x.
            columns = [self.x]
            headers = ["x"]
            for k, (y, dy, fx) in enumerate(zip(self.y, self.dy, theory)):
                columns.extend((y, dy, fx))
                headers.extend(("y[%d]"%(k+1), "dy[%d]"%(k+1), "fx[%d]"%(k+1)))
        else:
            # Single-valued y, dy for single valued x.
            headers = ["x", "y", "dy", "fy"]
            columns = [self.x, self.y, self.dy, theory]
        data = np.vstack(columns)
        outfile = basename + '.dat'
        with open(outfile, "w") as fd:
            fd.write("# " + "\t ".join(headers) + "\n")
            np.savetxt(fd, data.T)

    def plot(self, view=None):
        if self._plot is not None:
            kw = self._fetch_pars()
            self._plot(self.x, self.y, self.dy, self.theory(), view=view, **kw)
            return

        import pylab
        from .plotutil import coordinated_colors

        x = self.x
        if self.plot_x is not None:
            theory_x, theory_y = self.plot_x, self.theory(self.plot_x)
        else:
            theory_x, theory_y = x, self.theory()
        resid = self.residuals()

        if self._num_curves > 1:
            y, dy, theory_y, resid = self.y.T, self.dy.T, theory_y.T, resid.T
        else:
            y, dy, theory_y, resid = (v[:, None]
                                      for v in (self.y, self.dy, theory_y, resid))

        colors = tuple(coordinated_colors() for _ in range(self._num_curves))
        labels = self.labels

        #print "kw_plot",kw
        if view == 'residual':
            _plot_resids(x, resid, colors, labels=labels, view=view)
        else:
            plot_ratio = 4
            h = pylab.subplot2grid((plot_ratio, 1), (0, 0), rowspan=plot_ratio-1)
            for tick_label in h.get_xticklabels():
                tick_label.set_visible(False)
            _plot_fits(data=(x, y, dy), theory=(theory_x, theory_y),
                       colors=colors, labels=labels, view=view)
            #pylab.gca().xaxis.set_visible(False)
            #pylab.gca().spines['bottom'].set_visible(False)
            #pylab.gca().set_xticks([])

            pylab.subplot2grid((plot_ratio, 1), (plot_ratio-1, 0), sharex=h)
            _plot_resids(x, resid, colors=colors, labels=labels, view=view)

def _plot_resids(x, resid, colors, labels, view):
    import pylab
    pylab.axhline(y=1, ls='dotted', color='k')
    pylab.axhline(y=0, ls='solid', color='k')
    pylab.axhline(y=-1, ls='dotted', color='k')
    for k, color in enumerate(colors):
        pylab.plot(x, resid[:, k], '.', color=color['base'])
    pylab.gca().locator_params(axis='y', tight=True, nbins=4)
    pylab.xlabel(labels[0])
    pylab.ylabel("(f(x)-y)/dy")
    if view == 'logx':
        pylab.xscale('log')
    elif view == 'loglog':
        pylab.xscale('log')

def _plot_fits(data, theory, colors, labels, view):
    import pylab
    x, y, dy = data
    theory_x, theory_y = theory
    for k, color in enumerate(colors):
        pylab.errorbar(x, y[:, k], yerr=dy[:, k], fmt='.',
                       color=color['base'], label='_')
        pylab.plot(theory_x, theory_y[:, k], '-',
                   color=color['dark'], label=labels[k+2])
    # Note: no xlabel since it is supplied by the residual plot below this plot
    pylab.ylabel(labels[1])
    if len(colors) > 1:
        pylab.legend()
    if view == 'log':
        pylab.xscale('linear')
        pylab.yscale('log')
    elif view == 'logx':
        pylab.xscale('log')
        pylab.yscale('linear')
    elif view == 'logy':
        pylab.xscale('linear')
        pylab.yscale('log')
    elif view == 'loglog':
        pylab.xscale('log')
        pylab.yscale('log')
    else: # view == 'linear'
        pylab.xscale('linear')
        pylab.yscale('linear')

def plot_resid(x, resid):
    """
    **DEPRECATED**
    """
    import pylab
    pylab.axhline(y=1, ls='dotted', color='k')
    pylab.axhline(y=0, ls='solid', color='k')
    pylab.axhline(y=-1, ls='dotted', color='k')
    pylab.plot(x, resid, '.')
    pylab.gca().locator_params(axis='y', tight=True, nbins=4)
    pylab.ylabel("Residuals")

def plot_err(x, y, dy, fy, view=None, **kw):
    """
    **DEPRECATED**: subclass Curve and override the plot function.

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
