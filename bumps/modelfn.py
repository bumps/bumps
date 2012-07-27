"""
Build a BUMPS model from a function.
"""
import inspect

import numpy

from .parameter import Parameter

class ModelFunction(object):
    """
    Build a model from a function.

    This model can be fitted with any of the bumps optimizers.

    The function *fn* should return the negative log likelihood of seeing
    its input parameters.

    The fittable parameters are derived from the function definition, with
    the *name* prepended to each parameter.

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
        pnames,vararg,varkw,pvalues = inspect.getargspec(fn)
        if vararg or varkw:
            raise TypeError("Function cannot have *args or **kwargs in declaration")
        # Parameters default to zero
        init = dict( (p,0) for p in pnames)
        # If the function provides default values, use those
        if pvalues:
            init.update(zip(pnames[-len(pvalues):],pvalues))
        # Regardless, use any values specified in the constructor, but first
        # check that they exist as function parameters.
        invalid = set(kw.keys()) - set(pnames)
        if invalid:
            raise TypeError("Invalid initializers: %s"%", ".join(sorted(invalid)))
        init.update(kw)

        # Build parameters out of ranges and initial values
        pars = dict( (p,Parameter.default(init[p],name=name+p)) for p in pnames)

        # Make parameters accessible as model attributes
        for k,v in pars.items():
            if hasattr(self,k):
                raise TypeError("Parameter cannot be named %s"%k)
            setattr(self, k, v)

        # Remember the function, parameters, and number of parameters
        self._function = fn
        self._parameters = pars
        self._plot = plot

    def parameters(self):
        """
        Fittable parameters.
        """
        return self._parameters

    def __call__(self):
        kw = dict( (k,v.value) for k,v in self._parameters.items() )
        #print kw
        return self._function(**kw)

    def nllf(self):
        """
        Negative log probability of seeing model value.
        """
        return self()

    def update(self):
        """
        Parameters changed; clear cached values.
        """
        pass

    def plot(self, view=None):
        if self._plot:
            kw = dict( (k,v.value) for k,v in self._parameters.items() )
            self._plot(view=view, **kw)

    def numpoints(self):
        return len(self._parameters)+1

    def residuals(self):
        """
        Function residual.
        """
        return numpy.array([self()])
