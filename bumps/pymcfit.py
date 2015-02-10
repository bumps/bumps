r"""
Bumps wrapper for PyMC models.
"""
from __future__ import print_function

__all__ = ["PyMCProblem"]

from numpy import inf, array, asarray

def guess_bounds(p):
    # Delay referencing
    import pymc.distributions

    UNBOUNDED = lambda p: (-inf, inf)
    PYMC_BOUNDS = {
        pymc.distributions.DiscreteUniform: lambda p: (p['lower'],p['upper']),
        pymc.distributions.Uniform: lambda p: (p['lower'],p['upper']),
        pymc.distributions.Exponential: lambda p: (0,inf),
    }
    # pyMC doesn't provide info about bounds on the distributions
    # so we need a big table.
    return PYMC_BOUNDS.get(p.__class__, UNBOUNDED)(p.parents)

class PyMCProblem(object):
    def __init__(self, pars, conds):
        self.pars = pars
        self.conds = conds
        #self.dof = sum(len(c.value) for c in self.conds)
    def model_reset(self): pass
    def chisq(self):
        return self.nllf() # /self.dof
    def chisq_str(self):
        return "%g"%self.chisq()
    __call__ = chisq
    def nllf(self, pvec=None):
        if pvec is not None: self.setp(pvec)
        return -sum(p.logp for p in self.pars)-sum(c.logp for c in self.conds)
    def setp(self, values):
        for p,v in zip(self.pars,values):
            p.value = v
    def getp(self):
        return array([p.value for p in self.pars])
    def show(self):
        # maybe print graph of model
        print("[chisq=%g, nllf=%g]" % (self.chisq(), self.nllf()))
        print(self.summarize())
    def summarize(self):
        for p in self.pars:
            print(p.__name__, p.value)
    def labels(self):
        return [p.__name__ for p in self.pars]
    def randomize(self, N=None):
        if N is None:
            for p in self.pars:
                p.value = p.random()
        else:
            return asarray([[p.random() for _ in range(N)]
                                  for p in self.pars])
    def bounds(self):
        return asarray([guess_bounds(p) for p in self.pars], 'd').T
                 
    def plot(self, p=None, fignum=None, figfile=None):
        pass

    def __deepcopy__(self, memo): return self
