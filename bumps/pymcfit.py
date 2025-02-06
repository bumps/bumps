r"""
Bumps wrapper for PyMC models.
"""

__all__ = ["PyMCProblem"]

import numpy as np
from numpy import inf, array, asarray

# pyMC model attributes:
# - deterministics                     [Intermediate variables]
# - stochastics (with observed=False)  [Fitted parameters]
# - data (stochastic variables with observed=True)
# - variables                          [stochastics+
# - potentials
# - containers
# - nodes
# - all_objects
# - status: Not useful for the Model base class, but may be used by subclasses.


class PyMCProblem(object):
    def __init__(self, input):
        from pymc.Model import Model
        from pymc.Node import ZeroProbability

        self.model = Model(input)
        # sort parameters by name
        ordered_pairs = sorted((s.__name__, s) for s in self.model.stochastics)

        # Record parameter, shape, size, offset as a list
        pars = [v for k, v in ordered_pairs]
        shape = [v.shape for v in pars]
        size = [(np.prod(s) if s != () else 1) for s in shape]
        offset = np.cumsum([0] + size[:-1])
        self.pars = list(zip(pars, shape, size, offset))

        # List of cost functions contains both parameter and data, but not
        # intermediate deterministic points
        self.costs = self.model.variables - self.model.deterministics

        # Degrees of freedom is #data - #pars
        points = sum((np.prod(p.shape) if p.shape != () else 1) for p in self.costs)
        self.dof = points - 2 * offset[-1]

        self.ZeroProbability = ZeroProbability

    def model_reset(self):
        pass

    def chisq(self):
        return self.nllf()  # /self.dof

    def chisq_str(self):
        return "%g" % self.chisq()

    __call__ = chisq

    def nllf(self, pvec=None):
        if pvec is not None:
            self.setp(pvec)
        try:
            return -sum(c.logp for c in self.costs)
        except self.ZeroProbability:
            return inf

    def setp(self, values):
        for par, shape, size, offset in self.pars:
            if shape == ():
                par.value = values[offset]
                offset += 1
            else:
                par.value = array(values[offset : offset + size]).reshape(shape)
                offset += size

    def getp(self):
        return np.hstack(
            [(par.value.flatten() if shape != () else par.value) for par, shape, size, offset in self.pars]
        )

    def show(self):
        # maybe print graph of model
        print("[chisq=%g, nllf=%g]" % (self.chisq(), self.nllf()))
        print(self.summarize())

    def summarize(self):
        return "\n".join("%s=%s" % (par.__name__, par.value) for par, _, _, _ in self.pars)

    def labels(self):
        ret = []
        for par, _, _, _ in self.pars:
            ret.extend(_par_labels(par))
        return ret

    def randomize(self, N=None):
        if N is None:
            self.model.draw_from_prior()
        else:
            data = []
            for _ in range(N):
                self.model.draw_from_prior()
                data.append(self.getp())
            return asarray(data)

    def bounds(self):
        return np.vstack([_par_bounds(par) for par, _, _, _ in self.pars]).T

    def plot(self, p=None, fignum=None, figfile=None):
        pass

    def __deepcopy__(self, memo):
        return self


def _par_bounds(par):
    # Delay referencing
    import pymc.distributions

    UNBOUNDED = lambda p: (-inf, inf)
    PYMC_BOUNDS = {
        pymc.distributions.DiscreteUniform: lambda p: (p["lower"] - 0.5, p["upper"] + 0.5),
        pymc.distributions.Uniform: lambda p: (p["lower"], p["upper"]),
        pymc.distributions.Exponential: lambda p: (0, inf),
    }
    # pyMC doesn't provide info about bounds on the distributions
    # so we need a big table.
    bounds = PYMC_BOUNDS.get(par.__class__, UNBOUNDED)(par.parents)

    ret = np.tile(bounds, par.shape).flatten().reshape(-1, 2)
    return ret


def _par_labels(par):
    name = par.__name__
    dims = len(par.shape)
    if dims == 0:
        return [name]
    elif dims == 1:
        return ["%s[%d]" % (name, i) for i in range(par.shape[0])]
    elif dims == 2:
        return ["%s[%d,%d]" % (name, i, j) for i in range(par.shape[0]) for j in range(par.shape[1])]
    elif dims == 3:
        return [
            "%s[%d,%d,%d]" % (name, i, j, k)
            for i in range(par.shape[0])
            for j in range(par.shape[1])
            for k in range(par.shape[2])
        ]
    else:
        raise ValueError("limited to 3 dims for now")
