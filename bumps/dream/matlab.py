"""
Interface compatible with matlab dream.

Usage
-----

This interface is identical to dream in matlab::

    [Sequences, Reduced_Seq, X, output, hist_logp] = \
      dream(MCMCPar, ParRange, Measurement, ModelName, Extra, option)

With care, you will be able to use the same model definition file
on both platforms, with only minor edits required to switch between them.
Clearly, you won't be able to use if statements and loops since the
syntax for python and matlab are incompatible.  Similarly, you will
have to be careful about indentation and line breaks.  And create your
model without comments.

Python requires that structures be defined before you assign values to
their fields, so you will also need the following lines.  Note however,
that they are safe to use in matlab as well::

    MCMCPar = struct()
    Extra = struct()
    Measurement = struct()
    ParRange = struct()


Another challenge is *ModelName*.  In matlab this is the name of the
m-file that contains the model definition.  Following that convention,
we will try to using "from <ModelName> import <ModelName>", and if this
fails, assume that *ModelName* is actually the function itself.  For
this to work you will need to translate the function in ModelName.m
to the equivalent function in ModelName.py.

*option* is the same option number as before

IPython usage
-------------

Within ipython you can interact with your models something like
you do in matlab.  For example::

    $ ipython -pylab
    In [1]: from dream.matlab import *
    In [2]: from dream import *
    In [3]: %run example.m


You can now use various dream visualization tools or use the matlab-like
plotting functions from pylab::

    In [4]: out.state.save('modeloutput')
    In [5]: out.state.plot_state()

Command line usage
------------------

You can also run a suitable m-file example from the command line.  This will
place you at an ipython command line with all the variables from your
m-file available.  For example::

    python -m dream.matlab example.m
    In [1]: out.state.save('modeloutput')
    In [2]: out.state.plot_state()

Script usage
------------

You can create a driver script which calls the m-file example and
uses pylab commands to plot the results.  For example::

    -- example.py --
    #!/usr/bin/env python
    from pylab import *
    from dream.matlab import *
    execfile('example.m')

    from dream import *
    out.state.save('modeloutput')
    plot_state(out.state)

This can be run from the command prompt::

    $ python example.py

"""
from __future__ import print_function

__all__ = ['struct', 'dream', 'setup', 'convert_output']

import numpy as np

from .core import Dream
from .model import Density, LogDensity, Simulation
from .initpop import cov_init, lhs_init
from .crossover import Crossover, AdaptiveCrossover


class struct(object):
    """
    Matlab compatible structure creation.
    """
    def __init__(self, *pairs, **kw):
        for k, v in zip(pairs[::2], pairs[1::2]):
            setattr(self, k, v)
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, k):
        return None


def dream(MCMCPar, ParRange, Measurement, ModelName, Extra, option):
    """
    Emulate the matlab dream call.
    """
    dreamer = setup(MCMCPar, ParRange, Measurement, ModelName, Extra, option)
    dreamer.sample()
    return convert_state(dreamer.state)


def setup(MCMCPar, ParRange, Measurement, ModelName, Extra, option):
    """
    Convert matlab dream models to a python Dream object.
    """
    dreamer = Dream()

    # Problem specification
    bounds = ParRange.minn, ParRange.maxn
    dreamer.bounds_style = Extra.BoundHandling
    if ModelName == 'Banshp':
        # specific properties of the banana function
        # Extra.imat is computed from cmat
        # MCMCPar.n is implicit in Extra.cmat
        f = Banana(mu=Extra.mu.flatten(), bpar=Extra.bpar, cmat=Extra.cmat)
        option = 4
    else:
        try:
            f = None  # keep lint happy
            # Try matlab style of having the function in the same named file.
            exec("from "+ModelName+" import "+ModelName+" as f")
        except ImportError:
            # The import failed; hope the caller supplied a function instead.
            f = ModelName

    if option == 1:
        model = Density(f, bounds=bounds)
    elif option == 4:
        model = LogDensity(f, bounds=bounds)
    elif option in [2, 3, 5]:
        # Measurement.N is implicit in Measurement.MeasData
        model = Simulation(f, data=Measurement.MeasData, bounds=bounds,
                           sigma=Measurement.Sigma, gamma=MCMCPar.Gamma)
    else:
        raise ValueError("option should be in 1 to 5")
    dreamer.model = model

    # Sampling parameters
    if Extra.save_in_memory == 'Yes':
        thinning = 1
    elif Extra.reduced_sample_collection == 'Yes':
        thinning = Extra.T
    else:
        thinning = 1
    dreamer.thinning = thinning
    dreamer.draws = MCMCPar.ndraw

    # Outlier detection
    T = MCMCPar.outlierTest
    if T.endswith('_test'):
        T = T[:-5]
    dreamer.outlier_test = T

    # DE parameters
    dreamer.DE_steps = MCMCPar.steps
    dreamer.DE_pairs = MCMCPar.DEpairs
    dreamer.DE_eps = MCMCPar.eps

    # Initial population
    if Extra.InitPopulation == 'COV_BASED':
        pop = cov_init(N=MCMCPar.seq, x=Extra.muX.flatten(), cov=Extra.qcov)
    elif Extra.InitPopulation == 'LHS_BASED':
        pop = lhs_init(N=MCMCPar.seq, bounds=(ParRange.minn, ParRange.maxn))
    else:
        raise ValueError("Extra.InitPopulation must be COV_BASED or LHS_BASED")
    dreamer.population = pop

    # Crossover parameters
    if Extra.pCR == 'Update':
        CR = AdaptiveCrossover(MCMCPar.nCR)
    else:
        CR = Crossover(1./MCMCPar.nCR)
    dreamer.CR = CR

    # Delayed rejection parameters
    dreamer.use_delayed_rejection = (Extra.DR == 'Yes')
    dreamer.DR_scale = Extra.DRscale

    return dreamer


def convert_state(state):
    """
    Convert a completed dreamer run into a form compatible with the
    matlab dream interface::

        Sequences, Reduced_Seq, X, out, hist_logp

    The original state is stored in out.state
    """

    _, points, logp = state.sample()
    logp = logp[:, :, None]
    Sequences = np.concatenate((points, np.exp(logp), logp), axis=2)
    X = Sequences[-1, :, :]

    draws, logp = state.logp()
    hist_logp = np.concatenate((draws[:, None], logp), axis=1)

    out = struct()
    draws, R = state.R_stat()
    out.R_stat = np.concatenate((draws[:, None], R), axis=1)
    draws, AR = state.acceptance_rate()
    out.AR = np.concatenate((draws[:, None], AR[:, None]), axis=1)
    draws, w = state.CR_weight()
    out.CR = np.concatenate((draws[:, None], w), axis=1)
    out.outlier = state.outliers()[:, :2]

    # save the dreamer state data structure  as well
    out.state = state

    return Sequences, Sequences, X, out, hist_logp


def run_script(filename):
    exec(compile(open(filename).read(), filename, 'exec'))


class Banana(object):
    """
    Banana shaped function.

    Note that this is not one of the N dimensional Rosenbrock variants
    documented on wikipedia as it only operates "banana-like" in
    the x0-x1 plane.
    """
    def __init__(self, mu, bpar, cmat):
        self.mu, self.bpar, self.cmat = mu, bpar, cmat
        self.imat = np.linalg.inv(cmat)

    def __call__(self, x):
        x = x+0 # make a copy
        x[1] += self.bpar*(x[0]**2 - 100)
        ret = -0.5*np.dot(np.dot(x[None, :], self.imat), x[:, None])
        return ret[0, 0]


def main():
    import sys
    if len(sys.argv) == 2:
        import pylab
        run_script(sys.argv[1])
        user_ns = pylab.__dict__.copy().update(locals())
        import IPython
        IPython.Shell.IPShell(user_ns=user_ns).mainloop()
    else:
        print("usage: python -m dream.matlab model.m")

if __name__ == "__main__":
    main()
