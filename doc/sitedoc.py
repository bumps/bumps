import os

import numpy as np

import bumps.fitters as fit
from bumps.cli import load_model

SEED = 1

def example_dir():
    """
    Return the directory containing the rst file source for the current plot.
    """
    # Search through the call stack for the rstdir variable.
    #
    # This is an ugly hack which relies on internal structures of the python
    # interpreter and particular variables used in internal functions from
    # the matplotlib plot directive, and so is likely to break.
    #
    # If this code breaks, you could probably get away with searching up
    # the stack for the variable 'state_machine', which is a sphinx defined
    # variable, and use:
    #
    #  rstdir, _ = os.path.split(state_machine.document.attributes['source'])
    #
    # Even better would be to modify the plot directive to make rstdir
    # available to the inline plot directive, e.g., by adding it to the
    # locals context.  It is already implicitly available to the plot
    # file context because there is an explicit chdir to the directory
    # containing the plot.
    import inspect
    frame = inspect.currentframe()
    RSTDIR='rst_dir'
    while frame and RSTDIR not in frame.f_locals:
        frame = frame.f_back
        #print "checking",frame.f_code.co_name
    if not frame:
        raise RuntimeError('plot directive changed: %r no longer defined'%RSTDIR)
    return frame.f_locals[RSTDIR] if frame else ""

def plot_model(filename):
    from matplotlib import pyplot as plt
    #import sys; print >>sys.stderr, "in plot with",filename, example_dir()
    np.random.seed(SEED)
    p = load_model(os.path.join(example_dir(), filename))
    p.plot()
    plt.show()

def fit_model(filename):
    from matplotlib import pyplot as plt
    #import sys; print >>sys.stderr, "in plot with",filename, example_dir()
    np.random.seed(SEED)
    p =load_model(os.path.join(example_dir(),filename))
    #x.fx = fit.RLFit(p).solve(steps=1000, burn=99)
    #x,fx = fit.DEFit(p).solve(steps=200, pop=10)
    #x,fx = fit.PTFit(p).solve(steps=100,burn=400)
    #x.fx = fit.BFGSFit(p).solve(steps=200)
    x,fx = fit.SimplexFit(p).solve(steps=1000)
    chisq = p(x)
    print("chisq=%g"%chisq)
    if chisq>2:
        raise RuntimeError("Fit did not converge")
    p.plot()
    plt.show()
