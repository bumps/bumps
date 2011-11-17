import os
import numpy
import pylab
from refl1d.fitters import BFGSFit, DEFit, RLFit, PTFit
from refl1d.cli import load_problem

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
    while frame and 'rstdir' not in frame.f_locals:
        frame = frame.f_back
        #print "checking",frame.f_code.co_name
    if not frame:
        raise RuntimeException('plot directive changed: rstdir no longer defined')
    return frame.f_locals['rstdir'] if frame else ""

def plot_model(filename):
    numpy.random.seed(SEED)
    p = load_problem([os.path.join(example_dir(), filename)])
    p.plot()
    pylab.show()

def fit_model(filename):
    numpy.random.seed(SEED)
    p =load_problem([os.path.join(example_dir(),filename)])
    #x.fx = RLFit(p).solve(steps=1000, burn=99)
    x,fx = DEFit(p).solve(steps=200, pop=10)
    #x,fx = PTFit(p).solve(steps=100,burn=400)
    #x.fx = BFGSFit(p).solve(steps=200)
    chisq = p(x)
    print "chisq=",chisq
    if chisq>2:
        raise RuntimeError("Fit did not converge")
    p.plot()
    pylab.show()
