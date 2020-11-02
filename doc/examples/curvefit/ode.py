# Fitting an ODE
# ==============
#
# Bumps can fit black-box functions, such as odeint from scipy.
#
# The following example is adapted from:
#
#     https://people.duke.edu/~ccc14/sta-663/CalibratingODEs.html.
#
# | Instructor: Cliburn Chan cliburn.chan@duke..edu
# | Instructor: Janice McCarthy janice.mccarthy@duke.edu

from bumps.names import *
import numpy as np
from scipy.integrate import odeint

# Define the ODE

def g(t, x0, a, b):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    return odeint(dfdt, x0, t, args=(a, b)).flatten()

def dfdt(x, t, a, b):
    """Receptor synthesis-internalization model."""
    return a - b*x

# Simulate some data.
#
# Note that the function :func:`bumps.util.push_seed` is to set the random
# number generator to a known state so that this function will create the
# same data every time the simulation is run.  If not, then you wouldn't
# be able to resume a fit since each time you resumed you would be fitting
# different data.

def simulate():
    from bumps.util import push_seed

    # Fake some data
    a = 2.0
    b = 0.5
    x0 = 10.0
    t = np.linspace(0, 10, 10)
    dy = 0.2*np.ones_like(t)
    with push_seed(1):
        y = g(t, x0, a, b) + dy*np.random.normal(size=t.shape)
    #print(a, b, x0, t, dt, gt)
    return t, y, dy

t, y, dy = simulate()

# Define the fit problem.
#
# In this case :class:`bumps.curve.Curve` is initialized with *plot_x*
# as a vector of length 1000.  This is so that a smooth curve is drawn between
# the ten data points that were simulated in the fit.

M = Curve(g, t, y, dy, x0=1., a=1., b=1.,
          plot_x=np.linspace(t[0], t[-1], 1000))
M.x0.range(0, 100)
M.a.range(0, 10)
M.b.range(0, 10)

problem = FitProblem(M)
