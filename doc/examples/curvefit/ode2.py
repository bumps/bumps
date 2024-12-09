# Fitting a multi-valued function
# ===============================
#
# Like the ODE fit function, but this example fits a set of coupled ODEs.
# In this case, there are multiple values reported at each time step, two
# of which are measured and fitted.
#
# From SciPy cookbook coupled spring mass example:
#
#    https://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html
#

from bumps.names import *
from scipy.integrate import odeint

# Use ODEINT to solve the differential equations defined by the vector field


def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    m1, m2, k1, k2, L1, L2, b1, b2 = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1, (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1, y2, (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2]
    return f


# ODE solver parameters

abserr = 1.0e-8
relerr = 1.0e-6

# Curve function with all parameters exposed so that bumps knows their names.
# Only tracking x1, x2 with our measurements and not y1, y2, so returning
# components 0 and 2 of the *vectorfield* result.  The multi-valued y values
# are stacked into an array whose first axis matches t.  This is needed so that
# the plotter can sort out the different lines.


def f(t, x1, y1, x2, y2, m1, m2, k1, k2, L1, L2, b1, b2):
    # Pack up the parameters and initial conditions:
    p = [m1, m2, k1, k2, L1, L2, b1, b2]
    w0 = [x1, y1, x2, y2]

    # Call the ODE solver.
    wsol = odeint(vectorfield, w0, t, args=(p,), atol=abserr, rtol=relerr)
    return np.vstack((wsol[:, 0], wsol[:, 2]))


# Simulation parameter values

# Masses
m1 = 1.0
m2 = 1.5

# Spring constants
k1 = 8.0
k2 = 40.0

# Natural lengths
L1 = 0.5
L2 = 1.0

# Friction coefficients
b1 = 0.8
b2 = 0.5

# Initial conditions

# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
x1 = 0.5
y1 = 0.0
x2 = 2.25
y2 = 0.0

# Simulate data


def simulate():
    from bumps.util import push_seed

    # Create the time samples for the output of the ODE solver.
    # These are the times that the data is sampled, not the times at
    # which to evaluate the ode solver.
    t = np.linspace(0, 10, 100)

    # Pack up the parameters and initial conditions:
    p = [m1, m2, k1, k2, L1, L2, b1, b2]
    w0 = [x1, y1, x2, y2]
    ft = f(t, *(w0 + p))

    noise = 0.1 * np.ones_like(ft)
    with push_seed(1):  # Make sure that the simulated data is the same each run
        data = ft + noise * np.random.randn(*ft.shape)
    return t, data, noise


t, y, dy = simulate()

# Initial values for most parameters are known from system configuration.
# We are not including the spring constants or the friction coefficients
# since these will be fitted to the measured position over time.  *labels*
# allow you to set the labels for the x-axis and y-axis and the legend for
# the two data lines on the plot.

M = Curve(
    f,
    t,
    y,
    dy,
    m1=m1,
    m2=m2,
    L1=L1,
    L2=L2,
    x1=x1,
    y1=y1,
    x2=x2,
    y2=y2,
    labels=["time", "value", "x1", "x2"],
    plot_x=np.linspace(0, 10, 1000),
)

# Fitted parameters
#
# Only fitting spring constants and friction coefficients since these are
# not immediately measurable.  If we wanted to be fancy, we could set the
# prior on position and mass according to the uncertainty in our
# initial configuration and allow them to vary slightly.

# Masses: Allow mass estimate to be off by +/- 2% (1-sigma)  *untested*
# M.m1.dev(0.02*m1)
# M.m2.dev(0.02*m2)

# Spring constants
M.k1.range(0, 100)
M.k2.range(0, 100)

# Natural lengths
# M.L1.range(0, 10)
# M.L2.range(0, 10)

# Friction coefficients
M.b1.range(0, 1)
M.b2.range(0, 1)

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
# M.x1.range(0, 10)
# M.x2.range(0, 10)
# M.y1.range(0, 10)
# M.y2.range(0, 10)

problem = FitProblem(M)
