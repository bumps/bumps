# Anticorrelation demo
# ====================
#
# Model with strong correlations between the fitted parameters.
#
# We use a*x = y + N(0,1) made complicated by defining a=p1+p2.
#
# The expected distribution for p1 and p2 will be uniform, with p2 = a-p1 in
# each sample.  Because this distribution is inherently unbounded, artificial
# bounds are required on a least one of the parameters for finite duration
# simulations.
#
# The expected distribution for p1+p2 can be determined from the linear model
# y = a*x.  This is reported along with the values estimated from MCMC.

from bumps.names import *

# Anticorrelated function

def fn(x, a, b): return (a+b)*x

# Fake data

sigma = 1
x = np.linspace(-1., 1, 40)
dy = sigma*np.ones_like(x)
y = fn(x,5,5) + np.random.randn(*x.shape)*dy

# Wrap it in a curve fitter

M = Curve(fn, x, y, dy, a=(-20,20), b=(-20,20))

# Alternative representation, fitting a and S=a+b, and setting b=S-a.
#
# ::
#
#     S = Parameter((-20,20), name="sum")
#     M.b = S-M.a

problem = FitProblem(M)
