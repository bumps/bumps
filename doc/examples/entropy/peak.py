# Bayesian Experimental Design
# ============================
#
# Perform a tradeoff comparison between point density and counting time when
# measuring a peak in a poisson process.
#
# Usage:
#
# .. parsed-literal::
#
#    bumps peak.py N --entropy --store=/tmp/T1 --fit=dream
#
# The parameter N is the number of data points to use within the range.
#

from bumps.names import *
from numpy import exp, sqrt, pi, inf

# Define the peak shape as a gaussian plus background
def peak(x, scale, center, width, background):
    return scale*exp(-0.5*(x-center)**2/width**2)/sqrt(2*pi*width**2) + background

# Get the number of points from the command line
if len(sys.argv) == 2:
    npoints = int(sys.argv[1])
else:
    raise ValueError("Expected number of points n in the fit")

# set a constant number of counts, equally divided between points
x = np.linspace(5, 20, npoints)
scale = 10000/npoints

# Build the model, along with the valid fitting range. there is no data yet,
# so y is None
M = PoissonCurve(peak, x, y=None, scale=scale, center=15, width=1.5, background=1)
M.scale.range(0, inf)
dx = max(x)-min(x)
M.center.range(min(x) - 0.2*dx, max(x) + 0.2*dx)
M.width.range(0, 0.7*dx)
M.background.range(0, inf)

# Make a fake dataset from the give x spacing
M.simulate_data()

problem = FitProblem(M)


# Running this problem for a few values of the number of points is showing
# that adding points and reducing counting time per point is better able
# to recover the peak parameters.
