# Boundary check
# ==============
#
# Check probability at boundaries.
#
# In this case we define the probability density function (PDF) directly
# in an n-dimensional uniform box.
#
# Ideally, the correlation plots and variable distributions will be uniform.

from bumps.names import *

# Adjust scale from 1e-150 to 1e+150 and you will see that DREAM is equally
# adept at filling the box.

scale = 1

# Uniform cost function.

def box(x):
    return 0 if np.all(np.abs(x)<=scale) else np.inf

def diamond(x):
    return 0 if np.sum(np.abs(x))<=scale else np.inf

# Wrap it in a PDF object which turns an arbitrary probability density into
# a fitting function.  Give it a valid initial value, and set the bounds to
# a unit cube with one corner at the origin.

M = PDF(lambda a,b: box([a,b]))
#M = PDF(lambda a,b: diamond([a,b]))
M.a.range(-2*scale,2*scale)
M.b.range(-2*scale,2*scale)

# Make the PDF a fit problem that bumps can process.

problem = FitProblem(M)