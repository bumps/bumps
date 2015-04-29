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

scale = 10

# Uniform cost function.

def uniform(x):
    return 0 if np.all(np.abs(x)<=scale) else np.inf

# Wrap it in a PDF object which turns an arbitrary probability density into
# a fitting function.  Give it an initial value away from the cross.

M = SimplePDF(uniform)

# Set the range of values to include the cross.  You can skip the center of
# the cross by setting b.range to (1,3), and for reasonable values of sigma
# both arms will still be covered.  Extend the range too far (e.g.,
# a.range(-3000,3000), b.range(-1000,3000)), and like a value of sigma
# that is too small, only one arm of the cross will be filled.

M.a.range(-3*scale,3*scale)
M.b.range(-1*scale,3*scale)

# Make the PDF a fit problem that bumps can process.

problem = FitProblem(M)