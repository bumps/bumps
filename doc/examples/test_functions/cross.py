# Cross-shaped anti-correlation
# =============================
#
# Example model with strong correlations between the fitted parameters.
#
# In this case we define the probability density function (PDF) directly
# as an 'X' pattern, with width sigma.
#
# Ideally, the a-b correlation plot will show the 'X' completely filled
# within the bounds.

from bumps.names import *

# Adjust scale from 1e-150 to 1e+150 and you will see that DREAM is equally
# adept at filling the cross. However, if sigma gets too small relative to
# scale the fit will get stuck on one of the arms, and if sigma gets too
# large, then the whole space will be filled and the x will not form.

scale = 10
sigma = 0.1*scale
#sigma = 0.001*scale  # Too small
#sigma = 10*scale   # Too large

# Simple gaussian cost function based on the distance to the closest ridge
# *x=y* or *x=-y*.

def fn(a, b):
    return 0.5*min(abs(a+b),abs(a-b))**2/sigma**2 + 1

# Wrap it in a PDF object which turns an arbitrary probability density into
# a fitting function.  Give it an initial value away from the cross.

M = PDF(fn, a=3*scale, b=1.2*scale)

# Set the range of values to include the cross.  You can skip the center of
# the cross by setting b.range to (1,3), and for reasonable values of sigma
# both arms will still be covered.  Extend the range too far (e.g.,
# a.range(-3000,3000), b.range(-1000,3000)), and like a value of sigma
# that is too small, only one arm of the cross will be filled.

M.a.range(-3*scale,3*scale)
M.b.range(-1*scale,3*scale)

# Make the PDF a fit problem that bumps can process.

problem = FitProblem(M)
