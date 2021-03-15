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

# Adjust domain from 1e-150 to 1e+150 and you will see that DREAM is equally
# adept at filling the box.

domain = 1

# Uniform cost function.

def box(x):
    """
    A flat top mesa with a square border in [-1, 1].
    """
    return 0 if np.all(np.abs(x) <= domain) else np.inf

def ramp(x):
    """
    A ramp in the first parameter, all other parameters uniform over [-1, 1].
    """
    p = abs(x[0])/domain
    return -log(p) if np.all(np.abs(x) <= domain) else np.inf

def cone(x):
    """
    An inverted cone with peak probability at the rim of radius 1.
    """
    #r = np.sqrt(sum(xk**2 for xk in x[:2]))
    r = np.sqrt(sum(xk**2 for xk in x))
    return -log(r) if r <= domain else np.inf

def diamond(x):
    """
    A flat top mesa with a diamond border.
    """
    return 0 if np.sum(np.abs(x)) <= domain else np.inf

def sawtooth(x):
    """
    A symmetric sawtooth of frequency 1, phase 0, so f(0)=1, f(1/2)=0.
    """
    p = [2*abs(xk/domain%1 - 1/2) for xk in x]
    return -sum(np.log(pk) for pk in p)


def triangle_constraints():
    """
    The triangle below y=x.
    """
    a, b = M.a.value, M.b.value
    return 0 if a < b else 1e6 + (b-a)**2

def box_constraints():
    """
    A square over [-1/2, 1/2].
    """
    a, b = M.a.value, M.b.value
    return 0 if abs(a) <= domain/2 and abs(b) <= domain/2 else np.inf

def circle_constraints():
    """
    A circle of radius 1.
    """
    a, b = M.a.value, M.b.value
    r = np.sqrt(a**2 + b**2)
    return 0 if r <= domain*2/3 else np.inf

def ring_constraints():
    """
    A ring of inner radius 2/3.
    """
    a, b = M.a.value, M.b.value
    r = np.sqrt(a**2 + b**2)
    return 0 if domain*2/3 <= r <= domain else 1e6 + (r/domain - 1)**2

def sawtooth_constraints():
    """
    Sets one peak at the edge of the domain and another in the middle. Use
    this to investigate whether rejection outside the domain leads to
    distortion of the density at the boundary of the domain. You will need
    to modify the parameter view to show 100% of the range rather than
    the 95% CI cutoff in current plots (code in bumps.dream.varplot.plot_var).
    """
    a, b = M.a.value, M.b.value
    return (0 if all(0.0 < xk/domain < 1.5 for xk in (a, b))
            else 1e6 + sum((xk/domain)**2 for xk in (a, b)))

# Wrap it in a PDF object which turns an arbitrary probability density into
# a fitting function.  Give it a valid initial value, and set the bounds to
# a unit cube with one corner at the origin.

#M = PDF(lambda a, b: box([a, b]))
M = PDF(lambda a, b: diamond([a, b]))
#M = PDF(lambda a, b: ramp([a, b]))
#M = PDF(lambda a, b: cone([a, b]))
#M = PDF(lambda a, b: sawtooth([a, b]))

constraints = None
#constraints = triangle_constraints
constraints = box_constraints
#constraints = circle_constraints
#constraints = ring_constraints
#constraints = sawtooth_constraints

M.a.range(-2*domain, 2*domain)
M.b.range(-2*domain, 2*domain)

# Make the PDF a fit problem that bumps can process.
problem = FitProblem(M, constraints=constraints)
