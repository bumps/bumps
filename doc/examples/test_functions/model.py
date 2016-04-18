"""

Surjanovic, S. & Bingham, D. (2013).
Virtual Library of Simulation Experiments: Test Functions and Datasets.
Retrieved April 18, 2016, from http://www.sfu.ca/~ssurjano.
"""
from __future__ import print_function

import numpy as np
from numpy import sin, cos, linspace, meshgrid, e, pi, sqrt, exp, inf
from numpy.random import randn
from bumps.names import *
from functools import reduce

class ModelFunction(object):
    _registered = {}  # class mutable containing list of registered functions
    def __init__(self, f, xmin, fmin, bounds, dim):
        self.name = f.__name__
        self.xmin = xmin
        self.fmin = fmin
        self.bounds = bounds
        self.dim = dim
        self.f = f

        # register the function in the list of available functions
        ModelFunction._registered[self.name] = self

    def __call__(self, x):
        return self.f(x)

    def fk(self, k, **kw):
        args = ",".join("z%d"%j for j in range(2,k))
        if k == 1:
            def calculator(x): return self.f((x,), **kw)
        elif k == 2:
            def calculator(x, y): return self.f((x, y), **kw)
        else:
            calc= lambda x: self.f(x, **kw)
            eval("def calculator(x,y,%s): calc((x,y,%s))"%(args,args))
        return calculator

    @staticmethod
    def lookup(name):
        return ModelFunction._registered.get(name, None)

    @staticmethod
    def available():
        return list(sorted(ModelFunction._registered.keys()))

def columnize(L, indent="", width=79):
    # type: (List[str], str, int) -> str
    """
    Format a list of strings into columns.

    Returns a string with carriage returns ready for printing.
    """
    column_width = max(len(w) for w in L) + 1
    num_columns = (width - len(indent)) // column_width
    num_rows = len(L) // num_columns
    L = L + [""] * (num_rows*num_columns - len(L))
    columns = [L[k*num_rows:(k+1)*num_rows] for k in range(num_columns)]
    lines = [" ".join("%-*s"%(column_width, entry) for entry in row)
             for row in zip(*columns)]
    output = indent + ("\n"+indent).join(lines)
    return output

def model_function(xmin=None, fmin=None, bounds=(-inf, inf), dim=None):
    return lambda f: ModelFunction(f, xmin, fmin, bounds, dim)

def prod(L):
    return reduce(lambda x,y: x*y, L, 1)


def plot2d(fn, args=('x','y'), range=(-10,10)):
    """
    Return a mesh plotter for the given function.

    *args* are the function arguments that are to be meshed (usually the
    *x* and *y* arguments to the function).  *range* is the bounding box
    for the 2D mesh.

    All arguments except the meshed arguments are held fixed.
    """
    def plotter(view=None, **kw):
        import pylab
        x, y = kw[args[0]], kw[args[1]]
        r = linspace(range[0], range[1], 200)
        X, Y = meshgrid(x+r, y+r)
        kw['x'], kw['y'] = X, Y
        pylab.pcolormesh(x+r, y+r, fn(**kw))
        pylab.plot(x, y, 'o', hold=True, markersize=6,
                   markerfacecolor='red', markeredgecolor='black',
                   markeredgewidth=1, alpha=0.7)
    return plotter


# ================ Model functions ====================

@model_function(fmin=0.0, xmin=0.0)
def sphere(x):
    """
    Unimodal smooth well.
    """
    return sum(xi**2 for xi in x)


@model_function(fmin=0, xmin=3.)
def sin_plus_quadratic(x, c=3., d=2., m=5., h=2.):
    """
    Sin + quadratic.  Multimodal with global minimum.

    *c* is the center point where the function is minimized.

    *d* is the distance between modes, one per dimension.

    *h* is the sine wave amplitude, on per dimension, which controls
    the height of the barrier between modes.

    *m* is the curvature of the quadratic, one per dimension.
    """
    n = len(x)
    if np.isscalar(c): c = [c]*n
    if np.isscalar(d): d = [d]*n
    if np.isscalar(h): h = [h]*n
    if np.isscalar(m): m = [m]*n
    return (sum(hi*(sin((2*pi/di)*xi-ci)+1.) for xi,ci,di,hi in zip(x, c, d, h))
            + sum(((xi-ci)/float(mi))**2 for xi,ci,mi in zip(x, c, m)))

@model_function(fmin=0.0, xmin=0.0)
def stepped_well(x):
    return sum(np.floor(abs(xi)) for xi in x)

@model_function(xmin=0.0, fmin=0.0, bounds=(-32.768, 32.768))
def ackley(x, a=20., b=0.2, c=2*pi):
    """
    Multimodal with deep global minimum.
    """
    n = len(x)
    return (-a*exp(-b*sqrt(sum(xi**2 for xi in x)/n))
            - exp(sum(cos(c*xi) for xi in x)/n) + a + e)

@model_function(dim=2, xmin=(3, 0.5), fmin=0.0, bounds=(-4.5, 4.5))
def beale(xy):
    x, y = xy
    return (1.5 - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

_TRAY=1.34941
@model_function(dim=2, fmin=-2.06261, bounds=(-10,10),
                xmin=((_TRAY, _TRAY), (_TRAY, -_TRAY),
                      (-_TRAY, _TRAY), (-_TRAY, -_TRAY)))
def cross_in_tray(xy):
    x, y = xy
    return -0.0001*(abs(sin(x)*sin(y)*exp(abs(100-sqrt(x**2+y**2)/pi)))+1)**0.1

@model_function(bounds=(-600, 600), xmin=0.0, fmin=0.0)
def griewank(x):
    return (1 + sum(xi**2 for xi in x)**2/4000
            - prod(cos(xi/sqrt(i+1)) for i,xi in enumerate(x)))


@model_function(bounds=(-5.12,5.12), xmin=0.0, fmin=0.0)
def rastrigin(x, A=10.):
    """
    Multimodal with global minimum near local minima.
    """
    n = len(x)
    return A*n + sum(xi**2 - A*cos(2*pi*xi) for xi in x)


# could also use bounds=(-2.048, 2.048)
@model_function(bounds=(-5,10), xmin=1., fmin=0.)
def rosenbrock(x):
    """
    Unimodal with narrow parabolic valley.
    """
    return sum(100*(xn-xp**2)**2 + (xp-1)**2 for xp, xn in zip(x[:-1], x[1:]))


# ========================== wrapper ==================

USAGE = """\
Given the name of the test function followed by dimension.  Dimension
defaults to 2.  Available models are:

""" + columnize(ModelFunction.available(), indent="    ")

model = ModelFunction.lookup(sys.argv[1]) if 1 < len(sys.argv) < 4 else None
dim = int(sys.argv[2]) if len(sys.argv) > 2 else 2

if model is None:
    print(USAGE, file=sys.stderr)
    sys.exit(1)

nllf = model.fk(dim)

plot=plot2d(nllf, ('x', 'y'), range=(-10,10))

M = PDF(nllf, plot=plot)

for p in M.parameters().values():
    # TODO: really should pull value and range out of the bounds for the
    # function, if any are provided.
    p.value = 10*randn()
    p.range(-200,200)

problem = FitProblem(M)
