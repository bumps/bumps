import pylab
from numpy import sin, cos, linspace, meshgrid, e, pi, sqrt, array, exp
from bumps.names import *

def prod(L):
    return reduce(lambda x,y: x*y, L, 1) 
def plot2d(fn,args,range=(-10,10)):
    def plotter(view=None, **kw):
        x,y = kw[args[0]],kw[args[1]]
        r = linspace(range[0],range[1],200)
        X,Y = meshgrid(x+r,y+r)
        kw['x'],kw['y'] = X,Y
        pylab.pcolormesh(x+r,y+r,nllf(**kw))
        pylab.plot(x,y,'o',hold=True, markersize=6,markerfacecolor='red',markeredgecolor='black',markeredgewidth=1, alpha=0.7)
    return plotter 
def f2(fn):
    def cost(x,y): return fn((x,y))
    return cost
def fk(fn,k):
    args = ",".join("z%d"%j for j in range(k-2))
    eval("def cost(x,y,%s): return fn((x,y,%s))"%(args,args))
    return cost

def sin_plus_quadratic(x=0,y=0): 
    fx,fy = 2,3      # x,y frequency and between bowl barer height
    barrier = 2      # barrier height
    cx,cy = 3,1      # x,y center
    mx,my = 2,5      # x,y curvature
    width = 6        # size of the acceptance region
    return (barrier*(sin(fx*x) + sin(fy*y)+2) + ((x-cx)/mx)**2 + ((y-cy)/my)**2)/width

def ackley(x):
    n = len(x)
    return -20*exp(-0.2*sqrt(sum(xi**2 for xi in x)/n))-exp(sum(cos(2*pi*xi) for xi in x)/n) + 20 + e

def griewank(x):
    return 1 + sum(xi**2 for xi in x)**2/4000 - prod(cos(xi/sqrt(i+1)) for i,xi in enumerate(x))

def rastrigin(x):
    A = 10
    n = len(x)
    return A*n + sum(xi**2 - A*cos(2*pi*xi) for xi in x)

def uncoupled_rosenbrock(x):
    n = len(x)
    return sum(100*(x[2*i-1]**2 - x[2*i])**2 + (x[2*i-1] - 1)**2 for i in range(n/2))

def rosenbrock(x):
    n = len(x)
    return sum((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2 for i in range(n-1))

def gauss(x):
    n = len(x)
    return 1+0.5*sum(x[i]**2 for i in range(n))

try:
    nllf = fxy(rosenbrock) if len(sys.argv) < 2 else eval(sys.argv[1])
except:
    usage = """\
Provide a valid fitting function definition with x,y as fitting parameters,
such as:

    lambda x,y: (x+y-10)**2

A number of k-dimensional test functions are provided:

    ackley, griewank, rastrigin, rosenbrock, gauss

These can be selected using:

    f2(function)

or for the k-dimensional version:

    fk(function,k)
"""
    raise ValueError("Invalid function.\n"+usage)
plot=plot2d(nllf,('x','y'),range=(-1,1))
M = ModelFunction(nllf,plot=plot)
for p in M.parameters().values():
    p.value = 200
    p.range(-200,200)
problem = FitProblem(M)
