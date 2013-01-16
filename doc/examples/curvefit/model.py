import pylab
from numpy import sin, cos, linspace, meshgrid, e, pi, sqrt, array, exp
from bumps.names import *

def line(x, m, b):
    print "x",x
    print "m,b",m,b
    return m*x + b

x = [1,2,3,4,5,6]
y = [2.1,4.0,6.3,8.03,9.6,11.9]
dy = [0.05,0.05,0.2,0.05,0.2,0.2]

M = Curve(line,x,y,dy,m=2,b=2)
M.m.range(0,4)
M.b.range(-5,5)
problem = FitProblem(M)
