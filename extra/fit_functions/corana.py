#!/usr/bin/env python

"""
Corana's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from math import pow, floor, copysign


def corana1d(x):
    """Corana in 1D; coeffs = (x,0,0,0)"""
    return corana4d([x[0], 0, 0, 0])

def corana2d(x):
    """Corana in 2D; coeffs = (x,0,y,0)"""
    return corana4d([x[0], 0, x[1], 0])

def corana3d(x):
    """Corana in 3D; coeffs = (x,0,y,z)"""
    return corana4d([x[0], 0, x[1], x[2]])

def corana4d(x):
    """
    evaluates the Corana function on [x0,x1,x2,x3]

    minimum is f(x)=0.0 at xi=0.0
    """
    d = [1., 1000., 10., 100.]
    r = 0
    for xj,dj in zip(x,d):
        zj =  floor( abs(xj/0.2) + 0.49999 ) * copysign(0.2,xj)
        if abs(xj-zj) < 0.05:
            r += 0.15 * pow(zj - copysign(0.05,zj), 2) * dj
        else:
            r += dj * xj * xj
    return r

# End of file
