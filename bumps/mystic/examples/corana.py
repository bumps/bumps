"""
Ccorana's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from math import pow

from numpy import sign, floor, inf

def corana(coeffs):
    """
    evaluates the Corana function for a list of coeffs

    minimum is f(x)=0.0 at xi=0.0
    """
    d = [1., 1000., 10., 100.]
    x = coeffs
    r = 0
    for xj,dj in zip(x,d):
        zj =  floor( abs(xj/0.2) + 0.49999 ) * sign(xj) * 0.2
        if abs(xj-zj) < 0.05:
            r += 0.15 * pow(zj - 0.05*sign(zj), 2) * dj
        else:
            r += dj * xj ** 2
        return r


def corana1d(x):
    """Corana in 1D; coeffs = (x,0,0,0)"""
    return corana([x[0], 0, 0, 0])

def corana2d(x):
    """Corana in 2D; coeffs = (x,0,y,0)"""
    return corana([x[0], 0, x[1], 0])

def corana3d(x):
    """Corana in 3D; coeffs = (x,0,y,z)"""
    return corana([x[0], 0, x[1], x[2]])

Po = [1,1,1,1]
Plo = [-inf,-inf,-inf,-inf]
Phi = [inf,inf,inf,inf]

def s(V,*args):
    retval = []
    for i in args: retval.append(V[i])
    return retval
from .model import Function
corana1d = Function(f=corana1d, limits=(s(Plo,0),s(Phi,0)),
                    start=s(Po,0))
corana2d = Function(f=corana2d, limits=(s(Plo,0,2),s(Phi,0,2)),
                    start=s(Po,0,2))
corana3d = Function(f=corana3d, limits=(s(Plo,0,2,3),s(Phi,0,2,3)),
                    start=s(Po,0,2,3))
corana4d = Function(f=corana, limits=(Plo,Phi), start=Po)
# End of file
