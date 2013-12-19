#!/usr/bin/env python

"""
1d model representation for polynomials

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""

from numpy import sum as numpysum
from numpy import polyval

def poly(x, c):
    return polyval(c,x)

# coefficients for specific Chebyshev polynomials
chebyshev8coeffs = [128., 0., -256., 0., 160., 0., -32., 0., 1.]
chebyshev16coeffs = [32768., 0., -131072., 0., 212992., 0., -180224., 0., 84480., 0., -21504., 0., 2688., 0., -128., 0., 1]

class Chebyshev(Polynomial):
    """Chebyshev polynomial models and functions,
including specific methods for T8(z) & T16(z), Equation (27-33) of [2]

NOTE: default is T8(z)"""

    def __init__(self,order=8,name='poly',metric=lambda x: numpysum(x*x),sigma=1.0):
        Polynomial.__init__(self,name,metric,sigma)
        if order == 8:  self.coeffs = chebyshev8coeffs
        elif order == 16:  self.coeffs = chebyshev16coeffs
        else: raise NotImplementedError("provide self.coeffs 'by hand'")
        return

    def cost(self,trial,M=61):
        """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""# % (len(self.coeffs)-1)
        #XXX: throw error when len(trial) != len(self.coeffs) ?
        myCost = chebyshevcostfactory(self.coeffs)
        return myCost(trial,M)

    pass

# faster implementation
def chebyshevcostfactory(target):
    def chebyshevcost(trial,M=61):
        """The costfunction for order-n Chebyshev fitting.
M evaluation points between [-1, 1], and two end points"""

        result=0.0
        x=-1.0
        dx = 2.0 / (M-1)
        for i in range(M):
            px = polyeval(trial, x)
            if px<-1 or px>1:
                result += (1 - px) * (1 - px)
            x += dx

        px = polyeval(trial, 1.2) - polyeval(target, 1.2)
        if px<0: result += px*px

        px = polyeval(trial, -1.2) - polyeval(target, -1.2)
        if px<0: result += px*px

        return result
    return chebyshevcost
