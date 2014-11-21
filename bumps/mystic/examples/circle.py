"""
2d array representation of a circle

References::
    None
"""

import numpy as np
from numpy import arange
from numpy import random, sin, cos, pi, inf, sqrt

random.seed(123)

def circle(coeffs,interval=0.02):
    """
    generate a 2D array representation of a circle of given coeffs
    coeffs = (x,y,r)
    """
    xc,yc,r = coeffs
    theta = arange(0, 2*pi, interval)
    return r*cos(theta)+xc, r*sin(theta)+yc

def genpoints(coeffs, npts=20):
    """
    Generate a 2D dataset of npts enclosed in circle of given coeffs,
    where coeffs = (x,y,r).

    NOTE: if npts is None, constrain all points to circle of given radius
    """
    xo,yo,R = coeffs
    # Radial density varies as sqrt(x)
    r,theta = sqrt(random.rand(npts)), 2*pi*(random.rand(npts))
    x,y = r*cos(theta)+xo, r*sin(theta)+yo
    return x,y

def gendensity(coeffs,density=0.1):
    xo,yo,R = coeffs
    npts = int(0.2*pi*R**2)
    return genpoints(coeffs,npts)

from .model import Fitness
class MinimumCircle(Fitness):
    def __init__(self, data=None, limits=None, start=None):
        self.x,self.y = data
        self.dy = None
        self.limits = limits
        self.start = start

    def _residuals(self, p):
        x,y = self.x,self.y
        xc,yc,r = p
        d = sqrt((x-xc)**2 + (y-yc**2))
        d[d<r] = 0
        return d

    def profile(self, p):
        return circle(p)

    def residuals(self, p):
        """
        Residuals is used by Levenburg-Marquardt, so fake the
        penalty terms of the normal cost function for use here.
        """
        resid = self._residuals(p)
        # Throw r in the residual so that it is minimized, punish the circle
        # if there are too many points outside.
        d = np.concatenate((resid,self.r,sum(resid>0)))
        return d

    def __call__(self, p):
        xc,yc,r = p
        resid = self._residuals(p)
        # Penalties are the number
        # Add additional penalties for each point outside the circle
        return sum(resid>0) + sum(resid) + abs(r)

# prepared instances
Po = [0,0,1]
Plo=[-inf,-inf,0]
Phi=[inf,inf,inf]
dense_circle = MinimumCircle(data=gendensity([3,4,20],density=5),
                             limits=(Plo,Phi), start=Po)
sparse_circle = MinimumCircle(data=gendensity([3,-4,32],density=0.5),
                              limits=(Plo,Phi), start=Po)
minimal_circle = MinimumCircle(data=gendensity([0,0,10],density=5),
                               limits=(Plo,Phi), start=Po)
