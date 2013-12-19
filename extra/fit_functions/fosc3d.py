#!/usr/bin/env python

"""
the fOsc3D Mathematica function

References::
    [4] Mathematica guidebook
"""
from math import sin, exp

def fOsc3D(x,y):
    """
    fOsc3D Mathematica function:

    fOsc3D[x_,y_] := -4 Exp[(-x^2 - y^2)] + Sin[6 x] Sin[5 y]

    minimum?
    """

    func =  -4. * exp( -x*x - y*y ) + sin(6. * x) * sin(5. *y)
    penalty = 100.*y*y if y<0 else 0
    return func + penalty
