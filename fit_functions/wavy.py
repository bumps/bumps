#!/usr/bin/env python

"""
simple sine-based multi-minima functions

References::
    None.
"""

from numpy import absolute as abs
from numpy import asarray
from numpy import sin, pi


def wavy1(x):
    """
    Wave function #1: a simple multi-minima function

    minimum is f(x)=0.0 at xi=0.0
    """
    x = asarray(x)
    return abs(x+3.*sin(x+pi)+pi)

def wavy2(x):
    """
    Wave function #2: a simple multi-minima function

    minimum is f(x)=0.0 at xi=0.0
    """

    x = asarray(x)
    return 4 *sin(x)+sin(4*x) + sin(8*x)+sin(16*x)+sin(32*x)+sin(64*x)
