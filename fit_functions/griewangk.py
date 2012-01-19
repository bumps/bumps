#!/usr/bin/env python

"""
Griewangk's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""

from math import cos, sqrt

def griewangk(coeffs):
    """
    Griewangk function: a multi-minima function, Equation (23) of [2]

    minimum is f(x)=0.0 at xi=0.0
    """

    # ensure that there are 10 coefficients
    x = [0]*10
    x[:len(coeffs)]=coeffs

    term1 = sum([xi*xi for xi in x])/4000
    term2 = prod([cos(xi/sqrt(i+1.)) for i,xi in enumerate(x)])
    return term1 - term2 + 1

prod = lambda x: reduce(lambda a,b: a*b, x, 1.)