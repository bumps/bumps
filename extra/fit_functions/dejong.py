#!/usr/bin/env python

"""
Rosenbrock's function, De Jong's step function, De Jong's quartic function,
and Shekel's function

References::
    [1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces. Journal of Global
    Optimization 11: 341-359, 1997.

    [2] Storn, R. and Price, K.
    (Same title as above, but as a technical report.)
    http://www.icsi.berkeley.edu/~storn/deshort1.ps
"""
from six.moves import reduce

from numpy import sum as numpysum
from numpy import asarray
from math import floor
import random
from math import pow

def rosenbrock(x):
    """
    Rosenbrock function:

    A modified second De Jong function, Equation (18) of [2]

    minimum is f(x)=0.0 at xi=1.0
    """
    #ensure that there are 2 coefficients
    assert len(x) >= 2
    x = asarray(x)
    return numpysum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def step(x):
    """
    De Jong's step function:

    The third De Jong function, Equation (19) of [2]

    minimum is f(x)=0.0 at xi=-5-n where n=[0.0,0.12]
    """
    f = 30.
    for c in x:
        if abs(c) <= 5.12:
            f += floor(c)
        elif c > 5.12:
            f += 30 * (c - 5.12)
        else:
            f += 30 * (5.12 - c)
    return f

def quartic(x):
    """
    De Jong's quartic function:
    The modified fourth De Jong function, Equation (20) of [2]

    minimum is f(x)=random, but statistically at xi=0
    """
    f = 0.
    for j, c in enumerate(x):
        f += pow(c,4) * (j+1.0) + random.random()
    return f


def shekel(x):
    """
    Shekel: The modified fifth De Jong function, Equation (21) of [2]

    minimum is f(x)=0.0 at x(-32,-32)
    """

    A = [-32., -16., 0., 16., 32.]
    a1 = A * 5
    a2 = reduce(lambda x1,x2: x1+x2, [[c] * 5 for c in A])

    x1,x2 = x
    r = 0.0
    for i in range(25):
        r += 1.0/ (1.0*i + pow(x1-a1[i],6) + pow(x2-a2[i],6) + 1e-15)
    return 1.0/(0.002 + r)
