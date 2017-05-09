"""
Exponential power density parameter calculator.
"""

from __future__ import division

__all__ = ["exppow_pars"]

from math import sqrt

from scipy.special import gamma


def exppow_pars(B):
    r"""
    Return w(B) and c(B) for the exponential power density:

    .. math::

        p(v|S,B) = \frac{w(B)}{S} \exp\left(-c(B) |v/S|^{2/(1+B)}\right)

    *B* in (-1,1] is a measure of kurtosis::

        B = 1: double exponential
        B = 0: normal
        B -> -1: uniform

    [1] Thiemann, M., M. Trosser, H. Gupta, and S. Sorooshian (2001).
    *Bayesian recursive parameter estimation for hydrologic models*,
    Water Resour. Res. 37(10) 2521-2535.
    """

    # First calculate some dummy variables
    A1 = gamma(3*(1+B)/2)
    A2 = gamma((1+B)/2)
    # And use these to derive Cb and Wb
    cB = (A1/A2)**(1/(1+B))
    wB = sqrt(A1/A2**3)/(1+B)

    return cB, wB


def test():
    import math
    cB, wB = exppow_pars(13)
    assert abs(cB - 12.8587702619708) < 1e-13
    assert abs(wB - 5766.80847609837) < 1e-11
    # Check that beta=0 yields a normal distribution
    cB, wB = exppow_pars(0)
    assert abs(2*math.pi*wB**2 - 1) < 1e-14
    assert abs(cB - 0.5) < 1e-14


if __name__ == "__main__":
    #print calc_CbWb(13)
    test()
