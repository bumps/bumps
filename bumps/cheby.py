r"""
Freeform modeling with Chebyshev polynomials.

`Chebyshev polynomials <http://en.wikipedia.org/wiki/Chebyshev_polynomials>`_
$T_k$ form a basis set for functions over $[-1,1]$.  The truncated
interpolating polynomial $P_n$ is a weighted sum of Chebyshev polynomials
up to degree $n$:

.. math::

    f(x) \approx P_n(x) = \sum_{k=0}^n c_i T_k(x)

The interpolating polynomial exactly matches $f(x)$ at the chebyshev
nodes $z_k$ and is near the optimal polynomial approximation to $f$
of degree $n$ under the maximum norm.  For well behaved functions,
the coefficients $c_k$ decrease rapidly, and furthermore are independent
of the degree $n$ of the polynomial.

The models can either be defined directly in terms of the Chebyshev
coefficients $c_k$ with *method* = 'direct', or in terms of control
points $(z_k, f(z_k))$ at the Chebyshev nodes :func:`cheby_points`
with *method* = 'interp'.  Bounds on the parameters are easier to
control using 'interp', but the function may oscillate wildly outside
the bounds.  Bounds on the oscillation are easier to control using
'direct', but the shape of the profile is difficult to control.
"""

# TODO: clipping volume fraction to [0,1] distorts parameter space
# Option 0: clip to [0,1]
# - Bayesian analysis: parameter values outside the domain will be equally
#   probable out to infinity
# - Newton methods: the fit space is flat outside the domain, which leads
#   to a degenerate hessian.
# - Direct methods: won't fail, but will be subject to random walk
#   performance outside the domain.
# - trivial to implement!
# Option 1: compress (-inf,0.001] and [0.999,inf) into (0,0.001], [0.999,1)
# - won't address any of the problems of clipping
# Option 2: have chisq return inf for points outside the domain
# - Bayesian analysis: correctly assigns probability zero
# - Newton methods: degenerate Hessian outside domain
# - Direct methods: random walk outside domain
# - easy to implement
# Option 3: clip outside domain but add penalty based on amount of clipping
#   A profile based on clipping may have lower chisq than any profile that
#   can be described by a valid model (e.g., by having a sharper transition
#   than would be allowed by the model), leading to a minimum outside D.
#   Adding a penalty constant outside D would help, but there is no constant
#   that works everywhere.  We could use a constant greater than the worst
#   chisq seen so far in D, which can guarantee an arbitrarily low P(x) and
#   a global minimum within D, but for Newton methods, the boundary may still
#   have spurious local minima and objective value now depends on history.
#   Linear compression of profile to fit within the domain would avoid
#   unreachable profile shapes (this is just a linear transform on chebyshev
#   coefficients), and the addition of the penalty value would reduce
#   parameter correlations that result from having transformed parameters
#   resulting in identical profiles.  Returning T = ||A(x)|| from render,
#   with A being a transform that brings the profile within [0,1], the
#   objective function can return P'(x) = P(x)/(10*(1+sum(T_i)^4) for all
#   slabs i, or P(x) if no slabs return a penalty value.  So long as T is
#   monotonic with increasing badness, with value of 0 within D, and so long
#   as no values of x outside D can generate models that cannot be
#   expressed for any x within D, then any optimizer should return a valid
#   result at the global minimum.  There may still be local minima outside
#   the boundary, so information that the the value is outside the domain
#   still needs to pass through a local optimizer to the fitting program.
#   This approach could be used to transform a box constrained
#   problem to an unconstrained problem using clipping+penalty on the
#   parameter values and removing the need for constrained Newton optimizers.
# - Bayesian analysis: parameters outside D have incorrect probability, but
#   with a sufficiently large penalty, P(x) ~ 0; if the penalty value is
#   too low, details of the correlations outside D may leak into D.
# - Newton methods: Hessian should point back to domain
# - Direct methods: random walk should be biased toward the domain
# - moderately complicated
__all__ = ["profile", "cheby_approx", "cheby_val", "cheby_points", "cheby_coeff"]

import numpy as np
from numpy import real, exp, pi, cos, arange, asarray
from numpy.fft import fft


def profile(c, t, method):
    r"""
    Evaluate the chebyshev approximation c at points x.

    If method is 'direct' then $c_i$ are the coefficients for the chebyshev
    polynomials $T_i$ yielding $P = \sum_i{c_i T_i(x)}$.

    If method is 'interp' then $c_i$ are the values of the interpolated
    function $f$ evaluated at the chebyshev points returned by
    :func:`cheby_points`.
    """
    if method == "interp":
        c = cheby_coeff(c)
    return cheby_val(c, t)


def cheby_approx(n, f, range=(0, 1)):
    """
    Return the coefficients for the order n chebyshev approximation to
    function f evaluated over the range [low,high].
    """
    fx = f(cheby_points(n, range=range))
    return cheby_coeff(fx)


def cheby_val(c, x):
    r"""
    Evaluate the chebyshev approximation c at points x.

    The values $c_i$ are the coefficients for the chebyshev
    polynomials $T_i$ yielding $p(x) = \sum_i{c_i T_i(x)}$.
    """
    c = np.asarray(c)
    if len(c) == 0:
        return 0 * x

    # Crenshaw recursion from numerical recipes sec. 5.8
    y = 4 * x - 2
    d = dd = 0
    for c_j in c[:0:-1]:
        d, dd = y * d + (c_j - dd), d
    return y * (0.5 * d) + (0.5 * c[0] - dd)


def cheby_points(n, range=(0, 1)):
    r"""
    Return the points in at which a function must be evaluated to
    generate the order $n$ Chebyshev approximation function.

    Over the range [-1,1], the points are $p_k = \cos(\pi(2 k + 1)/(2n))$.
    Adjusting the range to $[x_L,x_R]$, the points become
    $x_k = \frac{1}{2} (p_k - x_L + 1)/(x_R-x_L)$.
    """
    return 0.5 * (cos(pi * (arange(n) + 0.5) / n) - range[0] + 1) / (range[1] - range[0])


def cheby_coeff(fx):
    """
    Compute chebyshev coefficients for a polynomial of order n given
    the function evaluated at the chebyshev points for order n.

    This can be used as the basis of a direct interpolation method where
    the n control points are positioned at cheby_points(n).
    """
    fx = asarray(fx)
    n = len(fx)
    w = exp((-0.5j * pi / n) * arange(n))
    y = np.hstack((fx[0::2], fx[1::2][::-1]))
    c = (2.0 / n) * real(fft(y) * w)
    return c
