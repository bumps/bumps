"""
Population initialization routines.

To start the analysis an initial population is required.  This will be
an array of size M x N, where M is the number of dimensions in the fitting
problem and N is the number of Markov chains.

Two functions are provided:

1. lhs_init(N, bounds) returns a latin hypercube sampling, which tests every
parameter at each of N levels.

2. cov_init(N, x, cov) returns a Gaussian sample along the ellipse
defined by the covariance matrix, cov.  Covariance defaults to
diag(dx) if dx is provided as a parameter, or to I if it is not.

Additional options are random box: rand(M, N) or random scatter: randn(M, N).
"""

from __future__ import division, print_function

__all__ = ['lhs_init', 'cov_init']

from numpy import eye, diag, asarray, empty
from . import util


def lhs_init(N, bounds):
    """
    Latin Hypercube Sampling

    Returns an array whose columns each have *N* samples from equally spaced
    bins between *bounds=(xmin, xmax)* for the column.  DREAM bounds
    objects, with bounds.low and bounds.high can be used as well.

    Note: Indefinite ranges are not supported.
    """
    try:
        xmin, xmax = bounds.low, bounds.high
    except AttributeError:
        xmin, xmax = bounds

    # Define the size of xmin
    nvar = len(xmin)
    # Initialize array ran with random numbers
    ran = util.rng.rand(N, nvar)

    # Initialize array s with zeros
    s = empty((N, nvar))

    # Now fill s
    for j in range(nvar):
        # Random permutation
        idx = util.rng.permutation(N)+1
        p = (idx-ran[:, j])/N
        s[:, j] = xmin[j] + p*(xmax[j]-xmin[j])

    return s


def cov_init(N, x, cov=None, dx=None):
    """
    Initialize *N* sets of random variables from a gaussian model.

    The center is at *x* with an uncertainty ellipse specified by the
    1-sigma independent uncertainty values *dx* or the full covariance
    matrix uncertainty *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, x=x, N=20)
    """
    #return mean + dot(util.rng.randn(N, len(mean)), chol(cov))
    if cov is None and dx is None:
        cov = eye(len(x))
    elif cov is None:
        cov = diag(asarray(dx)**2)
    return util.rng.multivariate_normal(mean=x, cov=cov, size=N)


def demo():
    from numpy import arange
    print("Three ways of calling cov_init:")
    print("with cov", cov_init(N=4, x=[5, 6], cov=diag([0.1, 0.001])))
    print("with dx", cov_init(N=4, x=[5, 6], dx=[0.1, 0.001]))
    print("with nothing", cov_init(N=4, x=[5, 6]))
    print("""
The following array should have four columns.  Column 1 should have the
numbers from 10 to 19, column 2 from 20 to 29, etc.  The columns are in
random order with a random fractional part.
""")
    pop = lhs_init(N=10, bounds=(arange(1, 5), arange(2, 6)))*10
    print(pop)


if __name__ == "__main__":
    demo()
