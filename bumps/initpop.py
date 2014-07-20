"""
Population initialization routines.

To start the analysis an initial population is required.  This will be
an array of size M x N, where M is the number of dimensions in the fitting
problem and N is the number of individuals in the population.

Three functions are provided:

1. lhs_init(n, pars) returns a latin hypercube sampling, which tests every
parameter at each of *n* levels.

2. cov_init(n, pars, cov) returns a Gaussian sample along the ellipse
defined by the covariance matrix, cov.  Covariance defaults to
diag(dx) if dx is provided as a parameter, or to I if it is not.

3. rand_init(n, pars) returns a random population following the
prior distribution of the parameter values.

Additional options are random box: rand(m,n) or random scatter: randn(m,n).
"""

# Note: borrowed from DREAM and extended.

from __future__ import division

__all__ = ['lhs_init', 'cov_init', 'random_init', 'eps_init', 'generate']

import math
import numpy as np
from numpy import eye, diag, asarray, empty, isinf, clip


def generate(problem, **options):
    """
    Population initializer.  Takes a problem and a set of initialization
    options.
    """
    initial = problem.getp()
    bounds = problem.bounds()
    pop_size = int(math.ceil(options['pop'] * len(initial)))
    # TODO: really need a continue option
    if options['init'] == 'random':
        population = random_init(
            pop_size, initial, problem, include_current=True)
    elif options['init'] == 'cov':
        cov = problem.cov()
        population = cov_init(
            pop_size, initial, bounds, include_current=True, cov=cov)
    elif options['init'] == 'lhs':
        population = lhs_init(
            pop_size, initial, bounds, include_current=True)
    elif options['init'] == 'eps':
        population = eps_init(
            pop_size, initial, bounds, include_current=True, eps=1e-6)
    else:
        raise ValueError(
            "Unknown population initializer '%s'" % options['init'])
    return population


def lhs_init(n, initial, bounds, include_current=False):
    """
    Latin Hypercube Sampling

    Returns an array whose columns and rows each have *n* samples from
    equally spaced bins between *bounds=(xmin, xmax)* for the column.
    Unlike random, this method guarantees a certain amount of coverage
    of the parameter space.  Consider, though that the diagonal matrix
    satisfies the LHS condition, and you can see that the guarantees are
    not very strong.  A better methods, similar to sudoku puzzles, would
    guarantee coverage in each block of the matrix, but this is not
    yet implmeneted.

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population, preserving the the
    LHS property.

    Note: Indefinite ranges are not supported.
    """
    xmin, xmax = bounds

    # Define the size of xmin
    nvar = len(xmin)

    # Initialize array ran with random numbers
    ran = np.random.rand(n, nvar)

    # Initialize array s with zeros
    s = empty((n, nvar))

    # Now fill s
    for j in range(nvar):
        if include_current:
            # Put current value at position 0 in population
            s[0, j] = initial[j]
            # Find which bin the current value belongs in
            xidx = int(n * initial[j] / (xmax[j] - xmin[j]))
            # Generate random permutation of remaining bins
            idx = np.random.permutation(n - 1)
            idx[idx >= xidx] += 1  # exclude current value bin
            # Assign random value within each bin
            p = (idx + ran[1:, j]) / n
            s[1:, j] = xmin[j] + p * (xmax[j] - xmin[j])
        else:
            # Random permutation of bins
            idx = np.random.permutation(n)
            # Assign random value within each bin
            p = (idx + ran[:, j]) / n
            s[:, j] = xmin[j] + p * (xmax[j] - xmin[j])

    return s


def cov_init(n, initial, bounds, include_current=False, cov=None, dx=None):
    """
    Initialize *n* sets of random variables from a gaussian model.

    The center is at *x* with an uncertainty ellipse specified by the
    1-sigma independent uncertainty values *dx* or the full covariance
    matrix uncertainty *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, pars=p, n=20)

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    # return mean + dot(RNG.randn(n,len(mean)), chol(cov))
    if cov is None and dx is None:
        cov = eye(len(initial))
    elif cov is None:
        cov = diag(asarray(dx) ** 2)
    population = np.random.multivariate_normal(
        mean=initial, cov=cov, size=n)
    if include_current:
        population[0] = initial
    # Make sure values are in bounds.
    population = clip(population, *bounds)
    return population


def random_init(n, initial, problem, include_current=False):
    """
    Generate a random population from the problem parameters.

    Values are selected at random from the bounds of the problem using a
    uniform distribution.  A certain amount of clustering is expected
    using this method.

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    population = problem.randomize(n)
    if include_current:
        population[0] = initial
    return population


def eps_init(n, initial, bounds, include_current=False, eps=1e-6):
    """
    Generate a random population using an epsilon ball around the current
    value.

    Since the initial population is contained in a small volume, this
    method is useful for exploring a local minimum around a point.  Over
    time the ball will expand to fill the minimum, and perhaps tunnel
    through barriers to nearby minima given enough burn-in time.

    eps is in proportion to the bounds on the parameter, or absolute if
    the parameter is unbounded.

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    x = initial
    xmin, xmax = bounds
    dx = (xmax - xmin) * eps
    dx[isinf(dx)] = eps
    population = x + dx * (2 * np.random.rand(n, len(xmin)) - 1)
    population = clip(population, xmin, xmax)
    if include_current:
        population[0] = x
    return population
