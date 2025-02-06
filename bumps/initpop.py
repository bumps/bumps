"""
Population initialization strategies.

To start the analysis an initial population is required.  This will be
an array of size M x N, where M is the number of dimensions in the fitting
problem and N is the number of individuals in the population.

Normally the initialization will use a call to :func:`generate` with
key-value pairs from the command line options.  This will include the
'init' option, with the name of the strategy used to initialize the
population.

Additional strategies like uniform box in [0,1] or standard norm
(rand(m,n) and randn(m,n) respectively), may also be useful.
"""

# Note: borrowed from DREAM and extended.

__all__ = ["generate", "cov_init", "eps_init", "lhs_init", "random_init"]

import math
import numpy as np
from numpy import diag, empty, isinf, isfinite, clip, inf

try:
    from typing import Optional
except ImportError:
    pass


def generate(problem, init="eps", pop=10, use_point=True, **options):
    # type: (Any, str, int, bool, ...) -> np.ndarray
    """
    Population initializer.

    *problem* is a fit problem with *getp* and *bounds* methods.

    *init* is 'eps', 'cov', 'lhs' or 'random', indicating which
    initializer should be used.

    *pop* is the population scale factor, generating *pop* individuals
    for each parameter in the fit. If *pop < 0*, generate a total of
    *-pop* individuals regardless of the number of parameters.

    *use_point* is True if the initial value should be a member of the
    population.

    Additional options are ignored so that generate can be called using
    all command line options.
    """
    initial = problem.getp()
    initial[~isfinite(initial)] = 1.0
    pop_size = int(math.ceil(pop * len(initial))) if pop > 0 else int(-pop)
    bounds = problem.bounds()
    if init == "random":
        population = random_init(pop_size, initial, bounds, use_point=use_point, problem=problem)
    elif init == "cov":
        cov = problem.cov()
        population = cov_init(pop_size, initial, bounds, use_point=use_point, cov=cov)
    elif init == "lhs":
        population = lhs_init(pop_size, initial, bounds, use_point=use_point)
    elif init == "eps":
        population = eps_init(pop_size, initial, bounds, use_point=use_point, eps=1e-6)
    else:
        raise ValueError("Unknown population initializer '%s'" % init)

    # Use LHS to initialize any "free" parameters
    # TODO: find a better way to "free" parameters on --resume/--pars
    undefined = getattr(problem, "undefined", None)
    if undefined is not None:
        del problem.undefined
        population[:, undefined] = lhs_init(pop_size, initial[undefined], bounds[:, undefined], use_point=False)

    return population


def lhs_init(n, initial, bounds, use_point=False):
    # type: (int, np.ndarray, np.ndarray, bool, float) -> np.ndarray
    """
    Latin hypercube sampling.

    Returns an array whose columns and rows each have *n* samples from
    equally spaced bins between *bounds=(xmin, xmax)* for the column.
    Unlike random, this method guarantees a certain amount of coverage
    of the parameter space.  Consider, though that the diagonal matrix
    satisfies the LHS condition, and you can see that the guarantees are
    not very strong.  A better methods, similar to sudoku puzzles, would
    guarantee coverage in each block of the matrix, but this is not
    yet implmeneted.

    If *use_point* is True, then the current value of the parameters
    is returned as the first point in the population, preserving the the
    LHS property.
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
        # Indefinite and semidefinite ranges need to be constrained.  Use
        # the initial value of the parameter as a hint.
        low, high = xmin[j], xmax[j]
        if np.isinf(low) and np.isinf(high):
            if initial[j] < 0.0:
                low, high = 2.0 * initial[j], 0.0
            elif initial[j] > 0.0:
                low, high = 0.0, 2.0 * initial[j]
            else:
                low, high = -1.0, 1.0
        elif np.isinf(low):
            if initial[j] != high:
                low, high = high - 2.0 * abs(high - initial[j]), high
            else:
                low, high = high - 2.0, high
        elif np.isinf(high):
            if initial[j] != high:
                low, high = low, low + 2.0 * abs(initial[j] - low)
            else:
                low, high = low, low + 2.0
        else:
            pass  # low, high = low, high

        if use_point:
            # Put current value at position 0 in population
            s[0, j] = clip(initial[j], low, high)
            # Find which bin the current value belongs in
            xidx = int(n * (s[0, j] - low) / (high - low))
            # Generate random permutation of remaining bins
            perm = np.random.permutation(n - 1)
            perm[perm >= xidx] += 1  # exclude current value bin
            idx = slice(1, None)
        else:
            # Random permutation of bins
            perm = np.random.permutation(n)
            idx = slice(0, None)

        # Assign random value within each bin
        p = (perm + ran[idx, j]) / n
        s[idx, j] = low + p * (high - low)

    return s


def cov_init(n, initial, bounds, use_point=False, cov=None, dx=None):
    # type: (int, np.ndarray, np.ndarray, bool, Optional[np.ndarray], Optional[np.ndarray]) -> np.ndarray
    """
    Initialize *n* sets of random variables from a gaussian model.

    The center is at *x* with an uncertainty ellipse specified by the
    1-sigma independent uncertainty values *dx* or the full covariance
    matrix uncertainty *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, pars=p, n=20)

    If *use_point* is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    # return mean + dot(RNG.randn(n,len(mean)), chol(cov))
    if cov is None:
        if dx is None:
            dx = _get_scale_factor(0.2, bounds, initial)
            # print("= dx",dx)
        cov = diag(dx**2)
    xmin, xmax = bounds
    initial = clip(initial, xmin, xmax)
    population = np.random.multivariate_normal(mean=initial, cov=cov, size=n)
    population = reflect(population, xmin, xmax)
    if use_point:
        population[0] = initial
    return population


def random_init(n, initial, bounds, use_point=False, problem=None):
    """
    Generate a random population from the problem parameters.

    Values are selected at random from the bounds of the problem according
    to the underlying probability density of each parameter.  Uniform
    semi-definite and indefinite bounds use the standard normal distribution
    for the underlying probability, with a scale factor determined by the
    initial value of the parameter.

    If *use_point* is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    population = np.ascontiguousarray(problem.randomize(n))
    if use_point:
        population[0] = clip(initial, *bounds)
    return population


def eps_init(n, initial, bounds, use_point=False, eps=1e-6):
    # type: (int, np.ndarray, np.ndarray, bool, float) -> np.ndarray
    """
    Generate a random population using an epsilon ball around the current
    value.

    Since the initial population is contained in a small volume, this
    method is useful for exploring a local minimum around a point.  Over
    time the ball will expand to fill the minimum, and perhaps tunnel
    through barriers to nearby minima given enough burn-in time.

    eps is in proportion to the bounds on the parameter, or the current
    value of the parameter if the parameter is unbounded.  This gives the
    initialization a bit of scale independence.

    If *use_point* is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    # Set the scale from the bounds, or from the initial value if the value
    # is unbounded.
    xmin, xmax = bounds
    scale = _get_scale_factor(eps, bounds, initial)
    # print("= scale", scale)
    initial = clip(initial, xmin, xmax)
    population = initial + scale * (2 * np.random.rand(n, len(xmin)) - 1)
    population = reflect(population, xmin, xmax)
    if use_point:
        population[0] = initial
    return population


def reflect(v, low, high):
    """
    Reflect v off the boundary, then clip to be sure it is within bounds
    """
    index = v < low
    v[index] = (2 * low - v)[index]
    index = v > high
    v[index] = (2 * high - v)[index]
    return clip(v, low, high)


def _get_scale_factor(scale, bounds, initial):
    # type: (float, np.ndarray, np.ndarray) -> np.ndarray
    xmin, xmax = bounds
    dx = (xmax - xmin) * scale  # type: np.ndarray
    dx[isinf(dx)] = abs(initial[isinf(dx)]) * scale
    dx[~isfinite(dx)] = scale
    dx[dx == 0] = scale
    # print("min,max,dx",xmin,xmax,dx)
    return dx


def demo_init(seed=1):
    # type: (Optional[int]) -> None
    from . import util
    from .bounds import init_bounds

    class Problem(object):
        def __init__(self, initial, bounds):
            self.initial = initial
            self._bounds = bounds

        def getp(self):
            return self.initial

        def bounds(self):
            return self._bounds

        def cov(self):
            return None

        def randomize(self, n=1):
            target = self.initial.copy()
            target[~isfinite(target)] = 1.0
            result = [init_bounds(pair).random(n, v) for v, pair in zip(self.initial, self._bounds.T)]
            return np.array(result).T

    bounds = np.array([(2.0, inf), (-inf, -2.0), (-inf, inf), (5.0, 6.0), (-2.0, 3.0)]).T
    # generate takes care of bad values
    # low = np.array([-inf]*5)
    # high = np.array([inf]*5)
    # bad = np.array([np.nan]*5)
    zero = np.array([0.0] * 5)
    below = np.array([-2.0, -4.0, -2.0, -3.0, -4.0])
    above = np.array([3.0, 4.0, 2.0, 8.0, 5.0])
    small = np.array([2.000001, -2.000001, 0.000001, 5.000001, -0.000001])
    large = np.array([2000001.0, -2000001.0, 2000001.0, 5.5, -2.000001])
    middle = np.array([100.0, -100.0, 100.0, 5.5, 0.5])
    starting_points = "zero below above small large middle".split()
    np.set_printoptions(linewidth=100000)
    with util.push_seed(seed):
        for init_type in ("cov", "random", "eps", "lhs"):
            print("bounds:")
            print(bounds)
            for name in starting_points:
                initial = locals()[name]
                M = Problem(initial, bounds)
                pop = generate(problem=M, init=init_type, pop=1)
                print("%s init from %s" % (init_type, name), str(initial))
                print(pop)


if __name__ == "__main__":
    demo_init(seed=None)
