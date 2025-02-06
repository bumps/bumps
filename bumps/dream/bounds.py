"""
Bounds handling.

Use bounds(low, high, style) to create a bounds handling object.  This
function operates on a point x, transforming it so that all dimensions
are within the bounds.  Options are available, including reflecting,
wrapping, clipping or randomizing the point, or ignoring the bounds.

The returned bounds object should have an apply(x) method which
transforms the point *x*.
"""

__all__ = ["make_bounds_handler", "Bounds", "ReflectBounds", "ClipBounds", "FoldBounds", "RandomBounds", "IgnoreBounds"]

import numpy as np
from numpy import inf, isinf
from . import util
from .compiled import dll

try:
    from numba import njit
except ImportError:

    def njit(*args, **kw):
        return lambda f: f


def make_bounds_handler(bounds, style="reflect"):
    """
    Return a bounds object which can update the bounds.

    Bounds handling *style* name is one of::

        reflect:   reflect off the boundary
        clip:      stop at the boundary
        fold:      wrap values to the other side of the boundary
        randomize: move to a random point in the bounds
        none:      ignore the bounds

    With semi-infinite intervals folding and randomizing aren't well
    defined, and reflection is used instead.

    With finite intervals the the reflected or folded point may still be
    outside the bounds (which can happen if the step size is too large),
    and a random uniform value is used instead.
    """
    if bounds is None:
        return IgnoreBounds()

    low, high = bounds

    # Do boundary handling -- what to do when points fall outside bound
    style = style.lower()
    if style == "reflect":
        f = ReflectBounds(low, high)
    elif style == "clip":
        f = ClipBounds(low, high)
    elif style == "fold":
        f = FoldBounds(low, high)
    elif style == "randomize":
        f = RandomBounds(low, high)
    elif style == "none" or style is None:
        f = IgnoreBounds()
    else:
        raise ValueError("bounds style %s is not valid" % style)
    return f


class Bounds(object):
    """
    Base class for all times of bounds objects.
    """

    c_interface = None  # type: Callable[[int, int, Any, Any, Any], None]
    low = None  # type: np.ndarray
    high = None  # type: np.ndarray

    @staticmethod
    def apply(minn, maxn, pop):
        """Force pop (population) values within bounds"""
        raise NotImplementedError

    def __call__(self, population):
        """
        Force parameter values within bounds for each member of the population
        (population is expected to be a 2d array of shape (M, N) where
         - M is the size of the population
         - N is then number of parameters
        Returns population for convenience.  E.g., y = bounds(x+0)
        """
        if self.c_interface is not None:
            self.c_interface(len(population), len(self.low), population.ctypes, self.low.ctypes, self.high.ctypes)
        else:
            self.apply(self.low, self.high, population)
        return population


class ReflectBounds(Bounds):
    """
    Reflect parameter values into bounded region
    """

    c_interface = dll.bounds_reflect if dll else None

    def __init__(self, low, high):
        self.low, self.high = [np.ascontiguousarray(v, "d") for v in (low, high)]

    @staticmethod
    @njit(cache=True)
    def apply(minn, maxn, pop):
        """
        Update pop so all values lie within bounds
        """
        for y in pop:
            # Reflect points which are out of bounds
            idx = y < minn
            y[idx] = 2 * minn[idx] - y[idx]
            idx = y > maxn
            y[idx] = 2 * maxn[idx] - y[idx]

            # Randomize points which are still out of bounds
            idx = (y < minn) | (y > maxn)
            y[idx] = minn[idx] + util.rng.rand(sum(idx)) * (maxn[idx] - minn[idx])


class ClipBounds(Bounds):
    """
    Clip values to bounded region
    """

    c_interface = dll.bounds_clip if dll else None

    def __init__(self, low, high):
        self.low, self.high = [np.ascontiguousarray(v, "d") for v in (low, high)]

    @staticmethod
    @njit(cache=True)
    def apply(minn, maxn, pop):
        for y in pop:
            idx = y < minn
            y[idx] = minn[idx]
            idx = y > maxn
            y[idx] = maxn[idx]


class FoldBounds(Bounds):
    """
    Wrap values into the bounded region
    """

    c_interface = dll.bounds_fold if dll else None

    def __init__(self, low, high):
        self.low, self.high = [np.ascontiguousarray(v, "d") for v in (low, high)]

    @staticmethod
    @njit(cache=True)
    def apply(minn, maxn, pop):
        for y in pop:
            # Deal with semi-infinite cases using reflection
            idx = (y < minn) & isinf(maxn)
            y[idx] = 2 * minn[idx] - y[idx]
            idx = (y > maxn) & isinf(minn)
            y[idx] = 2 * maxn[idx] - y[idx]

            # Wrap points which are out of bounds
            idx = y < minn
            y[idx] = maxn[idx] - (minn[idx] - y[idx])
            idx = y > maxn
            y[idx] = minn[idx] + (y[idx] - maxn[idx])

            # Randomize points which are still out of bounds
            idx = (y < minn) | (y > maxn)
            y[idx] = minn[idx] + util.rng.rand(sum(idx)) * (maxn[idx] - minn[idx])


class RandomBounds(Bounds):
    """
    Randomize values into the bounded region
    """

    c_interface = dll.bounds_random if dll else None

    def __init__(self, low, high):
        self.low, self.high = [np.ascontiguousarray(v, "d") for v in (low, high)]

    @staticmethod
    @njit(cache=True)
    def apply(minn, maxn, pop):
        for y in pop:
            # Deal with semi-infinite cases using reflection
            idx = (y < minn) & isinf(maxn)
            y[idx] = 2 * minn[idx] - y[idx]
            idx = (y > maxn) & isinf(minn)
            y[idx] = 2 * maxn[idx] - y[idx]

            # The remainder are selected uniformly from the bounded region
            idx = (y < minn) | (y > maxn)
            y[idx] = minn[idx] + util.rng.rand(sum(idx)) * (maxn[idx] - minn[idx])


class IgnoreBounds(Bounds):
    """
    Leave values outside the bounded region
    """

    c_interface = dll.bounds_ignore if dll else None

    def __init__(self, low=None, high=None):
        self.low, self.high = [np.ascontiguousarray(v, "d") for v in (low, high)]

    @staticmethod
    def apply(minn, maxn, pop):
        pass


def test():
    """bounds handlers test"""
    from numpy.linalg import norm
    from numpy import array

    bounds = list(zip([5, 10], [-inf, -10], [-5, inf], [-inf, inf]))
    v = np.ascontiguousarray([[6, -12, 6, -12]], "d")
    for t in "none", "reflect", "clip", "fold", "randomize":
        w = make_bounds_handler(bounds, t)
        assert norm(w(v + 0) - v) == 0
    v = np.ascontiguousarray([[12, 12, -12, -12]], "d")
    for t in "none", "reflect", "clip", "fold":
        w = make_bounds_handler(bounds, t)
        assert norm(w(v.repeat(3, axis=0)) - w(v + 0)) == 0
    assert norm(make_bounds_handler(bounds, "none")(v + 0) - v) == 0
    assert norm(make_bounds_handler(bounds, "reflect")(v + 0) - [8, -32, 2, -12]) == 0
    assert norm(make_bounds_handler(bounds, "clip")(v + 0) - [10, -10, -5, -12]) == 0
    assert norm(make_bounds_handler(bounds, "fold")(v + 0) - [7, -32, 2, -12]) == 0
    w = make_bounds_handler(bounds, "randomize")(v + 0)
    assert 5 <= w[0, 0] <= 10 and w[0, 1] == -32 and w[0, 2] == 2 and w[0, 3] == -12
    v = np.ascontiguousarray([[20, 1, 1, 1]], "d")
    w = make_bounds_handler(bounds, "reflect")(v + 0)
    assert 5 <= w[0, 0] <= 10
    v = np.ascontiguousarray([[20, 1, 1, 1]], "d")
    w = make_bounds_handler(bounds, "fold")(v + 0)
    assert 5 <= w[0, 0] <= 10


if __name__ == "__main__":
    test()
