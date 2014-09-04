"""
1-D and 2-D rebinning code.
"""

__all__ = ["bin_edges", "logbin_edges", "rebin", "rebin2d"]

import numpy as np

from . import _reduction


def bin_edges(C):
    r"""
    Construct bin edges *E* from equally spaced bin centers *C*.

    Assumes the edges lie half way between the centers.  This will only
    be true if the centers are evenly spaced, but may be good enough for
    visualization purposes even when they are not.  Ideally analysis would
    be performed on the raw data without rebinning.
    """
    E = 0.5 * (C[:-1] + C[1:])
    return np.hstack((C[0] - (E[0] - C[0]), E, C[-1] + (C[-1] - E[-1])))


def logbin_edges(L):
    r"""
    Construct bin edges *E* from logarithmically spaced bin centers *L*.

    Assuming fixed $\omega = \Delta\lambda/\lambda$ in the bins, the
    edges will be spaced logarithmically at:

    .. math::

        E_0     &= \min \lambda \\
        E_{i+1} &= E_i + \omega E_i = E_i (1+\omega)

    with centers $L$ half way between the edges:

    .. math::

        L_i = (E_i+E_{i+1})/2
            = (E_i + E_i (1+\omega))/2
            = E_i (2 + \omega)/2

    Solving for $E_i$, we can recover the edges from the centers:

    .. math::

        E_i = L_i \frac{2}{2+\omega}

    The final edge, $E_{n+1}$, does not have a corresponding center
    $L_{n+1}$ so we must determine it from the previous edge $E_n$:

    .. math::

        E_{n+1} = L_n \frac{2}{2+\omega}(1+\omega)

    The fixed $\omega$ can be retrieved from the ratio of any pair
    of bin centers using:

    .. math::

        \frac{L_{i+1}}{L_i} = \frac{ (E_{i+2}+E_{i+1})/2 }{ (E_{i+1}+E_i)/2 }
                          = \frac{ (E_{i+1}(1+\omega)+E_{i+1} }
                                  { (E_i(1+\omega)+E_i }
                          = \frac{E_{i+1}}{E_i}
                          = \frac{E_i(1+\omega)}{E_i} = 1 + \omega
    """
    if L[1] > L[0]:
        dLoL = L[1] / L[0] - 1
        last = (1 + dLoL)
    else:
        dLoL = L[0] / L[1] - 1
        last = 1. / (1 + dLoL)
    E = L * 2 / (2 + dLoL)
    return np.hstack((E, E[-1] * last))


def rebin(x, I, xo, Io=None, dtype=np.float64):
    """
    Rebin a vector.

    x are the existing bin edges
    xo are the new bin edges
    I are the existing counts (one fewer than edges)

    Io will be used if present, but be sure that it is a contiguous
    array of the correct shape and size.

    dtype is the type to use for the intensity vectors.  This can be
    integer (uint8, uint16, uint32) or real (float32 or f, float64 or d).
    The edge vectors are all coerced to doubles.

    Note that total intensity is not preserved for integer rebinning.
    The algorithm uses truncation so total intensity will be down on
    average by half the total number of bins.
    """
    # Coerce axes to float arrays
    x, xo = _input(x, dtype='d'), _input(xo, dtype='d')
    shape_in = np.array([x.shape[0] - 1])
    shape_out = np.array([xo.shape[0] - 1])

    # Coerce counts to correct type and check shape
    if dtype is None:
        try:
            dtype = I.dtype
        except AttributeError:
            dtype = np.float64
    I = _input(I, dtype=dtype)
    if shape_in != I.shape:
        raise TypeError("input array incorrect shape %s" % I.shape)

    # Create output vector
    Io = _output(Io, shape_out, dtype=dtype)

    # Call rebin on type if it is available
    try:
        rebincore = getattr(_reduction, 'rebin_' + I.dtype.name)
    except AttributeError:
        raise TypeError("rebin supports uint 8/16/32/64 and float 32/64, not "
                        + I.dtype.name)
    rebincore(x, I, xo, Io)
    return Io


def rebin2d(x, y, I, xo, yo, Io=None, dtype=None):
    """
    Rebin a matrix.

    x,y are the existing bin edges
    xo,yo are the new bin edges
    I is the existing counts (one fewer than edges in each direction)

    For example, with x representing the column edges in each row and
    y representing the row edges in each column, the following
    represents a uniform field::

        >>> #from bumps.rebin import rebin2d
        >>> x,y = [0,2,4,5], [0,1,3]
        >>> z = [[2,2,1], [4,4,2]]

    We can check this by rebinning with uniform size bins::

        >>> xo,yo = range(6),range(4)
        >>> rebin2d(y, x, z, yo, xo)
        array([[ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.]])

    dtype is the type to use for the intensity vectors.  This can be
    integer (uint8, uint16, uint32) or real (float32 or f, float64 or d).
    The edge vectors are all coerced to doubles.

    Note that total intensity is not preserved for integer rebinning.
    The algorithm uses truncation so total intensity will be down on
    average by half the total number of bins.

    Io will be used if present, if it is contiguous and if it has the
    correct shape and type for the input.  Otherwise it will raise a
    TypeError.  This will allow you to rebin the slices of an appropriately
    ordered matrix without making copies.
    """
    # Coerce axes to float arrays
    x, y, xo, yo = [_input(v, dtype='d') for v in (x, y, xo, yo)]
    shape_in = np.array([x.shape[0] - 1, y.shape[0] - 1])
    shape_out = np.array([xo.shape[0] - 1, yo.shape[0] - 1])

    # Coerce counts to correct type and check shape
    if dtype is None:
        try:
            dtype = I.dtype
        except AttributeError:
            dtype = np.float64
    I = _input(I, dtype=dtype)
    if (shape_in != I.shape).any():
        raise TypeError("input array incorrect shape %s" % str(I.shape))

    # Create output vector
    Io = _output(Io, shape_out, dtype=dtype)

    # Call rebin on type if it is available
    try:
        rebincore = getattr(_reduction, 'rebin2d_' + I.dtype.name)
    except AttributeError:
        raise TypeError("rebin2d supports uint 8/16/32/64 and float 32/64, not "
                        + I.dtype.name)
    # print x.shape, y.shape, I.shape, xo.shape, yo.shape, Io.shape
    # print x.dtype, y.dtype, I.dtype, xo.dtype, yo.dtype, Io.dtype
    rebincore(x, y, I, xo, yo, Io)
    return Io


def _input(v, dtype='d'):
    """
    Force v to be a contiguous array of the correct type, avoiding copies
    if possible.
    """
    return np.ascontiguousarray(v, dtype=dtype)


def _output(v, shape, dtype=np.float64):
    """
    Create a contiguous array of the correct shape and type to hold a
    returned array, reusing an existing array if possible.
    """
    if v is None:
        return np.empty(shape, dtype=dtype)
    if not (isinstance(v, np.ndarray)
            and v.dtype == np.dtype(dtype)
            and (v.shape == shape).all()
            and v.flags.contiguous):
        raise TypeError("output vector must be contiguous %s of size %s"
                        % (dtype, shape))
    return v


# ================ Test code ==================
# TODO: move test code to its own file
def _check_one_1d(from_bins, val, to_bins, target):
    target = _input(target)
    for (f, F) in [(from_bins, val), (from_bins[::-1], val[::-1])]:
        for (t, T) in [(to_bins, target), (to_bins[::-1], target[::-1])]:
            result = rebin(f, F, t)
            assert np.linalg.norm(T - result) < 1e-14, \
                "rebin failed for %s->%s %s" % (f, t, result)


def _check_all_1d():
    # Split a value
    _check_one_1d([1, 2, 3, 4], [10, 20, 30], [1, 2.5, 4], [20, 40])

    # bin is a superset of rebin
    _check_one_1d([0, 1, 2, 3, 4], [5, 10, 20, 30], [1, 2.5, 3], [20, 10])

    # bin is a subset of rebin
    _check_one_1d([1,   2,   3,   4,   5,   6],
                  [10,  20,  30,  40,  50],
                  [2.5,  3.5],
                  [25])

    # one bin to many
    _check_one_1d([1,   2,   3,   4,   5,  6],
                  [10,  20,  30,  40,  50],
                  [2.1, 2.2, 2.3, 2.4],
                  [2,   2,   2])

    # many bins to one
    _check_one_1d([1,   2,   3,   4,   5,  6],
                  [10,  20,  30,  40,  50],
                  [2.5,      4.5],
                  [60])


def _check_one_2d(x, y, z, xo, yo, zo):
    # print "checking"
    # print x, y, z
    # print xo, yo, zo
    result = rebin2d(x, y, z, xo, yo)
    target = np.array(zo, dtype=result.dtype)
    assert np.linalg.norm(target - result) < 1e-14, \
        "rebin2d failed for %s,%s->%s,%s\nexpected: %s\nbut got: %s" \
        % (x, y, xo, yo, zo, z)


def _check_uniform_2d(x, y):
    z = np.array([y], 'd') * np.array([x], 'd').T
    xedges = np.concatenate([(0,), np.cumsum(x)])
    yedges = np.concatenate([(0,), np.cumsum(y)])
    nx = np.round(xedges[-1])
    ny = np.round(yedges[-1])
    ox = np.arange(nx + 1)
    oy = np.arange(ny + 1)
    target = np.ones([nx, ny], 'd')
    _check_one_2d(xedges, yedges, z, ox, oy, target)


def _check_all_2d():
    x, y, I = [0, 3, 5, 7], [0, 1, 3], [[3, 6], [2, 4], [2, 4]]
    xo, yo, Io = range(8), range(4), [[1] * 3] * 7
    x, y, I, xo, yo, Io = [np.array(A, 'd') for A in (x, y, I, xo, yo, Io)]

    # Try various types and orders on a non-square matrix
    _check_one_2d(x, y, I, xo, yo, Io)
    _check_one_2d(x[::-1], y, I[::-1, :], xo, yo, Io)
    _check_one_2d(x, y[::-1], I[:, ::-1], xo, yo, Io)
    _check_one_2d(x, y, I, [7, 3, 0], yo, [[4] * 3, [3] * 3])
    _check_one_2d(x, y, I, xo, [3, 2, 0], [[1, 2]] * 7)
    _check_one_2d(y, x, I.T, yo, xo, Io.T)  # C vs. Fortran ordering

    # Test smallest possible result
    _check_one_2d([-1, 2, 4], [0, 1, 3], [[3, 6], [2, 4]],
                  [1, 2], [1, 2], [1])
    # subset/superset
    _check_one_2d([0, 1, 2, 3], [0, 1, 2, 3], [[1] * 3] * 3,
                  [0.5, 1.5, 2.5], [0.5, 1.5, 2.5], [[1] * 2] * 2)
    for dtype in ['uint8', 'uint16', 'uint32', 'float32', 'float64']:
        _check_one_2d([0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
                      np.array([[1] * 4] * 4, dtype=dtype),
                      [-2, -1, 2, 5, 6], [-2, -1, 2, 5, 6],
                      np.array([[0, 0, 0, 0], [0, 4, 4, 0],
                                   [0, 4, 4, 0], [0, 0, 0, 0]],
                                  dtype=dtype)
                      )
    # non-square test
    _check_uniform_2d([1, 2.5, 4, 0.5], [3, 1, 2.5, 1, 3.5])
    _check_uniform_2d([3, 2], [1, 2])


def _check_bin_edges():
    log_edges = np.asarray([2 * 1.2 ** c for c in range(10)])
    log_centers = (log_edges[1:] + log_edges[:-1]) / 2
    assert np.linalg.norm(logbin_edges(log_centers) - log_edges) < 1e-10

    lin_edges = np.linspace(2, 10, 10)
    lin_centers = (lin_edges[1:] + lin_edges[:-1]) / 2
    assert np.linalg.norm(bin_edges(lin_centers) - lin_edges) < 1e-10


def test():
    _check_all_1d()
    _check_all_2d()
    _check_bin_edges()

if __name__ == "__main__":
    test()
