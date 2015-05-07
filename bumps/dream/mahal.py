# This program is public domain
# Author: Paul Kienzle, April 2010
"""
Mahalanobis distance calculator

Compute the
`Mahalanobis distance <https://en.wikipedia.org/wiki/Mahalanobis_distance>`_
between observations and a reference set.  The principle components of the
reference set define the basis of the space for the observations.  The simple
Euclidean distance is used within this space.
"""

__all__ = ["mahalanobis"]

from numpy import dot, mean, sum
from numpy.linalg import svd

def mahalanobis(Y, X):
    """
    Returns the distances of the observations from a reference set.

    Observations are stored in rows *Y* and the reference set in *X*.
    """

    M = mean(X, axis=0)                 # mean
    Xc = X - mean(X, axis=0)            # center the reference
    W = dot(Xc.T, Xc)/(Xc.shape[0] - 1) # covariance of reference
    Yc = Y - M                          # center the observations
    # Distance is diag(Yc * inv(W) * Yc.H)
    # solve Wb = Yc.H using singular value decomposition because it is
    # the most accurate with numpy; QR decomposition does not use pivoting,
    # and is less accurate.  The built-in solve routine is between the two.
    u, s, vh = svd(W, 0)
    SVinv = vh.T.conj()/s
    Uy = dot(u.T.conj(), Yc.T.conj())
    b = dot(SVinv, Uy)

    D = sum(Yc.T * b, axis=0)  # compute distance
    return D


def test():
    from numpy import array
    from numpy.linalg import norm

    d = mahalanobis(array([[2, 3, 4], [2, 3, 4]]),
                    array([[1, 0, 0], [2, 1, 0], [1, 1, 0], [2, 0, 1]]))
    assert norm(d-[290.25, 290.25]) < 1e-12, "diff=%s" % str(d-[290.25, 290.25])


if __name__ == "__main__":
    test()
