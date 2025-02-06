# This code is in the public domain.
# choose_without_replacement is copyright (2007) Anne Archibald
"""
Utility functions

choose_without_replacement(m,n,repeats=None)

    sample from a set without replacement

runlength(v)

    return the values and lengths of runs in a vector

countunique(A)

    return the element and frequency of each unique value in an array
"""

import numpy as np


# Author: Anne Archibald
# Re: [Numpy-discussion] Generating random samples without repeats
# Fri, 19 Sep 2008 12:19:22 -0700
def choose_without_replacement(m, n, repeats=None):
    """
    Choose n nonnegative integers less than m without replacement

    Returns an array of shape n, or (n,repeats).
    """
    if repeats is None:
        r = 1
    else:
        r = repeats
    if n > m:
        raise ValueError("Cannot find %d nonnegative integers less than %d" % (n, m))
    elif n > m / 2:
        res = np.sort(np.random.rand(m, r).argsort(axis=0)[:n, :], axis=0)
    else:
        res = np.random.random_integers(0, m - 1, size=(n, r))
        while True:
            res = np.sort(res, axis=0)
            w = np.nonzero(np.diff(res, axis=0) == 0)
            nr = len(w[0])
            if nr == 0:
                break
            res[w] = np.random.random_integers(0, m - 1, size=nr)

    if repeats is None:
        return res[:, 0]
    else:
        return res


def runlength(v):
    """
    Return the run lengths for repeated values in a vector v.

    See also countunique.
    """
    if len(v) == 0:
        return [], []
    diffs = np.diff(v)
    steps = np.where(diffs != 0)[0] + 1
    ends = np.hstack([[0], steps, [len(v)]])
    vals = v[ends[:-1]]
    lens = np.diff(ends)
    return vals, lens


def countunique(A):
    """
    Returns the unique elements in an array and their frequency.
    """
    return runlength(np.sort(A.flatten()))


def zscore(A, axis=None):
    """
    Convert an array of data to zscores.

    Use *axis* to limit the calculation of mean and standard deviation to
    a particular axis.
    """
    return (A - np.mean(A, axis=axis)) / np.std(A, axis=axis, ddof=1)
