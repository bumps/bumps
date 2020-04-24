"""
Crossover ratios

The crossover ratio (CR) determines what percentage of parameters in the
target vector are updated with difference vector selected from the
population.  In traditional differential evolution a CR value is chosen
somewhere in [0, 1] at the start of the search and stays constant throughout.
DREAM extends this by allowing multiple CRs at the same time with different
probabilities.  Adaptive crossover adjusts the relative weights of the CRs
based on the average distance of the steps taken when that CR was used.  This
distance will be zero for unsuccessful metropolis steps, and so the relative
weights on those CRs which generate many unsuccessful steps will be reduced.

Usage
-----

1. Traditional differential evolution::

    crossover = Crossover(CR=CR)

2. Weighted crossover ratios::

    crossover = Crossover(CR=[CR1, CR2, ...], weight=[weight1, weight2, ...])

The weights are normalized to one, and default to equally weighted CRs.

3. Adaptive weighted crossover ratios::

    crossover = AdaptiveCrossover(N)

The CRs are set to *[1/N, 2/N, ... 1]*, and start out equally weighted.  The
weights are adapted during burn-in (10% of the runs) and fixed for the
remainder of the analysis.

Compatibility Notes
-------------------

For *Extra.pCR == 'Update'* in the matlab interface use::

    CR = AdaptiveCrossover(Ncr=MCMCPar.nCR)

For *Extra.pCR != 'Update'* in the matlab interface use::

    CR = Crossover(CR=[1./Ncr], pCR=[1])

"""
from __future__ import division, print_function

__all__ = ["Crossover", "AdaptiveCrossover", "LogAdaptiveCrossover"]

import numpy as np
from numpy import ones, zeros, arange, isscalar, std, trunc, log10, logspace


class Crossover(object):
    """
    Fixed weight crossover ratios.

    *CR* is a scalar if there is a single crossover ratio, or a vector of
    numbers in (0, 1].

    *weight* is the relative weighting of each CR, or None for equal weights.
    """
    def __init__(self, CR, weight=None):
        if isscalar(CR):
            CR, weight = [CR], [1]
        CR, weight = [np.asarray(v, 'd') for v in (CR, weight)]
        self.CR, self.weight = CR, weight/np.sum(weight)

    def reset(self):
        pass

    def update(self, xold, xnew, used):
        """
        Gather adaptation data on *xold*, *xnew* for each CR that was
        *used* in step *N*.
        """
        pass

    def adapt(self):
        """
        Update CR weights based on the available adaptation data.
        """
        pass


class BaseAdaptiveCrossover(object):
    """
    Adapted weight crossover ratios.
    """
    weight = None # type: np.ndarray
    _count = None # type: np.ndarray
    _distance = None # type: np.ndarray
    _generations = 0 # type: int
    _Nchains = 0 # type: int

    def _set_CRs(self, CR):
        self.CR = CR
        # Start with all CRs equally probable
        self.weight = ones(len(self.CR)) / len(self.CR)

        # No initial statistics for adaptation
        self._count = zeros(len(self.CR))
        self._distance = zeros(len(self.CR))
        self._generations = 0
        self._Nchains = 0

    def reset(self):
        # TODO: do we reset count and distance?
        pass

    def update(self, xold, xnew, used):
        """
        Gather adaptation data on *xold*, *xnew* for each CR that was
        *used* in step *N*.
        """
        # Calculate the standard deviation of each dimension of X
        r = std(xnew, ddof=1, axis=0)
        # Compute the Euclidean distance between new X and old X
        d = np.sum(((xold - xnew)/r)**2, axis=1)
        # Use this information to update sum_p2 to update N_CR
        count, total = distance_per_CR(self.CR, d, used)
        self._count += count
        self._distance += total
        self._generations += 1
        # [PAK] Accumulate number of steps so counts can be normalized
        self._Nchains += len(used)

    def adapt(self):
        """
        Update CR weights based on the available adaptation data.
        """
        # [PAK] Protect against completely stuck fits (no accepted updates
        # [PAK] and therefore a total distance of zero) by reusing the
        # [PAK] the existing weights.
        total_distance = np.sum(self._distance)
        if total_distance > 0 and self._Nchains > 0:
            # [PAK] Make sure no count is zero by adding one to all counts
            self.weight = self._distance/(self._count+1)
            self.weight *= self._Nchains/total_distance

        # [PAK] Adjust weights toward uniform; this guarantees nonzero weights.
        self.weight += 0.1*np.sum(self.weight)
        self.weight /= np.sum(self.weight)

class AdaptiveCrossover(BaseAdaptiveCrossover):
    """
    Adapted weight crossover ratios.

    *N* is the number of CRs to use.  CR is set to [1/N, 2/N, ..., 1], with
    initial weights [1/N, 1/N, ..., 1/N].
    """
    def __init__(self, N):
        if N < 2:
            raise ValueError("Need more than one CR for AdaptiveCrossover")
        self._set_CRs((arange(N)+1)/N)  # Equally spaced CRs


# [PAK] Add log spaced adaptive cross-over for high dimensional tightly
# constrained problems.
class LogAdaptiveCrossover(BaseAdaptiveCrossover):
    """
    Adapted weight crossover ratios, log-spaced.

    *dim* is the number of dimensions in the problem.
    *N* is the number of CRs to use per decade.

    CR is set to [k/dim] where k is log-spaced from 1 to dim.
    The CRs start equally weighted as [1, ..., 1]/len(CR).

    *N* should be around 4.5.  This gives good low end density, with 1, 2, 3,
    and 5 parameters changed at a time, and proceeds up to 60% and 100% of
    parameters each time.  Lower values of *N* give too few high density CRs,
    and higher values give too many low density CRs.
    """
    def __init__(self, dim, N=4.5):
        # Log spaced CR from 1/dim to dim/dim
        self._set_CRs(logspace(0, log10(dim), trunc(N*log10(dim)+1))/dim)

def distance_per_CR(available_CRs, distances, used):
    """
    Accumulate normalized Euclidean distance for each crossover value

    Returns the number of times each available CR was used and the total
    distance for that CR.
    """
    # TODO: could use sparse array trick to evaluate totals by CR
    # Set distance[k] to coordinate (k, used[k]), then sum by columns
    # Note: currently setting unused CRs to -1, so this won't work
    total = np.asarray([np.sum(distances[used == p]) for p in available_CRs])
    count = np.asarray([np.sum(used == p) for p in available_CRs])
    return count, total

