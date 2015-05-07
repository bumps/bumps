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
from __future__ import division

__all__ = ["Crossover", "AdaptiveCrossover", "LogAdaptiveCrossover"]

from numpy import hstack, empty, ones, zeros, cumsum, arange, \
    reshape, array, isscalar, asarray, std, sum, trunc, log10, logspace

from . import util


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
        CR, weight = [asarray(v, 'd') for v in (CR, weight)]
        self.CR, self.weight = CR, weight/sum(weight)

    def reset(self, Nsteps, Npop):
        """
        Generate CR samples for the next Nsteps over a population of size Npop.
        """
        self._CR_samples = gen_CR(self.CR, self.weight, Nsteps, Npop)

    def __getitem__(self, N):
        """
        Return CR samples for step N since reset.
        """
        return self._CR_samples[N]

    def update(self, N, xold, xnew, used):
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
    def _set_CRs(self, CR):
        self.CR = CR
        N = len(CR)
        self.weight = ones(N) / N  # Start with all CRs equally probable
        self._count = zeros(N)     # No statistics for adaptation
        self._distance = zeros(N)

    def reset(self, Nsteps, Npop):
        """
        Generate CR samples for the next Nsteps over a population of size Npop.
        """
        self._CR_samples = gen_CR(self.CR, self.weight, Nsteps, Npop)

    def __getitem__(self, step):
        """
        Return CR samples for step N since reset.
        """
        return self._CR_samples[step]

    def update(self, N, xold, xnew, used):
        """
        Gather adaptation data on *xold*, *xnew* for each CR that was
        *used* in step *N*.
        """
        # Calculate the standard deviation of each dimension of X
        r = std(xnew, ddof=1, axis=0)
        # Compute the Euclidean distance between new X and old X
        d = sum(((xold - xnew)/r)**2, axis=1)
        # Use this information to update sum_p2 to update N_CR
        N, Sd = distance_per_CR(self.CR, d, self._CR_samples[N], used)
        self._distance += Sd
        self._count += N

    def adapt(self):
        """
        Update CR weights based on the available adaptation data.
        """
        Npop = self._CR_samples.shape[1]
        self.weight = (self._distance/self._count) * (Npop/sum(self._distance))
        self.weight /= sum(self.weight)


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


def gen_CR(CR, weight, Nsteps, Npop):
    """
    Generates CR samples for Nsteps generations of size Npop.

    The frequency and value of the samples is based on the CR and weight
    """
    if len(CR) == 1:
        return CR[0] * ones( (Nsteps, Npop) )

    # Determine how many of each CR to use based on the weights
    L = util.rng.multinomial(Nsteps * Npop, weight)

    # Turn this into index boundaries within a CR location vector
    L = hstack((0, cumsum(L)))

    # Generate a random location vector for each CR in the sample
    r = util.rng.permutation(Nsteps * Npop)

    # Fill each location in the sample with the correct CR.
    sample = empty(r.shape)
    for i, v in enumerate(CR):
        # Select a range of elements in r
        idx = r[L[i]:L[i+1]]

        # Fill those elements with crossover ratio v
        sample[idx] = v

    # Now reshape CR
    sample = reshape(sample, (Nsteps, Npop) )

    return sample


def distance_per_CR(available_CRs, distances, CRs, used):
    """
    Accumulate normalized Euclidean distance for each crossover value

    Returns the number of times each available CR was used and the total
    distance for that CR.
    """
    total = array([sum(distances[(CRs==p)&used]) for p in available_CRs])
    count = array([sum((CRs==p)&used) for p in available_CRs])
    return count, total


def demo():
    CR, weight = array([.25, .5, .75, .1]), array([.1, .6, .2, .1])
    print(gen_CR(CR, weight, 5, 4))

if __name__ == "__main__":
    demo()
    # TODO: needs actual tests
