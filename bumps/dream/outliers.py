"""
Chain outlier tests.
"""

__all__ = ["identify_outliers"]

import os

from numpy import mean, std, sqrt, where, argmin, arange, array
from numpy import sort
from scipy.stats import t as student_t
from scipy.stats import scoreatpercentile

from .mahal import mahalanobis
from .acr import ACR

tinv = student_t.ppf

# Hack to adjust the aggressiveness of the interquartile range test.
# TODO: Document/remove BUMPS_INTERQUARTILES or replace with command option.
BUMPS_INTERQUARTILES = float(os.environ.get("BUMPS_INTERQUARTILES", "2.0"))


# CRUFT: scoreatpercentile not accepting array arguments in older scipy
def prctile(v, Q):
    v = sort(v)
    return [scoreatpercentile(v, Qi) for Qi in Q]


def identify_outliers(test, llf, x=None):
    """
    Determine which chains have converged on a local maximum much lower than
    the maximum likelihood.

    *test* is the name of the test to use (one of IQR, Grubbs, Mahal or none).
    IQR rejects any chains with mean log likelihood more than than twice the
    inter-quartile range below the value of the 25% quartile.  The Grubbs
    method uses a t-test to determine which chains have a mean log likelihood
    extremely far below the mean across all the chains.  The Mahal test looks
    at the head of the chain with the worst mean log likelihood and marks
    it as an outlier if it is far from the centroid of the population.  This
    assumes that the posterior is approximately gaussian, which is not true
    in general.

    *llf* is a set of log likelihood values for all chains, which is
    an array of shape (chain len, num chains)

    *x* is the current population with one point for each each, which is
    an array of shape (num chains, num vars).  This is only used for the
    Mahal test.

    Returns an integer array of outlier indices.
    """

    # Check whether any of these active chains are outlier chains
    test = test.lower()
    if test == "iqr":
        # Determine the mean log density of the active chains
        v = mean(llf, axis=0)
        # Derive the upper and lower quartile of the chain averages
        q1, q3 = prctile(v, [25.0, 75.0])
        # Derive the Inter Quartile Range (IQR)
        iqr = q3 - q1
        # See whether there are any outlier chains
        # 2017-10-06 [PAK] test against chain max rather than chain mean.
        # Chains wandering inside the active region should not be punished.
        # Since removing outliers will delay convergence tests until the
        # outlier removal effects have disappeared (i.e., an entire frame),
        # a less aggressive test will speed completion.
        vmax = llf.max(axis=0)
        outliers = where(vmax < q1 - BUMPS_INTERQUARTILES * iqr)[0]

    elif test == "grubbs":
        # Determine the mean log density of the active chains
        v = mean(llf, axis=0)
        # Compute zscore for chain averages
        zscore = (mean(v) - v) / std(v, ddof=1)
        # Determine t-value of one-sided interval
        n = len(v)
        t2 = tinv(1 - 0.01 / n, n - 2) ** 2  # 95% interval
        # Determine the critical value
        gcrit = ((n - 1) / sqrt(n)) * sqrt(t2 / (n - 2 + t2))
        # Then check against this
        outliers = where(zscore > gcrit)[0]

    elif test == "mahal":
        # Determine the mean log density of the active chains
        v = mean(llf, axis=0)
        # Find which chain has minimum log_density
        minidx = argmin(v)
        # Use the Mahalanobis distance to find outliers in the population
        alpha = 0.01
        npop, nvar = x.shape
        gcrit = ACR(nvar, npop - 1, alpha)
        # print "alpha", alpha, "nvar", nvar, "npop", npop, "gcrit", gcrit
        # check the Mahalanobis distance of the current point to other chains
        d1 = mahalanobis(x[minidx, :], x[minidx != arange(npop), :])
        # print "d1", d1, "minidx", minidx
        # and see if it is an outlier
        outliers = array([minidx]) if d1 > gcrit else array([])

    elif test == "none":
        outliers = array([])

    else:
        raise ValueError("Unknown outlier test " + test)

    return outliers


def test_outliers():
    from .walk import walk
    from numpy.random import multivariate_normal, seed
    from numpy import vstack, ones, eye

    seed(2)  # Remove uncertainty on tests
    # Set a number of good and bad chains
    ngood, nbad = 25, 2

    # Make chains mean-reverting chains with widely separated values for
    # bad and good; put bad chains first.
    chains = walk(1000, mu=[1] * nbad + [5] * ngood, sigma=0.45, alpha=0.1)

    # Check IQR and Grubbs
    assert (identify_outliers("IQR", chains, None) == arange(nbad)).all()
    assert (identify_outliers("Grubbs", chains, None) == arange(nbad)).all()

    # Put points for 'bad' chains at [-1,...,-1] and 'good' chains at [1,...,1]
    x = vstack(
        (multivariate_normal(-ones(4), 0.1 * eye(4), size=nbad), multivariate_normal(ones(4), 0.1 * eye(4), size=ngood))
    )
    assert identify_outliers("Mahal", chains, x)[0] in range(nbad)

    # Put points for _all_ chains at [1,...,1] and check that mahal return []
    xsame = multivariate_normal(ones(4), 0.2 * eye(4), size=ngood + nbad)
    assert len(identify_outliers("Mahal", chains, xsame)) == 0

    # Check again with large variance
    x = vstack(
        (multivariate_normal(-3 * ones(4), eye(4), size=nbad), multivariate_normal(ones(4), 10 * eye(4), size=ngood))
    )
    assert len(identify_outliers("Mahal", chains, x)) == 0

    # =====================================================================
    # Test replacement

    # Construct a state object
    from numpy.linalg import norm
    from .state import MCMCDraw

    ngen, npop = chains.shape
    npop, nvar = x.shape
    state = MCMCDraw(Ngen=ngen, Nthin=ngen, Nupdate=0, Nvar=nvar, Npop=npop, Ncr=0, thinning=0)
    # Fill it with chains
    for i in range(ngen):
        state._generation(new_draws=npop, x=x, logp=chains[i], accept=npop)

    # Make a copy of the current state so we can check it was updated
    nx, nlogp = x + 0, chains[-1] + 0
    # Remove outliers
    state.remove_outliers(nx, nlogp, test="IQR")
    # Check that the outliers were removed
    outliers = state.outliers()
    assert outliers.shape[0] == nbad
    for i in range(nbad):
        assert nlogp[outliers[i, 1]] == chains[-1][outliers[i, 2]]
        assert nx[outliers[i, 1], -1] == x[outliers[i, 2], -1]


if __name__ == "__main__":
    test_outliers()
