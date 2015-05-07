"""
Estimate entropy from an MCMC state vector.

Uses probabilities computed by the MCMC sampler normalized by a scale factor
computed from the kernel density estimate at a subset of the points.\ [#Kramer]_

.. [#Kramer]
    Kramer, A., Hasenauer, J., Allgower, F., Radde, N., 2010.
    Computation of the posterior entropy in a Bayesian framework
    for parameter estimation in biological networks,
    in: 2010 IEEE International Conference on Control Applications (CCA).
    Presented at the 2010 IEEE International Conference on
    Control Applications (CCA), pp. 493-498.
    doi:10.1109/CCA.2010.5611198
"""

__all__ = ["entropy"]

from numpy import mean, std, exp, log, max
from numpy.random import permutation
LN2 = log(2)


def scipy_stats_kde(data, points):
    from scipy.stats import gaussian_kde

    ## standardize data so that we can use uniform bandwidth
    ## Note: this didn't help with singular matrix
    #mu, sigma = mean(data, axis=0), std(data, axis=0)
    #data,points = (data - mu)/sigma, (points - mu)/sigma

    kde = gaussian_kde(data)
    return kde(points)


def sklearn_kde(data, points):
    from sklearn.neighbors import KernelDensity

    # Silverman bandwidth estimator
    n, d = data.shape
    bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4))

    # standardize data so that we can use uniform bandwidth
    mu, sigma = mean(data, axis=0), std(data, axis=0)
    data, points = (data - mu)/sigma, (points - mu)/sigma

    #print("starting grid search for bandwidth over %d points"%n)
    #from sklearn.grid_search import GridSearchCV
    #from numpy import logspace
    #params = {'bandwidth': logspace(-1, 1, 20)}
    #fitter = GridSearchCV(KernelDensity(), params)
    #fitter.fit(data)
    #kde = fitter.best_estimator_
    #print("best bandwidth: {0}".format(kde.bandwidth))
    #import time; T0 = time.time()
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth,
                        rtol=1e-6, atol=1e-6)
    #print("T:%6.3f   fitting"%(time.time()-T0))
    kde.fit(data)
    #print("T:%6.3f   estimating"%(time.time()-T0))
    log_pdf = kde.score_samples(points)
    #print("T:%6.3f   done"%(time.time()-T0))
    return exp(log_pdf)


# scipy kde fails with singular matrix, so we will use scikit.learn
density = sklearn_kde
#density = scipy_stats_kde


def entropy(state, N_data=10000, N_sample=2500):
    r"""
    Return entropy estimate and uncertainty from an MCMC draw.

    *state* is the MCMC state vector, with sample points and log likelihoods.

    *N_size* is the number of points $k$ to use to estimate the entropy
    normalization factor $P(D) = \hat N$, converting from $\log( P(D|M) P(M) )$
    to $\log( P(D|M)P(M)/P(D) )$. The relative uncertainty $\Delta\hat S/\hat S$
    scales with $\sqrt{k}$, with the default *N_size=2500* corresponding to 2%
    relative uncertainty.  Computation cost is $O(nk)$ where $n$ is number of
    points in the draw.

    If *N_random* is true, use a random draw from state when computing $\hat N$
    rather than the last $k$ points in the draw.
    """
    # Get the sample from the state
    points, logp = state.sample()

    # Use a random subset to estimate density
    if N_data >= len(logp):
        data = points
    else:
        idx = permutation(len(points))[:N_data]
        data = points[idx]

    # Use a different subset to estimate the scale factor between density
    # and logp.
    if N_sample >= len(logp):
        sample, logp_sample = points, logp
    else:
        idx = permutation(len(points))[:N_sample]
        sample, logp_sample = points[idx], logp[idx]

    # normalize logp to a peak probability of 1 so that exp() doesn't underflow
    logp_sample -= max(logp_sample)

    # Compute entropy and uncertainty in nats
    frac = exp(logp_sample)/density(data, sample)
    n_est, n_err = mean(frac), std(frac)
    s_est = (-mean(logp_sample) + log(n_est))
    s_err = n_err/n_est

    # return entropy and uncertainty in bits
    return s_est/LN2, s_err/LN2
