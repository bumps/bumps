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


def scipy_stats_density(sample_points, evaluation_points):
    """
    Estimate the probability density function from which a set of sample
    points was drawn and return the estimated density at the evaluation points.
    """
    from scipy.stats import gaussian_kde

    ## standardize data so that we can use uniform bandwidth
    ## Note: this didn't help with singular matrix
    #mu, sigma = mean(data, axis=0), std(data, axis=0)
    #data,points = (data - mu)/sigma, (points - mu)/sigma

    kde = gaussian_kde(sample_points)
    return kde(evaluation_points)


def sklearn_density(sample_points, evaluation_points):
    """
    Estimate the probability density function from which a set of sample
    points was drawn and return the estimated density at the evaluation points.
    """
    from sklearn.neighbors import KernelDensity

    # Silverman bandwidth estimator
    n, d = sample_points.shape
    bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4))

    # standardize data so that we can use uniform bandwidth
    mu, sigma = mean(sample_points, axis=0), std(sample_points, axis=0)
    data, points = (sample_points - mu)/sigma, (evaluation_points - mu)/sigma

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
#density = scipy_stats_density
density = sklearn_density


def entropy(state, N_entropy=10000, N_norm=2500):
    r"""
    Return entropy estimate and uncertainty from an MCMC draw.

    *state* is the MCMC state vector, with sample points and log likelihoods.

    *N_norm* is the number of points $k$ to use to estimate the entropy
    normalization factor $P(D) = \hat N$, converting from $\log( P(D|M) P(M) )$
    to $\log( P(D|M)P(M)/P(D) )$. The relative uncertainty $\Delta\hat S/\hat S$
    scales with $\sqrt{k}$, with the default *N_norm=2500* corresponding to 2%
    relative uncertainty.  Computation cost is $O(nk)$ where $n$ is number of
    points in the draw.

    If *N_entropy* is the number of points used to compute the entropy
    $\hat S = \int P(M|D) \log P(M|D)$ true, use a random draw from state when computing $\hat N$
    rather than the last $k$ points in the draw.
    """
    # Get the sample from the state
    points, logp = state.sample()

    # Use a random subset to estimate density
    if N_norm >= len(logp):
        norm_points = points
    else:
        idx = permutation(len(points))[:N_entropy]
        norm_points = points[idx]

    # Use a different subset to estimate the scale factor between density
    # and logp.
    if N_entropy >= len(logp):
        entropy_points, eval_logp = points, logp
    else:
        idx = permutation(len(points))[:N_entropy]
        entropy_points, eval_logp = points[idx], logp[idx]

    # Normalize p to a peak probability of 1 so that exp() doesn't underflow.
    #
    # This should be okay since for the normalizing constant C:
    #
    #      u' = e^(ln u + ln C) = e^(ln u)e^(ln C) = C u
    #
    # Using eq. 11 with u' substituted for u:
    #
    #      N_est = < u'/p > = < C u/p > = C < u/p >
    #
    #      S_est = - < ln q >
    #            = - < ln (u'/N_est) >
    #            = - < ln C + ln u - ln (C <u/p>) >
    #            = - < ln u + ln C - ln C  - ln <u/p> >
    #            = - < ln u - ln <u/p> >
    #            = - < ln u > + ln <u/p>
    #
    # Uncertainty comes from eq. 13:
    #
    #      N_err^2 = 1/(k-1) sum( (u'/p - <u'/p>)^2 )
    #              = 1/(k-1) sum( (C u/p - <C u/p>)^2 )
    #              = C^2 std(u/p)^2
    #      S_err = std(u'/p) / <u'/p> = (C std(u/p))/(C <u/p>) = std(u/p)/<u/p>
    #
    # So even though the constant C shows up in N_est, N_err, it cancels
    # again when S_est, S_err is formed.
    log_scale = max(eval_logp)
    # print("max log sample: %g"%log_scale)
    eval_logp -= log_scale

    # Compute entropy and uncertainty in nats
    frac = exp(eval_logp)/density(norm_points, entropy_points)
    n_est, n_err = mean(frac), std(frac)
    s_est = (-mean(eval_logp) + log(n_est))
    s_err = n_err/n_est

    # return entropy and uncertainty in bits
    return s_est/LN2, s_err/LN2
