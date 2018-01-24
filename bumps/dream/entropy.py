r"""
Estimate entropy after a fit.

The :func:`entropy` method computes the entropy directly from a set of
MCMC samples, normalized by a scale factor computed from the kernel density
estimate at a subset of the points.\ [#Kramer]_

The :func:`cov_entropy` method computes the entropy associated with the
covariance matrix.  This covariance matrix can be estimated during the
fitting procedure (BFGS updates an estimate of the Hessian matrix for example),
or computed by estimating derivatives when the fit is complete.

The :class:`MVNEntropy` estimates the covariance from an MCMC sample and
uses this covariance to estimate the entropy.  This gives a better
estimate of the entropy than the equivalent direct calculation, which requires
many more samples for a good kernel density estimate.  The *reject_normal*
attribute is *True* if the MCMC sample is significantly different from normal.

.. [#Kramer]
    Kramer, A., Hasenauer, J., Allgower, F., Radde, N., 2010.
    Computation of the posterior entropy in a Bayesian framework
    for parameter estimation in biological networks,
    in: 2010 IEEE International Conference on Control Applications (CCA).
    Presented at the 2010 IEEE International Conference on
    Control Applications (CCA), pp. 493-498.
    doi:10.1109/CCA.2010.5611198


.. [#Turjillo-Ortiz]
    Trujillo-Ortiz, A. and R. Hernandez-Walls. (2003). Mskekur: Mardia's
        multivariate skewness and kurtosis coefficients and its hypotheses
        testing. A MATLAB file. [WWW document].
        `<http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=3519>`_

.. [#Mardia1970]
    Mardia, K. V. (1970), Measures of multivariate skewnees and kurtosis with
        applications. Biometrika, 57(3):519-530.

.. [#Mardia1974]
    Mardia, K. V. (1974), Applications of some measures of multivariate skewness
        and kurtosis for testing normality and robustness studies. Sankhy A,
        36:115-128

.. [#Stevens]
    Stevens, J. (1992), Applied Multivariate Statistics for Social Sciences.
        2nd. ed. New-Jersey:Lawrance Erlbaum Associates Publishers. pp. 247-248.

"""
from __future__ import division

__all__ = ["entropy"]

import numpy as np
from numpy import mean, std, exp, log, max, sqrt, log2, pi, e
from numpy.random import permutation
from scipy.stats import norm, chi2
LN2 = log(2)


def scipy_stats_density(sample_points, evaluation_points):  # pragma: no cover
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

    # Standardize data so that we can use uniform bandwidth.
    # Note that we will need to scale the resulting density by sigma to
    # correct the area.
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
    return exp(log_pdf)/np.prod(sigma)  # undo the x scaling on the data points


# scipy kde fails with singular matrix, so we will use scikit.learn
#density = scipy_stats_density
density = sklearn_density


def entropy(points, logp, N_entropy=10000, N_norm=2500):
    r"""
    Return entropy estimate and uncertainty from a random sample.

    *points* is a set of draws from an underlying distribution, as returned
    by a Markov chain Monte Carlo process for example.

    *logp* is the log-likelihood for each draw.

    *N_norm* is the number of points $k$ to use to estimate the posterior
    density normalization factor $P(D) = \hat N$, converting
    from $\log( P(D|M) P(M) )$ to $\log( P(D|M)P(M)/P(D) )$. The relative
    uncertainty $\Delta\hat S/\hat S$ scales with $\sqrt{k}$, with the
    default *N_norm=2500* corresponding to 2% relative uncertainty.
    Computation cost is $O(nk)$ where $n$ is number of points in the draw.

    *N_entropy* is the number of points used to estimate the entropy
    $\hat S = - \int P(M|D) \log P(M|D)$ from the normalized log likelihood
    values.
    """

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

    """
    # Try again, just using the points from the high probability regions
    # to determine the scale factor
    N_norm = min(len(logp), 5000)
    N_entropy = int(0.8*N_norm)
    idx = np.argsort(logp)
    norm_points = points[idx[-N_norm:]]
    entropy_points = points[idx[-N_entropy:]]
    eval_logp = logp[idx[-N_entropy:]]
    """

    # Normalize p to a peak probability of 1 so that exp() doesn't underflow.
    #
    # This should be okay since for the normalizing constant C:
    #
    #      u' = e^(ln u + ln C) = e^(ln u)e^(ln C) = C u
    #
    # Using eq. 11 of Kramer with u' substituted for u:
    #
    #      N_est = < u'/p > = < C u/p > = C < u/p >
    #
    #      S_est = - < ln q >
    #            = - < ln (u'/N_est) >
    #            = - < ln C + ln u - ln (C <u/p>) >
    #            = - < ln u + ln C - ln C - ln <u/p> >
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
    rho = density(norm_points, entropy_points)
    frac = exp(eval_logp)/rho
    n_est, n_err = mean(frac), std(frac)
    s_est = log(n_est) - mean(eval_logp)
    s_err = n_err/n_est
    #print(n_est, n_err, s_est/LN2, s_err/LN2)
    ##print(np.median(frac), log(np.median(frac))/LN2, log(n_est)/LN2)
    if False:
        import pylab
        idx = pylab.argsort(entropy_points[:, 0])
        pylab.figure()
        pylab.subplot(221)
        pylab.hist(points[:, 0], bins=50, normed=True, log=True)
        pylab.plot(entropy_points[idx, 0], rho[idx], label='density')
        pylab.plot(entropy_points[idx, 0], exp(eval_logp+log_scale)[idx], label='p')
        pylab.ylabel("p(x)")
        pylab.legend()
        pylab.subplot(222)
        pylab.hist(points[:, 0], bins=50, normed=True, log=False)
        pylab.plot(entropy_points[idx, 0], rho[idx], label='density')
        pylab.plot(entropy_points[idx, 0], exp(eval_logp+log_scale)[idx], label='p')
        pylab.ylabel("p(x)")
        pylab.legend()
        pylab.subplot(212)
        pylab.plot(entropy_points[idx, 0], frac[idx], '.')
        pylab.xlabel("P[0] value")
        pylab.ylabel("p(x)/kernel density")

    # return entropy and uncertainty in bits
    return s_est/LN2, s_err/LN2


class MVNEntropy(object):
    """
    Multivariate normal entropy approximation.

    Uses Mardia's multivariate skewness and kurtosis test to estimate normality.

    *x* is a set of points

    *alpha* is the cutoff for the normality test.

    *max_points* is the maximum number of points to use when computing the
    entropy.  Since the normality test is $O(n^2)$ in memory and time,
    where $n$ is the number of points, *max_points* defaults to 1000.

    The returned object has the following attributes:

        *p_kurtosis* is the p-value for the kurtosis normality test

        *p_skewness* is the p-value for the skewness normality test

        *reject_normal* is True if either the the kurtosis or the skew test
        fails

        *entropy* is the estimated entropy of the best normal approximation
        to the distribution

    """
    def __init__(self, x, alpha=0.05, max_points=1000):
        # compute Mardia test coefficient
        n, p = x.shape   # num points, num dimensions
        mu = np.mean(x, axis=0)
        C = np.cov(x.T, bias=1) if p > 1 else np.array([[np.var(x.T, ddof=1)]])
        # squared Mahalanobis distance matrix
        # Note: this forms a full n x n matrix of distances, so will
        # fail for a large number of points.  Kurtosis only requires
        # the diagonal elements so can be computed cheaply.  If there
        # is no order to the points, skew could be estimated using only
        # the block diagonal
        dx = (x - mu[None, :])[:max_points]
        D = np.dot(dx, np.linalg.solve(C, dx.T))
        kurtosis = np.sum(np.diag(D)**2)/n
        skewness = np.sum(D**3)/n**2

        kurtosis_stat = (kurtosis - p*(p+2)) / sqrt(8*p*(p+2)/n)
        raw_skewness_stat = n*skewness/6
        # Small sample correction converges to 1 as n increases, so it is
        # always safe to apply it
        small_sample_correction = (p+1)*(n+1)*(n+3)/((p+1)*(n+1)*n - n*6)
        skewness_stat = raw_skewness_stat * small_sample_correction
        dof = (p*(p+1)*(p+2))/6   # degrees of freedom for chisq test

        self.p_kurtosis = 2*(1 - norm.cdf(abs(kurtosis_stat)))
        self.p_skewness = 1 - chi2.cdf(skewness_stat, dof)
        self.reject_normal = self.p_kurtosis < alpha or self.p_skewness < alpha
        #print("kurtosis", kurtosis, kurtosis_stat, self.p_kurtosis)
        #print("skewness", skewness, skewness_stat, self.p_skewness)
        # compute entropy
        self.entropy = cov_entropy(C)

    def __str__(self):
        return "H=%.1f bits%s"%(self.entropy, " (not normal)" if self.reject_normal else "")

def cov_entropy(C):
    """
    Entropy estimate from covariance matrix C
    """
    return 0.5 * (len(C) * log2(2*pi*e) + log2(abs(np.linalg.det(C))))

def mvn_entropy_test():
    """
    Test against results from the R MVN pacakge (using the web version)
    and the matlab Mskekur program (using Octave), both of which produce
    the same value.  Note that MVNEntropy uses the small sample correction
    for the skewness stat since it converges to the large sample value for
    large n.
    """
    x = np.array([
        [2.4, 2.1, 2.4],
        [4.5, 4.9, 5.7],
        [3.5, 1.8, 3.9],
        [3.9, 4.7, 4.7],
        [6.7, 3.6, 5.9],
        [4.0, 3.6, 2.9],
        [5.3, 3.3, 6.1],
        [5.7, 5.5, 6.2],
        [5.2, 4.1, 6.4],
        [2.4, 2.9, 3.2],
        [3.2, 2.7, 4.0],
        [2.7, 2.6, 4.1],
    ])
    M = MVNEntropy(x)
    #print M
    #print "%.15g %.15g %.15g"%(M.p_kurtosis, M.p_skewness, M.entropy)
    assert abs(M.p_kurtosis - 0.265317890462476) <= 1e-10
    assert abs(M.p_skewness - 0.773508066109368) <= 1e-10
    assert abs(M.entropy - 5.7920040570988) <= 1e-10


def _check_entropy(D, seed=1, N=10000, N_entropy=10000, N_norm=2500):
    """
    Check if entropy from a random draw matches analytic entropy.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        theta = D.rvs(size=N)
        logp_theta = D.logpdf(theta)
        logp_theta += 27  # result should be independent of scale factor
        if getattr(D, 'dim', 1) == 1:
            theta = theta.reshape(N, 1)
        S, Serr = entropy(theta, logp_theta, N_entropy=N_entropy, N_norm=N_norm)
    finally:
        np.random.set_state(state)
    #print "entropy", S, Serr, "target", D.entropy()/LN2
    assert Serr < 0.05*S
    assert abs(S - D.entropy()/LN2) < Serr

def test():
    """check entropy estimates from known distributions"""
    from scipy import stats
    _check_entropy(stats.norm(100, 8), N=2000)
    _check_entropy(stats.norm(100, 8), N=12000)
    _check_entropy(stats.multivariate_normal(cov=np.diag([1, 12**2, 0.2**2])))

if __name__ == "__main__":  # pragma: no cover
    test()
    mvn_entropy_test()
