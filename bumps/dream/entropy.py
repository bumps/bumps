r"""
Estimate entropy after a fit.

The :func:`gmm_entropy` function computes the entropy from a Gaussian mixture
model. This provides a reasonable estimate even for non-Gaussian distributions.
This is the recommended method for estimating the entropy of a sample.

The :func:`cov_entropy` method computes the entropy associated with the
covariance matrix.  This covariance matrix can be estimated during the
fitting procedure (BFGS updates an estimate of the Hessian matrix for example),
or computed by estimating derivatives when the fit is complete.

The :class:`MVNEntropy` class estimates the covariance from an MCMC sample and
uses this covariance to estimate the entropy.  This gives a better
estimate of the entropy than the equivalent direct calculation, which requires
many more samples for a good kernel density estimate.  The *reject_normal*
attribute is *True* if the MCMC sample is significantly different from normal.
Unfortunately, this almost always the case for any reasonable sample size that
isn't strictly gaussian.

The :func:`entropy` function computes the entropy directly from a set
of MCMC samples, normalized by a scale factor computed from the kernel density
estimate at a subset of the points.\ [#Kramer]_

There are many other entropy calculations implemented within this file, as
well as a number of sampling distributions for which the true entropy is known.
Furthermore, entropy was computed against dream output and checked for
consistency. None of the methods is truly excellent in terms of minimum
sample size, maximum dimensions and speed, but many of them are pretty
good.

The following is an informal summary of the results from different algorithms
applied to DREAM output::

        from .entropy import Timer as T

        # Try MVN ... only good for normal distributions, but very fast
        with T(): M = entropy.MVNEntropy(drawn.points)
        print("Entropy from MVN: %s"%str(M))

        # Try wnn ... no good.
        with T(): S_wnn, Serr_wnn = entropy.wnn_entropy(drawn.points, n_est=20000)
        print("Entropy from wnn: %s"%str(S_wnn))

        # Try wnn with bootstrap ... still no good.
        with T(): S_wnn, Serr_wnn = entropy.wnn_bootstrap(drawn.points)
        print("Entropy from wnn bootstrap: %s"%str(S_wnn))

        # Try wnn entropy with thinning ... still no good.
        #drawn = self.draw(portion=portion, vars=vars,
        #                  selection=selection, thin=10)
        with T(): S_wnn, Serr_wnn = entropy.wnn_entropy(points)
        print("Entropy from wnn: %s"%str(S_wnn))

        # Try wnn with gmm ... still no good
        with T(): S_wnn, Serr_wnn = entropy.wnn_entropy(drawn.points, n_est=20000, gmm=20)
        print("Entropy from wnn with gmm: %s"%str(S_wnn))

        # Try pure gmm ... pretty good
        with T(): S_gmm, Serr_gmm = entropy.gmm_entropy(drawn.points, n_est=10000)
        print("Entropy from gmm: %s"%str(S_gmm))

        # Try kde from statsmodels ... pretty good
        with T(): S_kde_stats = entropy.kde_entropy_statsmodels(drawn.points, n_est=10000)
        print("Entropy from kde statsmodels: %s"%str(S_kde_stats))

        # Try kde from sklearn ... pretty good
        with T(): S_kde = entropy.kde_entropy_sklearn(drawn.points, n_est=10000)
        print("Entropy from kde sklearn: %s"%str(S_kde))

        # Try kde from sklearn at points from gmm ... pretty good
        with T(): S_kde_gmm = entropy.kde_entropy_sklearn_gmm(drawn.points, n_est=10000)
        print("Entropy from kde+gmm: %s"%str(S_kde_gmm))

        # Try Kramer ... pretty good, but doesn't support marginal entropy
        with T(): S, Serr = entropy.entropy(drawn.points, drawn.logp, N_entropy=n_est)
        print("Entropy from Kramer: %s"%str(S))


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

__all__ = ["entropy", "gmm_entropy", "cov_entropy", "wnn_entropy", "MVNEntropy"]

import numpy as np
from numpy import e, exp, log, log2, mean, nan, pi, sqrt, std
from numpy.random import choice, permutation
from scipy import stats
from scipy.special import digamma, gammaln
from scipy.stats import chi2, norm

LN2 = log(2)


def standardize(x):
    """
    Standardize the points by removing the mean and scaling by the standard
    deviation.
    """
    # TODO: check if it is better to multiply by inverse covariance
    # That would serve to unrotate and unscale the dimensions together,
    # but squishing them down individually might be just as good.

    # compute zscores for the each variable independently
    mu, sigma = mean(x, axis=0), std(x, axis=0, ddof=1)
    # Protect against NaN when sigma is zero.  If sigma is zero
    # then all points are equal, so x == mu and z-score is zero.
    return (x - mu) / (sigma + (sigma == 0.0)), mu, sigma


def kde_entropy_statsmodels(points, n_est=None):
    """
    Use statsmodels KDEMultivariate pdf to estimate entropy.

    Density evaluated at sample points.

    Slow and fails for bimodal, dirichlet; poor for high dimensional MVN.
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate

    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
    else:
        x = points[permutation(n)[:n_est]]
        n = n_est

    predictor = KDEMultivariate(data=x, var_type="c" * d)
    p = predictor.pdf()
    H = -np.mean(log(p))
    return H / LN2


def kde_entropy_sklearn(points, n_est=None):
    """
    Use sklearn.neigbors.KernelDensity pdf to estimate entropy.

    Data is standardized before analysis.

    Sample points drawn from the kernel density estimate.

    Fails for bimodal and dirichlet, similar to statsmodels kde.
    """
    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
    else:
        x = points[permutation(n)[:n_est]]
        n = n_est

    # logp = sklearn_log_density(points, evaluation_points=n_est)
    logp = sklearn_log_density(x, evaluation_points=x)
    H = -np.mean(logp)
    return H / LN2


def kde_entropy_sklearn_gmm(points, n_est=None, n_components=None):
    """
    Use sklearn.neigbors.KernelDensity pdf to estimate entropy.

    Data is standardized before kde.

    Sample points drawn from gaussian mixture model from original points.

    Fails for bimodal and dirichlet, similar to statsmodels kde.
    """
    from sklearn.mixture import BayesianGaussianMixture as GMM

    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
    else:
        x = points[permutation(n)[:n_est]]
        n = n_est

    if n_components is None:
        n_components = int(5 * sqrt(d))

    predictor = GMM(
        n_components=n_components,
        covariance_type="full",
        # verbose=True,
        max_iter=1000,
    )
    predictor.fit(x)
    evaluation_points, _ = predictor.sample(n_est)

    logp = sklearn_log_density(x, evaluation_points=evaluation_points)
    H = -np.mean(logp)
    return H / LN2


def gmm_entropy(points, n_est=None, n_components=None):
    r"""
    Use sklearn.mixture.BayesianGaussianMixture to estimate entropy.

    *points* are the data points in the sample.

    *n_est* are the number of points to use in the estimation; default is
    10,000 points, or 0 for all the points.

    *n_components* are the number of Gaussians in the mixture. Default is
    $5 \sqrt{d}$ where $d$ is the number of dimensions.

    Returns estimated entropy and uncertainty in the estimate.

    This method uses BayesianGaussianMixture from scikit-learn to build a
    model of the point distribution, then uses Monte Carlo sampling to
    determine the entropy of that distribution. The entropy uncertainty is
    computed from the variance in the MC sample scaled by the number of
    samples. This does not incorporate any uncertainty in the sampling that
    generated the point distribution or the uncertainty in the GMM used to
    model that distribution.
    """
    # from sklearn.mixture import GaussianMixture as GMM
    from sklearn.mixture import BayesianGaussianMixture as GMM

    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = 10000
    elif n_est == 0:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
        n_est = n
    else:
        x = points[permutation(n)[:n_est]]
        n = n_est

    if n_components is None:
        n_components = int(5 * sqrt(d))

    ## Standardization doesn't seem to help
    ## Note: sigma may be zero
    # x, mu, sigma = standardize(x)   # if standardized
    predictor = GMM(
        n_components=n_components,
        covariance_type="full",
        # verbose=True,
        max_iter=1000,
    )
    predictor.fit(x)
    eval_x, _ = predictor.sample(n_est)
    weight_x = predictor.score_samples(eval_x)
    H = -np.mean(weight_x)
    # with np.errstate(divide='ignore'): H = H + np.sum(np.log(sigma))   # if standardized
    dH = np.std(weight_x, ddof=1) / sqrt(n)
    ## cross-check against own calcs
    # alt = GaussianMixture(predictor.weights_, mu=predictor.means_, sigma=predictor.covariances_)
    # print("alt", H, alt.entropy())
    # print(np.vstack((weight_x[:10], alt.logpdf(eval_x[:10]))).T)
    return H / LN2, dH / LN2


def wnn_bootstrap(points, k=None, weights=True, n_est=None, reps=10, parts=10):
    # raise NotImplementedError("deprecated; bootstrap doesn't help.")
    n, d = points.shape
    if n_est is None:
        n_est = n // parts

    results = [wnn_entropy(points, k=k, weights=weights, n_est=n_est) for _ in range(reps)]
    # print(results)
    S, Serr = list(zip(*results))
    return np.mean(S), np.std(S)


def wnn_entropy(points, k=None, weights=True, n_est=None, gmm=None):
    r"""
    Weighted Kozachenko-Leonenko nearest-neighbour entropy calculation.

    *k* is the number of neighbours to consider, with default $k=n^{1/3}$

    *n_est* is the number of points to use for estimating the entropy,
    with default $n_\rm{est} = n$

    *weights* is True for default weights, False for unweighted (using the
    distance to the kth neighbour only), or a vector of weights of length *k*.

    *gmm* is the number of gaussians to use to model the distribution using
    a gaussian mixture model.  Default is 0, and the points represent an
    empirical distribution.

    Returns entropy H in bits and its uncertainty.

    Berrett, T. B., Samworth, R.J., Yuan, M., 2016. Efficient multivariate
    entropy estimation via k-nearest neighbour distances.
    DOI:10.1214/18-AOS1688 https://arxiv.org/abs/1606.00304
    """
    from sklearn.neighbors import NearestNeighbors

    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = 10000
    elif n_est == 0:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
        n_est = n
    else:
        x = points[permutation(n)[:n_est]]
        n = n_est

    # Default k based on n
    if k is None:
        # Private communication: cube root of n is a good choice for k
        # Personal observation: k should be much bigger than d
        k = max(int(n ** (1 / 3)), 3 * d)

    # If weights are given then use them (setting the appropriate k),
    # otherwise use the default weights.
    if isinstance(weights, bool):
        weights = _wnn_weights(k, d, weights)
    else:
        k = len(weights)
    # print("weights", weights, sum(weights))

    # select knn algorithm
    algorithm = "auto"
    # algorithm = 'kd_tree'
    # algorithm = 'ball_tree'
    # algorithm = 'brute'

    n_components = 0 if gmm is None else gmm

    # H = 1/n sum_i=1^n sum_j=1^k w_j log E_{j,i}
    # E_{j,i} = e^-Psi(j) V_d (n-1) z_{j,i}^d = C z^d
    # logC = -Psi(j) + log(V_d) + log(n-1)
    # H = 1/n sum sum w_j logC + d/n sum sum w_j log(z)
    #   = sum w_j logC + d/n sum sum w_j log(z)
    #   = A + d/n B
    # H^2 = 1/n sum
    Psi = digamma(np.arange(1, k + 1))
    logVd = d / 2 * log(pi) - gammaln(1 + d / 2)
    logC = -Psi + logVd + log(n - 1)

    # TODO: standardizing points doesn't work.
    # Standardize the data so that distances conform.  This is equivalent to
    # a u-substitution u = sigma x + mu, so the integral needs to be corrected
    # for dU = det(sigma) dx.  Since the standardization squishes the dimensions
    # independently, sigma is a diagonal matrix, with the determinant equal to
    # the product of the diagonal elements.
    # x, mu, sigma = standardize(x)  # Note: sigma may be zero
    # detDU = np.prod(sigma)
    detDU = 1.0

    if n_components > 0:
        # Use Gaussian mixture to model the distribution
        from sklearn.mixture import GaussianMixture as GMM

        predictor = GMM(n_components=gmm, covariance_type="full")
        predictor.fit(x)
        eval_x, _ = predictor.sample(n_est)
        # weight_x = predictor.score_samples(eval_x)
        skip = 0
    else:
        # Empirical distribution
        # TODO: should we use the full draw for kNN and a subset for eval points?
        # Choose a subset for evaluating the entropy estimate, if desired
        # print(n_est, n)
        # eval_x = x if n_est >= n else x[permutation(n)[:n_est]]
        eval_x = x
        # weight_x = 1
        skip = 1

    tree = NearestNeighbors(algorithm=algorithm, n_neighbors=k + skip)
    tree.fit(x)
    dist, _ind = tree.kneighbors(eval_x, n_neighbors=k + skip, return_distance=True)
    # Remove first column. Since test points are in x, the first column will
    # be a point from x with distance 0, and can be ignored.
    if skip:
        dist = dist[:, skip:]
    # Find log distances.  This can be problematic for MCMC runs where a
    # step is rejected, and therefore identical points are in the distribution.
    # Ignore them by replacing these points with nan and using nanmean.
    # TODO: need proper analysis of duplicated points in MCMC chain
    dist[dist == 0] = nan
    logdist = log(dist)
    H_unweighted = logC + d * np.nanmean(logdist, axis=0)
    H = np.dot(H_unweighted, weights)[0]
    Hsq_k = np.nanmean((logC[-1] + d * logdist[:, -1]) ** 2)
    # TODO: abs shouldn't be needed?
    if Hsq_k < H**2:
        print("warning: avg(H^2) < avg(H)^2")
    dH = sqrt(abs(Hsq_k - H**2) / n_est)
    # print("unweighted", H_unweighted)
    # print("weighted", H, Hsq_k, H**2, dH, detDU, LN2)
    return H * detDU / LN2, dH * detDU / LN2


def _wnn_weights(k, d, weighted=True):
    # Private communication: ignore w_j = 0 constraints (they are in the
    # paper for mathematical nicety), and find the L2 norm of the
    # remaining underdeterimined system described in Eq 2.
    # Personal observation: k should be some small multiple of d
    # otherwise the weights blow up.
    if d < 4 or not weighted:
        # with few dimensions go unweighted with the kth nearest neighbour.
        return np.array([[0.0] * (k - 1) + [1.0]]).T
    j = np.arange(1, k + 1)
    sum_zero = [exp(gammaln(j + 2 * i / d) - gammaln(j)) for i in range(1, d // 4 + 1)]
    sum_one = [[1.0] * k]
    A = np.array(sum_zero + sum_one)
    b = np.array([[0.0] * (d // 4) + [1.0]]).T
    return np.dot(np.linalg.pinv(A), b)


def scipy_stats_density(sample_points, evaluation_points):  # pragma: no cover
    """
    Estimate the probability density function from which a set of sample
    points was drawn and return the estimated density at the evaluation points.
    """
    ## standardize data so that we can use uniform bandwidth
    ## Note: this didn't help with singular matrix
    ## Note: if re-enable, protect against sigma=0 in some dimensions
    # mu, sigma = mean(data, axis=0), std(data, axis=0)
    # data,points = (data - mu)/sigma, (points - mu)/sigma

    kde = stats.gaussian_kde(sample_points)
    return kde(evaluation_points)


def sklearn_log_density(sample_points, evaluation_points):
    """
    Estimate the log probability density function from which a set of sample
    points was drawn and return the estimated density at the evaluation points.

    *sample_points* is an [n x m] matrix.

    *evaluation_points* is the set of points at which to evaluate the kde.

    Note: if any dimension has all points equal then the entire distribution
    is treated as a dirac distribution with infinite density at each point.
    This makes the entropy calculation better behaved (narrowing the
    distribution increases the entropy) but is not so useful in other contexts.
    Other packages will (correctly) ignore dimensions of width zero.
    """
    # Ugly hack warning: if *evaluation_points* is an integer, then sample
    # that many points from the kde and return the log density at each
    # sampled point.  Since the code that uses this is looking only at
    # the mean log density, it doesn't need the sample points themselves.
    # This interface should be considered internal to the entropy module
    # and not used by outside functions.  If you need it externally, then
    # restructure the api so that the function always returns both the
    # points and the density, as well as any other function (such as the
    # denisty function and the sister function scipy_stats_density) so
    # that all share the new interface.

    from sklearn.neighbors import KernelDensity

    # Standardize data so we can use spherical kernels and uniform bandwidth
    data, mu, sigma = standardize(sample_points)

    # Note that sigma will be zero for dimensions w_o where all points are equal.
    # With P(w) = P(w, w_o) / P(w_o | w) and P(w_o) = 1 for all points in
    # the set, then P(w) = P(w, w_o) and we can ignore the zero dimensions.
    # However, as another ugly hack, we want the differential entropy to go
    # to -inf as the distribution narrows, so pretend that P = 0 everywhere.
    # Uncomment the following line to return the sample probability instead.
    ## sigma[sigma == 0.] = 1.

    # Silverman bandwidth estimator
    n, d = sample_points.shape
    bandwidth = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

    # print("starting grid search for bandwidth over %d points"%n)
    # from sklearn.grid_search import GridSearchCV
    # from numpy import logspace
    # params = {'bandwidth': logspace(-1, 1, 20)}
    # fitter = GridSearchCV(KernelDensity(), params)
    # fitter.fit(data)
    # kde = fitter.best_estimator_
    # print("best bandwidth: {0}".format(kde.bandwidth))
    # import time; T0 = time.time()
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, rtol=1e-6, atol=1e-6)
    kde.fit(data)

    if isinstance(evaluation_points, int):
        # For generated points, they already follow the distribution
        points = kde.sample(n)
    elif evaluation_points is not None:
        # Standardized evaluation points to match sample distribution
        # Note: for dimensions where all sample points are equal, sigma
        # has been artificially set equal to one.  This means that the
        # evaluation points which do not match the sample value will
        # use the simple differences for the z-score rather than
        # pushing them out to plus/minus infinity.
        points = (evaluation_points - mu) / (sigma + (sigma == 0.0))
    else:
        points = sample_points

    # Evaluate pdf, scaling the resulting density by sigma to correct the area.
    # If sigma is zero, return entropy as -inf;  this seems to not be the
    # case for discrete distributions (consider Bernoulli with p=1, q=0,
    #  => H = -p log p - q log q = 0), so need to do something else, both
    # for the kde and for the entropy calculation.
    with np.errstate(divide="ignore"):
        log_pdf = kde.score_samples(points) - np.sum(np.log(sigma))

    return log_pdf


def sklearn_density(sample_points, evaluation_points):
    """
    Estimate the probability density function from which a set of sample
    points was drawn and return the estimated density at the evaluation points.
    """
    return exp(sklearn_log_density(sample_points, evaluation_points))


# scipy kde fails with singular matrix, so we will use scikit.learn
# density = scipy_stats_density
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
    if N_entropy is None:
        N_entropy = 10000
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
    log_scale = np.max(eval_logp)
    # print("max log sample: %g"%log_scale)
    eval_logp -= log_scale

    # Compute entropy and uncertainty in nats
    # Note: if all values are the same in any dimension then we have a dirac
    # functional with infinite probability at every sample point, and the
    # differential entropy estimate will yield H = -inf.
    rho = density(norm_points, entropy_points)
    # print(rho.min(), rho.max(), eval_logp.min(), eval_logp.max())
    frac = exp(eval_logp) / rho
    n_est, n_err = mean(frac), std(frac)
    if n_est == 0.0:
        s_est, s_err = -np.inf, 0.0
    else:
        s_est = log(n_est) - mean(eval_logp)
        s_err = n_err / n_est
    # print(n_est, n_err, s_est/LN2, s_err/LN2)
    ##print(np.median(frac), log(np.median(frac))/LN2, log(n_est)/LN2)
    if False:
        import pylab

        idx = pylab.argsort(entropy_points[:, 0])
        pylab.figure()
        pylab.subplot(221)
        pylab.hist(points[:, 0], bins=50, density=True, log=True)
        pylab.plot(entropy_points[idx, 0], rho[idx], label="density")
        pylab.plot(entropy_points[idx, 0], exp(eval_logp + log_scale)[idx], label="p")
        pylab.ylabel("p(x)")
        pylab.legend()
        pylab.subplot(222)
        pylab.hist(points[:, 0], bins=50, density=True, log=False)
        pylab.plot(entropy_points[idx, 0], rho[idx], label="density")
        pylab.plot(entropy_points[idx, 0], exp(eval_logp + log_scale)[idx], label="p")
        pylab.ylabel("p(x)")
        pylab.legend()
        pylab.subplot(212)
        pylab.plot(entropy_points[idx, 0], frac[idx], ".")
        pylab.xlabel("P[0] value")
        pylab.ylabel("p(x)/kernel density")

    # return entropy and uncertainty in bits
    return s_est / LN2, s_err / LN2


class MVNEntropy(object):
    """
    Multivariate normal entropy approximation.

    Uses Mardia's multivariate skewness and kurtosis test to estimate normality.

    *x* is a set of points

    *alpha* is the cutoff for the normality test.

    *max_points* is the maximum number of points to use when checking
    normality.  Since the normality test is $O(n^2)$ in memory and time,
    where $n$ is the number of points, *max_points* defaults to 1000. The
    entropy is computed from the full dataset.

    The returned object has the following attributes:

        *p_kurtosis* is the p-value for the kurtosis normality test

        *p_skewness* is the p-value for the skewness normality test

        *reject_normal* is True if either the the kurtosis or the skew test
        fails

        *entropy* is the estimated entropy of the best normal approximation
        to the distribution

    """

    # TODO: use robust covariance estimator for mean and covariance
    # FastMSD is available in sklearn.covariance.MinDetCov. There are methods
    # such as (Zhahg, 2012), which may be faster if performance is an issue.
    # [1] Zhang (2012) DOI: 10.5539/ijsp.v1n2p119
    def __init__(self, x, alpha=0.05, max_points=1000):
        # compute Mardia test coefficient
        n, p = x.shape  # num points, num dimensions
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
        kurtosis = np.sum(np.diag(D) ** 2) / n
        skewness = np.sum(D**3) / n**2

        kurtosis_stat = (kurtosis - p * (p + 2)) / sqrt(8 * p * (p + 2) / n)
        raw_skewness_stat = n * skewness / 6
        # Small sample correction converges to 1 as n increases, so it is
        # always safe to apply it
        small_sample_correction = (p + 1) * (n + 1) * (n + 3) / ((p + 1) * (n + 1) * n - n * 6)
        skewness_stat = raw_skewness_stat * small_sample_correction
        dof = (p * (p + 1) * (p + 2)) / 6  # degrees of freedom for chisq test

        self.p_kurtosis = 2 * (1 - norm.cdf(abs(kurtosis_stat)))
        self.p_skewness = 1 - chi2.cdf(skewness_stat, dof)
        self.reject_normal = self.p_kurtosis < alpha or self.p_skewness < alpha
        # print("kurtosis", kurtosis, kurtosis_stat, self.p_kurtosis)
        # print("skewness", skewness, skewness_stat, self.p_skewness)
        # compute entropy
        self.entropy = cov_entropy(C)

    def __str__(self):
        return "H=%.1f bits%s" % (self.entropy, " (not normal)" if self.reject_normal else "")


def cov_entropy(C):
    """
    Entropy estimate from covariance matrix C
    """
    return 0.5 * (len(C) * log2(2 * pi * e) + log2(abs(np.linalg.det(C))))


def mvn_entropy_bootstrap(points, samples=50):
    """
    Use bootstrap method to estimate entropy and its uncertainty
    """
    n, d = points.shape

    results = []
    for _ in range(samples):
        # sample n points with replacement in 0 ... n-1.
        x = points[choice(n, size=n)]
        C = np.cov(x.T, bias=1) if d > 1 else np.array([[np.var(x.T, ddof=1)]])
        # print(f"cov {samples}, {x.shape}, {C.shape}")
        results.append(cov_entropy(C))

    return np.mean(results), np.std(results)


# ======================================================================
# Testing code
# ======================================================================

# Based on: Eli Bendersky https://stackoverflow.com/a/5849861
# Extended with tic/toc by Paul Kienzle
import time


class Timer(object):
    @staticmethod
    def tic(name=None):
        return Timer(name).toc

    def __init__(self, name=None):
        self.name = name
        self.step_number = 0
        self.tlast = self.tstart = time.time()

    def toc(self, step=None):
        self.step_number += 1
        if step is None:
            step = str(self.step_number)
        label = self.name + "-" + step if self.name else step
        tnext = time.time()
        total = tnext - self.tstart
        delta = tnext - self.tlast
        print("[%s] Elapsed: %s, Delta: %s" % (label, total, delta))
        self.tlast = tnext

    def __enter__(self):
        self.tlast = self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s]" % self.name, end="")
        print("Elapsed: %s" % (time.time() - self.tstart))


def entropy_mc(D, N=1000000):
    logp = D.logpdf(D.rvs(N))
    return -np.mean(logp)
    # return -np.mean(logp[np.isfinite(logp)])


# CRUFT: dirichlet needs transpose of theta for logpdf
class Dirichlet:
    def __init__(self, alpha):
        self.alpha = alpha
        self._dist = stats.dirichlet(alpha)
        self.dim = len(alpha)

    def logpdf(self, theta):
        return self._dist.logpdf(theta.T)

    def rvs(self, *args, **kw):
        x = self._dist.rvs(*args, **kw)
        # Dirichlet logpdf is failing if x=0 for any x when alpha<1.
        # The simplex check allows fudge of 1e-10.
        x[x == 0] = 1e-100
        return x

    def entropy(self, *args, **kw):
        return self._dist.entropy(*args, **kw)


class Box:
    def __init__(self, width=None, center=None):
        if width is None:
            width = np.ones(len(center), dtype="d")
        if center is None:
            center = np.zeros(len(width), dtype="d")
        self.center = center
        self.width = width
        self.dim = len(width)
        self._logpdf = -np.sum(np.log(self.width))

    def rvs(self, size=1):
        x = np.random.rand(size, len(self.width))
        x = (x - 0.5) * self.width + self.center
        return x

    def logpdf(self, theta):
        y = (theta - self.center) / self.width + 0.5
        logp = np.ones(len(theta)) * self._logpdf
        logp[np.any(y < 0, axis=1)] = -np.inf
        logp[np.any(y > 1, axis=1)] = -np.inf
        return logp

    def entropy(self):
        return -self._logpdf


# CRUFT: scipy MVN gives wrong entropy for singular (and near singular) matrices
# This solution gives wrong results for near-singular rvs(), but for the simple
# case of a diagonal Sigma with one zero it does what I need for the test.
class MVNSingular:
    def __init__(self, *args, **kw):
        kw["allow_singular"] = True
        self.dist = stats.multivariate_normal(*args, **kw)

    @property
    def dim(self):
        return self.dist.dim

    def pdf(self, theta):
        return self.dist.pdf(theta)

    def logpdf(self, theta):
        return self.dist.logpdf(theta)

    def rvs(self, size=1):
        return self.dist.rvs(size=size)

    def entropy(self, N=10000):
        # CRUFT scipy==1.10.0: scipy.stats briefly removed the dist.cov attribute.
        if hasattr(self.dist, "cov"):
            cov = self.dist.cov
        else:
            cov = self.dist.cov_object.covariance

        with np.errstate(divide="ignore"):
            return 0.5 * log(np.linalg.det((2 * pi * np.e) * cov))


class GaussianMixture:
    def __init__(self, w, mu=None, sigma=None):
        mu = np.asarray(mu)
        dim = mu.shape[1]
        if sigma is None:
            sigma = [None] * len(mu)
        sigma = [(np.ones(dim) if s is None else np.asarray(s)) for s in sigma]
        sigma = [(np.diag(s) if len(s.shape) == 1 else s) for s in sigma]
        self.dim = dim
        self.weight = np.asarray(w, "d") / np.sum(w)
        self.dist = [stats.multivariate_normal(mean=m, cov=s) for m, s in zip(mu, sigma)]

    def pdf(self, theta):
        return sum(w * D.pdf(theta) for w, D in zip(self.weight, self.dist))

    def logpdf(self, theta):
        return log(self.pdf(theta))

    def rvs(self, size=1):
        # TODO: should randomize the output
        sizes = partition(size, self.weight)
        draws = [D.rvs(size=n) for n, D in zip(sizes, self.dist)]
        return np.random.permutation(np.vstack(draws))

    def entropy(self, N=10000):
        # No analytic expression, so estimate entropy using MC integration
        return entropy_mc(self, N=N)


class MultivariateT:
    def __init__(self, mu=None, sigma=None, df=None):
        if sigma is not None:
            sigma = np.asarray(sigma)
        self.mu = np.zeros(sigma.shape[0]) if mu is None else np.asarray(mu)
        if sigma is None:
            sigma = np.ones(len(mu))
        if len(sigma.shape) == 1:
            sigma = np.diag(sigma)
        self.dim = len(self.mu)
        self.sigma = sigma
        self.df = df
        # Use scipy stats to compute |Sigma| and (x-mu)^T Sigma^{-1} (x - mu),
        # and to estimate dimension p from rank.  Formula for pdf from wikipedia
        # https://en.wikipedia.org/wiki/Multivariate_t-distribution
        from scipy.stats._multivariate import _PSD

        self._psd = _PSD(self.sigma)
        nu, p = self.df, self._psd.rank
        self._log_norm = gammaln((nu + p) / 2) - gammaln(nu / 2) - p / 2 * log(pi * nu) - self._psd.log_pdet / 2

    def logpdf(self, theta):
        dev = theta - self.mu
        maha = np.sum(np.square(np.dot(dev, self._psd.U)), axis=-1)
        nu, p = self.df, self._psd.rank
        return self._log_norm - (nu + p) / 2 * np.log1p(maha / nu)

    def pdf(self, theta):
        return exp(self.logpdf(theta))

    def rvs(self, size=1):
        # From farhawa on stack overflow
        # https://stackoverflow.com/questions/29798795/multivariate-student-t-distribution-with-python
        nu, p = self.df, len(self.mu)
        g = np.tile(np.random.gamma(nu / 2, 2 / nu, size=size), (p, 1)).T
        Z = np.random.multivariate_normal(np.zeros(p), self.sigma, size=size)
        return self.mu + Z / np.sqrt(g)

    def entropy(self, N=100000):
        # No analytic expression, so estimate entropy using MC integration
        return entropy_mc(self, N=N)


def MultivariateCauchy(mu=None, sigma=None):
    return MultivariateT(mu=mu, sigma=sigma, df=1)


class Joint:
    def __init__(self, distributions):
        # Note: list(x) converts any sequence, including generators, into a list
        self.distributions = list(distributions)
        self.dim = len(self.distributions)

    def rvs(self, size=1):
        return np.stack([D.rvs(size=size) for D in self.distributions], axis=-1)

    def pdf(self, theta):
        return exp(self.logpdf(theta))

    def logpdf(self, theta):
        return sum(D.logpdf(theta[..., k]) for k, D in enumerate(self.distributions))

    def cdf(self, theta):
        return exp(self.logcdf(theta))

    def logcdf(self, theta):
        return sum(D.logcdf(theta[..., k]) for k, D in enumerate(self.distributions))

    def sf(self, theta):
        return -np.expm1(self.logcdf(theta))

    def logsf(self, theta):
        return log(self.sf(theta))

    def entropy(self):
        return sum(D.entropy() for D in self.distributions)


def partition(n, w):
    # TODO: build an efficient algorithm for splitting n things into k buckets
    indices = np.arange(len(w), dtype="i")
    choices = np.random.choice(indices, size=n, replace=True, p=w)
    bins = np.arange(len(w) + 1, dtype="f") - 0.5
    sizes, _ = np.histogram(choices, bins=bins)
    return sizes


def _check_entropy(name, D, seed=1, N=10000, N_entropy=None, N_norm=2500, demo=False):
    """
    Check if entropy from a random draw matches analytic entropy.
    """
    use_kramer = use_mvn = use_wnn = use_gmm = use_kde = False
    if demo:
        # use_kramer = True
        # use_wnn = True
        use_mvn = True
        use_gmm = True
        use_kde = True
    else:
        use_kramer = True

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        theta = D.rvs(size=N)
        if getattr(D, "dim", 1) == 1:
            theta = theta.reshape(N, 1)
        if use_kramer:
            logp_theta = D.logpdf(theta)
            logp_theta += 27  # result should be independent of scale factor
            S, Serr = entropy(theta, logp_theta, N_entropy=N_entropy, N_norm=N_norm)
        if use_wnn:
            S_wnn, Serr_wnn = wnn_entropy(theta, n_est=N_entropy)
        if use_gmm:
            S_gmm, Serr_gmm = gmm_entropy(theta, n_est=N_entropy)
        if use_mvn:
            M = MVNEntropy(theta)
            S_mvn = M.entropy
        if use_kde:
            S_kde = kde_entropy_statsmodels(theta, n_est=N_entropy)
            # S_kde = kde_entropy_sklearn(theta, n_est=N_entropy)
            # S_kde = kde_entropy_sklearn_gmm(theta, n_est=N_entropy)
    finally:
        np.random.set_state(state)
    H = D.entropy() / LN2
    if demo:
        print("entropy", N, "~", name, H, end="")
        if use_kramer:
            print(" Kramer", S, Serr, end="")
        if use_wnn:
            print(" wnn", S_wnn, Serr_wnn, end="")
        if use_gmm:
            print(" gmm", S_gmm, Serr_gmm, end="")
        if use_mvn:
            print(" MVN", S_mvn, end="")
        if use_kde:
            print(" KDE", S_kde, end="")
        print()
    else:
        if use_kramer:
            # assert Serr < 0.05*S, "incorrect error est. for Kramer"
            if np.isfinite(H):
                assert abs(S - H) < 3 * Serr, "incorrect est. for Kramer"
            else:
                assert not np.isfinite(S), "incorrect est. for Kramer"
        if use_wnn:
            if np.isfinite(H):
                assert Serr_wnn < 0.05 * S_wnn, "incorrect error est. for wnn"
                assert abs(S_wnn - H) < 3 * Serr_wnn, "incorrect est. for wnn"
            else:
                assert not np.isfinite(S_wnn), "incorrect est. for wnn"


def _show_entropy(name, D, **kw):
    with Timer():
        return _check_entropy(name, D, seed=None, demo=True, **kw)


def _check_smoke(D):
    theta = D.rvs(size=1000)
    if getattr(D, "dim", 1) == 1:
        theta = theta.reshape(-1, 1)
    logp_theta = D.logpdf(theta)
    entropy(theta, logp_theta)
    wnn_entropy(theta)
    MVNEntropy(theta).entropy


def test():
    """check entropy estimates from known distributions"""
    # entropy test is optional: don't test if sklearn is not installed
    try:
        import sklearn
    except ImportError:
        return

    # Smoke test - do all the methods run in 1-D and 10-D?
    _check_smoke(stats.norm(10, 8))
    if hasattr(stats, "multivariate_normal"):
        _check_smoke(stats.multivariate_normal(cov=np.diag([1] * 10)))

    D = stats.norm(10, 8)
    _check_entropy("N[100,8]", D, N=2000)
    _check_entropy("N[100,8]", D, N=12000)
    if hasattr(stats, "multivariate_normal"):
        D = stats.multivariate_normal(cov=np.diag([1, 12**2, 0.2**2]))
        _check_entropy("MVN[1,12,0.2]", D)
        D = stats.multivariate_normal(cov=np.diag([1] * 10))
        _check_entropy("MVN[1]*10", D, N=10000)
        # Make sure zero-width dimensions return H = -inf
        D = MVNSingular(cov=np.diag([1, 1, 0]))
        _check_entropy("MVN[1,1,0]", D, N=10000)
    # raise TestFailure("make bumps testing fail so we know that test harness works")


def mvn_entropy_test():
    """
    Test against results from the R MVN pacakge (using the web version)
    and the matlab Mskekur program (using Octave), both of which produce
    the same value.  Note that MVNEntropy uses the small sample correction
    for the skewness stat since it converges to the large sample value for
    large n.
    """
    x = np.array(
        [
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
        ]
    )
    M = MVNEntropy(x)
    # print(M)
    # print("%.15g %.15g %.15g"%(M.p_kurtosis, M.p_skewness, M.entropy))
    assert abs(M.p_kurtosis - 0.265317890462476) <= 1e-10
    assert abs(M.p_skewness - 0.773508066109368) <= 1e-10
    assert abs(M.entropy - 5.7920040570988) <= 1e-10

    ## wnn_entropy doesn't work for small sample sizes (no surprise there!)
    # S_wnn, Serr_wnn = wnn_entropy(x)
    # assert abs(S_wnn - 5.7920040570988) <= 1e-10
    # print("wnn %.15g, target %g"%(S_wnn, 5.7920040570988))


def demo():
    # hide module load time from Timer
    from sklearn.neighbors import NearestNeighbors

    ## Bootstrap didn't help, but leave the test code in place for now
    # D = Dirichlet(alpha=[0.02]*20)
    # theta = D.rvs(size=1000)
    # S, Serr = wnn_bootstrap(D.rvs(size=200000))
    # print("bootstrap", S, D.entropy())
    # return
    if False:
        # Multivariate T distribution
        D = stats.t(df=4)
        _show_entropy("T;df=4", D, N=20000)
        D = MultivariateT(sigma=np.diag([1]), df=4)
        _show_entropy("MT[1];df=4", D, N=20000)
        D = MultivariateT(sigma=np.diag([1, 12, 0.2]) ** 2, df=4)
        _show_entropy("MT[1,12,0.2];df=4", D, N=10000)
        D = MultivariateT(sigma=np.diag([1] * 10), df=4)
        _show_entropy("MT[1]*10;df=4", D, N=10000)
        D = MultivariateT(sigma=np.diag([1, 12, 0.2, 1e2, 1e-2, 1]) ** 2, df=4)
        _show_entropy("MT[1,12,0.2,1e3,1e-3,1];df=4", D, N=10000)
        return

    if False:
        # Multivariate skew normal distribution
        D = stats.skewnorm(5)
        _show_entropy("skew=5 N[1]", D, N=20000)
        D = Joint(stats.skewnorm(5, 0, s) for s in [1, 12, 0.2])
        _show_entropy("skew=5 N[1,12,0.2]", D, N=10000)
        D = Joint(stats.skewnorm(5, 0, s) for s in [1] * 10)
        _show_entropy("skew=5 N[1]*10", D, N=10000)
        D = Joint(stats.skewnorm(5, 0, s) for s in [1, 12, 0.2, 1e2, 1e-2, 1])
        _show_entropy("skew=5 N[1,12,0.2,1e3,1e-3,1]", D, N=10000)
        # print("double check entropy", D.entropy()/LN2, entropy_mc(D)/LN2)
        return

    D = Box(center=[100] * 10, width=np.linspace(1, 10, 10))
    _show_entropy("Box 10!", D, N=10000)
    D = stats.norm(10, 8)
    # _show_entropy("N[100,8]", D, N=100)
    # _show_entropy("N[100,8]", D, N=200)
    # _show_entropy("N[100,8]", D, N=500)
    # _show_entropy("N[100,8]", D, N=1000)
    # _show_entropy("N[100,8]", D, N=2000)
    # _show_entropy("N[100,8]", D, N=5000)
    _show_entropy("N[100,8]", D, N=10000)
    # _show_entropy("N[100,8]", D, N=20000)
    # _show_entropy("N[100,8]", D, N=50000)
    # _show_entropy("N[100,8]", D, N=100000)
    D = stats.multivariate_normal(cov=np.diag([1, 12, 0.2]) ** 2)
    # _show_entropy("MVN[1,12,0.2]", D)
    D = stats.multivariate_normal(cov=np.diag([1] * 10) ** 2)
    # _show_entropy("MVN[1]*10", D, N=1000)
    _show_entropy("MVN[1]*10", D, N=10000)
    # _show_entropy("MVN[1]*10", D, N=100000)
    # _show_entropy("MVN[1]*10", D, N=200000, N_entropy=20000)
    D = stats.multivariate_normal(cov=np.diag([1, 12, 0.2, 1, 1, 1]) ** 2)
    # _show_entropy("MVN[1,12,0.2,1,1,1]", D, N=100)
    # _show_entropy("MVN[1,12,0.2,1,1,1]", D, N=1000)
    _show_entropy("MVN[1,12,0.2,1,1,1]", D, N=10000)
    # _show_entropy("MVN[1,12,0.2,1,1,1]", D, N=100000)
    D = stats.multivariate_normal(cov=np.diag([1, 12, 0.2, 1e2, 1e-2, 1]) ** 2)
    # _show_entropy("MVN[1,12,0.2,1e3,1e-3,1]", D, N=100)
    # _show_entropy("MVN[1,12,0.2,1e3,1e-3,1]", D, N=1000)
    _show_entropy("MVN[1,12,0.2,1e3,1e-3,1]", D, N=10000)
    # _show_entropy("MVN[1,12,0.2,1e3,1e-3,1]", D, N=100000)
    D = GaussianMixture([1, 10], mu=[[0] * 10, [100] * 10], sigma=[[10] * 10, [0.1] * 10])
    _show_entropy("bimodal mixture", D)
    D = Dirichlet(alpha=[0.02] * 20)
    # _show_entropy("Dirichlet[0.02]*20", D, N=1000)
    # _show_entropy("Dirichlet[0.02]*20", D, N=2000)
    # _show_entropy("Dirichlet[0.02]*20", D, N=5000)
    # _show_entropy("Dirichlet[0.02]*20", D, N=10000)
    _show_entropy("Dirichlet[0.02]*20", D, N=20000)
    # _show_entropy("Dirichlet[0.02]*20", D, N=50000)
    # _show_entropy("Dirichlet[0.02]*20", D, N=200000, N_entropy=20000)


if __name__ == "__main__":  # pragma: no cover
    demo()
