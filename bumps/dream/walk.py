# This program is in the public domain
# Author: Paul Kienzle
"""
Random walk functions.

:function:`walk` simulates a mean-reverting random walk.
"""

# This code was developed to test outlier detection

__all__ = ["walk"]

from numpy import asarray, ones_like, nan, isnan

from . import util


def walk(n=1000, mu=0, sigma=1, alpha=0.01, s0=nan):
    """
    Mean reverting random walk.

    Returns an array of n-1 steps in the following process::

        s[i] = s[i-1] + alpha*(mu-s[i-1]) + e[i]

    with e ~ N(0,sigma).

    The parameters are::

        *n* walk length
        *s0* starting value, defaults to N(mu,sigma)
        *mu* target mean, defaults to 0
        *sigma* volatility
        *alpha* in [0,1] reversion rate

    Use alpha=0 for a pure Gaussian random walk or alpha=1 independent
    samples about the mean.

    If *mu* is a vector, multiple streams are run in parallel.  In this
    case *s0*, *sigma* and *alpha* can either be scalars or vectors.

    If *mu* is an array, the target value is non-stationary, and the
    parameter *n* is ignored.

    Note: the default starting value should be selected from a distribution
    whose width depends on alpha.  N(mu,sigma) is too narrow.  This
    effect is illustrated in :function:`demo`, where the following choices
    of sigma and alpha give approximately the same histogram::

        sigma = [0.138, 0.31, 0.45, 0.85, 1]
        alpha = [0.01,  0.05, 0.1,  0.5,  1]
    """
    s0, mu, sigma, alpha = [asarray(v) for v in (s0, mu, sigma, alpha)]
    nchains = mu.shape[0] if mu.ndim > 0 else 1

    if mu.ndim < 2:
        if isnan(s0):
            s0 = mu + util.rng.randn(nchains) * sigma
        s = [s0 * ones_like(mu)]
        for i in range(n - 1):
            s.append(s[-1] + alpha * (mu - s[-1]) + sigma * util.rng.randn(nchains))
    elif mu.ndim == 2:
        if isnan(s0):
            s0 = mu[0] + util.rng.randn(nchains) * sigma
        s = [s0 * ones_like(mu[0])]
        for i in range(mu.shape[1]):
            s.append(s[-1] + alpha * (mu[i] - s[-1]) + sigma * util.rng.randn(nchains))
    else:
        raise ValueError("mu must be scalar, vector or 2D array")
    return asarray(s)


def demo():
    """
    Example showing the relationship between alpha and sigma in the random
    walk posterior distribution.

    The lag 1 autocorrelation coefficient R^2 is approximately 1-alpha.
    """
    from numpy import mean, std, sum
    import pylab
    from matplotlib.ticker import MaxNLocator

    pylab.seed(10)  # Pick a pretty starting point

    # Generate chains
    n = 5000
    mu = [0, 5, 10, 15, 20]
    sigma = [0.138, 0.31, 0.45, 0.85, 1]
    alpha = [0.01, 0.05, 0.1, 0.5, 1]
    chains = walk(n, mu=mu, sigma=sigma, alpha=alpha)

    # Compute lag 1 correlation coefficient
    m, s = mean(chains, axis=0), std(chains, ddof=1, axis=0)
    r2 = sum((chains[1:] - m) * (chains[:-1] - m), axis=0) / ((n - 2) * s**2)
    r2[abs(r2) < 0.01] = 0

    # Plot chains
    ax_data = pylab.axes([0.05, 0.05, 0.65, 0.9])  # x,y,w,h
    ax_data.plot(chains)
    textkw = dict(
        xytext=(30, 0), textcoords="offset points", verticalalignment="center", backgroundcolor=(0.8, 0.8, 0.8, 0.8)
    )
    label = r"$\ \alpha\,%.2f\ \ \sigma\,%.3f\ \ " r"R^2\,%.2f\ \ avg\,%.2f\ \ std\,%.2f\ $"
    for m, s, a, r2, em, es in zip(mu, sigma, alpha, r2, m, s):
        pylab.annotate(label % (a, s, r2, em - m, es), xy=(0, m), **textkw)

    # Plot histogram
    ax_hist = pylab.axes([0.75, 0.05, 0.2, 0.9], sharey=ax_data)
    ax_hist.hist(chains.flatten(), 100, orientation="horizontal")
    pylab.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.xaxis.set_major_locator(MaxNLocator(3))

    pylab.show()


if __name__ == "__main__":
    demo()
