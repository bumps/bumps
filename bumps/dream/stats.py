import numpy
from numpy import mean, std

def stats(x, weights=None):
    if weights == None:
        mean, std = numpy.mean(x), numpy.std(x,ddof=1)
    else:
        mean = numpy.mean(x*weights)/numpy.sum(weights)
        # TODO: this is biased by selection of mean; need an unbiased formula
        var = numpy.sum((weights*(x-mean))**2)/numpy.sum(weights)
        std = numpy.sqrt(var)

    return mean, std



def credible_interval(x, ci=0.95, weights=None):
    """
    Find the credible interval covering the portion *ci* of the data.

    Returns the minimum and maximum values of the interval.

    *x* are samples from the posterior distribution.
    *weights* is a vector of weights for each x, or None for unweighted
    *ci* is the portion in (0,1], and defaults to 0.95.

    This function is faster if the inputs are already sorted.

    For log likelihood data, setting weights to exp(max(logp)-logp) should
    give reasonable results.

    If *ci* is a vector, return a vector of intervals.
    """
    sorted = numpy.all(x[1:]>=x[:-1])
    if not sorted:
        idx = numpy.argsort(x)
        x = x[idx]
        if weights is not None:
            weights = weights[idx]

    #  w = exp(max(logp)-logp)
    if weights is not None:
        # convert weights to cdf
        w = numpy.cumsum(weights/sum(weights))
        # sample the cdf at every 0.001
        idx = numpy.searchsorted(w, numpy.arange(0,1,0.001))
        x = x[idx]

    # Simple solution: ci*N is the number of points in the interval, so
    # find the width of every interval of that size and return the smallest.
    if numpy.isscalar(ci):
        return _find_interval(x, ci)
    else:
        return [_find_interval(x, i) for i in ci]

def _find_interval(x,ci):
    """
    Find credible interval ci in sorted, unweighted x
    """
    size = ci * len(x)
    if size > len(x)-0.5:
        return x[0],x[-1]
    else:
        width = x[size:] - x[:-size]
        idx = numpy.argmin(width)
        return x[idx],x[idx+size]
