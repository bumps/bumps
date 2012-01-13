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



def credible_interval(x, ci=0.95, weights=None, unbiased=False):
    """
    Find the credible interval covering the portion *ci* of the data.

    Returns the minimum and maximum values of the interval.
    If *ci* is a vector, return a vector of intervals.

    *x* are samples from the posterior distribution.
    This function is faster if the inputs are already sorted.
    About 1e6 samples are needed for 2 digits of precision on a 95%
    credible interval, or 1e5 for 2 digits on a 1-sigma credible interval.

    *ci* is the interval size in (0,1], and defaults to 0.95.  For a
    1-sigma interval use *ci=erf(1/sqrt(2))*.

    *weights* is a vector of weights for each x, or None for unweighted.
    For log likelihood data, setting weights to exp(max(logp)-logp) should
    give reasonable results.

    if *unbiased* is True, then attempt to correct for the bias toward
    shorter intervals when the number of samples is small.  The bias
    correction increases the number of samples in the credible interval by 
    *sqrt(1-ci) log(n)* where *n* is the number of samples in *x*.  This 
    has been found empirically to given improved estimates for 1-sigma
    and 95% credible intervals for samples from the gaussian, gamma and
    cauchy distributions, but with overestimated intervals for *n<100*.
    The default will remain *unbiased=False* until a better estimate is
    available.
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
        return _find_interval(x, ci, unbiased=unbiased)
    else:
        return [_find_interval(x, i, unbiased=unbiased) for i in ci]

def _find_interval(x,ci, unbiased):
    """
    Find credible interval ci in sorted, unweighted x
    """
    n = len(x)
    size = int( ci*n + unbiased*numpy.sqrt(1-ci)*numpy.log(n) )
    if size >= n:
        return x[0],x[-1]
    else:
        width = x[size:] - x[:-size]
        idx = numpy.argmin(width)
        return x[idx],x[idx+size]
