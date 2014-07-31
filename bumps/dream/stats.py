import re

import numpy as np

from .formatnum import format_uncertainty

class VarStats(object):
    def __init__(self, **kw):
        self.__dict__ = kw

def var_stats(draw, vars=None):
    if vars is None: vars = range(draw.points.shape[1])
    return [_var_stats_one(draw, v) for v in vars]

ONE_SIGMA = 1 - 2*0.15865525393145705
def _var_stats_one(draw, var):
    weights, values = draw.weights, draw.points[:, var].flatten()

    best_idx = np.argmax(draw.logp)
    best = values[best_idx]

    # Choose the interval for the histogram
    p95,p68,p0 = credible_intervals(x=values, weights=weights, ci=[0.95,ONE_SIGMA,0.0])
    #open('/tmp/out','a').write("in vstats: p68=%s, p95=%s, p0=%s, value range=%s\n"%(p68,p95,p0,(min(values),max(values))))
    #if p0[0] != p0[1]: raise RuntimeError("wrong median %s"%(str(p0),))

    mean, std = stats(x=values, weights=weights)

    vstats = VarStats(label=draw.labels[var], index=var+1, p95=p95, p68=p68,
                      median=p0[0], mean=mean, std=std, best=best)

    return vstats

def format_num(x, place):
    precision = 10**place
    digits_after_decimal = abs(place) if place < 0 else 0
    return "%.*f"%(digits_after_decimal,
                   np.round(x/precision)*precision)

def format_vars(all_vstats):
    v = dict(parameter="Parameter",
             mean="mean", median="median", best="best",
             interval68="68% interval",
             interval95="95% interval")
    s = ["   %(parameter)20s %(mean)10s %(median)7s %(best)7s [%(interval68)15s] [%(interval95)15s]"%v]
    for v in all_vstats:
        # Make sure numbers are formatted with the appropriate precision
        place = int(np.log10(v.p95[1]-v.p95[0]))-2
        summary = dict(mean=format_uncertainty(v.mean,v.std),
                       median=format_num(v.median,place-1),
                       best=format_num(v.best,place-1),
                       lo68=format_num(v.p68[0],place),
                       hi68=format_num(v.p68[1],place),
                       loci=format_num(v.p95[0],place),
                       hici=format_num(v.p95[1],place),
                       parameter=v.label,
                       index=v.index)
        s.append("%(index)2d %(parameter)20s %(mean)10s %(median)7s %(best)7s [%(lo68)7s %(hi68)7s] [%(loci)7s %(hici)7s]"%summary)

    return "\n".join(s)

VAR_PATTERN = re.compile(r"""
   ^\ *
   (?P<parnum>[0-9]+)\ +
   (?P<parname>.+?)\ +
   (?P<mean>[0-9.-]+?)
   \((?P<err>[0-9]+)\)
   (e(?P<exp>[+-]?[0-9]+))?\ +
   (?P<median>[0-9.eE+-]+?)\ +
   (?P<best>[0-9.eE+-]+?)\ +
   \[\ *(?P<lo68>[0-9.eE+-]+?)\ +
   (?P<hi68>[0-9.eE+-]+?)\]\ +
   \[\ *(?P<lo95>[0-9.eE+-]+?)\ +
   (?P<hi95>[0-9.eE+-]+?)\]
   \ *$
   """, re.VERBOSE)

def parse_var(line):
    """
    Parse a line returned by format_vars back into the statistics for the
    variable on that line.
    """
    m = VAR_PATTERN.match(line)
    if m:
        exp = int(m.group('exp')) if m.group('exp') else 0
        return VarStats(index = int(m.group('parnum')),
                        name = m.group('parname'),
                        mean = float(m.group('mean')) * 10**exp,
                        median = float(m.group('median')),
                        best = float(m.group('best')),
                        p68 = (float(m.group('lo68')), float(m.group('hi68'))),
                        p95 = (float(m.group('lo95')), float(m.group('hi95'))),
                        )
    else:
        return None


def stats(x, weights=None):
    """
    Find mean and standard deviation of a set of weighted samples.

    Note that the median is not strictly correct (we choose an endpoint
    of the sample for the case where the median falls between two values
    in the sample), but this is good enough when the sample size is large.
    """
    if weights == None:
        x = np.sort(x)
        mean, std = np.mean(x), np.std(x,ddof=1)
    else:
        mean = np.mean(x*weights)/np.sum(weights)
        # TODO: this is biased by selection of mean; need an unbiased formula
        var = np.sum((weights*(x-mean))**2)/np.sum(weights)
        std = np.sqrt(var)

    return mean, std


def credible_intervals(x, ci, weights=None):
    """
    Find the credible interval covering the portion *ci* of the data.

    Returns a 2D array of credible intervals, the minimum and maximum values of the interval.
    If *ci* is a vector, return a vector of intervals.

    *x* are samples from the posterior distribution.

    This function is faster if the inputs are already sorted.

    *ci* is a set of intervals in [0,1].  For a $1-\sigma$ interval use
    *ci=erf(1/sqrt(2))*, or 0.68. About 1e5 samples are needed for 2 digits
    of  precision on a $1-\sigma$ credible interval.  For a 95% interval,
    about 1e6 samples are needed.

    *weights* is a vector of weights for each x, or None for unweighted.
    One could weight points according to temperature in a parallel tempering
    dataset.
    """
    from numpy import asarray, vstack, sort, cumsum, searchsorted, round, clip
    ci = asarray(ci, 'd')
    target = (1 + vstack((-ci, +ci))).T/2

    if weights is None:
        idx = clip(round(target*(x.size-1)), 0, x.size-1).astype('i')
        return sort(x)[idx]
    else:
        idx = np.argsort(x)
        x, weights = x[idx], weights[idx]
        # convert weights to cdf
        w = cumsum(weights/sum(weights))
        return x[searchsorted(w, target)]

