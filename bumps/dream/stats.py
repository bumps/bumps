"""
Statistics helper functions.
"""

__all__ = [
    "VarStats",
    "var_stats",
    "format_vars",
    "parse_var",
    "stats",
    "credible_interval",
    "shortest_credible_interval",
]

import re
import json

import numpy as np

from .formatnum import format_uncertainty


class VarStats(object):
    def __init__(self, **kw):
        self.__dict__ = kw


def var_stats(draw, vars=None):
    if vars is None:
        vars = range(draw.points.shape[1])
    return [_var_stats_one(draw, v) for v in vars]


ONE_SIGMA = 1 - 2 * 0.15865525393145705


def _var_stats_one(draw, var):
    weights, values = draw.weights, draw.points[:, var].flatten()

    integer = draw.integers is not None and draw.integers[var]
    if integer:
        values = np.floor(values)

    best_idx = np.argmax(draw.logp)
    best = values[best_idx]
    x_sort_index = draw.get_argsort_indices(var)

    # Choose the interval for the histogram
    # credible_interval = shortest_credible_interval
    p95, p68, p0 = credible_interval(x=values, weights=weights, ci=[0.95, ONE_SIGMA, 0.0], x_sort_index=x_sort_index)

    ## reporting uncertainty on credible intervals?
    ## might be nice to pair sd on credible intervals
    ## with the actual CIs, rather than use a separate param
    # from .digits import credible_inderval_sd
    # p95sd = credible_interval_sd(values, 0.95)
    # p68sd = credible_interval_sd(values, ONE_SIGMA)

    # open('/tmp/out','a').write(
    #     "in vstats: p68=%s, p95=%s, p0=%s, value range=%s\n"
    #     % (p68,p95,p0,(min(values),max(values))))
    # if p0[0] != p0[1]: raise RuntimeError("wrong median %s"%(str(p0),))

    mean, std = stats(x=values, weights=weights, x_sort_index=x_sort_index)

    vstats = VarStats(
        label=draw.labels[var],
        index=var + 1,
        p95=p95,
        p95_range=(p95[0], p95[1] + integer * 0.9999999999),
        p68=p68,
        p68_range=(p68[0], p68[1] + integer * 0.9999999999),
        # p95sd=p95sd, p68sd=p68sd,
        median=p0[0],
        mean=mean,
        std=std,
        best=best,
        integer=integer,
    )

    return vstats


def format_num(x, place):
    precision = 10**place
    digits_after_decimal = abs(place) if place < 0 else 0
    return "%.*f" % (digits_after_decimal, np.round(x / precision) * precision)


def format_vars(all_vstats):
    v = dict(
        parameter="Parameter",
        mean="mean",
        median="median",
        best="best",
        interval68="68% interval",
        interval95="95% interval",
    )
    s = ["   %(parameter)20s %(mean)10s %(median)7s %(best)7s " "[%(interval68)15s] [%(interval95)15s]" % v]
    for v in all_vstats:
        # Make sure numbers are formatted with the appropriate precision
        place = (
            int(np.log10(v.p95[1] - v.p95[0])) - 2
            if v.p95[1] > v.p95[0]
            else int(np.log10(abs(v.p95[0]))) - 3
            if v.p95[0] != 0
            else 0
        )
        summary = dict(
            mean=format_uncertainty(v.mean, v.std),
            median=format_num(v.median, place - 1),
            best=format_num(v.best, place - 1),
            lo68=format_num(v.p68[0], place),
            hi68=format_num(v.p68[1], place),
            loci=format_num(v.p95[0], place),
            hici=format_num(v.p95[1], place),
            parameter=v.label,
            index=v.index,
        )
        s.append(
            "%(index)2d %(parameter)20s %(mean)10s %(median)7s %(best)7s "
            "[%(lo68)7s %(hi68)7s] [%(loci)7s %(hici)7s]" % summary
        )

    return "\n".join(s)


def save_vars(all_vstats, filename):
    with open(filename, "w") as fid:
        json.dump(
            dict((v.label, v.__dict__) for v in all_vstats),
            fid,
            default=numpy_json,
            sort_keys=True,
            indent=2,
        )


def numpy_json(o):
    """
    JSON encoder for numpy data.

    To automatically convert numpy data to lists when writing a datastream
    use json.dumps(object, default=numpy_json).
    """
    try:
        return o.tolist()
    except AttributeError:
        raise TypeError


VAR_PATTERN = re.compile(
    r"""
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
   """,
    re.VERBOSE,
)


def parse_var(line):
    """
    Parse a line returned by format_vars back into the statistics for the
    variable on that line.
    """
    m = VAR_PATTERN.match(line)
    if m:
        exp = int(m.group("exp")) if m.group("exp") else 0
        return VarStats(
            index=int(m.group("parnum")),
            name=m.group("parname"),
            mean=float(m.group("mean")) * 10**exp,
            median=float(m.group("median")),
            best=float(m.group("best")),
            p68=(float(m.group("lo68")), float(m.group("hi68"))),
            p95=(float(m.group("lo95")), float(m.group("hi95"))),
        )
    else:
        return None


def stats(x, weights=None, x_sort_index=None):
    """
    Find mean and standard deviation of a set of weighted samples.

    Note that the median is not strictly correct (we choose an endpoint
    of the sample for the case where the median falls between two values
    in the sample), but this is good enough when the sample size is large.
    """
    if weights is None:
        if x_sort_index is None:
            x_sort_index = np.argsort(x)
        x = x[x_sort_index]
        mean, std = np.mean(x), np.std(x, ddof=1)
    else:
        mean = np.mean(x * weights) / np.sum(weights)
        # TODO: this is biased by selection of mean; need an unbiased formula
        var = np.sum((weights * (x - mean)) ** 2) / np.sum(weights)
        std = np.sqrt(var)

    return mean, std


def credible_interval(x, ci, weights=None, x_sort_index=None):
    r"""
    Find the credible interval covering the portion *ci* of the data.

    *x* are samples from the posterior distribution.

    *ci* is a set of intervals in [0,1].  For a $1-\sigma$ interval use
    *ci=erf(1/sqrt(2))*, or 0.68. About 1e5 samples are needed for 2 digits
    of  precision on a $1-\sigma$ credible interval.  For a 95% interval,
    about 1e6 samples are needed for 2 digits of precision.  At least 1000
    points are needed for an unbiased result, otherwise the resulting interval
    will be shorter than expected (tested on a variety of distributions
    including exponential, cauchy, gaussian, beta and gamma).

    *weights* is a vector of weights for each x, or None for unweighted.
    One could weight points according to temperature in a parallel tempering
    dataset.

    Returns an array *[[x1_low, x1_high], [l2_low, x2_high], ...]* where
    *[xi_low, xi_high]* are the starting and ending values for credible
    interval *i*.

    This function is faster if the inputs are already sorted.
    """
    n = x.size
    ci = np.asarray(ci, "d")
    target = (1 + np.vstack((-ci, +ci))).T / 2
    if x_sort_index is None:
        x_sort_index = np.argsort(x)

    if weights is None:
        cdf = np.linspace(0.5 / n, 1 - 0.5 / n, n)
        # cdf = np.linspace(1, n, n)/(n+1)
        result = np.interp(target, cdf, x[x_sort_index])
    else:
        x, weights = x[x_sort_index], weights[x_sort_index]
        # convert weights to cdf
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        cdf -= 0.5 * cdf[0]
        # cdf *= n/(cdf[-1]*(n+1))
        result = np.interp(target, cdf, x)
    return result if ci.shape else result[0]


def shortest_credible_interval(x, ci=0.95, weights=None):
    """
    Find the credible interval covering the portion *ci* of the data.

    *x* are samples from the posterior distribution.
    *ci* is the interval size in (0,1], and defaults to 0.95.
    For a 1-sigma interval use *ci=erf(1/sqrt(2))*.
    *weights* is a vector of weights for each x, or None for unweighted.

    Returns the minimum and maximum values of the interval.
    If *ci* is a vector, return a vector of intervals.

    This function is faster if the inputs are already sorted.

    About 1e6 samples are needed for 2 digits of precision on a 95%
    credible interval, or 1e5 for 2 digits on a 1-sigma credible interval.

    To remove bias towards toward smaller intervals, the midpoints between
    the surrounding intervals are used as the end points.
    """

    if weights is None:
        x = np.sort(x)
        # Simple solution: ci*N is the number of points in the interval, so
        # find the width of every interval of that size and return the smallest.
        if np.isscalar(ci):
            return _unweighted_hpd(x, ci)
        else:
            return [_unweighted_hpd(x, ci_k) for ci_k in ci]
    else:
        index = np.argsort(x)
        x, weights = x[index], weights[index]
        # Work from the empirical cdf, finding the corresponding right
        # interval for each possible left interval and choosing that with
        # the shortest distance.
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        # jcdf -= 0.5*cdf[0]
        if np.isscalar(ci):
            return _weighted_hpd(x, cdf, ci)
        else:
            return [_weighted_hpd(x, cdf, ci_k) for ci_k in ci]


def _unweighted_hpd(x, ci):
    """
    Find shortest credible interval ci in sorted, unweighted x
    """
    n = len(x)
    size = int(ci * n)
    if size >= n:
        return x[0], x[-1]
    else:
        width = x[size:] - x[:-size]
        index = np.argmin(width)
        # left, right = x[idx], x[idx+size]
        left = x[0] if index == 0 else (x[index - 1] + x[index]) / 2
        right = x[-1] if index + size == n - 1 else (x[index + size] + x[index + size + 1]) / 2
        return left, right


def _weighted_hpd(z, cdf, ci):  # extra one-half interval
    """
    Find shortest credible interval ci in sorted, weighted x
    """
    size = np.searchsorted(cdf, 1 - ci)
    if size == 0:
        return z[0], z[-1]
    p_left = cdf[:size]
    z_left = z[:size]
    # avoid spurious floating point bugs, e.g., where .1+0.9 > 1.0
    i_right = np.searchsorted(cdf[:-1], p_left + ci)
    z_right = z[i_right]
    index = np.argmin(z_right - z_left)
    left = z_left[0] if index == 0 else (z_left[index - 1] + z_left[index]) / 2
    right = z_right[-1] if index + 1 == len(z_right) else (z_right[index] + z_right[index + 1]) / 2
    return left, right
