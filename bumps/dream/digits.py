"""
Significant digits
------------------

Calculate the number significant digits in a statistic using boot
strapping/jackknife.

The 'gum' method from the GUM supplement[1] splits the sequence into n
independent sequences, computes the statistic for each sequence, and
returns the mean and variance of the statistic as the estimated value
with uncertainty.  Each block should have at least 10000 samples, or
100/(1-ci) samples if the desired credible interval ci > 0.99.

The 'jack' method uses

notes:

- the gum method gives a slight underestimate of the sd, while
  jackknife gives an overestimate.
- Jackknife performs better than gum on small samples
- in current implementation, gum is much faster
- in picking the number of cuts for jackknife, lower is usually better
  (but not too low...)

[1] JCGM, 2008. Evaluation of measurement data - Supplement 1 to the "Guide to
the expression of uncertainty in measurement" - Propagation of distributions
using a Monte Carlo method (No. JCGM 101:2008). Geneva, Switzerland.
"""

"""
THINGS TO DO:
- actually useful display of information in output
- an effective, fast, online version to use as a stoping crit
- use as a method to guess how many more samples:
    - provides estimate of sd of sampling distribution of statistic
    - sd of stat for n samples = sd for k samples / sqrt(n/k - 1)
    - not actually well tested...
- provide better user accessible control hooks
- do better deciding between jack and gum
- these don't deal with lists of CIs (as the actual ci functions do)
- very slow, needs a restructure
"""
# TODO: needs a refactor before going into production

import numpy as np
import numpy.ma as ma

from .stats import credible_interval


def credible_interval_sd(data, ci, fn=None, method=None, cuts=None):
    if fn is None:
        # ugly hack here
        fn = lambda *a, **kw: credible_interval(*a, **kw)[0]

    if method is None:
        # do something clever to choose gum or jack
        method = "gum"

    # Determine the number of cuts
    if cuts is None or cuts <= 0:
        # Use groups of 10000 unless looking for ci > 0.99, in which case
        # use groups of 100/(1-ci).
        M = max(int(100 / (1 - ci)), 10000)
        cuts = len(data) // M
    if cuts < 3:
        cuts = 3

    if method == "gum":
        return gum_sd(data, fn, ci)
    elif method == "jack":
        return jack_sd(data, fn, ci)
    else:
        raise ValueError("Unknown sd method" + method)


# TODO does not handle either CI function natively
# requires a 1d ndarray return value


def gum_sd(data, f, ci, cuts=10):
    """
    adaptation of method in section 7.9.4 of gum suppliment 1
    """
    # reshape data into blocks, skipping the first partial block
    M = len(data) // cuts
    data = data[-M * cuts :].reshape((cuts, M))
    # compute the variance of the statistic over the sub-blocks
    stat = np.apply_along_axis(f, 1, data, ci)
    var = np.var(stat, axis=0, ddof=1)
    # estimate standard deviation of the statistic for the full data set
    return np.sqrt(var / cuts)


def jack_sd(data, f, ci, cuts=10):
    """
    does an average of sd on smaller blocks to counteract skewed
    distribution of jackknife
    """
    # reshape data into blocks, skipping the first partial block
    M = len(data) // cuts
    data = data[-M * cuts :].reshape((cuts, M))
    # compute the standard deviation of the statistic over the sub-blocks
    std = np.apply_along_axis(fast_jack, 1, data, f, ci)
    # estimate standard deviation of the statistic for the full data set
    return np.mean(std, axis=0) / np.sqrt(cuts)


# TODO: does not deal well with a list of CIs
# returns the sd of a statistic, f
def fast_jack(data, f, *a, **kw):
    jack = FastJackknife(data=data, fn=f)
    jack.set_args(*a, **kw)
    return jack.std()


# TODO: this is coded badly and so is very slow
class FastJackknife(object):
    def __init__(self, **kw):
        self.fn = None
        self._x = None
        for k, v in kw.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                pass

    @property
    def data(self):
        return self._x

    @data.setter
    def data(self, value):
        self._x = np.sort(value)
        self.n = len(value)

    @property
    def fn(self):
        if self._fn is not None:
            return lambda x: self._fn(x, *self._args, **self._kwargs)
        else:
            return None

    @fn.setter
    def fn(self, f):
        self._fn = f

    def set_args(self, *a, **kw):
        self._args = a
        self._kwargs = kw

    def _f_drop(self, i):
        from numpy import zeros_like

        mask = zeros_like(self.data)
        mask[i] = 1
        mx = ma.array(self.data, mask=mask)
        return np.array(self.fn(mx.compressed()))

    def pooled(self):
        assert self.data is not None and self.fn is not None
        return self.fn(self.data)

    def std(self):
        from numpy import sqrt, var, empty, searchsorted

        true_stat = self.fn(self.data)
        drop_stats = empty((self.n, len(true_stat)))

        # this part is specific for CI
        idx_lo, idx_hi = searchsorted(self.data, true_stat)
        self._fill_interval(drop_stats, (0, idx_lo), true_stat)
        self._fill_interval(drop_stats, (idx_lo, idx_hi), true_stat)
        self._fill_interval(drop_stats, (idx_hi, self.n - 1), true_stat)

        return sqrt((self.n - 1) * var(drop_stats, axis=0, ddof=1))

    def _fill_interval(self, drop_stats, interval, ev, splits=None):
        lo, hi = interval
        v_hi = self._f_drop(hi)
        v_lo = self._f_drop(lo)
        a = np.array([v_lo, v_hi, ev]).T
        g = lambda ar: (ar[0] == ar).all()
        if np.apply_along_axis(g, 1, a).all():
            for i in range(lo, hi + 1):
                drop_stats[i] = v_hi
        elif hi == lo:
            drop_stats[hi] = v_hi
        elif hi == lo + 1:
            drop_stats[hi] = v_hi
            drop_stats[lo] = v_lo
        else:
            splits = max(int(np.ceil(np.log10(hi - lo))), 2)
            dx = (hi - lo) // splits
            intervals = [[lo + dx * i, lo + dx * (i + 1)] for i in range(splits)]
            intervals[-1][1] = hi
            for inter in intervals:
                self._fill_interval(drop_stats, inter, np.mean((v_lo, v_hi), axis=0))


def n_dig(sx, sd):
    """
    Converts standard deviation to # of significand digits.

    *sx* is standard deviation of the statistic
    *sd* is standard deviation of the sample

    Adapted from section 7.9.2 of gum suppliment 1
    """
    l = np.log10(4 * sx)
    return np.floor(np.log10(sd)) - l + 1  # return as float or round now?
