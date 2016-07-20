"""
Contains methods for calculating the number
significant digits in a statistic
"""
# TODO: needs a refactor before going into production

"""
notes:
- the gum method gives a slight underestimate of the sd, while
jackknife gives an overestimate.
- Jackknife performs better than gum on small samples
- in current implementation, gum is much faster
- in picking the number of cuts for jackknife, lower is usually better
(but not too low...)
"""

"""
things to do:
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

import numpy as np
import numpy.ma as ma


def credible_interval_sd(data, ci, fn=None, method=None):
    if fn is None:
        # ugly hack here
        from .stats import credible_intervals
        fn = lambda *a, **kw: credible_intervals(*a, **kw)[0]
    if method is None:
        # do something clever to choose gum or jack
        method = 'gum'
    
    if method == 'gum':
        return gum_sd(data, fn, ci)
    elif method == 'jack':
        return jack_sd(data, fn, ci)
    else:
        raise ValueError("Unknown sd method"+method)


# TODO does not handle either CI function natively
# requires a 1d ndarray return value

# adaptation of method in section 7.9.4 of gum suppliment 1
def gum_sd(data, f, ci, cuts=10):
    # rounds data length down to use equal block sizes
    k = len(data) // cuts
    data = data[-k*cuts:].reshape((cuts, k))
    v = np.var(np.apply_along_axis(f, 1, data, ci), axis=0) / (cuts - 1)
    return np.sqrt(v)


# does an average of sd on smaller blocks to counteract skewed
# distribution of jackknife
def jack_sd(data, f, ci, cuts=10):
    k = len(data) // cuts
    #round down so we can use equal length blocks
    data = data[-k*cuts:].reshape((cuts, k))
    return np.mean(np.apply_along_axis(fast_jack, 1, data, f, ci), axis=0) / np.sqrt(cuts - 1)


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
        for k,v in kw.items():
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
        
        return sqrt((self.n-1)*var(drop_stats, axis=0))
        
    def _fill_interval(self, drop_stats, inter, ev, splits=None):
        lo, hi = inter
        v_hi = self._f_drop(hi)
        v_lo = self._f_drop(lo)
        a = np.array([v_lo, v_hi, ev]).T
        g = lambda ar: (ar[0] == ar).all()
        if np.apply_along_axis(g, 1, a).all():
            for i in range(lo, hi+1):
                drop_stats[i] = v_hi
        elif hi == lo:
            drop_stats[hi] = v_hi
        elif hi == lo + 1:
            drop_stats[hi] = v_hi
            drop_stats[lo] = v_lo
        else:
            splits = max(int(np.ceil(np.log10(hi-lo))), 2)
            dx = (hi - lo)//splits
            intervals = [[lo+dx*i, lo+dx*(i+1)] for i in range(splits)]
            intervals[-1][1] = hi
            for inter in intervals:
                self._fill_interval(drop_stats, inter, np.mean((v_lo, v_hi), axis=0))

# converts standard deviation to # of significand digits
# adapted from section 7.9.2 of gum suppliment 1
# sx is standard deviation of the statistic
# sd is standard deviation of the sample 
def n_dig(sx, sd):
    l = np.log10(4*sx)
    return np.floor(np.log10(sd)) - l + 1 #return as float or round now?
