# This program is in the public domain
# Author: Paul Kienzle
"""
Parameter bounds and prior probabilities.

Parameter bounds encompass several features of our optimizers.

First and most trivially they allow for bounded constraints on
parameter values.

Secondly, for parameter values known to follow some distribution,
the bounds encodes a penalty function as the value strays from
its nominal value.  Using a negative log likelihood cost function
on the fit, then this value naturally contributes to the overall
likelihood measure.

Predefined bounds are::

    Unbounded
        range (-inf, inf)
    BoundedBelow
        range (base, inf)
    BoundedAbove
        range (-inf, base)
    Bounded
        range (low, high)
    Normal
        range (-inf, inf) with gaussian probability
    BoundedNormal
        range (low, high) with gaussian probability within
    SoftBounded
        range (low, high) with gaussian probability outside

New bounds can be defined following the abstract base class
interface defined in :class:`Bounds`, or using Distribution(rv)
where rv is a scipy.stats continuous distribution.

For generating bounds given a value, we provide a few helper
functions::

    v +/- d:  pm(x,dx) or pm(x,-dm,+dp) or pm(x,+dp,-dm)
        return (x-dm,x+dm) limited to 2 significant digits
    v +/- p%: pmp(x,p) or pmp(x,-pm,+pp) or pmp(x,+pp,-pm)
        return (x-pm*x/100, x+pp*x/100) limited to 2 sig. digits
    pm_raw(x,dx) or raw_pm(x,-dm,+dp) or raw_pm(x,+dp,-dm)
        return (x-dm,x+dm)
    pmp_raw(x,p) or raw_pmp(x,-pm,+pp) or raw_pmp(x,+pp,-pm)
        return (x-pm*x/100, x+pp*x/100)
    nice_range(lo,hi)
        return (lo,hi) limited to 2 significant digits
"""
from __future__ import division
__all__ = ['pm', 'pmp', 'pm_raw', 'pmp_raw', 'nice_range', 'init_bounds',
           'Bounds', 'Unbounded', 'Bounded', 'BoundedAbove', 'BoundedBelow',
           'Distribution', 'Normal', 'BoundedNormal', 'SoftBounded']

import math
from math import log, log10, sqrt, pi, ceil, floor

from numpy import inf, isinf, isfinite, clip
import numpy.random as RNG
try:
    from scipy.stats import norm as normal_distribution
except ImportError:
    # Normal distribution is an optional dependency.  Leave it as a runtime
    # failure if it doesn't exist.
    pass


def pm(v, plus, minus=None, limits=None):
    """
    Return the tuple (~v-dv,~v+dv), where ~expr is a 'nice' number near to
    to the value of expr.  For example::

        >>> r = pm(0.78421, 0.0023145)
        >>> print("%g - %g"%r)
        0.7818 - 0.7866

    If called as pm(value, +dp, -dm) or pm(value, -dm, +dp),
    return (~v-dm, ~v+dp).
    """
    return nice_range(limited_range(pm_raw(v, plus, minus), limits=limits))


def pmp(v, plus, minus=None, limits=None):
    """
    Return the tuple (~v-%v,~v+%v), where ~expr is a 'nice' number near to
    the value of expr.  For example::

        >>> r = pmp(0.78421, 10)
        >>> print("%g - %g"%r)
        0.7 - 0.87
        >>> r = pmp(0.78421, 0.1)
        >>> print("%g - %g"%r)
        0.7834 - 0.785

    If called as pmp(value, +pp, -pm) or pmp(value, -pm, +pp),
    return (~v-pm%v, ~v+pp%v).
    """
    return nice_range(limited_range(pmp_raw(v, plus, minus), limits=limits))

# Generate ranges using x +/- dx or x +/- p%*x


def pm_raw(v, plus, minus=None):
    """
    Return the tuple [v-dv,v+dv].

    If called as pm_raw(value, +dp, -dm) or pm_raw(value, -dm, +dp),
    return (v-dm, v+dp).
    """
    if minus is None:
        minus = -plus
    if plus < minus:
        plus, minus = minus, plus
    return v + minus, v + plus


def pmp_raw(v, plus, minus=None):
    """
    Return the tuple [v-%v,v+%v]

    If called as pmp_raw(value, +pp, -pm) or pmp_raw(value, -pm, +pp),
    return (v-pm%v, v+pp%v).
    """
    if minus is None:
        minus = -plus
    if plus < minus:
        plus, minus = minus, plus
    b1, b2 = v * (1 + 0.01 * minus), v * (1 + 0.01 * plus)

    return (b1, b2) if v > 0 else (b2, b1)

def limited_range(bounds, limits=None):
    """
    Given a range and limits, fix the endpoints to lie within the range
    """
    if limits is not None:
        return clip(bounds[0], *limits), clip(bounds[1], *limits)
    return bounds

def nice_range(bounds):
    """
    Given a range, return an enclosing range accurate to two digits.
    """
    step = bounds[1] - bounds[0]
    if step > 0:
        d = 10 ** (floor(log10(step)) - 1)
        return floor(bounds[0]/d)*d, ceil(bounds[1]/d)*d
    else:
        return bounds


def init_bounds(v):
    """
    Returns a bounds object of the appropriate type given the arguments.

    This is a helper factory to simplify the user interface to parameter
    objects.
    """
    # if it is none, then it is unbounded
    if v is None:
        return Unbounded()

    # if it isn't a tuple, assume it is a bounds type.
    try:
        lo, hi = v
    except TypeError:
        return v

    # if it is a tuple, then determine what kind of bounds we have
    if lo is None:
        lo = -inf
    if hi is None:
        hi = inf
    # TODO: consider issuing a warning instead of correcting reversed bounds
    if lo >= hi:
        lo, hi = hi, lo
    if isinf(lo) and isinf(hi):
        return Unbounded()
    elif isinf(lo):
        return BoundedAbove(hi)
    elif isinf(hi):
        return BoundedBelow(lo)
    else:
        return Bounded(lo, hi)


class Bounds(object):
    """
    Bounds abstract base class.

    A range is used for several purposes.  One is that it transforms parameters
    between unbounded and bounded forms depending on the needs of the optimizer.

    Another is that it generates random values in the range for stochastic
    optimizers, and for initialization.

    A third is that it returns the likelihood of seeing that particular value
    for optimizers which use soft constraints.  Assuming the cost function that
    is being optimized is also a probability, then this is an easy way to
    incorporate information from other sorts of measurements into the model.
    """
    limits = (-inf, inf)
    # TODO: need derivatives wrt bounds transforms

    def get01(self, x):
        """
        Convert value into [0,1] for optimizers which are bounds constrained.

        This can also be used as a scale bar to show approximately how close to
        the end of the range the value is.
        """

    def put01(self, v):
        """
        Convert [0,1] into value for optimizers which are bounds constrained.
        """

    def getfull(self, x):
        """
        Convert value into (-inf,inf) for optimizers which are unconstrained.
        """

    def putfull(self, v):
        """
        Convert (-inf,inf) into value for optimizers which are unconstrained.
        """

    def random(self, n=1, target=1.0):
        """
        Return a randomly generated valid value.

        *target* gives some scale independence to the random number
        generator, allowing the initial value of the parameter to influence
        the randomly generated value.  Otherwise fits without bounds have
        too large a space to search through.
        """

    def nllf(self, value):
        """
        Return the negative log likelihood of seeing this value, with
        likelihood scaled so that the maximum probability is one.

        For uniform bounds, this either returns zero or inf.  For bounds
        based on a probability distribution, this returns values between
        zero and inf.  The scaling is necessary so that indefinite and
        semi-definite ranges return a sensible value.  The scaling does
        not affect the likelihood maximization process, though the resulting
        likelihood is not easily interpreted.
        """

    def residual(self, value):
        """
        Return the parameter 'residual' in a way that is consistent with
        residuals in the normal distribution.  The primary purpose is to
        graphically display exceptional values in a way that is familiar
        to the user.  For fitting, the scaled likelihood should be used.

        To do this, we will match the cumulative density function value
        with that for N(0,1) and find the corresponding percent point
        function from the N(0,1) distribution.  In this way, for example,
        a value to the right of 2.275% of the distribution would correspond
        to a residual of -2, or 2 standard deviations below the mean.

        For uniform distributions, with all values equally probable, we
        use a value of +/-4 for values outside the range, and 0 for values
        inside the range.
        """

    def start_value(self):
        """
        Return a default starting value if none given.
        """
        return self.put01(0.5)

    def __contains__(self, v):
        return self.limits[0] <= v <= self.limits[1]

    def __str__(self):
        limits = tuple(num_format(v) for v in self.limits)
        return "(%s,%s)" % limits

    def to_dict(self):
        return dict(
            type=type(self).__name__,
            limits=self.limits,
            )

# CRUFT: python 2.5 doesn't format indefinite numbers properly on windows


def num_format(v):
    """
    Number formating which supports inf/nan on windows.
    """
    if isfinite(v):
        return "%g" % v
    elif isinf(v):
        return "inf" if v > 0 else "-inf"
    else:
        return "NaN"


class Unbounded(Bounds):
    """
    Unbounded parameter.

    The random initial condition is assumed to be between 0 and 1

    The probability is uniformly 1/inf everywhere, which means the negative
    log likelihood of P is inf everywhere.  A value inf will interfere
    with optimization routines, and so we instead choose P == 1 everywhere.
    """

    def random(self, n=1, target=1.0):
        scale = target + (target == 0.)
        return RNG.randn(n)*scale

    def nllf(self, value):
        return 0

    def residual(self, value):
        return 0

    def get01(self, x):
        return _get01_inf(x)

    def put01(self, v):
        return _put01_inf(v)

    def getfull(self, x):
        return x

    def putfull(self, v):
        return v


class BoundedBelow(Bounds):
    """
    Semidefinite range bounded below.

    The random initial condition is assumed to be within 1 of the maximum.

    [base,inf] <-> (-inf,inf) is direct above base+1, -1/(x-base) below
    [base,inf] <-> [0,1] uses logarithmic compression.

    Logarithmic compression works by converting sign*m*2^e+base to
    sign*(e+1023+m), yielding a value in [0,2048]. This can then be
    converted to a value in [0,1].

    Note that the likelihood function is problematic: the true probability
    of seeing any particular value in the range is infinitesimal, and that
    is indistinguishable from values outside the range.  Instead we say
    that P = 1 in range, and 0 outside.
    """

    def __init__(self, base):
        self.limits = (base, inf)
        self._base = base

    def start_value(self):
        return self._base + 1

    def random(self, n=1, target=1.):
        target = max(abs(target), abs(self._base))
        scale = target + (target == 0.)
        return self._base + abs(RNG.randn(n)*scale)

    def nllf(self, value):
        return 0 if value >= self._base else inf

    def residual(self, value):
        return 0 if value >= self._base else -4

    def get01(self, x):
        m, e = math.frexp(x - self._base)
        if m >= 0 and e <= _E_MAX:
            v = (e + m) / (2. * _E_MAX)
            return v
        else:
            return 0 if m < 0 else 1

    def put01(self, v):
        v = v * 2 * _E_MAX
        e = int(v)
        m = v - e
        x = math.ldexp(m, e) + self._base
        return x

    def getfull(self, x):
        v = x - self._base
        return v if v >= 1 else 2 - 1. / v

    def putfull(self, v):
        x = v if v >= 1 else 1. / (2 - v)
        return x + self._base


class BoundedAbove(Bounds):
    """
    Semidefinite range bounded above.

    [-inf,base] <-> [0,1] uses logarithmic compression
    [-inf,base] <-> (-inf,inf) is direct below base-1, 1/(base-x) above

    Logarithmic compression works by converting sign*m*2^e+base to
    sign*(e+1023+m), yielding a value in [0,2048].  This can then be
    converted to a value in [0,1].

    Note that the likelihood function is problematic: the true probability
    of seeing any particular value in the range is infinitesimal, and that
    is indistinguishable from values outside the range.  Instead we say
    that P = 1 in range, and 0 outside.
    """

    def __init__(self, base):
        self.limits = (-inf, base)
        self._base = base

    def start_value(self):
        return self._base - 1

    def random(self, n=1, target=1.0):
        target = max(abs(self._base), abs(target))
        scale = target + (target == 0.)
        return self._base - abs(RNG.randn(n)*scale)

    def nllf(self, value):
        return 0 if value <= self._base else inf

    def residual(self, value):
        return 0 if value <= self._base else 4

    def get01(self, x):
        m, e = math.frexp(self._base - x)
        if m >= 0 and e <= _E_MAX:
            v = (e + m) / (2. * _E_MAX)
            return 1 - v
        else:
            return 1 if m < 0 else 0

    def put01(self, v):
        v = (1 - v) * 2 * _E_MAX
        e = int(v)
        m = v - e
        x = -(math.ldexp(m, e) - self._base)
        return x

    def getfull(self, x):
        v = x - self._base
        return v if v <= -1 else -2 - 1. / v

    def putfull(self, v):
        x = v if v <= -1 else -1. / (v + 2)
        return x + self._base


class Bounded(Bounds):
    """
    Bounded range.

    [lo,hi] <-> [0,1] scale is simple linear
    [lo,hi] <-> (-inf,inf) scale uses exponential expansion

    While technically the probability of seeing any value within the
    range is 1/range, for consistency with the semi-infinite ranges
    and for a more natural mapping between nllf and chisq, we instead
    set the probability to 0.  This choice will not affect the fits.
    """

    def __init__(self, lo, hi):
        self.limits = (lo, hi)
        self._nllf_scale = log(hi - lo)

    def random(self, n=1, target=1.0):
        lo, hi = self.limits
        #print("= uniform",lo,hi)
        return RNG.uniform(lo, hi, size=n)

    def nllf(self, value):
        lo, hi = self.limits
        return 0 if lo <= value <= hi else inf
        # return self._nllf_scale if lo<=value<=hi else inf

    def residual(self, value):
        lo, hi = self.limits
        return -4 if lo > value else (4 if hi < value else 0)

    def get01(self, x):
        lo, hi = self.limits
        return float(x - lo) / (hi - lo) if hi - lo > 0 else 0

    def put01(self, v):
        lo, hi = self.limits
        return (hi - lo) * v + lo

    def getfull(self, x):
        return _put01_inf(self.get01(x))

    def putfull(self, v):
        return self.put01(_get01_inf(v))


class Distribution(Bounds):
    """
    Parameter is pulled from a distribution.

    *dist* must implement the distribution interface from scipy.stats.
    In particular, it should define methods rvs, nnlf, cdf and ppf and
    attributes args and dist.name.
    """

    def __init__(self, dist):
        self.dist = dist

    def random(self, n=1, target=1.0):
        return self.dist.rvs(n)

    def nllf(self, value):
        return -log(self.dist.pdf(value))

    def residual(self, value):
        return normal_distribution.ppf(self.dist.cdf(value))

    def get01(self, x):
        return self.dist.cdf(x)

    def put01(self, v):
        return self.dist.ppf(v)

    def getfull(self, x):
        return x

    def putfull(self, v):
        return v

    def __getstate__(self):
        # WARNING: does not preserve and restore seed
        return self.dist.__class__, self.dist.args, self.dist.kwds

    def __setstate__(self, state):
        cls, args, kwds = state
        self.dist = cls(*args, **kwds)

    def __str__(self):
        return "%s(%s)" % (self.dist.dist.name,
                           ",".join(str(s) for s in self.dist.args))

    def to_dict(self):
        return dict(
            type=type(self).__name__,
            limits=self.limits,
            # TODO: how to handle arbitrary distribution function in save/load?
            dist=type(self.dist).__name__,
            )


class Normal(Distribution):
    """
    Parameter is pulled from a normal distribution.

    If you have measured a parameter value with some uncertainty (e.g., the
    film thickness is 35+/-5 according to TEM), then you can use this
    measurement to restrict the values given to the search, and to penalize
    choices of this fitting parameter which are different from this value.

    *mean* is the expected value of the parameter and *std* is the 1-sigma
    standard deviation.
    """

    def __init__(self, mean=0, std=1):
        Distribution.__init__(self, normal_distribution(mean, std))
        self._nllf_scale = log(sqrt(2 * pi * std ** 2))

    def nllf(self, value):
        # P(v) = exp(-0.5*(v-mean)**2/std**2)/sqrt(2*pi*std**2)
        # -log(P(v)) = -(-0.5*(v-mean)**2/std**2 - log( (2*pi*std**2) ** 0.5))
        #            = 0.5*(v-mean)**2/std**2 + log(2*pi*std**2)/2
        mean, std = self.dist.args
        return 0.5 * ((value-mean)/std)**2 + self._nllf_scale

    def residual(self, value):
        mean, std = self.dist.args
        return (value-mean)/std

    def __getstate__(self):
        return self.dist.args  # args is mean,std

    def __setstate__(self, state):
        mean, std = state
        self.__init__(mean=mean, std=std)


class BoundedNormal(Bounds):
    """
    truncated normal bounds
    """

    def __init__(self, sigma=1, mu=0, limits=(-inf, inf)):
        self.limits = limits
        self.sigma, self.mu = sigma, mu

        self._left = normal_distribution.cdf((limits[0]-mu)/sigma)
        self._delta = normal_distribution.cdf((limits[1]-mu)/sigma) - self._left
        self._nllf_scale = log(sqrt(2 * pi * sigma ** 2)) + log(self._delta)

    def get01(self, x):
        """
        Convert value into [0,1] for optimizers which are bounds constrained.

        This can also be used as a scale bar to show approximately how close to
        the end of the range the value is.
        """
        v = ((normal_distribution.cdf((x-self.mu)/self.sigma) - self._left)
             / self._delta)
        return clip(v, 0, 1)

    def put01(self, v):
        """
        Convert [0,1] into value for optimizers which are bounds constrained.
        """
        x = v * self._delta + self._left
        return normal_distribution.ppf(x) * self.sigma + self.mu

    def getfull(self, x):
        """
        Convert value into (-inf,inf) for optimizers which are unconstrained.
        """
        raise NotImplementedError

    def putfull(self, v):
        """
        Convert (-inf,inf) into value for optimizers which are unconstrained.
        """
        raise NotImplementedError

    def random(self, n=1, target=1.0):
        """
        Return a randomly generated valid value, or an array of values
        """
        return self.get01(RNG.rand(n))

    def nllf(self, value):
        """
        Return the negative log likelihood of seeing this value, with
        likelihood scaled so that the maximum probability is one.
        """
        if value in self:
            return 0.5 * ((value-self.mu)/self.sigma)**2 + self._nllf_scale
        else:
            return inf

    def residual(self, value):
        """
        Return the parameter 'residual' in a way that is consistent with
        residuals in the normal distribution.  The primary purpose is to
        graphically display exceptional values in a way that is familiar
        to the user.  For fitting, the scaled likelihood should be used.

        For the truncated normal distribution, we can just use the normal
        residuals.
        """
        return (value - self.mu) / self.sigma

    def start_value(self):
        """
        Return a default starting value if none given.
        """
        return self.put01(0.5)

    def __contains__(self, v):
        return self.limits[0] <= v <= self.limits[1]

    def __str__(self):
        vals = (
            self.limits[0], self.limits[1],
            self.mu, self.sigma,
        )
        return "(%s,%s), norm(%s,%s)" % tuple(num_format(v) for v in vals)


class SoftBounded(Bounds):
    """
    Parameter is pulled from a stretched normal distribution.

    This is like a rectangular distribution, but with gaussian tails.

    The intent of this distribution is for soft constraints on the values.
    As such, the random generator will return values like the rectangular
    distribution, but the likelihood will return finite values based on
    the distance from the from the bounds rather than returning infinity.

    Note that for bounds constrained optimizers which force the value
    into the range [0,1] for each parameter we don't need to use soft
    constraints, and this acts just like the rectangular distribution.
    """

    def __init__(self, lo, hi, std=None):
        self._lo, self._hi, self._std = lo, hi, std
        self._nllf_scale = log(hi - lo + sqrt(2 * pi * std))

    def random(self, n=1, target=1.0):
        return RNG.uniform(self._lo, self._hi, size=n)

    def nllf(self, value):
        # To turn f(x) = 1 if x in [lo,hi] else G(tail)
        # into a probability p, we need to normalize by \int{f(x)dx},
        # which is just hi-lo + sqrt(2*pi*std**2).
        if value < self._lo:
            z = self._lo - value
        elif value > self._hi:
            z = value - self._hi
        else:
            z = 0
        return (z / self._std) ** 2 / 2 + self._nllf_scale

    def residual(self, value):
        if value < self._lo:
            z = self._lo - value
        elif value > self._hi:
            z = value - self._hi
        else:
            z = 0
        return z / self._std

    def get01(self, x):
        v = float(x - self._lo) / (self._hi - self._lo)
        return v if 0 <= v <= 1 else (0 if v < 0 else 1)

    def put01(self, v):
        return v * (self._hi - self._lo) + self._lo

    def getfull(self, x):
        return x

    def putfull(self, v):
        return v

    def __str__(self):
        return "box_norm(%g,%g,sigma=%g)" % (self._lo, self._hi, self._std)


_E_MIN = -1023
_E_MAX = 1024

def _get01_inf(x):
    """
    Convert a floating point number to a value in [0,1].

    The value sign*m*2^e to sign*(e+1023+m), yielding a value in [-2048,2048].
    This can then be converted to a value in [0,1].

    Sort order is preserved.  At least 14 bits of precision are lost from
    the 53 bit mantissa.
    """
    # Arctan alternative
    # Arctan is approximately linear in (-0.5, 0.5), but the
    # transform is only useful up to (-10**15,10**15).
    # return atan(x)/pi + 0.5
    m, e = math.frexp(x)
    s = math.copysign(1.0, m)
    v = (e - _E_MIN + m * s) * s
    v = v / (4 * _E_MAX) + 0.5
    v = 0 if _E_MIN > e else (1 if _E_MAX < e else v)
    return v


def _put01_inf(v):
    """
    Convert a value in [0,1] to a full floating point number.

    Sort order is preserved.  Reverses :func:`_get01_inf`, but with fewer
    bits of precision.
    """
    # Arctan alternative
    # return tan(pi*(v-0.5))

    v = (v - 0.5) * 4 * _E_MAX
    s = math.copysign(1., v)
    v *= s
    e = int(v)
    m = v - e
    x = math.ldexp(s * m, e + _E_MIN)
    # print "< x,e,m,s,v",x,e+_e_min,s*m,s,v
    return x
