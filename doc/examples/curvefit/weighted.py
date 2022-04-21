# Fitting uncertainty
# ===================
#
# Like curve fit, but with an unknown scale on the error bars.  This model
# allows you to explore the effect of systematic error on a fit to a sine wave.
#
# Options are set from the bumps command line using::
#
#    bumps weighted.py option=value ...
#
# Model parameters::
#
#   freq=0.2 in [0, 100] cycles per second over the time range (tmin, tmax)
#   phase=0.2 in [-pi, pi] phase offset in radians for sine
#   amplitude=5 in [0, 20] scale of sin wave
#   offset=10 in [-500, 500] DC offset
#
# Data generation::
#
#   tmin=0, tmax=10 are the time range for the curve
#   err=5 is the measurement uncertainty as a percentage of y range
#   dy_scale=1 scales the estimated measurement uncertainty
#   y_slope=0 adds a tilt over time as a percentage of y range
#   t_jitter=0 is deviation from nominal time as a percentage of the interval
#   weight=1 is the weight on the nominal uncertainty
#
# Fitting::
#
#   fit_weight=1 if the uncertainty should be adjusted in the fit
#
# Together this forms the model::
#
#    y = amplitude sin(2 pi freq t' + phase) + offset + delta + tilt
#    t' ~ N(mu=t, sigma=t_jitter (t[1]-t[0]))
#    delta ~ N(mu=0, sigma=(2 amplitude err/100 dy_scale))
#    tilt = y_slope/100 (t - tmin)/(tmax - tmin)
#
# Run using::
#
#    bumps weighted.py --fit=dream --samples=50000 --burn=3000 --store=path opt=value ...
#
# Without fitting the weights the various systematic errors in the model
# lead to increased $\chi^2$ and decreased credible intervals on the
# parameters. For example, compare *dy_scale=3 fit_weight=0* to 
# *dy_scale=3 fit_weight=1*.
#
# Surprisingly, when fitting weights the residual normalized
# $\chi^2 \approx 1$ almost always. Using the full log likelihood function
# in the weighted curve, the decrease in $\chi^2$ from increased y uncertainty
# exactly matches the log(scale) penalty added to the likelihood. The resulting
# normalized residuals plot show the majority of the points lying in [-1, 1].
# Set *y_slope=10* to see the clear structure in the normalized residuals,
# with one end of the curve systematically above zero and the other end below.
#
# The weighting process does not appear to distort the uncertainties. Compare
# *err=1 dy_scale=1/5 fit_weight=1* to *err=1/5 dy_scale=1 fit_weight=0* to
# see that the posterior distributions on the model parameters are very
# close.
#
# For a system where statistical uncertainty is high the fitted weight is
# close to one and again the posterior distributions were minimally changed.
# Compare *t_jitter=10* to *t_jitter=0* with and without *fit_weight=0*.
# Add *err=0.1* to the mix. Here the systematic uncertainty dominates and the
# various conditions give different distributions, with the fitted weight
# showing broader parameter distributions.
#
# The form of the weight parameter is not yet unresolved. Check the current
# definition in Curve.dy_scale. It may have one of four forms:
#
# * $\sigma$ is a direct scaling of dy, so it is easy to understand.
# * $1/\sigma$ is an inverse scaling. This is an unusual definition,
#   but is useful when fitting systematic uncertainty since constraining the
#   weight to [0,1] forces the systematic uncertainty to be an increase on the
#   statistical uncertainty. This form matches the weight used in simultaneous
#   fits (bumps 0.8.1, though this may have changed). The value
#   $h = 1 / \sqrt{2 sigma^2}$ was used in the original development of the
#   normal distribution by Gauss. Some sources call this $\tau$.
# * $1/\sigma^2$ is the usual form of the weights in statistics sources. Many
#   sources call this $\tau$.
# * $Var = \sigma^2$ expresses the weight directly as the variance on the
#   individual measurements, particularly default uncertainty dy=1 is used
#   for the data.
#
# The squared forms perform poorly in this simulation, with skewed posterior
# distributions, and marginal maximum likelihood distinct from marginal
# likelihood (the green line is offset from the histogram in the individual
# variables plot). This effect is weaker on the unsquared distributions.
# It seems that the squared forms have poorer convergence in MCMC, but this
# hasn't been rigourously verified.
#

from bumps.names import *

# Set up a sine function model

def sinx(x, freq, phase, amplitude, offset):
    return amplitude*sin(2*pi*freq*x + phase) + offset

# Target model parmaters. These can be set from the command line using
# arguments of the form par=value.

class Options:
    tmin, tmax = 0, 10 # time range (s)
    freq = 0.2 # signal frequency (Hz)
    phase = 0.2 # phase shift (radians)
    amplitude = 5 # signal amplitude (arbitrary)
    offset = 10 # dc offset (amplitude units)
    weight = 1 # weight on the nominal uncertainty
    err = 5 # Size of errorbar (% yrange)
    dy_scale = 1 # Use w<1 to simulate underestimated measurement error
    y_slope = 0 # Add x% slope as systematic error
    t_jitter = 0 # Add %jitter in x position, relative to dx
    fit_weight = 1 # Use fit_weight=0 to suppress weight fitting

# Parse command line options

opt = Options()
for arg in sys.argv[1:]:
    try:
        name, value_str = arg.split('=')
        assert hasattr(opt, name)
        value = float(value_str)
    except Exception:
        print(f"Cannot parse {arg}")
        sys.exit(1)
    setattr(opt, name, value)

# Define some data. Fix the random number sequence so it is reproducible.

with push_seed(1):
    t = np.linspace(opt.tmin, opt.tmax, 100)
    dy = opt.err*2*opt.amplitude/100
    t_err = np.random.randn(*t.shape)*(opt.t_jitter/100*(t[1]-t[0]))
    y_err = np.random.randn(*t.shape)*(opt.dy_scale*dy)
    y = sinx(t+t_err, opt.freq, opt.phase, opt.amplitude, opt.offset) + y_err
    # Add x% slope to the signal as a systematic error
    y += 2*opt.amplitude*opt.y_slope/100*(t-opt.tmin)/(opt.tmax - opt.tmin)

# Prepare the fit

#M = Curve(sinx, t, y, amplitude=amplitude, phase=phase, freq=freq, offset=offset)
M = Curve(sinx, t, y, dy=dy, weight=opt.weight, amplitude=1, phase=0, freq=1, offset=0)
M.amplitude.range(0, 20)
M.offset.range(-500, 500)
M.phase.range(-pi, pi)
M.freq.range(0, 1)
if opt.fit_weight != 0.:
    #M.weight.range(0.1, 10) # allow dy to scale by 10x (w=sigma or w=1/sigma)
    M.weight.range(0.01, 100) # allow dy to scale by 10x (w=sigma^2 or w=1/sigma^2)

problem = FitProblem(M)
