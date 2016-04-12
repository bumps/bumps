# Fitting Poisson data
# ====================
#
# Data from poisson processes, such as the number of counts per unit time
# or counts per unit area, do not have the same pattern of uncertainties
# as data from gaussian processes.  Poisson data consists of natural
# numbers occurring at some underlying rate.  The fitting process checks
# if the number of counts observed is consistent with the proposed rate
# for each point in the dataset, much like the fitting process for gaussian
# data checks if the observed value is consistent with the proposed value
# within the measurement uncertainty.
#
# Using :class:`bumps.curve.PoissonCurve` instead of :class:`bumps.curve.Curve`,
# we can fit a set of *counts* at conditions *x* using a function
# *f(x, p1, p2, ...)* to propose rates for the various *x* values given the
# parameters, yielding parameter values *p1, p2, ...* that are most consistent
# with the *counts* at *x*. When measuring poisson processes, the underlying
# rate is not known, so the measurement variance, which is a property of the
# rate, is not associated with the data but instead associated with the
# theory function which predicts the rates.  This is opposite from what we
# have with gaussian data, in which the uncertainty is associated with the
# measurement device, and explains why the call to PoissonCurve only accepts
# *x* and *counts*, not *x*, *y*, and *dy*.
#
# One property of the Poisson distribution is that it is well approximated
# by a gaussian distribution for values above about 10.   It will never be
# perfect match since numbers from a poisson distribution can never be
# negative, whereas gaussian numbers can always be negative, albeit with
# vanishingly small probability some of the time.  Below 10, there are
# various ways you can approximate the poisson distribution with a gaussian.
# This example explores some of the options.
#
# In particular, the handling of zero counts can be problematic when treating
# the measurement as gaussian.  You cannot simply drop the points with zero
# counts. Once you've done various reduction steps, the resulting non-zero
# value for the uncertainty will carry meaning.  The longer you count,
# the smaller the uncertainty should be, once you've normalized for counting
# time or monitor.  Being off by a factor of 2 on the residuals is much
# better than being off by a factor of infinity using uncertainty = zero,
# and better than dropping the point altogether.
#
# There are a few things you can do with zero counts without being
# completely arbitrary:
#
#   1) $\lambda = (k+1) \pm \sqrt{k+1}$ for all $k$
#   2) $\lambda = (k+1/2) \pm \sqrt{k+1/4}$ for all k
#   3) $\lambda = k \pm \sqrt{k+1}$ for all k
#   4) $\lambda = k \pm \sqrt{k}$ for $k>0$, $1/2 \pm 1/2$ for $k = 0$
#   5) $\lambda = k \pm \sqrt{k}$ for $k>0$, $0 \pm 1$ for $k = 0$
#
# See the notes from the CDF Statistics Committee for details at
# `<http://www-cdf.fnal.gov/physics/statistics/notes/pois_eb.txt>`_.
#
# Of these, option 5 works slightly better for fitting, giving the best
# estimate of the background.
#
# The ideal case is to have your model produce an expected number of counts
# on the detector.  It is then trivial to compute the probability of
# seeing the observed counts from the expected counts and fit the parameters
# using PoissonCurve.  Unfortunately, this means incorporating all
# instrumental effects when modelling the measurement rather than correcting
# for instrumental effects in a data reduction program, and using a common
# sample model independent of instrument.
#
# Setting $\lambda = k$ is good since that is the maximum likelihood value
# for $\lambda$ given observed $k$, but this breaks down at $k=0$, giving zero
# uncertainty regardless of how long we measured.
#
# Since the Poisson distribution is slightly skew, a good estimate is
# $\lambda = k+1$ (option 1 above).  This follows from the formula for the
# expected value of a distribution:
#
# .. math::
#
#    E[x] = \int_{-infty}^\infty x P(x) dx
#
# For the poisson distribution, this is:
#
# .. math::
#
#    E[\lambda] = \int_0^\infty \lambda \frac{\lambda^k e^{-\lambda}}{k!} d\lambda
#
# Running some simulations, we can see that $\hat\lambda=(k+1)\pm\sqrt{k+1}$
# (see `sim.py <sim.html>`_). This is the best fit rms value to the distribution
# of possible $\lambda$ values that could give rise to the observed $k$.
#
# Convincing the world to accept $\lambda = k+1$ would be challenging since
# the expected value is not the most likely value.  As a compromise, one can
# use $0 \pm 1$ for zero counts, and $k \pm \sqrt{k}$ for other values.  A
# minor problem is that this permits negative count rates for zero without
# significant penalty.
#
# Note that from the simulation, the variance on $\lambda$ given $\lambda=k$
# is also $k+1$.
#
# Another suggestion is to choose the center and bounds so that the
# uncertainty covers $1-\sigma$ from the distribution (68%).  A simple
# approximation which does this is $(n+1/2) \pm \sqrt{n+1/4}$.
#
# Again, hard to convince the world to do, so one could compromise and
# choose $1/2 \pm 1/2$ for $k=0$, and the usual $k \pm \sqrt{k}$ otherwise.
#
# What follows is a model which allows us to fit a simulated peak using
# these various definitions of $\lambda$ and see which version best recovers
# the true parameters which generated the peak.

from bumps.names import *

# Define the peak shape.  We are using a simple gaussian with center, width,
# scale and background.

def peak(x, scale, center, width, background):
    return scale*np.exp(-0.5*(x-center)**2/width**2) + background

# Generate simulated peak data with poisson noise.  When running the fit,
# you can choose various values for the peak intensity.  We are using a
# large number of points so that the peak is highly constrained by the
# data, and the returned parameters are consistent from run to run.  Real
# data is likely not so heavily sampled.

x = np.linspace(5,20,345)
#y = np.random.poisson(peak(x, 1000, 12, 1.0, 1))
#y = np.random.poisson(peak(x, 300, 12, 1.5, 1))
y = np.random.poisson(peak(x, 3, 12, 1.5, 1))

# Define the various conditions.  These can be selected on the command
# line by listing the condition name after the model file.  Note that
# bumps will make any option not preceded by "-" available to the model
# file as elements of *sys.argv*.  *sys.argv[0]* is the model file itself.
#
# The options correspond to the five options listed above, with an additional
# option "poisson" which is used to select PoissonCurve rather than Curve
# in the fit.

cond = sys.argv[1] if len(sys.argv) > 1 else "pearson"
if cond=="poisson": # option 0: use PoissonCurve rather than Curve to fit
    pass
elif cond=="expected": # option 1: L = (y+1) +/- sqrt(y+1)
    y += 1
    dy = np.sqrt(y)
elif cond=="pearson": # option 2: L = (y + 0.5)  +/- sqrt(y + 1/4)
    dy = np.sqrt(y+0.25)
    y = y + 0.5
elif cond=="expected_mle": # option 3: L = y +/- sqrt(y+1)
    dy = np.sqrt(y+1)
elif cond=="pearson_zero": # option 4: L = y +/- sqrt(y); L[0] = 0.5 +/- 0.5
    dy = np.sqrt(y)
    y = np.asarray(y, 'd')
    y[y==0] = 0.5
    dy[y==0] = 0.5
elif cond=="expected_zero": # option 5: L = y +/- sqrt(y);  L[0] = 0 +/- 1
    dy = np.sqrt(y)
    dy[y==0] = 1.0
else:
    raise RuntimeError("Need to select uncertainty: pearson, pearson_zero, expected, expected_zero, expected_mle, poisson")

# Build the fitter, and set the range on the fit parameters.

if cond == "poisson":
    M = PoissonCurve(peak,x,y,scale=1,center=2,width=2,background=0)
else:
    M = Curve(peak,x,y,dy,scale=1,center=2,width=2,background=0)
dx = max(x)-min(x)
M.scale.range(0,max(y)*1.5)
M.center.range(min(x)-0.2*dx,max(x)+0.2*dx)
M.width.range(0,0.7*dx)
M.background.range(0,max(y))

# Set the fit problem as usual.

problem = FitProblem(M)

# We can now load and run the fit.  Be sure to substitute COND for one of the
# conditions defined above:
#
# .. parsed-literal::
#
#    $ bumps.py poisson.py --fit=dream --burn=600 --store=/tmp/T1 COND
#
# Comparing the results for the various conditions, we can see that all methods
# yield a good fit to the underlying center, scale and width.  It is only the
# background that causes problems.  Using poisson statistics for the fit gives
# the proper background estimate, and using the traditional method of
# $\lambda = k \pm \sqrt{k}$ for $k>0$, and $0 \pm 1$ for $k=1$ gives the
# best gaussian approximation.
#
# .. table:: Fit results
#
#     = ================= ==========
#     # method            background
#     = ================= ==========
#     0 poisson           1.0
#     1 expected          1.55
#     2 pearson           0.16
#     3 expected_mle      0.55
#     4 pearson_zero      0.34
#     5 expected_zero     0.75
#     = ================= ==========
