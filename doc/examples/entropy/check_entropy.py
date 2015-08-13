# Check the entropy calculator
# ============================
#
# Many of the probability distributions in scipy.stats include a method
# to compute the entropy of the distribution.  We can use these to test
# the values from bumps against known good values.

from math import log
from scipy.stats import distributions
from bumps.names import *

# Create the distribution using the name and parameters from the command line.
# Provide some handy help if the no distribution is given.

USAGE = """
Usage: bumps check_entropy.py dist p1 p2 ...

where dist is one of the distributions in scipy.stats.distributions and
p1, p2, ... are the arguments for the distribution in the order that they
appear. For example, for the normal distribution, x ~ N(3, 0.8), use:

    bumps --fit=dream --entropy  --store=/tmp/T1 check_entropy.py norm 3 0.2
"""
if len(sys.argv) > 1:
    D_class = getattr(distributions, sys.argv[1])
    args = [float(v) for v in sys.argv[2:]]
    D = D_class(*args)
else:
    print(USAGE)
    sys.exit()

# Set the fitting problem using the direct PDF method.  In this case, bumps
# is not being used to fit data, but instead to explore the probability
# distribution directly through the negative log likelihood function.  The
# only argument to this function is the parameter value x, which becomes the
# fitting parameter.  This model file will not work for multivariate
# distributions.

def D_nllf(x):
    return -D.logpdf(x)
M = PDF(D_nllf, x=0.9)
M.x.range(-inf, inf)

problem = FitProblem(M)

# Before fitting, print the expected entropy from the fit.

print("*** Expected entropy: %.4f"%(D.entropy()/log(2)))

# To exercise the entropy calculator, try fitting some non-normal
# distributions:
#
# .. parsed-literal::
#
#       t 84            # close to normal
#       t 4             # high kurtosis
#       uniform -5 100  # high entropy
#       cauchy 0 1      # undefined variance
#       expon 0.1 0.2   # asymmetric, narrow
#       beta 0.5 0.5    # 'antimodal' u-shaped pdf
#       beta 2 5        # skewed
#
# Ideally, the entropy estimated by bumps will match the predicted entropy
# when using *--fit=dream*.  This is not the case for *beta 0.5 0.5*.  For
# the other distributions, the estimated entropy is within uncertainty of
# actual value, but the uncertainty is a bit high.
#
# The other fitters, which use the curvature at the peak to estimate
# the entropy, do not work reliably when the fit is not normal.  Try
# the same distributions with *--fit=amoeba* to see this.
#
