# Check the entropy calculator
# ============================
#
# A single measure for a multivariate distribution is the entropy
#
# .. math:
#    E = - \int_\Omega \log(p(x)) p(x)\,\text{d}x
#
# By comparing the entropy of the prior distribution (usually a box uniform
# distribution with entropy $\sum_{i=1}^n \log(w_i)$ where $w_i$ is the
# range on parameter $i$ and $n$ is the number of paramters, but maybe
# lower if explicit priors are given for any of the parameters based on
# information from other sources) to the entropy computed from the posterior,
# you can estimate the number of bits of information from the fit to the data.
#
# Note that bumps calculates the entropy expected from the closest multivariate
# normal distribution (MVN) as well as directly from the samples.  The sample
# derived entropy has more variability, particularly in high dimensions.
#
# Many of the probability distributions in scipy.stats include a method
# to compute the entropy of the distribution.  We can use these to test
# the values from bumps against known good values.

import numpy as np
from math import log
from scipy.stats import distributions, multivariate_normal
from bumps.names import *
from bumps.dream.entropy import Box, MultivariateT, Joint

# Create the distribution using the name and parameters from the command line.
# Provide some handy help if the no distribution is given.

# TODO: create version of dirichlet that we can sample from.
# For dirichlet, need to enforce x_k in [0,1] and sum(x) = 1.  By reducing the
# number of parameters by 1 and setting

USAGE = """
Usage: bumps check_entropy.py dist p1 p2 ...

where dist is one of the distributions in scipy.stats.distributions and
p1, p2, ... are the arguments for the distribution in the order that they
appear. For example, for the normal distribution, x ~ N(3, 0.8), use:

    bumps --fit=dream --entropy  --store=/tmp/T1 check_entropy.py norm 3 0.2
"""
def _mu_sigma(mu, sigma):
    sigma = np.asarray(sigma)
    if len(sigma.shape) == 1:
        sigma = np.diag(sigma**2)
    if mu is None:
        mu = np.zeros(sigma.shape[0])
    return mu, sigma

def mvn(sigma, mu=None):
    mu, sigma = _mu_sigma(mu, sigma)
    return multivariate_normal(mean=mu, cov=sigma)

def mvskewn(alpha, sigma, mu=None):
    sigma = np.asarray(sigma)
    assert len(sigma.shape) == 1
    if mu is None:
        mu = np.zeros(sigma.shape[0])
    Dk = [distributions.skewnorm(alpha, m, s) for m, s in zip(mu, sigma)]
    return Joint(Dk)

def mvt(df, sigma, mu=None):
    mu, sigma = _mu_sigma(mu, sigma)
    return MultivariateT(mu=mu, sigma=sigma, df=df)

def mvcauchy(sigma, mu=None):
    mu, sigma = _mu_sigma(mu, sigma)
    return MultivariateT(mu=mu, sigma=sigma, df=1)

DISTS = {
    'mvn': mvn,
    'mvt': mvt,
    'mvskewn': mvskewn,
    'mvcauchy': mvcauchy,
    'mvu': Box,
}
if len(sys.argv) > 1:
    dist_name = sys.argv[1]
    D_class = DISTS.get(dist_name, None)
    if D_class is None:
        D_class = getattr(distributions, dist_name, None)
    if D_class is None:
        print("unknown distribution " + dist_name)
        sys.exit()
    args = [[[float(vjk) for vjk in vj.split(',')] for vj in v.split(',')] if ';' in v
            else [float(vj) for vj in v.split(',')] if ',' in v
            else float(v)
            for v in sys.argv[2:]]
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
dim = getattr(D, 'dim', 1)
if dim == 1:
    M = PDF(D_nllf, x=0.9)
    M.x.range(-inf, inf)
else:
    M = VectorPDF(D_nllf, np.ones(dim))
    for k in range(dim):
        getattr(M, 'p'+str(k)).range(-inf, inf)

if dist_name == "mvskewn":
    for k in range(dim):
        getattr(M, 'p'+str(k)).value = D.distributions[k].mean()

problem = FitProblem(M)

# Before fitting, print the expected entropy from the fit.

entropy = D.entropy()
print("*** Expected entropy: %.4f bits %.4f nats"%(entropy/log(2), entropy))

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
#       mvn 1,1,1 1,2,3 # 3-D multivariate standard normal at (1,2,3)
#       mvt 4 1,1,1,1,1 # 5-D multivariate t-distribution with df=4 at origin
#       mvu 1,1,1,1,1   # 5-D unit uniform distribution centered at origin
#       mvcauchy 1,1,1  # 3-D multivariate Cauchy distribution at origin
#       mvskewn 5 1,1,1 # 3-D multivariate skew normal with alpha=5 at origin
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
