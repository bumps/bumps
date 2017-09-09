"""
MCMC step acceptance test.
"""
from __future__ import with_statement

__all__ = ["metropolis", "metropolis_dr"]

from numpy import exp, sqrt, minimum, where, cov, eye, array, dot, errstate
from numpy.linalg import norm, cholesky, inv
from . import util

import os
BUMPS_TEMPERATURE = float(os.environ.get('BUMPS_TEMPERATURE', '1'))

def paccept(logp_old, logp_try):
    """
    Returns the probability of taking a metropolis step given two
    log density values.
    """
    return exp(minimum(logp_try-logp_old, 0)/BUMPS_TEMPERATURE)


def metropolis(xtry, logp_try, xold, logp_old, step_alpha):
    """
    Metropolis rule for acceptance or rejection

    Generates the next generation, *newgen* from::

        x_new[k] = x[k]     if U > alpha
                 = x_old[k] if U <= alpha

    where alpha is p/p_old and accept is U > alpha.

    Returns x_new, logp_new, alpha, accept
    """
    with errstate(under='ignore'):
        alpha = paccept(logp_try=logp_try, logp_old=logp_old)
        alpha *= step_alpha
    accept = alpha > util.rng.rand(*alpha.shape)
    logp_new = where(accept, logp_try, logp_old)
    ## The following only works for vectors:
    # xnew = where(accept, xtry, xold)
    xnew = xtry+0
    for i, a in enumerate(accept):
        if not a:
            xnew[i] = xold[i]

    return xnew, logp_new, alpha, accept


def dr_step(x, scale):
    """
    Delayed rejection step.
    """

    # Compute the Cholesky Decomposition of X
    nchains, npars = x.shape
    r = (2.38/sqrt(npars)) * cholesky(cov(x.T) + 1e-5*eye(npars))

    # Now do a delayed rejection step for each chain
    delta_x = dot(util.rng.randn(*x.shape), r)/scale

    # Generate ergodicity term
    eps = 1e-6 * util.rng.randn(*x.shape)

    # Update x_old with delta_x and eps;
    return x + delta_x + eps, r


def metropolis_dr(xtry, logp_try, x, logp, xold, logp_old, alpha12, R):
    """
    Delayed rejection metropolis
    """

    # Compute alpha32 (note we turned x and xtry around!)
    alpha32 = paccept(logp_try=logp, logp_old=logp_try)

    # Calculate alpha for each chain
    l2 = paccept(logp_try=logp_try, logp_old=logp_old)
    iR = inv(R)
    q1 = array([exp(-0.5*(norm(dot(x2-x1, iR))**2 - norm(dot(x1-x0, iR))**2))
                for x0, x1, x2 in zip(xold, x, xtry)])
    alpha13 = l2*q1*(1-alpha32)/(1-alpha12)

    accept = alpha13 > util.rng.rand(*alpha13.shape)
    logp_new = where(accept, logp_try, logp)
    ## The following only works for vectors:
    # xnew = where(accept, xtry, x)
    xnew = xtry+0
    for i, a in enumerate(accept):
        if not a:
            xnew[i] = x[i]

    return xnew, logp_new, alpha13, accept
