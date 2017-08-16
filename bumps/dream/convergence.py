r"""
Convergence diagnostics
=======================

The function :func:`burn_point` returns the point within the MCMC
chain at which the chain can be said to have converged, or -1 if
the log probabilities are still improving throughout the chain.
"""
from __future__ import division, print_function

__all__ = ["burn_point"]

import numpy as np
from scipy.stats import ks_2samp
from numpy.random import choice

# TODO is cramer von mises better than a KS test?

def burn_point(state, method='window', n=5, **kwargs):
    r"""
    Determines the point at which the MCMC chain seems to have converged.

    *state* contains the MCMC chain information.

    *method="window"* is the name of the convergence diagnostic (see below).

    *n=5* number of trials of the diagnostic method.

    The remaining arguments are method-specific.

    Returns the median value of the computed burn points, or -1 if no good
    burn point is found.

    **Kolmogorov-Smirnov sliding window**

    The "window" method detects convergence by comparing the distribution of
    $\log(p)$ values in a small window at the start of the chains and the
    values in a section at the end of the chain using a Kolmogorov-Smirnov
    test.  It accepts the following arguments:

    *sample=0.1* is the proportion of samples to select from the window

    *alpha=0.01* is the significance level for the test

    *window=100* is the size of the window. If the window is too big the
    test will falsly end burn when the start of the window is still
    converging.  If the window is too small the test will take a long time,
    and will start to show effects of autocorrelation (efficient MCMC
    samplers move slowly across the posterior probability space, showing
    short term autocorrelation between samples.)

    *reserved=0.5* portion of chain to reserve at the end
    """
    if method == 'window':
        _, logp = state.logp()
        trials = [_ks_sliding_window(logp, **kwargs) for _ in range(n)]
    else:
        raise ValueError("Unknown convergence test "+method)

    if -1 in trials:
        return logp.shape[0]//2
    else:
        return np.median(trials)

def ks_converged(state, n=5, sample=0.1, alpha=0.01, window=100):
    """
    Return True if the MCMC has converged according to the K-S window test.

    Since we are only looking at the distribution of logp values, and not the
    individual points, we should be relatively stable regardless of the
    properties of the sampler.  The main reason for failure will be "stuck"
    fits which have not had a chance to jump to a lower minimum.
    """
    # Make sure we have the desired number of draws
    if state.generation < state.Ngen:
        return False
    # Grab a window at the start and the end`
    head = state.logp_slice(window).flatten()
    tail = state.logp_slice(-window).flatten()
    # Do a few draws from that window, seeing if any fail
    n_draw = int(head.shape[0]*sample)
    for _ in range(n):
        f_samp = choice(head, n_draw, replace=True)
        r_samp = choice(tail, n_draw, replace=True)
        p_val = ks_2samp(f_samp, r_samp)[1]
        if p_val < alpha:
            return False
    return True

def _ks_sliding_window(logp, sample=0.1, alpha=0.01, window=100, reserved=0.5):
    """
    *logp* list of logp values for each chain with shape (len_chain x n_chains)
    """

    len_chain, n_chain = logp.shape
    n_draw = int(sample*window*n_chain)

    idx_res = int(reserved*len_chain)
    idx_burn = 0

    # check if we have enough samples
    if window >= idx_res:
        return -1

    while idx_burn <= idx_res:
        # [PAK] Using replace=True since it is more efficient and practically
        # indistinguishable for a small sampling portion such as 10% or less.
        f_samp = choice(logp[idx_burn:idx_burn+window].flatten(),
                        n_draw, replace=True)
        r_samp = choice(logp[idx_res:].flatten(), n_draw, replace=True)

        p_val = ks_2samp(f_samp, r_samp)[1]

        if p_val > alpha:
            break
        idx_burn += window

    return idx_burn if idx_burn <= idx_res else -1
