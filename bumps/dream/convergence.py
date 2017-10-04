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

def ks_converged(state, n=5, density=0.6, alpha=0.1, samples=1000):
    # type: ("MCMCDraw", int, float, float, int) -> bool
    """
    Return True if the MCMC has converged according to the K-S window test.

    Since we are only looking at the distribution of logp values, and not the
    individual points, we should be relatively stable regardless of the
    properties of the sampler.  The main reason for failure will be "stuck"
    fits which have not had a chance to jump to a lower minimum.
    """
    # Make sure we have the desired number of draws
    if state.generation < state.Ngen or not state.stable_best():
        return False
    if state.Nsamples < 2*samples:
        window_size = state.Ngen//2
    else:
        window_size = samples//state.Npop + 1
    n_draw = int(density * window_size * state.Npop)
    # Grab a window at the start and the end
    head = state.logp_slice(window_size).flatten()
    tail = state.logp_slice(-window_size).flatten()

    # Quick fail if logp head is worse than logp tail
    if np.min(head) < state.min_slice(-state.Ngen//2):
        print("fast reject", np.min(head), state.min_slice(-state.Ngen//2))
        return False

    # Do a few draws from that window, seeing if any fail
    for _ in range(n):
        f_samp = choice(head, n_draw, replace=True)
        r_samp = choice(tail, n_draw, replace=True)
        p_val = ks_2samp(f_samp, r_samp)[1]
        if p_val < alpha:
            print("ks not converged", p_val, alpha)
            # head and tail are significantly different, so not converged
            return False
        print("ks converged", p_val, alpha)
    return True

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

    *density=0.1* is the proportion of samples to select from the window

    *alpha=0.01* is the significance level for the test

    *samples=5000* is the size of the sample window. If the window is too big
    the test will falsly end burn when the start of the window is still
    converging.  If the window is too small the test will take a long time,
    and will start to show effects of autocorrelation (efficient MCMC
    samplers move slowly across the posterior probability space, showing
    short term autocorrelation between samples.)

    *reserved=0.5* portion of chain to reserve at the end
    """
    if method == 'window':
        trials = [_ks_sliding_window(state, **kwargs) for _ in range(n)]
    else:
        raise ValueError("Unknown convergence test "+method)

    #print("trials", trials)
    #return 0
    return -1 if -1 in trials else np.amax(trials)

def _ks_sliding_window(state, density=0.1, alpha=0.01, samples=1000, reserved=0.5):
    if state.Nsamples < 2*samples:
        window_size = state.Ngen//2
    else:
        window_size = samples//state.Npop + 1
    n_draw = int(density * window_size * state.Npop)
    cutoff_point = state.Ngen//2
    _, logp = state.logp()

    tail = logp[cutoff_point:].flatten()
    min_tail = np.min(tail)

    # Check in large bunches
    for index in range(0, cutoff_point, window_size):
        # [PAK] make sure the worst point is not in the first window.
        # Stastically this will introduce some bias (by chance the max could
        # happen to occur in the first window) but it will be small when the
        # window is small relative to the full pool.  A better test would
        # count the number of samples worse than the all the tail, compute
        # the probability, and reject according to a comparison with a uniform
        # number in [0,1].  To much work for so little bias.
        window = logp[index:index+window_size].flatten()
        if np.min(window) < min_tail:
            continue

        # [PAK] Using replace=True since it is more efficient and practically
        # indistinguishable for a small sampling portion such as 10% or less.
        f_samp = choice(window, n_draw, replace=True)
        r_samp = choice(tail, n_draw, replace=True)

        p_val = ks_2samp(f_samp, r_samp)[1]

        if p_val > alpha:
            # head and tail are not significantly different, so break
            break
        #print("big step", index, window_size)

    if index >= cutoff_point:
        return -1

    # check in smaller steps for fine tuned stopping
    tiny_window = window_size//11 + 1
    for index in range(index, index+window_size, tiny_window):
        window = logp[index:index+tiny_window].flatten()
        if np.min(window) < min_tail:
            continue

        # [PAK] Using replace=True since it is more efficient and practically
        # indistinguishable for a small sampling portion such as 10% or less.
        f_samp = choice(window, n_draw, replace=True)
        r_samp = choice(tail, n_draw, replace=True)

        p_val = ks_2samp(f_samp, r_samp)[1]

        if p_val > alpha:
            # head and tail are not significantly different, so break
            break
        #print("little step", index, tiny_window)

    return index
