r"""
Convergence diagnostics
=======================

The function :func:`burn_point` returns the point within the MCMC
chain at which the chain can be said to have converged, or -1 if
the log probabilities are still improving throughout the chain.
"""

__all__ = ["burn_point", "ks_converged"]

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import choice
from scipy.stats import chi2, ks_2samp, kstest

if TYPE_CHECKING:
    from .state import MCMCDraw

# TODO is cramer von mises better than a KS test?

# defaults should match for ks_converged and burn_point trimming
DENSITY = 0.6
ALPHA = 0.01
SAMPLES = 1000
TRIALS = 5
MIN_WINDOW = 100


def ks_converged(state, trials=TRIALS, density=DENSITY, alpha=ALPHA, samples=SAMPLES):
    # type: ("MCMCDraw", int, float, float, int) -> bool
    """
    Return True if the MCMC has converged according to the K-S window test.

    Since we are only looking at the distribution of logp values, and not the
    individual points, we should be relatively stable regardless of the
    properties of the sampler.  The main reason for failure will be "stuck"
    fits which have not had a chance to jump to a lower minimum.

    *state* contains the MCMC chain information.

    *trials* is the number of times to run the K-S test.

    *density* is the proportion of samples to select from the window.  Prefer
    lower density from larger number of samples so the sets chosen for the
    K-S test have fewer duplicates.  For density=0.1 about 5% of samples will
    be duplicates.  For density=0.6 about 25% will be duplicates.

    *alpha* is the significance level for the test.  With smaller alpha
    values the K-S test is less likely to reject the current window when
    testing against the tail of the distribution, and so the fit will end
    earlier, with more samples after the burn point.

    *samples* is the size of the sample window. If the window is too big
    the test will falsly end burn when the start of the window is still
    converging.  If the window is too small the test will take a long time,
    and will start to show effects of autocorrelation (efficient MCMC
    samplers move slowly across the posterior probability space, showing
    short term autocorrelation between samples.)  A minimum of 10 generations
    and a maximum of 1/2 the generations will be used.

    There is a strong interaction between density, alpha, samples and trials.
    If the K-S test has too many points (=density*samples), it will often
    reject simply because the different portions of the Markov chain are
    not identical (Markov chains can have short range correlations yet
    still have the full chain as a representative draw from the posterior
    distribution) unless alpha is reduced.  With fewer points, the estimated
    K-S statistic will have more variance, and so more trials will be needed
    to avoid spurious accept/reject decisions.
    """
    # Make sure we are testing for convergence
    if alpha == 0.0:
        return False

    # Make sure we have the desired number of draws
    if state.generation < state.Ngen:
        return False

    # Quick fail if best occurred within draw
    if not state.stable_best():
        # print(state.generation, "best gen", state._best_gen, "start", state.generation - state.Ngen)
        return False

    # Grab a window at the start and the end
    window_size = min(max(samples // state.Npop + 1, MIN_WINDOW), state.Ngen // 2)

    head = state.logp_slice(window_size).flatten()
    tail = state.logp_slice(-window_size).flatten()

    # Quick fail if logp head is worse than logp tail
    if np.min(head) < state.min_slice(-state.Ngen // 2):
        # print(state.generation, "improving worst", np.min(head), state.min_slice(-state.Ngen//2))
        return False

    n_draw = int(density * samples)
    reject = _robust_ks_2samp(head, tail, n_draw, trials, alpha)
    if reject:
        return False

    return True


def check_nllf_distribution(state):
    """
    Check if the nllf distribution looks like chisq.

    Note: test is not used.  It is only true for gaussian.  It fails pretty
    badly for doc/examples/test_functions.py griewank 2.  It fails even
    worse with pure integer models such as the OpenBugs asia example, which
    has discrete levels in the posterior pdf corresponding to the various
    binary configurations of the model values.
    """

    # Check that likelihood distribution looks like chi2
    # Note: cheating, and looking at the stored logp without unrolling
    reject = _check_nllf_distribution(data=-state._gen_logp.flatten(), df=state.Nvar, n_draw=10, trials=5, alpha=0.01)
    return not reject


def _check_nllf_distribution(data, df, n_draw, trials, alpha):
    # fit the best chisq to the data given df
    float_df, loc, scale = chi2.fit(data, f0=df)
    df = int(float_df + 0.5)
    cdf = lambda x: chi2.cdf(x, df, loc, scale)

    # check the quality of the fit (i.e., does the set of nllfs look vaguely
    # like the fitted chisq distribution).  Repeat the test a few times on
    # small data sets for consistency.
    p_vals = []
    for _ in range(trials):
        f_samp = choice(data, n_draw, replace=True)
        p_vals.append(kstest(data, cdf)[1])

    print("llf dist", p_vals, df, loc, scale)
    return alpha > np.mean(p_vals)


def burn_point(state, method="window", trials=TRIALS, **kwargs):
    # type: ("MCMCDraw", str, int, **dict) -> int

    r"""
    Determines the point at which the MCMC chain seems to have converged.

    *state* contains the MCMC chain information.

    *method="window"* is the name of the convergence diagnostic (see below).

    *trials* is the number of times to run the K-S test.

    Returns the index of the burn points, or -1 if no good burn point is found.

    **Kolmogorov-Smirnov sliding window**

    The "window" method detects convergence by comparing the distribution of
    $\log(p)$ values in a small window at the start of the chains and the
    values in a section at the end of the chain using a Kolmogorov-Smirnov
    test.  See :func:`ks_converged` for a description of the parameters.
    """
    if method == "window":
        index = _ks_sliding_window(state, trials=trials, **kwargs)
    else:
        raise ValueError("Unknown convergence test " + method)
    if index < 0:
        print("Did not converge!")
    return index


def _ks_sliding_window(state, trials=TRIALS, density=DENSITY, alpha=ALPHA * 0.01, samples=SAMPLES):
    _, logp = state.logp()

    window_size = min(max(samples // state.Npop + 1, MIN_WINDOW), state.Ngen // 2)
    tiny_window = window_size // 11 + 1
    half = state.Ngen // 2
    max_index = len(logp) - half - window_size

    if max_index < 0:
        return -1
    tail = logp[-half:].flatten()
    min_tail = np.min(tail)

    # Check in large bunches
    n_draw = int(density * samples)
    for index in range(0, max_index + 1, window_size):
        # [PAK] make sure the worst point is not in the first window.
        # Stastically this will introduce some bias (by chance the max could
        # happen to occur in the first window) but it will be small when the
        # window is small relative to the full pool.  A better test would
        # count the number of samples worse than the all the tail, compute
        # the probability, and reject according to a comparison with a uniform
        # number in [0,1].  To much work for so little bias.
        window = logp[index : index + window_size].flatten()
        if np.min(window) < min_tail:
            # print("step llf", index, window_size, len(window), np.min(window), min_tail)
            continue

        # if head and tail are different, slide to the next window
        reject = _robust_ks_2samp(window, tail, n_draw, trials, alpha)
        if reject:
            continue

        # Head and tail are not significantly different, so break.
        # Index is not yet updated, so the tiny step loop will start with
        # the first rejected window.
        break

    if index >= max_index:
        return -1

    # check in smaller steps for fine tuned stopping
    for index in range(index, index + window_size, tiny_window):
        window = logp[index : index + tiny_window].flatten()
        if np.min(window) < min_tail:
            # print("tiny llf", index, tiny_window, len(window), np.min(window), min_tail)
            continue

        p_val = _robust_ks_2samp(window, tail, n_draw, trials, alpha)
        # print("tiny ks", index, tiny_window, len(window), p_val, alpha)
        if p_val > alpha:
            # head and tail are not significantly different, so break
            break

    return index


def _robust_ks_2samp(f_data, r_data, n_draw, trials, alpha):
    """
    Repeat ks test n times for a more robust statistic.

    Returns True if f_data is significantly different from r_data.
    """
    p_vals = []
    for _ in range(trials):
        # [PAK] Using replace=True since it is more efficient and practically
        # indistinguishable for a small sampling portion such as 10% or less.
        # When drawing 60% of the sample size, 25% of the samples are repeated.
        f_samp = choice(f_data, n_draw, replace=True)
        r_samp = choice(r_data, n_draw, replace=True)
        p_vals.append(ks_2samp(f_samp, r_samp)[1])
    return alpha > np.mean(p_vals)
    # return any(alpha > p for p in p_vals)
