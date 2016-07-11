"""
convergence diagnostics for DREAM
"""

__all__ = ["burn_point"]

import numpy as np


def burn_point(state, method='window', n=5, *args):
    """
    attempts to find a good point to end burn in. Takes a median
    of several runs of a diagnostic

    *state* is a dream state
    *method* is the name of the convergence diagnostic (only 'window')
    *n* number of runs of convergence test to perform

    returns the index to end burn, or -1 if no point found
    """
    if method == 'window':
        _, logp = state.logp()
        trials = [_ks_sliding_window(logp, *args) for _ in range(n)]
    else:
        raise ValueError("Unknown convergence test "+method)
        
    if -1 in trials:
        return -1
    else:
        return np.median(trials)


"""
logp is the list of logp values for each chain
with shape chain length x n chains

sample is the proportion of total samples in the window
to be selected from

alpha is the significance level of the test

window is the size of the window
"""


# effects of window size:
# too big - falsly end burn if only part of window is converged
# too small - take a long time, start to see effects of autocorrelation


# TODO is cramer von mises better than a KS test?
# TODO modify so that the reserved end portion can be smaller that last 50%

def _ks_sliding_window(logp, sample=0.1, alpha=0.01, window=100):
    from scipy.stats import ks_2samp
    
    len_chain, n_chain = logp.shape
    n_draw = int(sample*window*n_chain)
    
    idx_half = len_chain // 2
    idx_burn = 0
    
    while idx_burn <= idx_half:
        f_samp = np.random.choice(logp[idx_burn:idx_burn+window].flatten(), n_draw)
        r_samp = np.random.choice(logp[idx_half:].flatten(), n_draw)
        
        p_val = ks_2samp(f_samp, r_samp)[1]
        print idx_burn, p_val
        if p_val > alpha: break
        idx_burn += window
    
    return idx_burn + window if idx_burn <= idx_half else -1

