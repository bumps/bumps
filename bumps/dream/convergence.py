"""
convergence diagnostics for DREAM
"""

__all__ = ["burn_point"]

import numpy as np


def burn_point(state, method='window', n=5, **kwargs):
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
        trials = [_ks_sliding_window(logp, **kwargs) for _ in range(n)]
    else:
        raise ValueError("Unknown convergence test "+method)
        
    if -1 in trials:
        return -1
    else:
        return np.median(trials)


"""
detects convergence by comparing the distribution of logp values in a small
window at the start of the chains and the values in a section at the end of
the chain using a kolmogorov-smirnov test.

*logp* list of logp values for each chain with shape (len_chain x n_chains)
*sample* is the proportion of total samples in the window to be selected from
*alpha* is the significance level for the test
*window* is the size of the window
*reserved* portion of chain to reserve at the end

returns an index at which the chain can be considered converged, or -1 if
no such index found
"""

# effects of window size:
# too big - falsly end burn when the start of the window is still converging
# too small - take a long time, start to see effects of autocorrelation

# TODO is cramer von mises better than a KS test?

def _ks_sliding_window(logp, sample=0.1, alpha=0.01, window=100, reserved=0.5):
    from scipy.stats import ks_2samp
    
    len_chain, n_chain = logp.shape
    n_draw = int(sample*window*n_chain)
    
    idx_res = int(reserved*len_chain)
    idx_burn = 0
    
    # check if we have enough samples
    assert window < idx_res
    
    while idx_burn <= idx_res:
        f_samp = np.random.choice(logp[idx_burn:idx_burn+window].flatten(),
                                  n_draw, replace=False)
        r_samp = np.random.choice(logp[idx_res:].flatten(), n_draw, replace=False)
        
        p_val = ks_2samp(f_samp, r_samp)[1]

        if p_val > alpha: break
        idx_burn += window
    
    return idx_burn + window if idx_burn <= idx_res else -1

