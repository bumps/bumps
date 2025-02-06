"""
Convergence test statistic from Gelman and Rubin, 1992.[1]

[1] Gelman, Andrew, and Donald B. Rubin.
    "Inference from Iterative Simulation Using Multiple Sequences."
    Statistical Science 7, no. 4 (November 1, 1992): 457-72.
    https://doi.org/10.2307/2246093.
"""

__all__ = ["gelman"]

from numpy import var, mean, ones, sqrt


def gelman(sequences, portion=0.5):
    """
    Calculates the R-statistic convergence diagnostic

    For more information please refer to: Gelman, A. and D.R. Rubin, 1992.
    Inference from Iterative Simulation Using Multiple Sequences,
    Statistical Science, Volume 7, Issue 4, 457-472.
    doi:10.1214/ss/1177011136
    """

    # Find the size of the sample
    chain_len, nchains, nvar = sequences.shape
    # print sequences[:20, 0, 0]

    # Only use the last portion of the sample
    chain_len = int(chain_len * portion)
    sequences = sequences[-chain_len:]

    if chain_len < 2:
        # Set the R-statistic to a large value
        r_stat = -2 * ones(nvar)
    else:
        # Step 1: Determine the sequence means
        mean_seq = mean(sequences, axis=0)

        # Step 1: Determine the variance between the sequence means
        b = chain_len * var(mean_seq, axis=0, ddof=1)

        # Step 2: Compute the variance of the various sequences
        var_seq = var(sequences, axis=0, ddof=1)

        # Step 2: Calculate the average of the within sequence variances
        w = mean(var_seq, axis=0)

        # Step 3: Estimate the target mean
        # mu = mean(mean_seq)

        # Step 4: Estimate the target variance (Eq. 3)
        sigma2 = ((chain_len - 1) / chain_len) * w + (1 / chain_len) * b

        # TODO: the second term, -(N-1)/(K N), doesn't appear in [1]
        # Step 5: Compute the R-statistic
        r_stat = sqrt((nchains + 1) / nchains * sigma2 / w - (chain_len - 1) / nchains / chain_len)
        # par=2
        # print chain_len,b[par],var_seq[...,par],w[par],r_stat[par]

    return r_stat


def test():
    from numpy import reshape, arange, transpose
    from numpy.linalg import norm

    # Targe values computed from octave:
    #    format long
    #    s = reshape([1:15*6*7],[15,6,7]);
    #    r = gelman(s,struct('n',6,'seq',7))
    s = reshape(arange(1.0, 15 * 6 * 7 + 1) ** -2, (15, 6, 7), order="F")
    s = transpose(s, [0, 2, 1])
    target = [
        1.06169861367116,
        2.75325774624905,
        4.46256647696399,
        6.12792266170178,
        7.74538715553575,
        9.31276519155232,
    ]
    r = gelman(s, portion=1)
    # print r
    # print "target", array(target), "\nactual", r
    assert norm(r - target) < 1e-14
    r = gelman(s, portion=0.1)
    assert norm(r - [-2, -2, -2, -2, -2, -2]) == 0


if __name__ == "__main__":
    test()
