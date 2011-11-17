"""
Convergence test statistic from Gelman and Rubin, 1992.
"""

from __future__ import division

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
    chain_len,Nchains,Nvar = sequences.shape

    # Only use the last portion of the sample
    chain_len = int(chain_len*portion)
    sequences = sequences[-chain_len:]

    if chain_len < 10:
        # Set the R-statistic to a large value
        R_stat = -2 * ones(Nvar)
    else:
        # Step 1: Determine the sequence means
        meanSeq = mean(sequences, axis=0)

        # Step 1: Determine the variance between the sequence means
        B = chain_len * var(meanSeq, axis=0, ddof=1)

        # Step 2: Compute the variance of the various sequences
        varSeq = var(sequences, axis=0, ddof=1)

        # Step 2: Calculate the average of the within sequence variances
        W = mean(varSeq,axis=0)

        # Step 3: Estimate the target mean
        #mu = mean(meanSeq)

        # Step 4: Estimate the target variance (Eq. 3)
        sigma2 = ((chain_len - 1)/chain_len) * W + (1/chain_len) * B

        # Step 5: Compute the R-statistic
        R_stat = sqrt((Nchains + 1)/Nchains * sigma2 / W - (chain_len-1)/Nchains/chain_len);

    return R_stat

def test():
    from numpy import reshape, arange, array, transpose
    from numpy.linalg import norm
    # Targe values computed from octave:
    #    format long
    #    S = reshape([1:15*6*7],[15,6,7]);
    #    R = gelman(S,struct('n',6,'seq',7))
    S = reshape(arange(1.,15*6*7+1)**-2, (15, 6, 7), order='F')
    S = transpose(S, (0,2,1))
    target = [1.06169861367116,   2.75325774624905,   4.46256647696399,
              6.12792266170178,   7.74538715553575,   9.31276519155232]
    R = gelman(S, portion=1)
    #print "target", array(target), "\nactual", R
    assert norm(R-target) < 1e-14
    R = gelman(S, portion=.1)
    assert norm(R - [-2, -2, -2, -2, -2, -2]) == 0

if __name__ == "__main__":
    test()
