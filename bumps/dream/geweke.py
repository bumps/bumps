"""
Convergence test statistic from Gelman and Rubin, 1992.
"""

__all__ = ["geweke"]

from numpy import var, mean, ones, sqrt, reshape, log10, abs


def geweke(sequences, portion=0.25):
    """
    Calculates the Geweke convergence diagnostic

    Refer to:

        pymc-devs.github.com/pymc/modelchecking.html#informal-methods
        support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introbayes_sect008.html

    """

    # Find the size of the sample
    chain_len, nchains, nvar = sequences.shape
    z_stat = -2 * ones(nvar)
    if chain_len >= 2:
        # Only use the last portion of the sample
        try:
            front_portion, back_portion = portion
        except TypeError:
            front_portion = back_portion = portion
        front_len = int(chain_len * front_portion)
        back_len = int(chain_len * back_portion)
        # print "STARTING SHAPE", sequences.shape
        seq1 = reshape(sequences[:front_len, :, :], (front_len * nchains, nvar))
        seq2 = reshape(sequences[-back_len:, :, :], (back_len * nchains, nvar))
        # print "SEQ1", seq1.shape, 'SEQ2', seq2.shape
        # Step 1: Determine the sequence means
        meanseq1 = mean(seq1, axis=0)
        meanseq2 = mean(seq2, axis=0)
        # print "SHAPEs", meanseq1.shape, meanseq2.shape
        var1 = var(seq1, axis=0)
        var2 = var(seq2, axis=0)
        denom = sqrt(var1 + var2)
        index = denom > 0
        z_stat[index] = (meanseq1 - meanseq2)[index] / denom[index]

        # z_stat is now the Z score for every chain and parameter
        # in that with shape (chains, vars)

        # To make it easier to look at, return the average for the vars.
        if 0:
            avg_z = mean(z_stat, axis=0)
            lavg_z = log10(abs(avg_z))
            return lavg_z.tolist()
        if 0:
            avg_z = z_stat
            lavg_z = log10(abs(avg_z))
            return lavg_z.flatten().tolist()
        else:
            return z_stat.flatten().tolist()

    # TODO: code is wrong if chain length is 1, since lavg_z is not defined
    return lavg_z.tolist()
