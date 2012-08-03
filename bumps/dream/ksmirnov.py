"""
Kolmogorov-Smirnov test comparing the distribution of values at the front of the
chain to that at the end of the chain.
"""

from __future__ import division

from numpy import reshape,apply_along_axis
from scipy.stats import ks_2samp                                                                                                                                                                                                                                                                                                
def ksmirnov(seq, portion=0.25, filter_order=15):
    """
    Kolmogorov-Smirnov test of similarity between the empirical distribution at the start
    and at the end of the chain.  Apply a median filter (filter=15) on neighbouring K-S 
    values to reduce variation in the test statistic value.
    """
    chlen,nchains,nvars = seq.shape
    count = portion*chlen*nchains
    n = filter_order
    def ksm(chain):
        #return ks_2samp(chain[:count], chain[-count:])
        Ks,p = zip(*[ks_2samp(chain[i:count+i],chain[-count-n+i:-n+i]) for i in range(n)])
        return sorted(Ks)[(n-1)//2],sorted(p)[(n-1)//2]
    Ks,p = apply_along_axis(ksm,0,reshape(seq, (chlen*nchains,nvars)))
    return Ks,p
