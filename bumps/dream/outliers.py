"""
Chain outlier tests.

Given a
"""

from numpy import mean, std, sqrt, argsort, where, argmin, arange
from matplotlib.mlab import prctile
from scipy.stats import t as student_t
tinv = student_t.ppf
from mahal import mahalanobis
from acr import ACR
from . import util

def identify_outliers(test, chains, x):
    """
    Determine which chains have converged on a local maximum much lower than
    the maximum likelihood.

    *test* is the name of the test to use (one of IQR, Grubbs, Mahal or none).
    *chains* is a set of log likelihood values of shape (chain len, num chains)
    *x* is the current population of shape (num vars, num chains)

    See :module:`outliers` for details.
    """
    # Determine the mean log density of the active chains
    v = mean(chains, axis=0)

    # Check whether any of these active chains are outlier chains
    test = test.lower()
    if test == 'iqr':
        # Derive the upper and lower quartile of the chain averages
        Q1,Q3 = prctile(v,[25,75])
        # Derive the Inter Quartile Range (IQR)
        IQR = Q3 - Q1
        # See whether there are any outlier chains
        outliers = where(v < Q1 - 2*IQR)[0]

    elif test == 'grubbs':
        # Compute zscore for chain averages
        zscore = (mean(v) - v) / std(v, ddof=1)
        # Determine t-value of one-sided interval
        N = len(v)
        t2 = tinv(1 - 0.01/N,N-2)**2; # 95% interval
        # Determine the critical value
        Gcrit = ((N - 1)/sqrt(N)) * sqrt(t2/(N-2 + t2))
        # Then check against this
        outliers = where(zscore > Gcrit)[0]

    elif test == 'mahal':
        # Use the Mahalanobis distance to find outliers in the population
        alpha = 0.01
        Npop, Nvar = x.shape
        Gcrit = ACR(Nvar,Npop-1,alpha)
        #print "alpha",alpha,"Nvar",Nvar,"Npop",Npop,"Gcrit",Gcrit
        # Find which chain has minimum log_density
        minidx = argmin(v)
        # Then check the Mahalanobis distance of the current point to other chains
        d1 = mahalanobis(x[minidx,:], x[minidx!=arange(Npop),:])
        #print "d1",d1,"minidx",minidx
        # and see if it is an outlier
        outliers = [minidx] if d1 > Gcrit else []

    elif test == 'none':
        outliers = []

    else:
        raise ValueError("Unknown outlier test "+test)

    return outliers

def test():
    from walk import walk
    from numpy.random import multivariate_normal, seed
    from numpy import vstack, ones, eye
    seed(2) # Remove uncertainty on tests
    # Set a number of good and bad chains
    Ngood,Nbad = 25,2

    # Make chains mean-reverting chains with widely separated values for
    # bad and good; put bad chains first.
    chains = walk(1000, mu=[1]*Nbad+[5]*Ngood, sigma=0.45, alpha=0.1)

    # Check IQR and Grubbs
    assert (identify_outliers('IQR',chains,None) == range(Nbad)).all()
    assert (identify_outliers('Grubbs',chains,None) == range(Nbad)).all()

    # Put points for 'bad' chains at [-1,...,-1] and 'good' chains at [1,...,1]
    x = vstack( (multivariate_normal(-ones(4),.1*eye(4),size=Nbad),
                 multivariate_normal(ones(4),.1*eye(4),size=Ngood)) )
    assert identify_outliers('Mahal',chains,x)[0] in range(Nbad)

    # Put points for _all_ chains at [1,...,1] and check that mahal return []
    xsame = multivariate_normal(ones(4),.2*eye(4),size=Ngood+Nbad)
    assert identify_outliers('Mahal',chains,xsame) == []

    # Check again with large variance
    x = vstack( (multivariate_normal(-3*ones(4),eye(4),size=Nbad),
                 multivariate_normal(ones(4),10*eye(4),size=Ngood)) )
    assert identify_outliers('Mahal',chains,x) == []


    # =====================================================================
    # Test replacement

    # Construct a state object
    from numpy.linalg import norm
    from state import MCMCDraw
    Ngen, Npop = chains.shape
    Npop, Nvar = x.shape
    state = MCMCDraw(Ngen=Ngen, Nthin=Ngen, Nupdate=0,
                     Nvar=Nvar, Npop=Npop, Ncr=0, thin_rate=0)
    # Fill it with chains
    for i in range(Ngen):
        state._generation(new_draws=Npop, x=x, logp=chains[i], accept=Npop)

    # Make a copy of the current state so we can check it was updated
    nx, nlogp = x+0,chains[-1]+0
    # Remove outliers
    remove_outliers(state, nx, nlogp, test='IQR', portion=0.5)
    # Check that the outliers were removed
    outliers = state.outliers()
    assert outliers.shape[0] == Nbad
    for i in range(Nbad):
        assert nlogp[outliers[i,1]] == chains[-1][outliers[i,2]]
        assert norm(nx[outliers[i,1],:] - x[outliers[i,2],:]) == 0


if __name__ == "__main__":
    test()
