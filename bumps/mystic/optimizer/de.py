# This code is public domain
# Author: Paul Kienzle

## Differential Evolution Solver Class
## Based on algorithms developed by Dr. Rainer Storn & Kenneth Price
## Influenced by
##    Lester E. Godwin (godwin@pushcopr.com)  1998: C++ version
##    James R. Phillips (zunzun@zunzun.com)   2002: Python conversion
##    Patrick Hung                            2006: cleanup
##    Mike McKerns (mmckerns@caltech.edu)     2008: parallel version, bounds
##    Paul Kienzle                            2009: rewrite

"""
Differential evolution optimizer.

This module contains a collection of optimization routines based on
Storn and Price's differential evolution algorithm.

Minimal function interface to optimization routines::

    x = diffev(f,xo)

Stepping interface::

    DifferentialEvolution


References
==========

[1] Storn, R. and Price, K. Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces. Journal of Global
Optimization 11: 341-359, 1997.

[2] Price, K., Storn, R., and Lampinen, J. - Differential Evolution,
A Practical Approach to Global Optimization. Springer, 1st Edition, 2005

"""

# Symbols required for simple interface
__all__ = ['de','stop']

import numpy as np

from .. import stop
from .. import solver

CROSSOVER = 'c_exp','c_bin'
MUTATE = 'best1','best1u','best2','randtobest1','rand1','rand2'

#############################################################################
#  Code below are the different crossovers/mutation strategies
#############################################################################
def c_exp(ndim, CR):
    """
    Select a sequence of dimensions.

    The length of the sequence follows the geometric distribution for 1-CR.

    This is equivalent to flipping the first heads after n flips with
    a weighted coin.

    The sequence starts at a random position and wraps if necessary.
    """
    # Note: this is different from Patrick Hung's version in that it forces
    # at least one success.
    L = min(abs(np.random.geometric(1-CR)),ndim)
    idx = np.zeros(ndim,'bool')
    n = np.random.randint(ndim)
    idx[np.arange(n,n+L)%ndim] = True
    return idx

def c_bin(ndim, CR):
    """
    Select random dimensions.

    The probability of selecting any dimension is CR.  At least one dimension
    will be selected.
    """
    n = np.random.randint(ndim)
    idx = np.random.rand(ndim) < CR
    idx[n] = True
    return idx


def best1(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = best + F*(r1-r2)
    """
    r1,r2 = _candidates(pop, 2, exclude=idx)
    return best[dims] + F*(r1[dims]-r2[dims])

def best1u(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = best + U*(r1-r2),  U ~ Uniform[0,F]
    """
    r1,r2 = _candidates(pop,2,exclude=idx)
    return best[dims] + F*np.random.rand()*(r1[dims]-r2[dims])

def best2(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = best + F*(r1+r2-r3-r4)
    """
    r1,r2,r3,r4 = _candidates(pop, 4, exclude=idx)
    return best[dims] + F*(r1[dims]+r2[dims]-r3[dims]-r4[dims])

def randtobest1(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = F*(best-old + r1-r2)
    """
    r1,r2 = _candidates(pop,2,exclude=idx)
    return F*(best[dims]-pop[idx][dims] + r1[dims]-r2[dims])

def rand1(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = r0 + F*(r1-r2)
    """
    r0,r1,r2 = _candidates(pop, 3, exclude=idx)
    return r0[dims] + F*(r1[dims]-r2[dims])

def rand1u(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = r0 + U*(r1-r2), U ~ Uniform[0,F]
    """
    r0,r1,r2 = _candidates(pop, 3, exclude=idx)
    return r0[dims] + F*np.random.rand()*(r1[dims]-r2[dims])

def rand2(F, best, pop, idx, dims):
    """
    Differential evolution mutation T = r0 + F*(r1+r2-r3-r4)
    """
    r0,r1,r2,r3,r4 = _candidates(pop, 5, exclude=idx)
    return r0[dims] + F*(r1[dims]+r2[dims]-r3[dims]-r4[dims])

############################################################


def _candidates(pop, k, exclude=None):
    """
    Select *n* random candidates from *pop*, not including the
    candidate at index *exclude*.
    """
    n = len(pop)
    if exclude is not None:
        selection = np.arange(n-1, dtype='i')
        if exclude < n-1:
            selection[exclude] = n-1
    else:
        selection = np.arange(n, dtype='i')
    np.random.shuffle(selection)
    return pop[selection[:k]]

##########################################################################

class DifferentialEvolution(solver.Strategy):
    """
    Differential evolution optimization.

    *CR*  float in [0-1]
        Crossover rate.
    *F* float in (0,inf)
        Crossover step size.
    *npop* float
        The size of the population is npop times the number of dimensions
        in the problem.
    *crossover* func(ndim, CR) -> index vector
        Crossover selection.  Returns the index vector of dimensions which
        should be mutated.
    *mutate* fn(F, best, pop, idx, dims) -> new[dims]
        Mutation strategy.  Selects the crossover population members and
        returns the mutated portion of the trial point in the new population.
        *F* is the scale factor, *best* is the best point seen so far, *pop*
        is the current population, *idx* is the vector being updated and
        *dims* is the set of dimensions to update.

    Available crossover functions (block is default)::

        c_exp:  start at dimension n and continue until U[0,1] >= CR
        c_bin:  select dimension i if U[0,1] >= CR

    Available mutation functions (best1u is default)::

        best1u: T = best + U(F)*(r1-r2),  U(F) ~ Uniform in [0,F]
        best1:  T = best + F*(r1-r2)
        best2:  T = best + F*(r1+r2-r3-r4)
        randtobest1:  T = F*(best-old) + F*(r1-r2)
        rand1:  T = r0 + F*(r1-r2)
        rand2:  T = r0 + F*(r1+r2-r3-r4)
    """
    requires = [('mystic','0.9')]

    def __init__(self, CR=0.5, F=2.0, npop=3,
                 crossover=c_exp, mutate=best1u):
        self.crossover = crossover
        self.mutate = mutate
        self.CR, self.F = CR, F
        self.npop = npop

    def default_termination_conditions(self, problem):
        success = stop.Cf(tol=1e-7,scaled=False)
        #maxiter = 100
        maxiter = len(problem.getp())*200
        #maxfun  = self.npop*maxiter
        failure = stop.Steps(maxiter)
        return success,failure

    def config_history(self, history):
        """
        Indicates how much history is required.
        """
        history.requires(value=1, population_points=2, population_values=2)

    def start(self, problem):
        """
        Generate the initial population.

        Returns a matrix *P* of points to be evaluated.
        """
        # Generate a random population
        current = problem.getp()
        ndim = len(current)
        population = problem.randomize(int(self.npop * ndim))
        population[0] = current

        # Return the population
        return population

    def step(self, history):
        """
        Generate the next population.

        Returns a matrix *P* of points to be evaluated.

        *history* contains the previous history of the computation,
        including any fields placed by :meth:`update`.
        """

        best = history.point[0]
        pop = history.population_points[0]
        pop_size,ndim = pop.shape

        trial = pop.copy()
        for idx,vec in enumerate(trial):
            dims = self.crossover(ndim, self.CR)
            vec[dims] = self.mutate(self.F, best, pop, idx, dims)
        return trial

    def update(self, history):
        """
        Update population, keeping old points that are better than
        the trial points.
        """
        #print "result",history.step[0]
        #for i,v in enumerate(history.population_values[0]):
        #    print history.population_points[0][i],'=',v
        if len(history.population_points) > 1:
            oldpop = history.population_points[1]
            oldval = history.population_values[1]
            newpop = history.population_points[0]
            newval = history.population_values[0]

            worse = newval > oldval
            newpop[worse] = oldpop[worse]
            newval[worse] = oldval[worse]

#minimizer_function(strategy=DifferentialEvolution,
#                   success=stop.Df(1e-5,n=10),
#                   failure=stop.Steps(100))
