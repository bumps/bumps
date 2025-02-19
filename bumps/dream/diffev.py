"""
Differential evolution MCMC stepper.
"""

__all__ = ["de_step"]

from numpy import zeros, ones, empty, sqrt, sum
from numpy import where, array
from numpy.linalg import norm

from .util import draw, rng

try:
    # raise ImportError("skip numpy")
    from numba import njit, prange

    @njit(cache=True)
    def pchoice(choices, size=0, replace=True, p=None):
        if p is None:
            return rng.choice(choices, size=size, replace=replace)
        # TODO: if choices is an array, then shape should be an array
        result = empty(size, dtype=choices.dtype)
        num_choices = choices.shape[0]
        for index in prange(size):
            u = rng.rand()
            cdf = 0.0
            for k in range(num_choices):
                cdf += p[k]
                if u <= cdf:
                    result[index] = choices[k]
            # else: should never get here if choices sum to 1
        return result

except ImportError:

    def njit(*args, **kw):
        return lambda f: f

    prange = range
    pchoice = rng.choice

EPS = 1e-6
_DE, _SNOOKER, _DIRECT = 0, 1, 2


@njit(cache=True)
def de_step(Nchain, pop, CR, max_pairs=2, eps=0.05, snooker_rate=0.1, noise=1e-6, scale=1.0):
    """
    Generates offspring using METROPOLIS HASTINGS monte-carlo markov chain

    *Nchain* is the number of simultaneous changes that are running.

    *pop* is an array of shape [Npop x Nvar] providing the active points used
    to generate the next proposal for each chain. This may be larger than the
    Nchains if the caller is using ancestor generations for active population.
    The current population is assumed to be the first *Nchain* rows of *pip*..

    *CR* is an array of [Ncrossover x 2] crossover ratios with weights. The
    crossover ratio is the probability of selecting a particular dimension when
    generating the difference vector. The weights are used to select the
    crossover ratio. The weights are adjusted dynamically during the fit based
    on the acceptance rate of points generated with each crossover ratio.

    *max_pairs* determines the maximum number of pairs which contribute to the
    differential evolution step. The number of pairs is chosen at random, with
    the difference vectors between the pairs averaged when creating the DE step.

    *eps* determines the jitter added to the DE step.

    *snooker_rate* determines the probability of using the snooker stepper.
    Otherwise use DE stepper 80% of the time, or apply the difference between
    pairs the other 20% of the time.

    *scale=1* scales the difference vector (constant, not stochastic)

    *noise=1e-6* adds random noise to the non-zero components of the
    difference vector. This noise is relative rather than absolute to allow
    for parameter values far from 1.0. Noise is also scaled by *scale*.
    """
    Npop, Nvar = pop.shape

    # Initialize the delta update to zero
    delta_x = zeros((Nchain, Nvar))
    step_alpha = ones(Nchain)

    # Choose snooker, de or direct according` to snooker_rate, and 80:20
    # ratio of de to direct.
    u = rng.rand(Nchain)
    de_rate = 0.8 * (1 - snooker_rate)
    alg = zeros(Nchain, dtype="int")  # _DE = 0
    alg[u >= de_rate] = _SNOOKER
    alg[u >= de_rate + snooker_rate] = _DIRECT
    # alg = select([u < snooker_rate, u < snooker_rate+de_rate],
    #             [_SNOOKER, _DE], default=_DIRECT)
    # [PAK] CR selection moved from crossover into DE step
    CR_used = pchoice(CR[:, 0], size=Nchain, replace=True, p=CR[:, 1])

    # Chains evolve using information from other chains to create offspring
    for qq in prange(Nchain):
        if alg[qq] == _DE:  # Use DE with cross-over ratio
            # Select to number of vector pair differences to use in update
            # using k ~ discrete U[1, max pairs]
            k = rng.randint(max_pairs) + 1
            # [PAK: same as k = DEversion[qq, 1] in matlab version]

            # TODO: make sure we don't fail if too few chains.
            # Select 2*k members at random different from the current member
            perm = draw(2 * k, Npop - 1)
            # TODO: rewrite draw so that it accepts a not_matching int
            perm[perm >= qq] += 1
            r1, r2 = perm[:k], perm[k : 2 * k]

            # Select the dims to update based on the crossover ratio, making
            # sure at least one dim is selected
            vars = where(rng.rand(Nvar) <= CR_used[qq])[0]
            if len(vars) == 0:
                vars = array([rng.randint(Nvar)])
            # print("for chain", qq, CR_used[qq], "% update", vars)

            # Weight the size of the jump inversely proportional to the
            # number of contributions, both from the parameters being
            # updated and from the population defining the step direction.
            gamma = 2.38 / sqrt(2 * len(vars) * k)
            # [PAK: same as F=Table_JumpRate[len(vars), k] in matlab version]

            # Find and average step from the selected pairs
            step = sum(pop[r1] - pop[r2], axis=0)

            # Apply that step with F scaling and noise
            jiggle = 1 + eps * (2 * rng.rand(*step.shape) - 1)
            delta_x[qq, vars] = (jiggle * gamma * step)[vars]

        elif alg[qq] == _SNOOKER:  # Use snooker update
            # Select current and three others
            perm = draw(3, Npop - 1)
            perm[perm >= qq] += 1
            xi = pop[qq]
            z, R1, R2 = [pop[i] for i in perm]

            # Find the step direction and scale it to the length of the
            # projection of R1-R2 onto the step direction.
            # TODO: population sometimes not unique!
            step = xi - z
            denom = sum(step**2)
            if denom == 0:  # identical points; should be extremely rare
                step = EPS * rng.randn(*step.shape)
                denom = sum(step**2)
            step_scale = sum((R1 - R2) * step) / denom

            # Step using gamma of 2.38/sqrt(2) + U(-0.5, 0.5)
            gamma = 1.2 + rng.rand()
            delta_x[qq] = gamma * step_scale * step

            # Scale Metropolis probability by (||xi* - z||/||xi - z||)^(d-1)
            step_alpha[qq] = (norm(delta_x[qq] + step) / norm(step)) ** ((Nvar - 1) / 2)
            CR_used[qq] = 0.0

        elif alg[qq] == _DIRECT:  # Use one pair and all dimensions
            # Note that there is no F scaling, dimension selection or noise
            perm = draw(2, Npop - 1)
            perm[perm >= qq] += 1
            delta_x[qq, :] = pop[perm[0], :] - pop[perm[1], :]
            CR_used[qq] = 0.0

        else:
            raise RuntimeError("Select failed...should never happen")

        # Didn't implement this in the compiled version (yet)
        ## If no step was specified (exceedingly unlikely!), then
        ## select a delta at random from a gaussian approximation to the
        ## current population
        # if all(delta_x[qq] == 0):
        #    from numpy.linalg import cholesky, LinAlgError
        #    try:
        #        #print "No step"
        #        # Compute the Cholesky Decomposition of x_old
        #        R = (2.38/sqrt(Nvar)) * cholesky(cov(pop.T) + EPS*eye(Nvar))
        #        # Generate jump using multinormal distribution
        #        delta_x[qq] = dot(rng.randn(*(1, Nvar)), R)
        #    except LinAlgError:
        #        print("Bad cholesky")
        #        delta_x[qq] = rng.randn(Nvar)

    # Update x_old with delta_x and noise
    delta_x *= scale
    # print("alg", alg)
    # print("CR_used", CR_used)
    # print("delta_x", delta_x)

    # [PAK] The noise term needs to depend on the fitting range
    # of the parameter rather than using a fixed noise value for all
    # parameters.  The  current parameter value is a pretty good proxy
    # in most cases (i.e., relative noise), but it breaks down if the
    # parameter is zero, or if the range is something like 1 +/- eps.

    # absolute noise
    # x_new = pop[:Nchain] + delta_x + scale*noise*rng.randn(Nchain, Nvar)

    # relative noise
    x_new = pop[:Nchain] * (1 + scale * noise * rng.randn(Nchain, Nvar)) + delta_x

    # no noise
    # x_new = pop[:Nchain] + delta_x

    return x_new, step_alpha, CR_used


def _check():
    from numpy import arange, vstack, ascontiguousarray

    max_pairs, snooker_rate, eps, noise, scale = 2, 0.1, 0.05, 1e-6, 1.0
    nchain, npop, nvar, ncr = 4, 10, 7, 4

    pop = 100 * arange(npop * nvar, dtype="d").reshape((npop, nvar))
    pop = pop * (1 + rng.rand(*pop.shape) * 0.1)
    ratios = 1.0 / (rng.randint(4, size=ncr) + 1)  # 4 => ratios in [0.2, 0.25, 0.333, 0.5, 1.0]
    weights = [1 / ncr] * ncr  # equal-weight for each CR
    # print(ratios, weights)
    CR = ascontiguousarray(vstack((ratios, weights)).T, dtype="d")
    work = lambda: de_step(nchain, pop, CR, max_pairs=max_pairs, eps=eps)
    x_new, _step_alpha, used = work()
    # print(f"{pop=}")
    # print(f"{x_new=}")
    # print(f"{_step_alpha=}")
    # print(f"{used=}")
    print("""\
The following table shows the expected portion of the dimensions that
are changed and the rounded value of the change for each point in the
population.
""")
    for k, u in enumerate(used):
        rstr = f"{int(u * 100):4d}% " if u else " full "
        vstr = " ".join(f"{v:.2f}" for v in x_new[k] - pop[k])
        print(rstr + vstr)

    if 1:  # timing check
        from timeit import timeit
        from ctypes import c_double
        from .compiled import dll

        dll_work = lambda: dll.de_step(
            nchain,
            nvar,
            len(CR),
            pop.ctypes,
            CR.ctypes,
            max_pairs,
            c_double(eps),
            c_double(snooker_rate),
            c_double(noise),
            c_double(scale),
            x_new.ctypes,
            _step_alpha.ctypes,
            used.ctypes,
        )

        print("small pop time (ms)", timeit(work, number=10000) / 10)
        if dll:
            print("small pop time compiled (ms)", timeit(dll_work, number=10000) / 10)
        else:
            print("no dlls")

        nchain, nvar = 1000, 50
        npop = nchain
        pop = 100 * arange(npop * nvar, dtype="d").reshape((npop, nvar))
        pop = pop * (1 + rng.rand(*pop.shape) * 0.1)
        print("large pop time (ms)", timeit(work, number=100))
        if dll:
            x_new, _step_alpha, used = work()  # need this line to define new return vectors for dll
            print("large pop time compiled (ms)", timeit(dll_work, number=100))


if __name__ == "__main__":
    _check()
