"""
Differential evolution MCMC stepper.
"""
from __future__ import division, print_function

__all__ = ["de_step"]

from numpy import zeros, ones, dot, cov, eye, sqrt, sum, all
from numpy import where, select
from numpy.linalg import norm, cholesky, LinAlgError
from .util import draw, rng

EPS = 1e-6
_SNOOKER, _DE, _DIRECT = 0, 1, 2


def de_step(Nchain, pop, CR, max_pairs=2, eps=0.05,
            snooker_rate=0.1, noise=1e-6, scale=1.0):
    """
    Generates offspring using METROPOLIS HASTINGS monte-carlo markov chain

    The number of chains may be smaller than the population size if the
    population is selected from both the current generation and the
    ancestors.
    """
    Npop, Nvar = pop.shape

    # Initialize the delta update to zero
    delta_x = zeros((Nchain, Nvar))
    step_alpha = ones(Nchain)

    # Choose snooker, de or direct according to snooker_rate, and 80:20
    # ratio of de to direct.
    u = rng.rand(Nchain)
    de_rate = 0.8 * (1-snooker_rate)
    alg = select([u < snooker_rate, u < snooker_rate+de_rate],
                 [_SNOOKER, _DE], default=_DIRECT)
    # [PAK] CR selection moved from crossover into DE step
    CR_used = rng.choice(CR[:, 0], size=Nchain, replace=True, p=CR[:, 1])

    # Chains evolve using information from other chains to create offspring
    for qq in range(Nchain):

        if alg[qq] == _DE:  # Use DE with cross-over ratio

            # Select to number of vector pair differences to use in update
            # using k ~ discrete U[1, max pairs]
            k = rng.randint(max_pairs)+1
            # [PAK: same as k = DEversion[qq, 1] in matlab version]

            # Select 2*k members at random different from the current member
            perm = draw(2*k, Npop-1)
            perm[perm >= qq] += 1
            r1, r2 = perm[:k], perm[k:2*k]

            # Select the dims to update based on the crossover ratio, making
            # sure at least one dim is selected
            vars = where(rng.rand(Nvar) > CR_used[qq])
            if len(vars) == 0:
                vars = [rng.randint(Nvar)]

            # Weight the size of the jump inversely proportional to the
            # number of contributions, both from the parameters being
            # updated and from the population defining the step direction.
            gamma = 2.38/sqrt(2 * len(vars) * k)
            # [PAK: same as F=Table_JumpRate[len(vars), k] in matlab version]

            # Find and average step from the selected pairs
            step = sum(pop[r1]-pop[r2], axis=0)

            # Apply that step with F scaling and noise
            jiggle = 1 + eps * (2 * rng.rand(*step.shape) - 1)
            delta_x[qq, vars] = (jiggle*gamma*step)[vars]

        elif alg[qq] == _SNOOKER:  # Use snooker update

            # Select current and three others
            perm = draw(3, Npop-1)
            perm[perm >= qq] += 1
            xi = pop[qq]
            z, R1, R2 = [pop[i] for i in perm]

            # Find the step direction and scale it to the length of the
            # projection of R1-R2 onto the step direction.
            # TODO: population sometimes not unique!
            step = xi - z
            denom = sum(step**2)
            if denom == 0:  # identical points; should be extremely rare
                step = EPS*rng.randn(*step.shape)
                denom = sum(step**2)
            step_scale = sum((R1-R2)*step) / denom

            # Step using gamma of 2.38/sqrt(2) + U(-0.5, 0.5)
            gamma = 1.2 + rng.rand()
            delta_x[qq] = gamma * step_scale * step

            # Scale Metropolis probability by (||xi* - z||/||xi - z||)^(d-1)
            step_alpha[qq] = (norm(delta_x[qq]+step)/norm(step))**((Nvar-1)/2)
            CR_used[qq] = 0.0

        elif alg[qq] == _DIRECT:  # Use one pair and all dimensions

            # Note that there is no F scaling, dimension selection or noise
            perm = draw(2, Npop-1)
            perm[perm >= qq] += 1
            delta_x[qq, :] = pop[perm[0], :] - pop[perm[1], :]
            CR_used[qq] = 0.0

        else:
            raise RuntimeError("Select failed...should never happen")

        # If no step was specified (exceedingly unlikely!), then
        # select a delta at random from a gaussian approximation to the
        # current population
        if all(delta_x[qq] == 0):
            try:
                #print "No step"
                # Compute the Cholesky Decomposition of x_old
                R = (2.38/sqrt(Nvar)) * cholesky(cov(pop.T) + EPS*eye(Nvar))
                # Generate jump using multinormal distribution
                delta_x[qq] = dot(rng.randn(*(1, Nvar)), R)
            except LinAlgError:
                print("Bad cholesky")
                delta_x[qq] = rng.randn(Nvar)

    # Update x_old with delta_x and noise
    delta_x *= scale

    # [PAK] The noise term needs to depend on the fitting range
    # of the parameter rather than using a fixed noise value for all
    # parameters.  The  current parameter value is a pretty good proxy
    # in most cases (i.e., relative noise), but it breaks down if the
    # parameter is zero, or if the range is something like 1 +/- eps.

    # absolute noise
    #x_new = pop[:Nchain] + delta_x + scale*noise*rng.randn(Nchain, Nvar)

    # relative noise
    x_new = pop[:Nchain] * (1 + scale*noise*rng.randn(Nchain, Nvar)) + delta_x

    # no noise
    #x_new = pop[:Nchain] + delta_x

    return x_new, step_alpha, CR_used


def _check():
    from numpy import arange
    nchain, npop, nvar = 4, 10, 3

    pop = 100*arange(npop*nvar).reshape((npop, nvar))
    pop += rng.rand(*pop.shape)*1e-6
    cr = 1./(rng.randint(4, size=nvar)+1)
    x_new, _step_alpha, used = de_step(nchain, pop, cr, max_pairs=2, eps=0.05)
    print("""\
The following table shows the expected portion of the dimensions that
are changed and the rounded value of the change for each point in the
population.
""")
    for r, i, u in zip(cr, range(8), used):
        rstr = ("%3d%% " % (r*100)) if u else "full "
        vstr = " ".join("%4d" % (int(v/100+0.5)) for v in x_new[i]-pop[i])
        print(rstr+vstr)


if __name__ == "__main__":
    _check()
