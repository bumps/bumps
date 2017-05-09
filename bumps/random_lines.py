"""
Random Lines Algorithm finds the optimal minimum of a function.

Sahin, I. (2013). Minimization over randomly selected lines.  An
International  Journal Of Optimization And Control: Theories &
Applications (IJOCTA), 3(2), 111-119.
http://dx.doi.org/10.11121/ijocta.01.2013.00167
"""

# Author : Ismet Sahin
from __future__ import print_function

__all__ = ["random_lines", "particle_swarm"]

from itertools import count

from numpy import zeros, ones, asarray, sqrt, arange, isfinite
from numpy.random import rand, random_integers


def print_every_five(step, x, fx, k):
    if step % 5 == 0:
        print(step, ":", fx[k], x[k])


def random_lines(cfo, NP, CR=0.9, epsilon=1e-10, abort_test=None, maxiter=1000):
    """
    Random lines is a population based optimizer which using quadratic
    fits along randomly oriented directions.

    *cfo* is the cost function object.  This is a dictionary which contains
    the following keys:

        *cost* is the function to be optimized.  If *parallel_cost* exists,
        it should accept a list of points, not just a single point on each
        evaluation.

        *n* is the problem dimension

        *x0* is the initial point

        *x1* and *x2* are lower and upper bounds for each parameter

        *monitor* is a callable which is called each iteration using
        *callback(step, x, fx, k)*, where *step* is the iteration number,
        *x* is the population, *fx* is value of the cost function for each
        member of the population and *k* is the index of the best point in
        the population.

        *f_opt* is the target value of the optimization

    *NP* is the number of fit parameters

    *CR* is the cross-over ratio, which is the proportion of dimensions
    that participate in any random orientation vector.

    *epsilon* is the convergence criterion.

    *abort_test* is a callable which indicates whether an external processes
    requests the fit to stop.

    *maxiter* is the maximum number of generations

    Returns success, num_evals, f(x_best), x_best.
    """
    if 'parallel_cost' in cfo:
        mapper = lambda v: asarray(cfo['parallel_cost'](v.T), 'd')
    else:
        mapper = lambda v: asarray(list(map(cfo['cost'], v.T)), 'd')
    monitor = cfo.get('monitor', print_every_five)

    n = cfo['n']

    X = rand(n, NP)            # will hold original vectors

    # CREATE FIRST GENERATION WITH LEGAL PARAMETER VALUES AND EVALUATE COSTS
    # m th member of the population
    for m in range(0, NP):
        X[:, m] = cfo['x1'] + (cfo['x2'] - cfo['x1']) * X[:, m]
    if 'x0' in cfo:
        X[:, 0] = cfo['x0']
    f = mapper(X)

    n_feval = NP
    f_best, i_best = min(zip(f, count()))

    # CHECK INITIAL STOPPING CRITERIA
    if abs(cfo['f_opt'] - f_best) < epsilon:
        satisfied_sc = 1
        x_best = X[:, i_best]
        return satisfied_sc, n_feval, f_best, x_best

    for L in range(1, maxiter + 1):

        # finding destination vector
        i_Xj = random_integers(0, NP - 2, NP)
        i_ge = (i_Xj >= arange(0, NP))
        i_Xj[i_ge] += 1

        # choosing muk
        muk = 0.01 + 0.49 * rand(NP)
        inx = rand(NP) < 0.5
        muk[inx] = -muk[inx]

        # find xk and fk s
        Xi = X
        Xj = X[:, i_Xj]
        P = Xj - Xi
        Xk = Xi + (ones((n, 1)) * muk) * P
        fk = mapper(Xk)
        n_feval = n_feval + NP

        # find quadratic models
        if any(muk == 0) or any(muk == 1):
            satisfied_sc = 0
            x_best = X[:, i_best]
            print('muk cannot be zero or one !!!')
            return satisfied_sc, n_feval, f_best, x_best

        fi = f
        fj = f[i_Xj]
        b = (muk/(muk-1))*fj - ((muk+1)/muk)*fi - (1/(muk*(muk-1)))*fk
        a = fj - fi - b

        crossovers = []
        for k in range(0, NP):
            if (abs(a[k]) < 1e-30
                    or (a[k] < 0 and fk[k] > fi[k] and fk[k] > fj[k])
                    or not isfinite(a[k])):
                # xi survives
                continue
            else:
                # xi may not survive
                mustar = -b[k] / (2 * a[k])
                xstar = Xi[:, k] + mustar * P[:, k]

                # choosing random numbers for crossover
                rn = rand(n)
                indi = (rn < 0.5 * (1 - CR))
                indj = (rn > 0.5 * (1 + CR))
                xstar[indi] = Xi[indi, k]
                xstar[indj] = Xj[indj, k]

                # map into feasible set
                inx = xstar < cfo['x1']
                xstar[inx] = cfo['x1'][inx]
                inx = xstar > cfo['x2']
                xstar[inx] = cfo['x2'][inx]

                crossovers.append((k, xstar))

        if len(crossovers) > 0:
            idx, xstar = [asarray(v) for v in zip(*crossovers)]
            fstar = mapper(xstar.T)
            n_feval += len(crossovers)

            # xi does not survive, xstar replaces it
            update = fstar < fi[idx]
            f[idx[update]] = fstar[update]
            X[:, idx[update]] = xstar[update, :].T

        # CHECKING STOPPING CRITERIA
        f_best, i_best = min(zip(f, count()))
        if abs(cfo['f_opt'] - f_best) < epsilon:
            satisfied_sc = 1
            x_best = X[:, i_best]
            return satisfied_sc, n_feval, f_best, x_best
        if abort_test():
            break

        monitor(L, X, f, i_best)

    return 1, n_feval, f_best, X[:, i_best]


def particle_swarm(cfo, NP, epsilon=1e-10, maxiter=1000):
    """
    Particle swarm is a population based optimizer which uses force and
    momentum to select candidate points.

    *cfo* is the cost function object.  This is a dictionary which contains
    the following keys:

        *cost* is the function to be optimized.  If *parallel_cost* exists,
        it should accept a list of points, not just a single point on each
        evaluation.

        *n* is the problem dimension

        *x0* is the initial point

        *x1* and *x2* are lower and upper bounds for each parameter

        *monitor* is a callable which is called each iteration using
        *callback(step, x, fx, k)*, where *step* is the iteration number,
        *x* is the population, *fx* is value of the cost function for each
        member of the population and *k* is the index of the best point in
        the population.

        *f_opt* is the target value of the optimization

    *NP* is the number of fit parameters

    *epsilon* is the convergence criterion.

    *abort_test* is a callable which indicates whether an external processes
    requests the fit to stop.

    *maxiter* is the maximum number of generations

    Returns success, num_evals, f(x_best), x_best.
    """

    if 'parallel_cost' in cfo:
        mapper = lambda v: asarray(cfo['parallel_cost'](v.T), 'd')
    else:
        mapper = lambda v: asarray(list(map(cfo['cost'], v.T)), 'd')
    monitor = cfo.get('monitor', print_every_five)

    n = cfo['n']
    c1 = 2.8
    c2 = 1.3
    phi = c1 + c2
    K = 2 / abs(2 - phi - sqrt(phi * phi - 4 * phi))

    X = rand(n, NP)            # will hold original vectors
    V = zeros((n, NP))

    # CREATE FIRST GENERATION WITH LEGAL PARAMETER VALUES AND EVALUATE COSTS
    rn1 = rand(n, NP)
    # m th member of the population
    for m in range(0, NP):
        extend = cfo['x2'] - cfo['x1']
        X[:, m] = cfo['x1'] + extend * X[:, m]
        V[:, m] = 2 * rn1[:, m] * extend - extend

    if 'x0' in cfo:
        X[:, 0] = cfo['x0']
    f = mapper(X)

    n_feval = NP
    P = X[:]

    f_best, i_best = min(zip(f, count()))
    for L in range(2, maxiter + 1):

        rn2 = rand(n, NP)
        for i in range(0, NP):
            #r = rand(2)
            r = rn2[:, i]
            V[:, i] = V[:, i] + r[0] * c1 * \
                (P[:, i] - X[:, i]) + r[1] * c2 * (P[:, i_best] - X[:, i])
            V[:, i] = K * V[:, i]

            X[:, i] = X[:, i] + V[:, i]

        f_temp = mapper(X)
        idx = f_temp < f
        f[idx] = f_temp[idx]
        P[:, idx] = X[:, idx]

        n_feval = n_feval + NP

        # CHECKING STOPPING CRITERIA
        f_best, i_best = min(zip(f, count()))
        if abs(cfo['f_opt'] - f_best) < epsilon:
            satisfied_sc = 1
            return satisfied_sc, n_feval, f_best, X[:, i_best]

        monitor(L, X, f, i_best)

    satisfied_sc = 0
    return satisfied_sc, n_feval, f_best, X[:, i_best]


def example_call(optimizer=random_lines):
    from numpy.random import seed
    seed(1)
    cost = lambda x: x[0] ** 2 + x[1] ** 2
    n = 2
    x1 = -5 * ones(n)
    x2 = 5 * ones(n)
    f_opt = 0
    cfo = {'cost': cost, 'n': n, 'x1': x1, 'x2': x2, 'f_opt': f_opt}

    NP = 10 * n
    satisfied_sc, n_feval, f_best, x_best = optimizer(cfo, NP)
    print(satisfied_sc, "n:%d" % n_feval, f_best, x_best)


def main():
    print("=== Random Lines")
    example_call(random_lines)
    print("=== Particle Swarm")
    example_call(particle_swarm)

if __name__ == "__main__":
    main()
