from __future__ import division, print_function

import sys
import argparse
import time

import numpy as np
from numpy import inf
import emcee

from bumps.cli import load_model, load_best
from bumps import initpop
from bumps.dream import stats, views
import matplotlib.pyplot as plt

class Draw(object):
    def __init__(self, logp, points, weights, labels, vars=None, integers=None):
        self.logp = logp
        self.weights = weights
        self.points = points[:,vars] if vars else points
        self.labels = [labels[v] for v in vars] if vars else labels
        if integers is not None:
            self.integers = integers[vars] if vars else integers
        else:
            self.integers = None

class State(object):
    def __init__(self, draw, nwalkers, title):
        # attributes of state that are used by bumps.dream.views
        self.title = title
        self.Nvar = draw.points.shape[-1]
        self.labels = draw.labels
        self._good_chains = slice(None, None)

        # private attributes for fake state
        chain_len = len(draw.logp)//nwalkers
        self._draw = draw
        self._samples_per_iteration = nwalkers*np.arange(1, chain_len+1, dtype='i')
        self._logp = draw.logp.reshape((nwalkers, -1)).T

    def logp(self, full=False):
        return self._samples_per_iteration, self._logp

    def chains(self):
        return self._samples_per_iteration, self._points, self._logp

    def draw(self): #, portion=1, vars=None, selection=None):
        return self._draw


def walk(problem, burn=100, steps=400, ntemps=30, npop=10, nthin=1,
         init='eps', state=None):
    betas = (np.arange(1, ntemps+1)/ntemps)**5
    p0 = problem.getp()
    dim = len(p0)
    nwalkers = npop*dim
    bounds = problem.bounds()
    log_prior = lambda p: 0 if ((p>=bounds[0])&(p<=bounds[1])).all() else -inf
    log_likelihood = lambda p: -problem.nllf(p)
    sampler = emcee.PTSampler(
        ntemps=ntemps, nwalkers=nwalkers, dim=dim,
        logl=log_likelihood, logp=log_prior,
        betas=betas,
        )

    # initial population
    if state is None:
        pop = initpop.generate(problem, init=init, pop=npop*ntemps)
        #lnprob, lnlike = None, None
    else:
        logp, samples = state
        pop = samples[:,:,-1,:]
        #lnprob, lnlike = logp[:,:,-1], logp[:,:,-1]
    p = pop.reshape(ntemps, nwalkers, -1)

    iteration = 0
    next_t = time.time() + 1

    # Burn-in
    if burn:
        print("=== burnin ", burn)
        for p, lnprob, lnlike in sampler.sample(p,
                #lnprob0=lnprob, lnlike0=lnlike,
                iterations=burn):
            t = time.time()
            if t >= next_t:
                print(iteration, -np.max(lnlike)/problem.dof)
                next_t = t + 1
            iteration += 1
    elif steps:
        # TODO: why can't we set lnprob, lnlike from saved state?
        for p, lnprob, lnlike in sampler.sample(p, iterations=1):
            pass

    sampler.reset()

    # Collect
    if steps:
        print("=== collect ", steps)
        for p, lnprob, lnlike in sampler.sample(p,
                lnprob0=lnprob, lnlike0=lnlike,
                iterations=nthin*steps, thin=nthin):
            t = time.time()
            if t >= next_t:
                print(iteration, -np.max(lnlike)/problem.dof)
                next_t = t + 1
            iteration += 1

    #assert sampler.chain.shape == (ntemps, nwalkers, steps, dim)
    return sampler

def process_vars(title, draw, nwalkers):
    import matplotlib.pyplot as plt
    vstats = stats.var_stats(draw)
    print("=== %s ==="%title)
    print(stats.format_vars(vstats))
    plt.figure()
    views.plot_vars(draw, vstats)
    plt.suptitle(title)
    plt.figure()
    views.plot_corrmatrix(draw)
    plt.suptitle(title)

    state = State(draw, nwalkers, title)
    plt.figure()
    views.plot_logp(state)


def plot_results(problem, sampler, tail=None):
    labels = problem.labels()
    dim = len(problem.getp())
    ntemps = len(sampler.betas)
    if sampler.chain is not None:
        samples = np.reshape(sampler.chain, (ntemps, -1, dim))
        logp = np.reshape(sampler.lnlikelihood, (ntemps, -1))
    else:
        samples = np.empty((ntemps, 0, dim), 'd')
        logp = np.empty((ntemps, 0), 'd')

    # Join results from the previous run
    if tail is not None:
        tail_samples = tail[:,1:].reshape((ntemps, -1, dim))
        tail_logp = tail[:,0].reshape((ntemps, -1))
        samples = np.hstack((tail_samples, samples))
        logp = np.hstack((tail_logp, logp))

    # process derived parameters
    visible_vars = getattr(problem, 'visible_vars', None)
    integer_vars = getattr(problem, 'integer_vars', None)
    derived_vars, derived_labels = getattr(problem, 'derive_vars', (None, None))
    if derived_vars:
        samples = np.reshape(samples, (-1, dim))
        new_vars = np.asarray(derived_vars(samples.T)).T
        samples= np.hstack((samples, new_vars))
        labels += derived_labels
        dim += len(derived_labels)
        samples = np.reshape(samples, (ntemps, -1, dim))

    # identify visible and integer variables
    visible = [labels.index(p) for p in visible_vars] if visible_vars else None
    integers = np.array([var in integer_vars for var in labels]) if integer_vars else None

    # plot the results, but only for the lowest and highest temperature
    title = problem.name + " (T=%g)"%(1/sampler.betas[0])
    draw = Draw(logp[0], samples[0], None, labels, vars=visible, integers=integers)
    process_vars(title, draw, sampler.nwalkers)
    if len(sampler.betas) > 1:
        title = problem.name + " (T=%g)"%(1/sampler.betas[-1])
        draw = Draw(logp[-1], samples[-1], None, labels, vars=visible, integers=integers)
        process_vars(title, draw, sampler.nwalkers)

    p = samples.reshape(-1, dim)[np.argmax(logp)]
    plt.figure()
    problem.plot(p)

def save_state(filename, sampler, tail=None, labels=None):
    if sampler.chain is None:
        # If no samples were generated don't bother to save state
        return

    logp = sampler.lnlikelihood.reshape(-1, 1)
    samples = sampler.chain.reshape(-1, sampler.dim)
    data = np.hstack((logp, samples))
    if tail is not None and tail.size:
        data = np.vstack((tail, data))
    np.savetxt(filename, data)

    # Save the best in the population
    with open("mc.par", 'wt') as fid:
        p = samples[np.argmax(logp)]
        pardata = "".join("%s %.15g\n" % (name, value)
                        for name, value in zip(labels, p))
        fid.write(pardata)


def load_state(opts, dim):
    if opts.resume:
        data = np.loadtxt(opts.resume)
        nwalkers = opts.npop*dim
        logp = data[:,0].reshape(opts.nT, nwalkers, -1)
        samples = data[:,1:].reshape(opts.nT, nwalkers, -1, dim)
        state = logp, samples
        preserved = min(opts.steps, max(samples.shape[2] - opts.burn, 0))
        #print(samples.shape[3], opts.steps, opts.burn, preserved)
        if preserved > 0:
            rows = preserved * opts.nT * nwalkers
            tail = data[-rows:]
        else:
            tail = None
        return preserved, state, tail
    else:
        return 0, None, None

def main():
    parser = argparse.ArgumentParser(
        description="run bumps model through emcee",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-b', '--burn', type=int, default=100, help='Number of burn iterations')
    parser.add_argument('-n', '--steps', type=int, default=400, help='Number of collection iterations')
    parser.add_argument('-i', '--init', choices='eps lhs cov random'.split(), default='eps', help='Population initialization method')
    parser.add_argument('-k', '--npop', type=int, default=2, help='Population multiplier (must be even)')
    parser.add_argument('-p', '--pars', type=str, default="", help='retrieve starting point from .par file')
    parser.add_argument('-t', '--nT', type=int, default=30, help='Number of temperatures')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Resume from file')
    parser.add_argument('-s', '--store', type=str, default='mc.out', help='Save to file')
    parser.add_argument('-x', '--thin', type=int, default=1, help='Number of iterations between collected points')
    parser.add_argument('modelfile', type=str, nargs=1, help='bumps model file')
    opts = parser.parse_args()

    problem = load_model(opts.modelfile[0])
    if opts.pars:
        load_best(problem, opts.pars)
    dim = len(problem.getp())
    preserved, state, tail = load_state(opts, dim)
    sampler = walk(problem,
                   init=opts.init, state=state,
                   burn=opts.burn if not preserved else 0,
                   steps=opts.steps-preserved, nthin=opts.thin,
                   ntemps=opts.nT, npop=opts.npop)
    save_state(opts.store, sampler, tail, labels=problem.labels())
    plot_results(problem, sampler, tail)
    plt.show()

if __name__ == "__main__":
    main()