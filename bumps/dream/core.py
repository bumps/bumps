"""
DiffeRential Evolution Adaptive Metropolis algorithm

DREAM runs multiple different chains simultaneously for global exploration,
and automatically tunes the scale and orientation of the proposal
distribution using differential evolution.  The algorithm maintains
detailed balance and ergodicity and works well and efficient for a large
range of problems, especially in the presence of high-dimensionality and
multimodality.

DREAM developed by Jasper A. Vrugt and Cajo ter Braak

This algorithm has been described in:

   Vrugt, J.A., C.J.F. ter Braak, M.P. Clark, J.M. Hyman, and B.A. Robinson,
      *Treatment of input uncertainty in hydrologic modeling: Doing hydrology
      backward with Markov chain Monte Carlo simulation*,
      Water Resources Research, 44, W00B09, 2008.
      `doi:10.1029/2007WR006720 <http://dx.doi.org/10.1029/2007WR006720>`_

   Vrugt, J.A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson,
       and J.M. Hyman,
       *Accelerating Markov chain Monte Carlo simulation by differential
       evolution with self-adaptive randomized subspace sampling*,
       International Journal of Nonlinear Sciences and Numerical Simulation,
       10(3), 271-288, 2009.

   Vrugt, J.A., C.J.F. ter Braak, H.V. Gupta, and B.A. Robinson,
       *Equifinality of formal (DREAM) and informal (GLUE) Bayesian approaches
       in hydrologic modeling*,
       Stochastic Environmental Research and Risk Assessment,
       1-16, 2009, In Press.
       `doi:10.1007/s00477-008-0274-y
       <http://dx.doi.org/10.1007/s00477-008-0274-y>`_

For more information please read:

   Ter Braak, C.J.F.,
       *A Markov Chain Monte Carlo version of the genetic algorithm Differential
       Evolution: easy Bayesian computing for real parameter spaces*,
       Stat. Comput., 16, 239 - 249, 2006.
       `doi:10.1007/s11222-006-8769-1
       <http://dx.doi.org/10.1007/s11222-006-8769-1>`_

   Vrugt, J.A., H.V. Gupta, W. Bouten and S. Sorooshian,
       *A Shuffled Complex Evolution Metropolis algorithm for optimization
       and uncertainty assessment of hydrologic model parameters*,
       Water Resour. Res., 39 (8), 1201, 2003.
       `doi:10.1029/2002WR001642 <http://dx.doi.org/10.1029/2002WR001642>`_

   Ter Braak, C.J.F., and J.A. Vrugt,
       *Differential Evolution Markov Chain with snooker updater
       and fewer chains*,
       Statistics and Computing, 2008.
       `doi:10.1007/s11222-008-9104-9
       <http://dx.doi.org/2008.10.1007/s11222-008-9104-9>`_

   Vrugt, J.A., C.J.F. ter Braak, and J.M. Hyman,
       *Differential evolution adaptive Metropolis with snooker update and
       sampling from past states*,
       SIAM journal on Optimization, 2009.

   Vrugt, J.A., C.J.F. ter Braak, and J.M. Hyman,
       *Parallel Markov chain Monte Carlo simulation on distributed computing
       networks using multi-try Metropolis with sampling from past states*,
       SIAM journal on Scientific Computing, 2009.

   G. Schoups, and J.A. Vrugt,
       *A formal likelihood function for Bayesian inference of hydrologic
       models with correlated, heteroscedastic and non-Gaussian errors*,
       Water Resources Research, 2010, In Press.

   G. Schoups, J.A. Vrugt, F. Fenicia, and N.C. van de Giesen,
       *Inaccurate numerical solution of hydrologic models corrupts efficiency
       and robustness of MCMC simulation*,
       Water Resources Research, 2010, In Press.

Copyright (c) 2008, Los Alamos National Security, LLC
All rights reserved.

Copyright 2008. Los Alamos National Security, LLC. This software was produced
under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
Laboratory (LANL), which is operated by Los Alamos National Security, LLC
for the U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF
THIS SOFTWARE.  If software is modified to produce derivative works, such
modified software should be clearly marked, so as not to confuse it with
the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Los Alamos National Security, LLC, Los Alamos National
  Laboratory, LANL the U.S. Government, nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

MATLAB code written by Jasper A. Vrugt, Center for NonLinear Studies (CNLS)

Written by Jasper A. Vrugt: vrugt@lanl.gov

Version 0.5: June 2008
Version 1.0: October 2008  Adaption updated and generalized CR implementation



2010-04-20 Paul Kienzle
* Convert to python
"""
from __future__ import division, print_function

__all__ = ["Dream", "run_dream"]

import sys
import time

import numpy as np

from .state import MCMCDraw
from .metropolis import metropolis, metropolis_dr, dr_step
from .gelman import gelman
from .crossover import AdaptiveCrossover, LogAdaptiveCrossover
from .diffev import de_step
from .bounds import make_bounds_handler

# Everything should be available in state, but lets be lazy for now
LAST_TIME = 0


def console_monitor(state, pop, logp):
    global LAST_TIME
    if state.generation == 1:
        print("#gen", "logp(x)",
              " ".join("<%s>" % par for par in state.labels))
        LAST_TIME = time.time()

    current_time = time.time()
    if current_time >= LAST_TIME + 1:
        LAST_TIME = current_time
        print(state.generation, state._best_logp,
              " ".join("%.15g" % v for v in state._best_x))
        sys.stdout.flush()


class Dream(object):
    """
    Data structure containing the details of the running DREAM analysis code.
    """
    model = None
    # Sampling parameters
    burn = 0
    draws = 100000
    thinning = 1
    outlier_test = "IQR"
    population = None
    # DE parameters
    DE_steps = 10
    DE_pairs = 3
    DE_eps = 0.05
    DE_snooker_rate = 0.1
    DE_noise = 1e-6
    bounds_style = 'reflect'
    # Crossover parameters
    CR = None
    CR_spacing = 'linear'  # 'log' or 'linear'
    # Delay rejection parameters
    use_delayed_rejection = False
    DR_scale = 1  # 1-sigma step size using cov of population
    # Local optimizer best fit injection  The optimizer has
    # the following interface:
    #    x, fx = goalseek(mapper, bounds_handler, pop, fpop)
    # where:
    #    x, fx are the local optimum point and its value
    #    pop is the starting population
    #    fpop is the nllf for each point in pop
    #    mapper is a function which takes pop and returns fpop
    #    bounds_handler takes pop and forces all points into the range
    goalseek_optimizer = None
    goalseek_interval = 1e100  # close enough to never
    goalseek_minburn = 1000

    def __init__(self, **kw):
        self.monitor = console_monitor
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError("Unknown attribute "+k)

        self._initialized = False

    def sample(self, state=None, abort_test=None):
        """
        Pull the requisite number of samples from the distribution
        """
        if not self._initialized:
            self._initialized = True
        self.state = state
        try:
            run_dream(self, abort_test=abort_test)
        except KeyboardInterrupt:
            pass
        return self.state


def run_dream(dream, abort_test=None):

    # Step 1: Sample s points in the parameter space
    # [PAK] I moved this out of dream so that the user can use whatever
    # complicated sampling scheme they want.  Unfortunately, this means
    # the user needs to know some complex sampling scheme.
    if dream.population is None:
        raise ValueError("initial population not defined")

    # Remember the problem dimensions
    n_gen, n_chain, n_var = dream.population.shape
    n_pop = n_gen*n_chain

    if dream.CR is None:
        if dream.CR_spacing == 'log':
            dream.CR = LogAdaptiveCrossover(n_var)
        else:  # linear
            dream.CR = AdaptiveCrossover(3)

    # Step 2: Calculate posterior density associated with each value in x
    apply_bounds = make_bounds_handler(dream.model.bounds,
                                       style=dream.bounds_style)

    # Record initial state
    allocate_state(dream)
    state = dream.state
    state.labels = dream.model.labels
    previous_draws = state.draws
    if previous_draws:
        x, logp = state._last_gen()
    else:
        # No initial state, so evaluate initial population
        for x in dream.population:
            apply_bounds(x)
# ********************** MAP *****************************
            logp = dream.model.map(x)
            state._generation(new_draws=n_chain, x=x,
                              logp=logp, accept=n_chain,
                              force_keep=True)
            dream.monitor(state, x, logp)

    # Skip r_stat and pCR until we have some data data to analyze
    state._update(R_stat=-2, CR_weight=dream.CR.weight)

    # Now start drawing samples
    #print "previous draws", previous_draws, "new draws", dream.draws+dream.burn
    last_goalseek = (dream.draws + dream.burn)/n_pop - dream.goalseek_minburn
    next_goalseek = state.generation + dream.goalseek_interval \
        if dream.goalseek_optimizer else 1e100

    #need_outliers_removed = True
    scale = 1.0
    #serial_time = parallel_time = 0.
    #last_time = time.time()
    while state.draws < dream.draws + dream.burn:

        # Age the population using differential evolution
        dream.CR.reset(Nsteps=dream.DE_steps, Npop=n_chain)
        for gen in range(dream.DE_steps):

            # Define the current locations and associated posterior densities
            xold, logp_old = x, logp
            pop = state._draw_pop()

            # Generate candidates for each sequence
            xtry, step_alpha, used \
                = de_step(n_chain, pop, dream.CR[gen],
                          max_pairs=dream.DE_pairs,
                          eps=dream.DE_eps,
                          snooker_rate=dream.DE_snooker_rate,
                          noise=dream.DE_noise,
                          scale=scale)

            # PAK: Try a local optimizer every N generations
            if next_goalseek <= state.generation <= last_goalseek:
                best, logp_best = dream.goalseek_optimizer(
                    dream.model.map, apply_bounds, xold, logp_old)
                xtry[0] = best
                # Note: it is slightly inefficient to throw away logp_best,
                # but it makes the the code cleaner if we do
                next_goalseek += dream.goalseek_interval

            # Compute the likelihood of the candidates
            apply_bounds(xtry)
# ********************** MAP *****************************
            #next_time = time.time()
            #serial_time += next_time - last_time
            #last_time = next_time
            logp_try = dream.model.map(xtry)
            #next_time = time.time()
            #parallel_time  += next_time - last_time
            #last_time = next_time
            draws = len(logp_try)

            # Apply the metropolis acceptance/rejection rule
            x, logp, alpha, accept \
                = metropolis(xtry, logp_try,
                             xold, logp_old,
                             step_alpha)

            # Process delayed rejection
            # PAK NOTE: this updates according to the covariance matrix of the
            # current sample, which may be useful on unimodal systems, but
            # doesn't seem to be of any value in general; the DREAM papers
            # found that the acceptance rate did indeed improve with delayed
            # rejection, but the overall performance did not.  Worse, this
            # requires a linear system solution O(nPop^3) which can be near
            # singular for complex posterior distributions.
            if dream.use_delayed_rejection and not accept.all():
                # Generate alternate candidates using the covariance of xold
                xdr, r = dr_step(x=xold, scale=dream.DR_scale)

                # Compute the likelihood of the new candidates
                reject = ~accept
                apply_bounds(xdr)
# ********************** MAP *****************************
                logp_xdr = dream.model.map(xdr[reject])
                draws += len(logp_xdr)

                # Apply the metropolis delayed rejection rule.
                x[reject], logp[reject], alpha[reject], accept[reject] = \
                    metropolis_dr(xtry[reject], logp_try[reject],
                                  x[reject], logp[reject],
                                  xold[reject], logp_old[reject],
                                  alpha[reject], r)

            # els = zip(logp_old, logp_try, logp, x[:, -2], x[:, -1], accept)
            #print "pop", "\n ".join((("%12.3e "*(len(el)-1))%el[:-1])
            #                       +("T " if el[-3]<=el[-2] else "  ")
            #                       +("accept" if el[-1] else "")
            #                       for el in els)

            # Update Sequences with the new population.
            state._generation(draws, x, logp, accept)
# ********************** NOTIFY **************************
            dream.monitor(state, x, logp)
            #print state.generation, ":", state._best_logp

            # Keep track of which CR ratios were successful
            if state.draws <= dream.burn:
                dream.CR.update(gen, xold, x, used)
            
            if abort_test():
                break

        #print("serial&parallel",serial_time,parallel_time)
        # End of differential evolution aging
        # ---------------------------------------------------------------------

        # Calculate Gelman and Rubin convergence diagnostic
        #_, points, _ = state.chains()
        #r_stat = gelman(points, portion=0.5)
        r_stat = 0.  # Suppress for now since it is broken, and it costs to unroll

        #if state.draws <= 0.1 * dream.draws:
        if state.draws <= dream.burn:
            # Adapt the crossover ratio, but only during burn-in.
            dream.CR.adapt()
        # See whether there are any outlier chains, and remove
        # them to current best value of X
        #if need_outliers_removed and state.draws > 0.5*dream.burn:
        #    state.remove_outliers(x, logp, test=dream.outlier_test)
        #    need_outliers_removed = False

        if False:
            # Suppress scale update until we have a chance to verify that it
            # doesn't skew the resulting statistics.
            _, r = state.acceptance_rate()
            ravg = np.mean(r[-dream.DE_steps:])
            if ravg > 0.4:
                scale *= 1.01
            elif ravg < 0.2:
                scale /= 1.01



        # Save update information
        state._update(R_stat=r_stat, CR_weight=dream.CR.weight)
        
        if abort_test():
            break


def allocate_state(dream):
    """
    Estimate the size of the output
    """
    # Determine problem dimensions from the initial population
    n_pop, n_chain, n_var = dream.population.shape
    steps = dream.DE_steps
    thinning = dream.thinning
    n_cr = len(dream.CR.CR)
    draws = dream.draws

    n_update = int(draws/(steps*n_chain)) + 1
    n_gen = n_update * steps
    n_thin = int(n_gen/thinning) + 1
    #print n_gen, n_thin, n_update, draws, steps, Npop, n_var

    if dream.state is not None:
        dream.state.resize(
            n_gen, n_thin, n_update, n_var, n_chain, n_cr, thinning)
    else:
        dream.state = MCMCDraw(
            n_gen, n_thin, n_update, n_var, n_chain, n_cr, thinning)
