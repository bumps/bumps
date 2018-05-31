"""
Interfaces to various optimizers.
"""
from __future__ import print_function, division

import sys
import time
from copy import copy

import numpy as np

from . import monitor
from . import initpop
from . import lsqerror

from .history import History
from .formatnum import format_uncertainty
from .fitproblem import nllf_scale

from .dream import MCMCModel


class ConsoleMonitor(monitor.TimedUpdate):
    """
    Display fit progress on the console
    """
    def __init__(self, problem, progress=1, improvement=30):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = problem

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = format_uncertainty(scale*history.value[0], err)
        print("step", history.step[0], "cost", chisq)
        sys.stdout.flush()

    def show_improvement(self, history):
        # print "step",history.step[0],"chisq",history.value[0]
        p = self.problem.getp()
        try:
            self.problem.setp(history.point[0])
            print(self.problem.summarize())
        finally:
            self.problem.setp(p)
        sys.stdout.flush()


class StepMonitor(monitor.Monitor):
    """
    Collect information at every step of the fit and save it to a file.

    *fid* is the file to save the information to
    *fields* is the list of "step|time|value|point" fields to save

    The point field should be last in the list.
    """
    FIELDS = ['step', 'time', 'value', 'point']

    def __init__(self, problem, fid, fields=FIELDS):
        if any(f not in self.FIELDS for f in fields):
            raise ValueError("invalid monitor field")
        self.fid = fid
        self.fields = fields
        self.problem = problem
        self._pattern = "%%(%s)s\n" % (")s %(".join(fields))
        fid.write("# " + ' '.join(fields) + '\n')

    def config_history(self, history):
        history.requires(time=1, value=1, point=1, step=1)

    def __call__(self, history):
        point = " ".join("%.15g" % v for v in history.point[0])
        time = "%g" % history.time[0]
        step = "%d" % history.step[0]
        scale, _ = nllf_scale(self.problem)
        value = "%.15g" % (scale * history.value[0])
        out = self._pattern % dict(point=point, time=time,
                                   value=value, step=step)
        self.fid.write(out)

class MonitorRunner(object):
    """
    Adaptor which allows solvers to accept progress monitors.
    """
    def __init__(self, monitors, problem):
        if monitors is None:
            monitors = [ConsoleMonitor(problem)]
        self.monitors = monitors
        self.history = History(time=1, step=1, point=1, value=1,
                               population_points=1, population_values=1)
        for M in self.monitors:
            M.config_history(self.history)
        self._start = time.time()

    def __call__(self, step, point, value,
                 population_points=None, population_values=None):
        self.history.update(time=time.time() - self._start,
                            step=step, point=point, value=value,
                            population_points=population_points,
                            population_values=population_values)
        for M in self.monitors:
            M(self.history)


class FitBase(object):
    """
    FitBase defines the interface from bumps models to the various fitting
    engines available within bumps.

    Each engine is defined in its own class with a specific set of attributes
    and methods.

    The *name* attribute is the name of the optimizer.  This is just a simple
    string.

    The *settings* attribute is a list of pairs (name, default), where the
    names are defined as fields in FitOptions.  A best attempt should be
    made to map the fit options for the optimizer to the standard fit options,
    since each of these becomes a new command line option when running
    bumps.  If that is not possible, then a new option should be added
    to FitOptions.  A plugin architecture might be appropriate here, if
    there are reasons why specific problem domains might need custom fitters,
    but this is not yet supported.

    Each engine takes a fit problem in its constructor.

    The :meth:`solve` method runs the fit.  It accepts a
    monitor to track updates, a mapper to distribute work and
    key-value pairs defining the settings.

    There are a number of optional methods for the fitting engines.  Basically,
    all the methods in :class:`FitDriver` first check if they are specialized
    in the fit engine before performing a default action.

    The *load*/*save* methods load and save the fitter state in a given
    directory with a specific base file name.  The fitter can choose a file
    extension to add to the base name.  Some care is needed to be sure that
    the extension doesn't collide with other extensions such as .mon for
    the fit monitor.

    The *plot* method shows any plots to help understand the performance of
    the fitter, such as a convergence plot showing the the range of values
    in the population over time, as well as plots of the parameter uncertainty
    if available.  The plot should work within  is given a figure canvas to work with

    The *stderr*/*cov* methods should provide summary statistics for the
    parameter uncertainties.  Some fitters, such as MCMC, will compute these
    directly from the population.  Others, such as BFGS, will produce an
    estimate of the uncertainty as they go along.  If the fitter does not
    provide these estimates, then they will be computed from numerical
    derivatives at the minimum in the FitDriver method.
    """
    def __init__(self, problem):
        """Fit the models and show the results"""
        self.problem = problem

    def solve(self, monitors=None, mapper=None, **options):
        raise NotImplementedError


class MultiStart(FitBase):
    """
    Multi-start monte carlo fitter.

    This fitter wraps a local optimizer, restarting it a number of times
    to give it a chance to find a different local minimum.  If the keep_best
    option is True, then restart near the best fit, otherwise restart at
    random.
    """
    name = "Multistart Monte Carlo"
    settings = [('starts', 100)]

    def __init__(self, fitter):
        FitBase.__init__(self, fitter.problem)
        self.fitter = fitter

    def solve(self, monitors=None, mapper=None, **options):
        # TODO: need better way of tracking progress
        import logging
        starts = options.pop('starts', 1)
        reset = not options.pop('keep_best', True)
        f_best = np.inf
        x_best = self.problem.getp()
        for _ in range(max(starts, 1)):
            logging.info("multistart round %d", _)
            x, fx = self.fitter.solve(monitors=monitors, mapper=mapper,
                                      **options)
            if fx < f_best:
                x_best, f_best = x, fx
                logging.info("multistart f(x),x: %s %s", str(fx), str(x_best))
            if reset:
                self.problem.randomize()
            else:
                # Jitter
                self.problem.setp(x_best)
                pop = initpop.eps_init(1, self.problem.getp(),
                                       self.problem.bounds(),
                                       use_point=False, eps=1e-3)
                self.problem.setp(pop[0])
        return x_best, f_best


class DEFit(FitBase):
    """
    Classic Storn and Price differential evolution optimizer.
    """
    name = "Differential Evolution"
    id = "de"
    settings = [('steps', 1000), ('pop', 10), ('CR', 0.9), ('F', 2.0),
                ('ftol', 1e-8), ('xtol', 1e-6), #('stop', ''),
               ]

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        from .mystic.optimizer import de
        from .mystic.solver import Minimizer
        from .mystic import stop
        if monitors is None:
            monitors = [ConsoleMonitor(self.problem)]
        if mapper is not None:
            _mapper = lambda p, v: mapper(v)
        else:
            _mapper = lambda p, v: list(map(self.problem.nllf, v))
        resume = hasattr(self, 'state')
        steps = options['steps'] + (self.state['step'][-1] if resume else 0)
        strategy = de.DifferentialEvolution(npop=options['pop'],
                                            CR=options['CR'],
                                            F=options['F'],
                                            crossover=de.c_bin,
                                            mutate=de.rand1u)
        success = parse_tolerance(options)
        failure = stop.Steps(steps)
        self.history = History()
        # Step adds to current step number if resume
        minimize = Minimizer(strategy=strategy, problem=self.problem,
                             history=self.history, monitors=monitors,
                             success=success, failure=failure)
        if resume:
            self.history.restore(self.state)
        x = minimize(mapper=_mapper, abort_test=abort_test, resume=resume)
        #print(minimize.termination_condition())
        #with open("/tmp/evals","a") as fid:
        #   print >>fid,minimize.history.value[0],minimize.history.step[0],\
        #       minimize.history.step[0]*options['pop']*len(self.problem.getp())
        return x, self.history.value[0]

    def load(self, input_path):
        self.state = load_history(input_path)

    def save(self, output_path):
        save_history(output_path, self.history.snapshot())


def parse_tolerance(options):
    from .mystic import stop
    if options.get('stop', ''):
        return stop.parse_condition(options['stop'])

    xtol, ftol = options['xtol'], options['ftol']
    if xtol == 0:
        if ftol == 0:
            return None
        if ftol < 0:
            return stop.Rf(-ftol, scaled=True)
        return stop.Rf(ftol, scaled=False)
    else:
        if xtol == 0:
            return None
        if xtol < 0:
            return stop.Rx(-xtol, scaled=True)
        return stop.Rx(xtol, scaled=False)


def _history_file(path):
    return path + "-history.json"


def load_history(path):
    """
    Load fitter details from a history file.
    """
    import json
    with open(_history_file(path), "r") as fid:
        return json.load(fid)


def save_history(path, state):
    """
    Save fitter details to a history file as JSON.

    The content of the details are fitter specific.
    """
    import json
    with open(_history_file(path), "w") as fid:
        json.dump(state, fid)


class BFGSFit(FitBase):
    """
    BFGS quasi-newton optimizer.
    """
    name = "Quasi-Newton BFGS"
    id = "newton"
    settings = [('steps', 3000), ('starts', 1),
                ('ftol', 1e-6), ('xtol', 1e-12)]

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        from .quasinewton import quasinewton
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        result = quasinewton(fn=self.problem.nllf,
                             x0=self.problem.getp(),
                             monitor=self._monitor,
                             abort_test=abort_test,
                             itnlimit=options['steps'],
                             gradtol=options['ftol'],
                             steptol=1e-12,
                             macheps=1e-8,
                             eta=1e-8,
                            )
        self.result = result
        #code = result['status']
        #from .quasinewton import STATUS
        #print("%d: %s, x=%s, fx=%s"
        #      % (code, STATUS[code], result['x'], result['fx']))
        return result['x'], result['fx']

    # BFGS estimates hessian and its cholesky decomposition, but initial
    # tests give uncertainties quite different from the directly computed
    # jacobian in levenburg-marquardt or the hessian estimated at the
    # minimum by numdifftools
    def Hstderr(self):
        return lsqerror.chol_stderr(self.result['L'])

    def Hcov(self):
        return lsqerror.chol_cov(self.result['L'])

    def _monitor(self, step, x, fx):
        self._update(step=step, point=x, value=fx,
                     population_points=[x],
                     population_values=[fx])
        return True


class PSFit(FitBase):
    """
    Particle swarm optimizer.
    """
    name = "Particle Swarm"
    id = "ps"
    settings = [('steps', 3000), ('pop', 1)]

    def solve(self, monitors=None, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: list(map(self.problem.nllf, x))
        from .random_lines import particle_swarm
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        low, high = self.problem.bounds()
        cfo = dict(parallel_cost=mapper,
                   n=len(low),
                   x0=self.problem.getp(),
                   x1=low,
                   x2=high,
                   f_opt=0,
                   monitor=self._monitor)
        npop = int(cfo['n'] * options['pop'])

        result = particle_swarm(cfo, npop, maxiter=options['steps'])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best

    def _monitor(self, step, x, fx, k):
        self._update(step=step, point=x[:, k], value=fx[k],
                     population_points=x.T, population_values=fx)
        return True


class RLFit(FitBase):
    """
    Random lines optimizer.
    """
    name = "Random Lines"
    id = "rl"
    settings = [('steps', 3000), ('starts', 20), ('pop', 0.5), ('CR', 0.9)]

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: list(map(self.problem.nllf, x))
        from .random_lines import random_lines
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        low, high = self.problem.bounds()
        cfo = dict(parallel_cost=mapper,
                   n=len(low),
                   x0=self.problem.getp(),
                   x1=low,
                   x2=high,
                   f_opt=0,
                   monitor=self._monitor)
        npop = max(int(cfo['n'] * options['pop']), 3)

        result = random_lines(cfo, npop, abort_test=abort_test,
                              maxiter=options['steps'], CR=options['CR'])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best

    def _monitor(self, step, x, fx, k):
        # print "rl best",k, x.shape,fx.shape
        self._update(step=step, point=x[:, k], value=fx[k],
                     population_points=x.T, population_values=fx)
        return True


class PTFit(FitBase):
    """
    Parallel tempering optimizer.
    """
    name = "Parallel Tempering"
    id = "pt"
    settings = [('steps', 400), ('nT', 24), ('CR', 0.9),
                ('burn', 100), ('Tmin', 0.1), ('Tmax', 10)]

    def solve(self, monitors=None, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from .partemp import parallel_tempering
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        t = np.logspace(np.log10(options['Tmin']),
                        np.log10(options['Tmax']),
                        options['nT'])
        history = parallel_tempering(nllf=self.problem.nllf,
                                     p=self.problem.getp(),
                                     bounds=self.problem.bounds(),
                                     # logfile="partemp.dat",
                                     T=t,
                                     CR=options['CR'],
                                     steps=options['steps'],
                                     burn=options['burn'],
                                     monitor=self._monitor)
        return history.best_point, history.best

    def _monitor(self, step, x, fx, P, E):
        self._update(step=step, point=x, value=fx,
                     population_points=P, population_values=E)
        return True


class SimplexFit(FitBase):
    """
    Nelder-Mead simplex optimizer.
    """
    name = "Nelder-Mead Simplex"
    id = "amoeba"
    settings = [('steps', 1000), ('starts', 1), ('radius', 0.15),
                ('xtol', 1e-6), ('ftol', 1e-8)]

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        from .simplex import simplex
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        # TODO: no mapper??
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        # print "bounds",self.problem.bounds()
        result = simplex(f=self.problem.nllf, x0=self.problem.getp(),
                         bounds=self.problem.bounds(),
                         abort_test=abort_test,
                         update_handler=self._monitor,
                         maxiter=options['steps'],
                         radius=options['radius'],
                         xtol=options['xtol'],
                         ftol=options['ftol'])
        # Let simplex propose the starting point for the next amoeba
        # fit in a multistart amoeba context.  If the best is always
        # used, the fit can get stuck in a local minimum.
        self.problem.setp(result.next_start)
        #print("amoeba %s %s"%(result.x,result.fx))
        return result.x, result.fx

    def _monitor(self, k, n, x, fx):
        self._update(step=k, point=x[0], value=fx[0],
                     population_points=x, population_values=fx)
        return True

class MPFit(FitBase):
    """
    MPFit optimizer.
    """
    name = "MPFit"
    id = "mp"
    settings = [('steps', 200), ('ftol', 1e-10), ('xtol', 1e-10)]

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        from .mpfit import mpfit
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        self._low, self._high = self.problem.bounds()
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        self._abort = abort_test
        x0 = self.problem.getp()
        parinfo = []
        for low, high in zip(*self.problem.bounds()):
            parinfo.append({
                #'value': None,  # passed in by xall instead
                #'fixed': False,  # everything is varying
                'limited': (np.isfinite(low), np.isfinite(high)),
                'limits': (low, high),
                #'parname': '',  # could probably ask problem for this...
                # From the code, default step size is sqrt(eps)*abs(value)
                # or eps if value is 0.  This seems okay.  The other
                # other alternative is to limit it by bounds.
                #'step': 0,  # compute step automatically
                #'mpside': 0,  # 1, -1 or 2 for right-, left- or 2-sided deriv
                #'mpmaxstep': 0.,  # max step for this parameter
                #'tied': '',  # parameter expressions tying fit parameters
                #'mpprint': 1,  # print the parameter value when iterating
            })

        result = mpfit(
            fcn=self._residuals,
            xall=x0,
            parinfo=parinfo,
            autoderivative=True,
            fastnorm=True,
            #damp=0,  # no damping when damp=0
            # Stopping conditions
            ftol=options['ftol'],
            xtol=options['xtol'],
            #gtol=1e-100, # exclude gtol test
            maxiter=options['steps'],
            # Progress monitor
            iterfunct=self._monitor,
            nprint=1,  # call monitor each iteration
            quiet=True,  # leave it to monitor to print any info
            # Returns values
            nocovar=True,  # use our own covar calculation for consistency
        )

        if result.status > 0:
            x, fx = result.params, result.fnorm
        else:
            x, fx = None, None

        return x, fx


    def _monitor(self, fcn, p, k, fnorm,
                 functkw=None, parinfo=None,
                 quiet=0, dof=None, **extra):
        self._update(k, p, fnorm)

    def _residuals(self, p, fjac=None):
        if self._abort():
            return -1, None

        self.problem.setp(p)
        # treat prior probabilities on the parameters as additional
        # measurements
        residuals = np.hstack(
            (self.problem.residuals().flat, self.problem.parameter_residuals()))
        # Tally costs for broken constraints
        extra_cost = self.problem.constraints_nllf()
        # Spread the cost over the residuals.  Since we are smoothly increasing
        # residuals as we leave the boundary, this should push us back into the
        # boundary (within tolerance) during the lm fit.
        residuals += np.sign(residuals) * (extra_cost / len(residuals))
        return 0, residuals


class LevenbergMarquardtFit(FitBase):
    """
    Levenberg-Marquardt optimizer.
    """
    name = "Levenberg-Marquardt"
    id = "lm"
    settings = [('steps', 200), ('ftol', 1.5e-8), ('xtol', 1.5e-8)]
    # LM also has
    #    gtol: orthoganality between jacobian columns
    #    epsfcn: numerical derivative step size
    #    factor: initial radius
    #    diag: variable scale factors to bring them near 1

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        from scipy import optimize
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        self._low, self._high = self.problem.bounds()
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        x0 = self.problem.getp()
        maxfev = options['steps']*(len(x0)+1)
        result = optimize.leastsq(self._bounded_residuals,
                                  x0,
                                  ftol=options['ftol'],
                                  xtol=options['xtol'],
                                  maxfev=maxfev,
                                  epsfcn=1e-8,
                                  full_output=True)
        x, cov_x, info, mesg, success = result
        if not 1 <= success <= 4:
            # don't treat "reached maxfev" as a true failure
            if "reached maxfev" in mesg:
                # unless the x values are bad
                if not np.all(np.isfinite(x)):
                    x = None
                    mesg = "Levenberg-Marquardt fit failed with bad values"
            else:
                x = None
        self._cov = cov_x if x is not None else None
        # compute one last time with x forced inside the boundary, and using
        # problem.nllf as returned by other optimizers.  We will ignore the
        # covariance output and calculate it again ourselves.  Not ideal if
        # f is expensive, but it will be consistent with other optimizers.
        if x is not None:
            x += self._stray_delta(x)
            self.problem.setp(x)
            fx = self.problem.nllf()
        else:
            fx = None
        return x, fx

    def _bounded_residuals(self, p):
        # Force the fit point into the valid region
        stray = self._stray_delta(p)
        stray_cost = np.sum(stray**2)
        if stray_cost > 0:
            stray_cost += 1e6
        self.problem.setp(p + stray)
        # treat prior probabilities on the parameters as additional
        # measurements
        residuals = np.hstack(
            (self.problem.residuals().flat, self.problem.parameter_residuals()))
        # Tally costs for straying outside the boundaries plus other costs
        extra_cost = stray_cost + self.problem.constraints_nllf()
        # Spread the cost over the residuals.  Since we are smoothly increasing
        # residuals as we leave the boundary, this should push us back into the
        # boundary (within tolerance) during the lm fit.
        residuals += np.sign(residuals) * (extra_cost / len(residuals))
        return residuals

    def _stray_delta(self, p):
        """calculate how far point is outside the boundary"""
        return (np.where(p < self._low, self._low - p, 0)
                + np.where(p > self._high, self._high - p, 0))

    def cov(self):
        return self._cov


class SnobFit(FitBase):
    name = "SNOBFIT"
    id = "snobfit"
    settings = [('steps', 200)]

    def solve(self, monitors=None, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from snobfit.snobfit import snobfit
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        x, fx, _ = snobfit(self.problem, self.problem.getp(),
                           self.problem.bounds(),
                           fglob=0, callback=self._monitor)
        return x, fx

    def _monitor(self, k, x, fx, improved):
        # TODO: snobfit does have a population...
        self._update(step=k, point=x, value=fx,
                     population_points=[x], population_values=[fx])


class DreamModel(MCMCModel):

    """
    DREAM wrapper for fit problems.
    """

    def __init__(self, problem=None, mapper=None):
        """
        Create a sampling from the multidimensional likelihood function
        represented by the problem set using dream.
        """
        # print "dream"
        self.problem = problem
        self.bounds = self.problem.bounds()
        self.labels = self.problem.labels()

        self.mapper = mapper if mapper else lambda p: list(map(self.nllf, p))

    def log_density(self, x):
        return -self.nllf(x)

    def nllf(self, x):
        """Negative log likelihood of seeing models given *x*"""
        # Note: usually we will be going through the provided mapper, and
        # this function will never be called.
        # print "eval",x; sys.stdout.flush()
        return self.problem.nllf(x)

    def map(self, pop):
        # print "calling mapper",self.mapper
        return -np.array(self.mapper(pop))


class DreamFit(FitBase):
    name = "DREAM"
    id = "dream"
    settings = [('samples', int(1e4)), ('burn', 100), ('pop', 10),
                ('init', 'eps'), ('thin', 1),
                ('steps', 0),  # deprecated: use --samples instead
               ]

    def __init__(self, problem):
        FitBase.__init__(self, problem)
        self.dream_model = DreamModel(problem)
        self.state = None

    def solve(self, monitors=None, abort_test=None, mapper=None, **options):
        from .dream import Dream
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)

        if mapper:
            self.dream_model.mapper = mapper
        self._update = MonitorRunner(problem=self.dream_model.problem,
                                     monitors=monitors)

        population = initpop.generate(self.dream_model.problem, **options)
        pop_size = population.shape[0]
        draws, steps = int(options['samples']), options['steps']
        if steps == 0:
            steps = (draws + pop_size-1) // pop_size
        # TODO: need a better way to announce number of steps
        # maybe somehow print iteration # of # iters in the monitor?
        print("# steps: %d, # draws: %d"%(steps, pop_size*steps))
        population = population[None, :, :]
        sampler = Dream(model=self.dream_model, population=population,
                        draws=pop_size * steps,
                        burn=pop_size * options['burn'],
                        thinning=options['thin'],
                        monitor=self._monitor,
                        DE_noise=1e-6)

        self.state = sampler.sample(state=self.state, abort_test=abort_test)
        self.state.mark_outliers()
        self.state.keep_best()
        self.state.title = self.dream_model.problem.name

        # TODO: Temporary hack to apply a post-mcmc action to the state vector
        # The problem is that if we manipulate the state vector before saving
        # it then we will not be able to use the --resume feature.  We can
        # get around this by just not writing state for the derived variables,
        # at which point we can remove this notice.
        # TODO: Add derived/visible variable support to other optimizers
        fn, labels = getattr(self.problem, 'derive_vars', (None, None))
        if fn is not None:
            self.state.derive_vars(fn, labels=labels)
        visible_vars = getattr(self.problem, 'visible_vars', None)
        if visible_vars is not None:
            self.state.set_visible_vars(visible_vars)
        integer_vars = getattr(self.problem, 'integer_vars', None)
        if integer_vars is not None:
            self.state.set_integer_vars(integer_vars)

        x, fx = self.state.best()

        # Check that the last point is the best point
        #points, logp = self.state.sample()
        #assert logp[-1] == fx
        #print(points[-1], x)
        #assert all(points[-1, i] == xi for i, xi in enumerate(x))
        return x, -fx

    def entropy(self, **kw):
        return self.state.entropy(**kw)

    def _monitor(self, state, pop, logp):
        # Get an early copy of the state
        self._update.history.uncertainty_state = state
        step = state.generation
        x, fx = state.best()
        self._update(step=step, point=x, value=-fx,
                     population_points=pop, population_values=-logp)
        return True

    def stderr(self):
        """
        Approximate standard error as 1/2 the 68% interval fo the sample,
        which is a more robust measure than the mean of the sample for
        non-normal distributions.
        """
        from .dream.stats import var_stats

        vstats = var_stats(self.state.draw())
        return np.array([(v.p68[1] - v.p68[0]) / 2 for v in vstats], 'd')

    #def cov(self):
    #    # Covariance estimate from final 1000 points
    #    return np.cov(self.state.draw().points[-1000:])

    def load(self, input_path):
        from .dream.state import load_state
        print("loading saved state (this might take awhile) ...")
        fn, labels = getattr(self.problem, 'derive_vars', (None, []))
        self.state = load_state(input_path, report=100, derived_vars=len(labels))

    def save(self, output_path):
        self.state.save(output_path)

    def plot(self, output_path):
        self.state.show(figfile=output_path)
        self.error_plot(figfile=output_path)

    def show(self):
        pass

    def error_plot(self, figfile):
        # Produce error plot
        import pylab
        from . import errplot
        # TODO: shouldn't mix calc and display!
        res = errplot.calc_errors_from_state(self.dream_model.problem,
                                             self.state)
        if res is not None:
            pylab.figure()
            errplot.show_errors(res)
            pylab.savefig(figfile + "-errors.png", format='png')


class Resampler(FitBase):
    # TODO: why isn't cli.resynth using this?

    def __init__(self, fitter):
        self.fitter = fitter
        raise NotImplementedError

    def solve(self, **options):
        starts = options.pop('starts', 1)
        restart = options.pop('restart', False)
        x, fx = self.fitter.solve(**options)
        points = _resampler(self.fitter, x, samples=starts,
                            restart=restart, **options)
        self.points = points  # save points for later plotting
        return x, fx


def _resampler(fitter, xinit, samples=100, restart=False, **options):
    """
    Refit the result multiple times with resynthesized data, building
    up an array in Result.samples which contains the best fit to the
    resynthesized data.  *samples* is the number of samples to generate.
    *fitter* is the (local) optimizer to use. **kw are the parameters
    for the optimizer.
    """
    x = xinit
    points = []
    try:  # TODO: some solvers already catch KeyboardInterrupt
        for _ in range(samples):
            # print "== resynth %d of %d" % (i, samples)
            fitter.problem.resynth_data()
            if restart:
                fitter.problem.randomize()
            else:
                fitter.problem.setp(x)
            x, fx = fitter.solve(**options)
            points.append(np.hstack((fx, x)))
            # print self.problem.summarize()
            # print "[chisq=%g]" % (nllf*2/self.problem.dof)
    except KeyboardInterrupt:
        # On keyboard interrupt we can declare that we are finished sampling
        # without it being an error condition, so let this exception pass.
        pass
    finally:
        # Restore the state of the problem
        fitter.problem.restore_data()
        fitter.problem.setp(xinit)
        fitter.problem.model_update()
    return points


class FitDriver(object):

    def __init__(self, fitclass=None, problem=None, monitors=None,
                 abort_test=None, mapper=None, **options):
        self.fitclass = fitclass
        self.problem = problem
        self.options = options
        self.monitors = monitors
        self.abort_test = abort_test
        self.mapper = mapper if mapper else lambda p: list(map(problem.nllf, p))
        self.fitter = None

    def fit(self, resume=None):

        if hasattr(self, '_cov'):
            del self._cov
        if hasattr(self, '_stderr'):
            del self._stderr
        fitter = self.fitclass(self.problem)
        if resume:
            fitter.load(resume)
        starts = self.options.get('starts', 1)
        if starts > 1:
            fitter = MultiStart(fitter)
        t0 = time.clock()
        self.fitter = fitter
        x, fx = fitter.solve(monitors=self.monitors,
                             abort_test=self.abort_test,
                             mapper=self.mapper,
                             **self.options)
        self.time = time.clock() - t0
        self.result = x, fx
        if x is not None:
            self.problem.setp(x)
        return x, fx

    def entropy(self):
        if hasattr(self.fitter, 'entropy'):
            return self.fitter.entropy()
        else:
            from .dream import entropy
            return entropy.cov_entropy(self.cov()), 0

    def cov(self):
        r"""
        Return an estimate of the covariance of the fit.

        Depending on the fitter and the problem, this may be computed from
        existing evaluations within the fitter, or from numerical
        differentiation around the minimum.  The numerical differentiation
        will use the Hessian estimated from nllf.   If the problem uses
        $\chi^2/2$ as its nllf, then you may want to instead compute
        the covariance from the Jacobian::

            J = lsqerror.jacobian(fitdriver.result[0])
            cov = lsqerror.cov(J)

        This should be faster and more accurate than the Hessian of nllf
        when you can use it.
        """
        if not hasattr(self, '_cov'):
            self._cov = None
            if hasattr(self.fitter, 'cov'):
                self._cov = self.fitter.cov()
        if self._cov is None:
            if hasattr(self.problem, 'residuals'):
                J = lsqerror.jacobian(self.problem, self.result[0])
                self._cov = lsqerror.cov(J)
            else:
                H = lsqerror.hessian(self.problem, self.result[0])
                H, L = lsqerror.perturbed_hessian(H)
                self._cov = lsqerror.chol_cov(L)
        return self._cov

    def stderr(self):
        """
        Return an estimate of the standard error of the fit.

        Depending on the fitter and the problem, this may be computed from
        existing evaluations within the fitter, or from numerical
        differentiation around the minimum.
        """
        if not hasattr(self, '_stderr'):
            self._stderr = None
            if hasattr(self.fitter, 'stderr'):
                self._stderr = self.fitter.stderr()
        if self._stderr is None:
            # If no stderr from the fitter then compute it from the covariance
            self._stderr = lsqerror.stderr(self.cov())
        return self._stderr

    def show(self):
        if hasattr(self.fitter, 'show'):
            self.fitter.show()
        if hasattr(self.problem, 'show'):
            self.problem.show()

    def show_err(self):
        """
        Display the error approximation from the numerical derivative.

        Warning: cost grows as the cube of the number of parameters.
        """
        # TODO: need cheaper uncertainty estimate
        # Note: error estimated from hessian diagonal is insufficient.
        err = lsqerror.stderr(self.cov())
        norm = np.sqrt(self.problem.chisq())
        print("=== Uncertainty est. from curvature: par    dx           dx/sqrt(chisq) ===")
        for k, v, dv in zip(self.problem.labels(), self.problem.getp(), err):
            print("%40s %-15s %-15s" % (k,
                  format_uncertainty(v, dv),
                  format_uncertainty(v, dv/norm)))
        print("="*75)

    def save(self, output_path):
        # print "calling driver save"
        if hasattr(self.fitter, 'save'):
            self.fitter.save(output_path)
        if hasattr(self.problem, 'save'):
            self.problem.save(output_path)

    def load(self, input_path):
        # print "calling driver save"
        if hasattr(self.fitter, 'load'):
            self.fitter.load(input_path)
        if hasattr(self.problem, 'load'):
            self.problem.load(input_path)

    def plot(self, output_path, view=None):
        # print "calling fitter.plot"
        if hasattr(self.problem, 'plot'):
            self.problem.plot(figfile=output_path, view=view)
        if hasattr(self.fitter, 'plot'):
            self.fitter.plot(output_path=output_path)



def _fill_defaults(options, settings):
    """
    Returns options dict with missing values filled from settings.
    """
    result = dict(settings)  # settings is a list of (key,value) pairs
    result.update(options)
    return result

# List of (parameter,factory value) required for each algorithm
FITTERS = [
    SimplexFit,
    DEFit,
    DreamFit,
    BFGSFit,
    LevenbergMarquardtFit,
    MPFit,
    PSFit,
    PTFit,
    RLFit,
    SnobFit,
    ]

FIT_AVAILABLE_IDS = [f.id for f in FITTERS]


FIT_ACTIVE_IDS = [
    SimplexFit.id,
    DEFit.id,
    DreamFit.id,
    BFGSFit.id,
    LevenbergMarquardtFit.id,
    MPFit.id,
    ]

FIT_DEFAULT_ID = SimplexFit.id

assert FIT_DEFAULT_ID in FIT_ACTIVE_IDS
assert all(f in FIT_AVAILABLE_IDS for f in FIT_ACTIVE_IDS)

def fit(problem, method=FIT_DEFAULT_ID, verbose=False, **options):
    """
    Simplified fit interface.

    Given a fit problem, the name of a fitter and the fitter options,
    it will run the fit and return the best value and standard error of
    the parameters.  If *verbose* is true, then the console monitor will
    be enabled, showing progress through the fit and showing the parameter
    standard error at the end of the fit, otherwise it is completely
    silent.

    Returns an *OptimizeResult* object containing "x" and "dx".  The
    dream fitter also includes the "state" object, allowing for more
    detailed uncertainty analysis.  Optimizer information such as the
    stopping condition and the number of function evaluations are not
    yet included.
    """
    from scipy.optimize import OptimizeResult

    #verbose = True
    if method not in FIT_AVAILABLE_IDS:
        raise ValueError("unknown method %r not one of %s"
                         % (method, ", ".join(sorted(FIT_ACTIVE_IDS))))
    for fitclass in FITTERS:
        if fitclass.id == method:
            break
    monitors = None if verbose else []  # default is step monitor
    driver = FitDriver(
        fitclass=fitclass, problem=problem, monitors=monitors,
        **options)
    x0 = problem.getp()
    x, fx = driver.fit()
    problem.setp(x)
    dx = driver.stderr()
    if verbose:
        print("final chisq", problem.chisq_str())
        driver.show_err()
    result = OptimizeResult(
        x=x, dx=driver.stderr(),
        fun=fx,
        success=True, status=0, message="successful termination",
        #nit=0, # number of iterations
        #nfev=0, # number of function evaluations
        #njev, nhev # jacobian and hessian evaluations
        #maxcv=0, # max constraint violation
        )
    if hasattr(driver.fitter, 'state'):
        result.state = driver.fitter.state
    return result
