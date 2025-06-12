"""
Interfaces to various optimizers.
"""

import sys
import warnings
from time import perf_counter

import numpy as np

from . import monitor
from . import initpop
from . import lsqerror

from .history import History
from .formatnum import format_uncertainty
from .util import NDArray, format_duration

# For typing
from typing import List, Tuple, Dict, Any, Optional
from numpy.typing import NDArray
from h5py import Group
from bumps.dream.state import MCMCDraw


class ConsoleMonitor(monitor.TimedUpdate):
    """
    Display fit progress on the console
    """

    def __init__(self, problem, progress=1, improvement=30):
        monitor.TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.problem = problem

    def _print_chisq(self, k, fx, final=False):
        print(f"step {k} cost {self.problem.chisq_str(nllf=fx)}{' [final]' if final else ''}")

    def _print_pars(self, x):
        p = self.problem.getp()
        try:
            self.problem.setp(x)
            print(self.problem.summarize())
        finally:
            self.problem.setp(p)

    def show_progress(self, history):
        self._print_chisq(history.step[0], history.value[0])
        sys.stdout.flush()

    def show_improvement(self, history):
        self._print_pars(history.point[0])
        sys.stdout.flush()

    def final(self, history: History, best: Dict[str, Any]):
        self._print_chisq(history.step[0], best["value"], final=True)
        self._print_pars(best["point"])
        print(f"time {format_duration(history.time[0])}")
        sys.stdout.flush()

    def info(self, message: str):
        print(message)
        sys.stdout.flush()


class CheckpointMonitor(monitor.TimedUpdate):
    """
    Periodically save fit state so that it can be resumed later.
    """

    #: Function to call at each checkpoint.
    checkpoint = None  # type: Callable[None, None]

    def __init__(self, checkpoint, progress=60 * 30):
        monitor.TimedUpdate.__init__(self, progress=progress, improvement=np.inf)
        self.checkpoint = checkpoint
        self._first = True

    def show_progress(self, history):
        # Skip the first checkpoint since it only contains the
        # start/resume state
        if self._first:
            self._first = False
        else:
            self.checkpoint(history)

    def show_improvement(self, history):
        pass


class StepMonitor(monitor.Monitor):
    """
    Collect information at every step of the fit and save it to a file.

    *fid* is the file to save the information to
    *fields* is the list of "step|time|value|point" fields to save

    The point field should be last in the list.
    """

    FIELDS = ["step", "time", "value", "point"]

    def __init__(self, problem, fid, fields=FIELDS):
        if any(f not in self.FIELDS for f in fields):
            raise ValueError("invalid monitor field")
        self.fid = fid
        self.fields = fields
        self.problem = problem
        self._pattern = "%%(%s)s\n" % (")s %(".join(fields))
        fid.write("# " + " ".join(fields) + "\n")

    def config_history(self, history):
        history.requires(time=1, value=1, point=1, step=1)

    def update(self, history):
        point = " ".join("%.15g" % v for v in history.point[0])
        time = "%g" % history.time[0]
        step = "%d" % history.step[0]
        value = "%.15g" % (self.problem.chisq(nllf=history.value[0]))
        out = self._pattern % dict(point=point, time=time, value=value, step=step)
        self.fid.write(out)

    __call__ = update


class MonitorRunner(object):
    """
    Adaptor which allows solvers to accept progress monitors.

    The stopping() method manages checks for abort and timeout.
    """

    def __init__(self, monitors: List[monitor.Monitor], problem, abort_test=None, max_time=0.0):
        self.monitors = monitors
        self.history = History(time=1, step=1, point=1, value=1, population_points=1, population_values=1)
        # Pre-populate history.time so we can call stopping() before the first update.
        self.history.update(time=0.0)
        for M in self.monitors:
            M.config_history(self.history)
        self._start = perf_counter()
        self.max_time = max_time
        self.abort_test = abort_test if abort_test is not None else lambda: False

    def update(
        self,
        step: int,
        point: NDArray,
        value: float,
        population_points: Optional[NDArray] = None,
        population_values: Optional[NDArray] = None,
    ):
        # Note: DEFit doesn't use MonitorRunner for config/update
        self.history.update(
            time=perf_counter() - self._start,
            step=step,
            point=point,
            value=value,
            population_points=population_points,
            population_values=population_values,
        )
        for M in self.monitors:
            M(self.history)

    __call__ = update

    def stopping(self):
        return self.abort_test() or (self.history.time[0] >= self.max_time > 0)

    def info(self, message: str):
        for M in self.monitors:
            monitor_message = getattr(M, "info", None)
            if monitor_message is not None:
                monitor_message(message)

    def final(self, point: NDArray, value: float):
        best = dict(point=point, value=value)
        for M in self.monitors:
            monitor_final = getattr(M, "final", None)
            if monitor_final is not None:
                monitor_final(self.history, best)


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

    name: str
    """Display name for the fit method"""
    id: str
    """Short name for the fit method, used as --id on the command line."""
    # TODO: Replace list of tuples with an ordered dictionary?
    settings: List[Tuple[str, Any]]
    """Available fitting options and their default values."""
    state: Any = None
    """
    Internal fit state. If the state object has a draw method this should return
    a set of points from the posterior probability distribution for the fit.
    """

    def __init__(self, problem):
        """Fit the models and show the results"""
        self.problem: "bumps.fitproblem.FitProblem" = problem

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        raise NotImplementedError()

    @staticmethod
    def h5dump(group: "Group", state: Any) -> None:
        """
        Store fitter.state into the given HDF5 Group.

        This will be restored by the corresponding h5load, then passed
        to the fitter to resume from its current state. This strategy
        is particularly useful for MCMC analysis where you may need more
        iterations for the chains to reach equilibrium.  It is also the
        basis of checkpoint/restore operations for fitters such as de
        and amoeba which manage a population, though in those cases the
        best point seen so far may be good enough.
        """
        # Default is nothing to save because resume isn't supported for the fitter
        ...

    @staticmethod
    def h5load(group: "Group") -> Any:
        """
        Load internal fit state from the group saved by h5dump. Note that
        this function will be responsible for migrating state from older
        versions to newer versions of the saved representation.
        """
        return None


class MultiStart(FitBase):
    """
    Multi-start monte carlo fitter.

    This fitter wraps a local optimizer, restarting it a number of times
    to give it a chance to find a different local minimum.  If the jump
    radius is non-zero, then restart near the best fit, otherwise restart at
    random.
    """

    name = "Multistart Monte Carlo"
    settings = [("starts", 100), ("jump", 0.0)]

    def __init__(self, fitter):
        FitBase.__init__(self, fitter.problem)
        self.fitter = fitter

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        starts = max(options.pop("starts", 1), 1)
        jump = options.pop("jump", 0.0)
        x_best, f_best, chisq_best = None, np.inf, None
        for k in range(starts):
            x, fx = self.fitter.solve(monitors=monitors, mapper=mapper, **options)
            chisq = self.problem.chisq_str(nllf=fx)
            if fx < f_best:
                x_best, f_best, chisq_best = x, fx, chisq
                monitors.info(f"fit {k+1} of {starts}: {chisq} [new best]")
            else:
                monitors.info(f"fit {k+1} of {starts}: {chisq} [best={chisq_best}]")
            if k >= starts - 1 or monitors.stopping():
                break
            if jump == 0.0:
                self.problem.randomize()
            else:
                pop = initpop.eps_init(1, x_best, self.problem.bounds(), use_point=False, eps=jump)
                self.problem.setp(pop[0])
            # print(f"jump={jump} moving from {x} to {self.problem.getp()}")
        return x_best, f_best


class DEFit(FitBase):
    """
    Classic Storn and Price differential evolution optimizer.
    """

    name = "Differential Evolution"
    id = "de"
    settings = [
        ("steps", 1000),
        ("pop", 10),
        ("CR", 0.9),
        ("F", 2.0),
        ("ftol", 1e-8),
        ("xtol", 1e-6),  # ('stop', ''),
    ]
    state = None

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        from .mystic.optimizer import de
        from .mystic.solver import Minimizer
        from .mystic import stop

        if mapper is not None:
            _mapper = lambda p, v: mapper(v)
        else:
            _mapper = lambda p, v: list(map(self.problem.nllf, v))
        resume = self.state is not None
        steps = options["steps"] + (self.state["step"][-1] if resume else 0)
        strategy = de.DifferentialEvolution(
            npop=options["pop"], CR=options["CR"], F=options["F"], crossover=de.c_bin, mutate=de.rand1u
        )
        success = parse_tolerance(options)
        failure = stop.Steps(steps)
        # Step adds to current step number if resume
        minimize = Minimizer(
            strategy=strategy,
            problem=self.problem,
            # TODO: use MonitorRunner update within DE
            history=monitors.history,
            monitors=monitors.monitors,
            success=success,
            failure=failure,
        )
        if self.state is not None:
            monitors.history.restore(self.state)
        x = minimize(mapper=_mapper, abort_test=monitors.stopping, resume=resume)
        self.state = monitors.history.snapshot()
        # print("final de state", self.state)
        # print(minimize.termination_condition())
        # with open("/tmp/evals","a") as fid:
        #   print >>fid,minimize.history.value[0],minimize.history.step[0],\
        #       minimize.history.step[0]*options['pop']*len(self.problem.getp())
        return x, monitors.history.value[0]

    def load(self, input_path):
        self.state = _de_load_history(input_path)

    def save(self, output_path):
        _de_save_history(output_path, self.state)

    @staticmethod
    def h5load(group: Group) -> Any:
        from .webview.server.state_hdf5_backed import read_json

        return read_json(group, "DE_history")

    @staticmethod
    def h5dump(group: Group, state: Dict[str, Any]):
        from .webview.server.state_hdf5_backed import write_json

        write_json(group, "DE_history", state)


def parse_tolerance(options):
    from .mystic import stop

    if options.get("stop", ""):
        return stop.parse_condition(options["stop"])

    xtol, ftol = options["xtol"], options["ftol"]
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


def _de_history_file(path):
    return path + "-history.json"


def _de_load_history(path):
    """
    Load fitter details from a history file.
    """
    import json

    with open(_de_history_file(path), "r") as fid:
        return json.load(fid)


def _de_save_history(path, state):
    """
    Save fitter details to a history file as JSON.

    The content of the details are fitter specific.
    """
    import json

    with open(_de_history_file(path), "w") as fid:
        json.dump(state, fid)


class BFGSFit(FitBase):
    """
    BFGS quasi-newton optimizer.

    BFGS estimates Hessian and its Cholesky decomposition, but initial
    tests give uncertainties quite different from the directly computed
    Jacobian in Levenburg-Marquardt or the Hessian estimated at the
    minimum by numerical differentiation.

    To use the internal 'H' and 'L' and save some computation time, then
    use::

        C = lsqerror.chol_cov(fit.result['L'])
        stderr = lsqerror.stderr(C)
    """

    name = "Quasi-Newton BFGS"
    id = "newton"
    settings = [("steps", 3000), ("ftol", 1e-6), ("xtol", 1e-12), ("starts", 1), ("jump", 0.0)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        from .quasinewton import quasinewton

        def update(step, x, fx):
            monitors(step=step, point=x, value=fx, population_points=[x], population_values=[fx])
            return not monitors.stopping()

        result = quasinewton(
            fn=self.problem.nllf,
            x0=self.problem.getp(),
            monitor=update,
            itnlimit=options["steps"],
            gradtol=options["ftol"],
            steptol=1e-12,
            macheps=1e-8,
            eta=1e-8,
        )
        self.result = result
        # code = result['status']
        # from .quasinewton import STATUS
        # print("%d: %s, x=%s, fx=%s"
        #      % (code, STATUS[code], result['x'], result['fx']))
        return result["x"], result["fx"]


class PSFit(FitBase):
    """
    Particle swarm optimizer.
    """

    name = "Particle Swarm"
    id = "ps"
    settings = [("steps", 3000), ("pop", 1)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        from .random_lines import particle_swarm

        options = _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: list(map(self.problem.nllf, x))

        def update(step, x, fx, k):
            monitors(step=step, point=x[:, k], value=fx[k], population_points=x.T, population_values=fx)
            return not monitors.stopping()

        low, high = self.problem.bounds()
        cfo = dict(
            parallel_cost=mapper,
            n=len(low),
            x0=self.problem.getp(),
            x1=low,
            x2=high,
            f_opt=0,
            monitor=update,
        )
        npop = int(cfo["n"] * options["pop"])

        result = particle_swarm(cfo, npop, maxiter=options["steps"])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best


class RLFit(FitBase):
    """
    Random lines optimizer.
    """

    name = "Random Lines"
    id = "rl"
    settings = [("steps", 3000), ("pop", 0.5), ("CR", 0.9), ("starts", 20), ("jump", 0.0)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        from .random_lines import random_lines

        options = _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: list(map(self.problem.nllf, x))

        def update(step, x, fx, k):
            monitors(step=step, point=x[:, k], value=fx[k], population_points=x.T, population_values=fx)
            return not monitors.stopping()

        low, high = self.problem.bounds()
        cfo = dict(
            parallel_cost=mapper,
            n=len(low),
            x0=self.problem.getp(),
            x1=low,
            x2=high,
            f_opt=0,
            monitor=update,
        )
        npop = max(int(cfo["n"] * options["pop"]), 3)

        result = random_lines(cfo, npop, maxiter=options["steps"], CR=options["CR"])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best


class PTFit(FitBase):
    """
    Parallel tempering optimizer.
    """

    name = "Parallel Tempering"
    id = "pt"
    settings = [("steps", 400), ("nT", 24), ("CR", 0.9), ("burn", 100), ("Tmin", 0.1), ("Tmax", 10.0)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from .partemp import parallel_tempering

        def update(step, x, fx, P, E):
            monitors(step=step, point=x, value=fx, population_points=P, population_values=E)
            return not monitors.stopping()

        t = np.logspace(np.log10(options["Tmin"]), np.log10(options["Tmax"]), options["nT"])
        history = parallel_tempering(
            nllf=self.problem.nllf,
            p=self.problem.getp(),
            bounds=self.problem.bounds(),
            # logfile="partemp.dat",
            T=t,
            CR=options["CR"],
            steps=options["steps"],
            burn=options["burn"],
            monitor=update,
        )
        return history.best_point, history.best


class SimplexFit(FitBase):
    """
    Nelder-Mead simplex optimizer.
    """

    name = "Nelder-Mead Simplex"
    id = "amoeba"
    settings = [("steps", 1000), ("radius", 0.15), ("xtol", 1e-6), ("ftol", 1e-8), ("starts", 1), ("jump", 0.01)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        from .simplex import simplex

        options = _fill_defaults(options, self.settings)

        # TODO: no mapper??
        # print("bounds", self.problem.bounds())
        def update(k, n, x, fx):
            monitors(step=k, point=x[0], value=fx[0], population_points=x, population_values=fx)

        result = simplex(
            f=self.problem.nllf,
            x0=self.problem.getp(),
            bounds=self.problem.bounds(),
            abort_test=monitors.stopping,
            update_handler=update,
            maxiter=options["steps"],
            radius=options["radius"],
            xtol=options["xtol"],
            ftol=options["ftol"],
        )
        # Let simplex propose the starting point for the next amoeba
        # fit in a multistart amoeba context.  If the best is always
        # used, the fit can get stuck in a local minimum.
        self.problem.setp(result.next_start)
        # print("amoeba %s %s"%(result.x,result.fx))
        return result.x, result.fx


class MPFit(FitBase):
    """
    MPFit optimizer.
    """

    name = "Levenberg-Marquardt"
    id = "lm"
    settings = [("steps", 200), ("ftol", 1e-10), ("xtol", 1e-10), ("starts", 1), ("jump", 0.0)]

    def solve(self, monitors=None, mapper=None, **options):
        from .mpfit import mpfit

        options = _fill_defaults(options, self.settings)
        self._low, self._high = self.problem.bounds()
        self._stopping = monitors.stopping
        x0 = self.problem.getp()
        parinfo = []
        for low, high in zip(*self.problem.bounds()):
            parinfo.append(
                {
                    #'value': None,  # passed in by xall instead
                    #'fixed': False,  # everything is varying
                    "limited": (np.isfinite(low), np.isfinite(high)),
                    "limits": (low, high),
                    #'parname': '',  # could probably ask problem for this...
                    # From the code, default step size is sqrt(eps)*abs(value)
                    # or eps if value is 0.  This seems okay.  The other
                    # other alternative is to limit it by bounds.
                    #'step': 0,  # compute step automatically
                    #'mpside': 0,  # 1, -1 or 2 for right-, left- or 2-sided deriv
                    #'mpmaxstep': 0.,  # max step for this parameter
                    #'tied': '',  # parameter expressions tying fit parameters
                    #'mpprint': 1,  # print the parameter value when iterating
                }
            )

        def update(fcn, p, k, fnorm, functkw=None, parinfo=None, quiet=0, dof=None, **extra):
            # The mpfit residuals are set up so that fnorm = sumsq residuals = 2*nllf.
            monitors(step=k, point=p, value=fnorm / 2)
            if monitors.stopping():
                return -1

        result = mpfit(
            fcn=self._residuals,
            xall=x0,
            parinfo=parinfo,
            autoderivative=True,
            fastnorm=True,
            double=0,  # use single precision machine epsilon for derivative step
            # damp=0,  # no damping when damp=0
            # Stopping conditions
            ftol=options["ftol"],
            xtol=options["xtol"],
            # gtol=1e-100, # exclude gtol test
            maxiter=options["steps"],
            # Progress monitor
            iterfunct=update,
            nprint=1,  # call monitor each iteration
            quiet=True,  # leave it to monitor to print any info
            # Returns values
            nocovar=True,  # use our own covar calculation for consistency
        )
        # Note that result.perror contains dx and result.covar contains cov.
        # See mpfit.py:781 for status codes. We are returning -1 for user abort.
        if result.status > 0 or result.status == -1:
            x = result.params
            # TODO: mpfit sometimes returns root chisq and sometimes chisq
            # Use nllf() as the resulting cost function for consistency with other fitters.
            # Should be able to use fnorm/2 but it is broken in mpfit.
            if not (self.problem.getp() == x).all():
                self.problem.setp(x)
            fx = self.problem.nllf()
        else:
            x, fx = None, None

        return x, fx

    def _residuals(self, p, fjac=None):
        # # Returning -1 here stops immediately rather than completing the step. This is
        # # different from the other fitters, which wait for the step to complete.
        # if self._stopping():
        #     return -1, None

        # Evaluating with new data point so update
        self.problem.setp(p)

        # Tally costs for residuals and broken constraints. Treat prior probabilities on
        # the parameters and broken constraints as additional measurements. The result
        # should be that fnorm = sumsq residuals = 2 * nllf
        extra_cost, failing_constraints = self.problem.constraints_nllf()
        residuals = np.hstack((self.problem.residuals().flat, self.problem.parameter_residuals(), np.sqrt(extra_cost)))
        # print("sumsq resid", np.sum(residuals**2), "nllf", self.problem.nllf()*2)

        # # Spread the cost over the residuals.  Since we are smoothly increasing
        # # residuals as we leave the boundary, this should push us back into the
        # # boundary (within tolerance) during the lm fit.
        # residuals += np.sign(residuals) * (extra_cost / len(residuals))
        return 0, residuals


class LevenbergMarquardtFit(FitBase):
    """
    Levenberg-Marquardt optimizer.
    """

    name = "Levenberg-Marquardt (scipy.leastsq)"
    id = "scipy.leastsq"
    settings = [("steps", 200), ("ftol", 1.5e-8), ("xtol", 1.5e-8)]
    # LM also has
    #    gtol: orthoganality between jacobian columns
    #    epsfcn: numerical derivative step size
    #    factor: initial radius
    #    diag: variable scale factors to bring them near 1

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        from scipy import optimize

        options = _fill_defaults(options, self.settings)
        self._low, self._high = self.problem.bounds()
        x0 = self.problem.getp()
        maxfev = options["steps"] * (len(x0) + 1)
        monitors(step=0, point=x0, value=self.problem.nllf())
        result = optimize.leastsq(
            self._bounded_residuals,
            x0,
            ftol=options["ftol"],
            xtol=options["xtol"],
            maxfev=maxfev,
            epsfcn=1e-8,
            full_output=True,
        )
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
        monitors(step=1, point=x, value=self.problem.nllf())
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
        residuals = np.hstack((self.problem.residuals().flat, self.problem.parameter_residuals()))
        # Tally costs for straying outside the boundaries plus other costs
        constraints_cost, failing_constraints = self.problem.constraints_nllf()
        extra_cost = stray_cost + constraints_cost
        # Spread the cost over the residuals.  Since we are smoothly increasing
        # residuals as we leave the boundary, this should push us back into the
        # boundary (within tolerance) during the lm fit.
        residuals += np.sign(residuals) * (extra_cost / len(residuals))
        return residuals

    def _stray_delta(self, p):
        """calculate how far point is outside the boundary"""
        return np.where(p < self._low, self._low - p, 0) + np.where(p > self._high, self._high - p, 0)

    def cov(self):
        return self._cov


class SnobFit(FitBase):
    name = "SNOBFIT"
    id = "snobfit"
    settings = [("steps", 200)]

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        options = _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from snobfit.snobfit import snobfit

        def update(k, x, fx, improved):
            # TODO: snobfit does have a population...
            monitors(step=k, point=x, value=fx, population_points=[x], population_values=[fx])

        x, fx, _ = snobfit(self.problem, self.problem.getp(), self.problem.bounds(), fglob=0, callback=update)
        return x, fx


class DreamModel:
    """
    DREAM wrapper for fit problems. Implements dream.core.Model protocol.
    """

    labels: List[str]
    bounds: NDArray

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

    def map(self, pop):
        # print "calling mapper",self.mapper
        return -np.array(self.mapper(pop))


class DreamFit(FitBase):
    name = "DREAM"
    id = "dream"
    settings = [
        ("samples", int(1e4)),
        ("burn", 100),
        ("pop", 10),
        ("init", "eps"),
        ("thin", 1),
        ("alpha", 0.0),
        ("outliers", "none"),
        ("trim", False),
        ("steps", 0),  # deprecated: use --samples instead
    ]

    def __init__(self, problem):
        FitBase.__init__(self, problem)
        self.state = None

    def solve(self, monitors: MonitorRunner, mapper=None, **options):
        from .dream import Dream

        options = _fill_defaults(options, self.settings)

        def update(state, pop, logp):
            # Get an early copy of the state
            self.state = monitors.history.fit_state = state
            step = state.generation
            x, fx = state.best()
            monitors(step=step, point=x, value=-fx, population_points=pop, population_values=-logp)
            return True

        population = initpop.generate(self.problem, **options)
        pop_size = population.shape[0]
        draws, steps = int(options["samples"]), options["steps"]
        if steps == 0:
            steps = (draws + pop_size - 1) // pop_size
        monitors.info(f"# burn: {options['burn']} # steps: {steps}, # draws: {pop_size * steps}")
        population = population[None, :, :]
        # print(f"Running dream with {population.shape=} {pop_size=} {steps=}")
        sampler = Dream(
            model=DreamModel(self.problem, mapper),
            population=population,
            draws=pop_size * steps,
            burn=pop_size * options["burn"],
            thinning=options["thin"],
            monitor=update,
            alpha=options["alpha"],
            outlier_test=options["outliers"],
            DE_noise=1e-6,
        )

        self.state = sampler.sample(state=self.state, abort_test=monitors.stopping)
        # print("<<< Dream is done sampling >>>")

        # If "trim" option is enabled, automatically set the portion, otherwise use
        # the default 100% that was set at the start of sampler.sample.
        if options.get("trim", False):
            self.state.portion = self.state.trim_portion()
        # print("trimming", options['trim'], self._trimmed)
        self.state.mark_outliers()
        self.state.keep_best()
        self.state.title = self.problem.name

        # TODO: Add derived/visible/integer variable support to other optimizers.
        # TODO: Serialize derived/visible/integer variable support with fitproblem.
        # TODO: Use parameter expressions for derived vars rather than a function.
        # TODO: Allow fixed parameters as part of the derived variable function.
        fn, labels = getattr(self.problem, "derive_vars", (None, None))
        if fn is not None:
            self.state.set_derived_vars(fn, labels=labels)
        visible_vars = getattr(self.problem, "visible_vars", None)
        if visible_vars is not None:
            self.state.set_visible_vars(visible_vars)
        integer_vars = getattr(self.problem, "integer_vars", None)
        if integer_vars is not None:
            self.state.set_integer_vars(integer_vars)

        x, fx = self.state.best()

        # Check that the last point is the best point
        # points, logp = self.state.sample()
        # assert logp[-1] == fx
        # print(points[-1], x)
        # assert all(points[-1, i] == xi for i, xi in enumerate(x))
        return x, -fx

    def entropy(self, **kw):
        return self.state.entropy(**kw)

    def stderr(self):
        """
        Approximate standard error as 1/2 the 68% interval fo the sample,
        which is a more robust measure than the mean of the sample for
        non-normal distributions.
        """
        from .dream.stats import var_stats

        vstats = var_stats(self.state.draw())
        return np.array([(v.p68[1] - v.p68[0]) / 2 for v in vstats], "d")

    # def cov(self):
    #    # Covariance estimate from final 1000 points
    #    return np.cov(self.state.draw().points[-1000:])

    def load(self, input_path):
        from .dream.state import load_state, path_contains_saved_state

        if path_contains_saved_state(input_path):
            print("loading saved state from %s (this might take awhile) ..." % (input_path,))
            fn, labels = getattr(self.problem, "derive_vars", (None, []))
            self.state = load_state(input_path, report=100, derived_vars=len(labels))
        else:
            # Warn if mc files are not found on --resume path
            warnings.warn("No mcmc found; ignoring --resume=%r" % input_path)

    def save(self, output_path):
        self.state.save(output_path)

    @staticmethod
    def h5load(group: Group) -> MCMCDraw:
        from .dream.state import h5load

        return h5load(group)

    @staticmethod
    def h5dump(group: Group, state: MCMCDraw):
        from .dream.state import h5dump

        h5dump(group, state)

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
        res = errplot.calc_errors_from_state(problem=self.problem, state=self.state)
        if res is not None:
            pylab.figure()
            errplot.show_errors(res)
            pylab.savefig(figfile + "-errors.png", format="png")


class Resampler(FitBase):
    # TODO: why isn't cli.resynth using this?

    def __init__(self, fitter):
        self.fitter = fitter
        raise NotImplementedError()

    def solve(self, **options):
        starts = options.pop("starts", 1)
        restart = options.pop("restart", False)
        x, fx = self.fitter.solve(**options)
        points = _resampler(self.fitter, x, samples=starts, restart=restart, **options)
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
            # print "[chisq=%.2f]" % self.problem.chisq(nllf=fx))
    except KeyboardInterrupt:
        # On keyboard interrupt we can declare that we are finished sampling
        # without it being an error condition, so let this exception pass.
        pass
    finally:
        # Restore the state of the problem
        fitter.problem.restore_data()
        fitter.problem.setp(xinit)
    return points


class FitDriver(object):
    def __init__(self, fitclass=None, problem=None, monitors=None, abort_test=None, mapper=None, time=0.0, **options):
        self.fitclass = fitclass
        self.problem = problem
        self.options = options
        self.monitors = [ConsoleMonitor(problem)] if monitors is None else monitors
        self.max_time = time * 3600  # Timeout converted from hours to seconds.
        self.abort_test = abort_test
        self.mapper = mapper if mapper else lambda p: list(map(problem.nllf, p))
        self.fitter = None
        self.result = None
        self._reset_cache()

    def _reset_cache(self):
        """
        Cached values. Deleted by fit() to force recomputation when the new fit is complete.
        """
        self._cov = None
        self._stderr = None
        self._stderr_from_cov = None
        self._chisq = None

    def fit(self, resume=None, fit_state=None):
        """
        Providing *fit_state* allows the fit to resume from a previous state. If None
        then the fit will be started from a clean state.

        The *fitclass* object should provide static methods for *h5dump/h5load* for
        saving and loading the internal state of the fitter to a specific group in an
        hdf5 file. The result of *h5load(group)* can be passed as *fit_state* to resume a
        fit with whatever new options are provided. It is up to the fitter to decide
        how to interpret this. The state can be retrieved from state=driver.fitter.state
        at the end of the fit and saved using *h5dump(group, state)*.

        *resume* (= resume_path / problem.name) is used by the pre-1.0 command line
        interface to provide the base path for the fit state files. For dream this can
        be replaced by the following::

            fn, labels = getattr(problem, "derive_vars", (None, []))
            fit_state = load_state(input_path, report=100, derived_vars=len(labels))
        """
        self._reset_cache()

        # Awkward interface for dump/load state. The structure of the state depends on
        # the fit method, so we need to delegate dump/load to the Fitter class. However,
        # the fitter is not instantiated outside of the fit method, so dump/load must
        # be static methods on fitclass, with load called before the fit and dump after.
        # A further complication is checkpointing, which requires that the state be
        # available and up to date when the checkpoint is requested.
        fitter = self.fitclass(self.problem)
        fitter.state = fit_state
        if resume and hasattr(fitter, "load"):
            fitter.load(resume)
        starts = self.options.get("starts", 1)
        if starts > 1:
            fitter = MultiStart(fitter)
        # TODO: better interface for history management?
        # Keep a handle to the fitter which has state and monitor_runner which has history
        self.fitter = fitter
        self.monitor_runner = MonitorRunner(
            problem=self.problem, monitors=self.monitors, abort_test=self.abort_test, max_time=self.max_time
        )
        x, fx = fitter.solve(
            monitors=self.monitor_runner, abort_test=self.abort_test, mapper=self.mapper, **self.options
        )
        if x is not None:
            self.problem.setp(x)
        self.wall_time = self.monitor_runner.history.time[0]
        self.result = x, fx
        self.monitor_runner.final(point=x, value=fx)
        return x, fx

    def clip(self):
        """
        Force parameters within bounds so constraints are finite.

        The problem is updated with the new parameter values.

        Returns a list of parameter names that were clipped.
        """
        labels = self.problem.labels()
        values = self.problem.getp()
        bounds = self.problem.bounds()
        new_values = np.clip(values, bounds[0], bounds[1])
        clipped = [name for name, old, new in zip(labels, values, new_values) if old != new]
        self.problem.setp(new_values)
        return clipped

    def entropy(self, method=None):
        if hasattr(self.fitter, "entropy"):
            return self.fitter.entropy(method=method)
        else:
            from .dream import entropy

            return entropy.cov_entropy(self.cov()), 0

    def chisq(self):
        if self._chisq is None:
            self._chisq = self.problem.chisq()
        return self._chisq

    def cov(self):
        r"""
        Return an estimate of the covariance of the fit.

        Depending on the fitter and the problem, this may be computed from
        existing evaluations within the fitter, or from numerical
        differentiation around the minimum.

        If the problem uses $\chi^2/2$ as its nllf, then the covariance
        is derived from the Jacobian::

            x = fit.problem.getp()
            J = bumps.lsqerror.jacobian(fit.problem, x)
            cov = bumps.lsqerror.jacobian_cov(J)

        Otherwise, the numerical differentiation will use the Hessian
        estimated from nllf::

            x = fit.problem.getp()
            H = bumps.lsqerror.hessian(fit.problem, x)
            cov = bumps.lsqerror.hessian_cov(H)
        """
        # Note: if fit() has not been run then self.fitter is None and in
        # particular, self.fitter will not have a covariance matrix.  In
        # this case, the code will fall through to computing the covariance
        # matrix directly from the problem.  It will use the initial value
        # stored in the problem parameters because results will also be None.
        if self._cov is None and hasattr(self.fitter, "cov"):
            self._cov = self.fitter.cov()
            # print("fitter cov", self._cov)
        if self._cov is None:
            # Use Jacobian if residuals are available because it is faster
            # to compute.  Otherwise punt and use Hessian.  The has_residuals
            # attribute should be True if present.  It may be false if
            # the problem defines a residuals method but doesn't really
            # have residuals (e.g. to allow levenberg-marquardt to run even
            # though it is not fitting a sum-square problem).
            if hasattr(self.problem, "has_residuals"):
                has_residuals = self.problem.has_residuals
            else:
                has_residuals = hasattr(self.problem, "residuals")
            x = self.problem.getp() if self.result is None else self.result[0]
            if has_residuals:
                J = lsqerror.jacobian(self.problem, x)
                # print("Jacobian", J)
                self._cov = lsqerror.jacobian_cov(J)
            else:
                H = lsqerror.hessian(self.problem, x)
                # print("Hessian", H)
                self._cov = lsqerror.hessian_cov(H)
        return self._cov

    def stderr(self):
        """
        Return an estimate of the standard error of the fit.

        Depending on the fitter and the problem, this may be computed from
        existing evaluations within the fitter, or from numerical
        differentiation around the minimum.
        """
        # Note: if fit() has not been run then self.fitter is None and in
        # particular, self.fitter will not have a stderr method defined so
        # it will compute stderr from covariance.
        if self._stderr is None and hasattr(self.fitter, "stderr"):
            self._stderr = self.fitter.stderr()
        if self._stderr is None:
            # If no stderr from the fitter then compute it from the covariance
            self._stderr = self.stderr_from_cov()
        return self._stderr

    def stderr_from_cov(self):
        """
        Return an estimate of standard error of the fit from covariance matrix.

        Unlike stderr, which uses the estimate from the underlying
        fitter (DREAM uses the MCMC sample for this), *stderr_from_cov*
        estimates the error from the diagonal of the covariance matrix.
        Here, the covariance matrix may have been estimated by the fitter
        instead of the Hessian.
        """
        if self._stderr_from_cov is None:
            self._stderr_from_cov = lsqerror.stderr(self.cov())
        return self._stderr_from_cov

    def show(self):
        if hasattr(self.fitter, "show"):
            self.fitter.show()
        if hasattr(self.problem, "show"):
            self.problem.show()

    # TODO: reenable the "implied variance" calculation
    def _unused_show_err(self):
        """
        Display the error approximation from the numerical derivative.

        Warning: cost grows as the cube of the number of parameters.
        """
        # TODO: need cheaper uncertainty estimate
        # Note: error estimated from hessian diagonal is insufficient.
        err = self.stderr_from_cov()
        # TODO: citation needed
        # The "implied variance" column is obtained by scaling the covariance
        # matrix so that chisq = DOF. Any excess chisq implies increased
        # variance in the measurements, so increased variance in the parameters.
        # This is well defined for linear systems with equal but unknown
        # variance in each measurement, and assumed to be approximately true
        # for nonlinear systems, with the unexplained variance distributed
        # proportionately amongst the measurement uncertainties.
        norm = np.sqrt(self.chisq())
        print("=== Uncertainty from curvature:     name value(unc.)     value(unc./chi)) ===")
        for k, v, dv in zip(self.problem.labels(), self.problem.getp(), err):
            print("%40s %-15s %-15s" % (k, format_uncertainty(v, dv), format_uncertainty(v, dv / norm)))
        print("=" * 75)

    def show_err(self):
        """
        Display the error approximation from the numerical derivative.

        Warning: cost grows as the cube of the number of parameters.
        """
        # TODO: need cheaper uncertainty estimate
        # Note: error estimated from hessian diagonal is insufficient.
        err = self.stderr_from_cov()
        print("=== Uncertainty from curvature:     name   value(unc.) ===")
        for k, v, dv in zip(self.problem.labels(), self.problem.getp(), err):
            print(f"{k:>40s}   {format_uncertainty(v, dv):<15s}")
        print("=" * 58)

    def show_cov(self):
        cov = self.cov()
        maxn = 1000  # max array dims to print
        cov_str = np.array2string(
            cov,
            max_line_width=20 * maxn,
            threshold=maxn * maxn,
            precision=6,  # suppress_small=True,
            separator=", ",
        )
        print("=== Covariance matrix ===")
        print(cov_str)
        print("=========================")

    def show_entropy(self, method=None):
        print("Calculating entropy...")
        S, dS = self.entropy(method=method)
        print("Entropy: %s bits" % format_uncertainty(S, dS))

    def save(self, output_path):
        # print "calling driver save"
        if hasattr(self.fitter, "save"):
            self.fitter.save(output_path)
        if hasattr(self.problem, "save"):
            self.problem.save(output_path)

    def load(self, input_path):
        # print "calling driver save"
        if hasattr(self.fitter, "load"):
            self.fitter.load(input_path)
        if hasattr(self.problem, "load"):
            self.problem.load(input_path)

    def plot(self, output_path, view=None):
        # print "calling fitter.plot"
        if hasattr(self.problem, "plot"):
            self.problem.plot(figfile=output_path, view=view)
        if hasattr(self.fitter, "plot"):
            self.fitter.plot(output_path=output_path)

    def _save_fit_cov(self, output_path):
        model = getattr(self.problem, "name", self.problem.__class__.__name__)
        fitter = self.fitclass.id
        cov = self.cov()
        err = self.stderr_from_cov()
        chisq = self.chisq()

        state = {
            "model": model,
            "fitter": fitter,
        }


def _fill_defaults(options, settings):
    """
    Returns options dict with missing values filled from settings.
    """
    result = dict(settings)  # settings is a list of (key,value) pairs
    result.update(options)
    return result


FITTERS = []
FIT_AVAILABLE_IDS = []
FIT_ACTIVE_IDS = []


def register(fitter, active=True):
    """
    Register a new fitter with bumps, if it is not already there.

    *active* is False if you don't want it showing up in the GUI selector.
    """
    # Check if already registered.
    if fitter in FITTERS:
        return

    # Check that there is no other fitter of that name
    if fitter.id in FIT_AVAILABLE_IDS:
        raise ValueError("There is already a fitter registered as %r" % fitter.id)

    # Register the fitter.
    FITTERS.append(fitter)
    FIT_AVAILABLE_IDS.append(fitter.id)

    # Make it "active" by listing it in the help menu.
    if active:
        FIT_ACTIVE_IDS.append(fitter.id)


# Register the fitters
register(SimplexFit)
register(DEFit)
register(DreamFit)
register(BFGSFit)
register(MPFit)
register(PTFit, active=False)
# register(PSFit, active=False)
# register(RLFit, active=False)
# register(LevenbergMarquardtFit, active=True)
# register(SnobFit, active=False)

FIT_DEFAULT_ID = SimplexFit.id

assert FIT_DEFAULT_ID in FIT_ACTIVE_IDS
assert all(f in FIT_AVAILABLE_IDS for f in FIT_ACTIVE_IDS)


def fit(
    problem,
    method=FIT_DEFAULT_ID,
    export=None,
    resume=None,
    store=None,
    name=None,
    verbose=False,
    parallel=1,
    **options,
):
    """
    Simplified fit interface.

    Given a fit problem, the name of a fitter and the fitter options,
    it will run the fit and return the best value and standard error of
    the parameters.  If *verbose* is true, then the console monitor will
    be enabled, showing progress through the fit and showing the parameter
    standard error at the end of the fit, otherwise it is completely
    silent.

    Returns a scipy *OptimizeResult* object containing "x" and "dx".  Some
    fitters also include a "state" object. For dream this can be used in
    the call *bumps.dream.views.plot_all(result.state)* to generate the
    uncertainty plots. Note: success=True and status=0 for now since the
    stopping condition is not yet available from the fitters.

    If *resume=result* is provided, then attempt to resume the fit from the
    previous result.

    If *export=path* is provided, generate the standard plots and export files
    to the specified directory. This uses *name* as the basename for the output
    files, or *problem.name* if name is not provided. Name defaults to "problem".

    If *parallel=n* is provided, then run on *n* separate cpus. By default
    *parallel=1* to run on a single cpu. For slow functions set *parallel=0*
    to run on all cpus. You want to run on a single cpu if your function is
    already parallel (for example using multiprocessing or using gpu code),
    or if your function is so fast that the overhead of transfering data is
    higher than cost of *n* function calls.
    """
    from pathlib import Path
    from scipy.optimize import OptimizeResult
    from .serialize import serialize
    from .mapper import MPMapper, SerialMapper
    from .webview.server.fit_thread import ConvergenceMonitor
    from .webview.server.state_hdf5_backed import State, FitResult, ProblemState

    # verbose = True
    # Options parser stores --fit=fitter in fit_options["fit"] rather than fit_options["method"]
    if "fit" in options:
        method = options.pop("fit")
    if method not in FIT_AVAILABLE_IDS:
        raise ValueError("unknown fit method %r not one of %s" % (method, ", ".join(sorted(FIT_ACTIVE_IDS))))
    for fitclass in FITTERS:
        if fitclass.id == method:
            break

    Mapper = MPMapper if parallel != 1 else SerialMapper
    mapper = Mapper.start_mapper(problem, [], cpus=parallel)
    convergence = ConvergenceMonitor(problem)
    monitors = [convergence]
    if verbose:
        monitors.append(ConsoleMonitor(problem))
    driver = FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, mapper=mapper, **options)
    driver.clip()  # make sure fit starts within domain
    # x0 = problem.getp()
    if resume is not None:
        problem.setp(resume.x)
    x, fx = driver.fit(fit_state=None if resume is None else resume.state)
    problem.setp(x)
    if verbose:
        print("final chisq", problem.chisq_str())
        driver.show_err()

    # TODO: can we put this in a function in state_hdf5_backed?
    if store is not None:
        # TODO: strip non-options such as mapper from fit options
        store = Path(store)
        state = State()
        if store.exists():
            state.read_session_file(store)
        fitting = FitResult(
            method=method, options=options, convergence=np.array(convergence.quantiles), fit_state=driver.fitter.state
        )
        try:
            serialize(problem)
            serializer = "dataclass"
        except Exception as exc:
            # import traceback; traceback.print_exc()
            if verbose:
                print("Problem stored using dill. It may not load in newer python versions.")
                print(f"error: {exc}")
            serializer = "dill"
        state.problem = ProblemState(fitProblem=problem, serializer=serializer)
        state.fitting = fitting
        state.save_to_history(label="fit")
        state.write_session_file(store)

    if export is not None:
        from .webview.server.api import _export_results

        _export_results(Path(export), problem, driver.fitter.state, serializer="dill", name=name)

    result = OptimizeResult(
        x=x,
        dx=driver.stderr(),
        fun=fx,
        # TODO: need better success/status/message handling
        success=True,
        status=0,
        message="successful termination",
        nit=driver.monitor_runner.history.step[0],  # number of iterations
        # nfev=0, # number of function evaluations
        # njev, nhev # jacobian and hessian evaluations
        # maxcv=0, # max constraint violation
    )
    # Non-standard result
    result.state = driver.fitter.state
    return result


def test_fitters():
    """
    Run the fit tests to make sure they work.
    """
    from .curve import Curve
    from .fitproblem import FitProblem

    x = [1, 2, 3, 4, 5, 6]
    y = [2.1, 4.0, 6.3, 8.03, 9.6, 11.9]
    dy = [0.05, 0.05, 0.2, 0.05, 0.2, 0.2]

    def line(x, m, b=0):
        return m * x + b

    M = Curve(line, x, y, dy, m=2, b=2)
    M.m.range(0, 4)
    M.b.range(-5, 5)

    problem = FitProblem(M)

    # Set the tolerance for the tests (relative)
    fit_value_tol = 1e-3
    fit_error_tol = 1e-3
    expected_value = [1.106e-1, 1.970]
    expected_error = [5.799e-2, 2.055e-2]

    store = None
    export = None
    verbose = False
    parallel = 1  # Serial fit
    # TODO: test store and export as normal tests rather than one-off tests
    # store = "/tmp/teststore.h5"
    # export = "/tmp/testexport"
    # verbose = True
    # parallel = 0 # Parallel fit
    for fitter_name in FIT_ACTIVE_IDS:
        # print(f"Running {fitter_name}")
        result = fit(
            problem,
            method=fitter_name,
            verbose=verbose,
            store=store,
            export=export,
            parallel=parallel,
            name=fitter_name,
        )
        assert np.allclose(result.x, expected_value, rtol=fit_value_tol)
        if fitter_name != "dream":
            # dream error bars vary too much to test
            assert np.allclose(result.dx, expected_error, rtol=fit_error_tol)
