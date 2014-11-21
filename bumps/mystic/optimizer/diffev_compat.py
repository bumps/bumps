from __future__ import print_function

from ..solver import Minimizer
from ...parameter import Parameter
from .. import stop

from .de import DifferentialEvolution, best1

class Function(object):
    def __init__(self, f, ndim=None, po=None, bounds=None, args=()):
        if bounds is not None and po is not None:
            self.parameters = [Parameter(value=v,bounds=b)
                               for v,b in zip(po,bounds)]
        elif bounds is not None:
            self.parameters = [Parameter(b) for b in bounds]
        elif po is not None:
            self.parameters = [Parameter(v) for v in po]
        elif ndim is not None:
            self.parameters = [Parameter() for _ in range(ndim)]
        else:
            raise TypeError("Need ndim, po or bounds to get problem dimension")
        if ((ndim is not None and ndim != len(self.parameters))
            or (po is not None and len(po) != len(self.parameters))
            or (bounds is not None and len(bounds) != len(self.parameters))):
            raise ValueError("Inconsistent dimensions for ndim, po and bounds")
        if po is None:
            po = [p.start_value() for p in self.parameters]

        self.f = f
        self.bounds = bounds
        self.po = po
        self.args = args

    def guess(self):
        if self.po is not None:
            return self.po
        else:
            return [p.start_value() for p in self.parameters]
    def __call__(self, p):
        return self.f(p, *self.args)


def diffev(func, x0=None, npop=10, args=(), bounds=None,
           ftol=5e-3, gtol=None,
           maxiter=None, maxfun=None, CR=0.9, F=0.8,
           full_output=0, disp=1, retall=0, callback=None):
    """\
Minimize a function using differential evolution.

Inputs::

    *func* -- the callable function to be minimized.
    *x0* -- the initial guess, or none for entirely random population.
    *npop* -- points per dimension.  Population size is npop*nparameters.

Additional Inputs::

    *args* -- extra arguments for func.
    *bounds* -- list of bounds (min,max), one pair for each parameter.
    *ftol* -- acceptable relative error in func(xopt) for convergence.
    *gtol* -- maximum number of iterations to run without improvement.
    *maxiter* -- the maximum number of iterations to perform.
    *maxfun* -- the maximum number of function evaluations.
    *CR* -- the probability of cross-parameter mutations
    *F* -- multiplier for impact of mutations on trial solution.
    *full_output* -- non-zero if fval and warnflag outputs are desired.
    *disp* -- non-zero to print convergence messages.
    *retall* -- non-zero to return list of solutions at each iteration.
    *callback* -- function(xk) to call after each iteration.

Returns:: (xopt, {fopt, iter, funcalls, status}, [allvecs])

    *xopt* -- the best point
    *fopt* -- value of function at the best point: fopt = func(xopt)
    *iter* -- number of iterations
    *funcalls* -- number of function calls
    *status* -- termination status
        0 : Function converged within tolerance.
        1 : Maximum number of function evaluations.
        2 : Maximum number of iterations.
    *allvecs* -- a list of solutions at each iteration
"""

    problem = Function(f=func, args=(), po=x0, bounds=None)
    strategy = DifferentialEvolution(CR=CR, F=F, npop=npop, mutate=best1)
    ndim = len(x0)

    # Determine success and failure conditions
    if gtol: # Improvement window specified
        # look for ftol improvement over gtol generations
        success = stop.Df(tol=ftol,n=gtol,scaled=False)
    else:
        # look for f < ftol.
        success = stop.Cf(tol=ftol,scaled=False)
    if maxiter is None: maxiter = ndim*100
    if maxfun is None: maxfun = npop*maxiter
    failure = stop.Calls(maxfun)|stop.Steps(maxiter)

    monitors = []
    #if callback is not None:
    #    monitors.append(CallbackMonitor(callback))
    #if retall:
    #    population_monitor = StepMonitor('population_values')
    #    monitors.append(population_monitor)
    minimize = Minimizer(problem=problem, strategy=strategy,
                         monitors=monitors, success=success, failure=failure)

    # Preserve history for output (must be after solver.reset())
    hist = minimize.history
    hist.requires(point=1,value=1,calls=1,step=1)


    # Run the solver
    minimize()


    # Generate return values (must be after call to minimize)
    if hist.calls[0] > maxfun:
        status = 1
        msg = "Warning: Maximum number of function evaluations exceeded."
    elif hist.step[0] > maxiter:
        status = 2
        msg = "Warning: Maximum number of iterations exceeded."
    else:
        status = 0
        msg = """Optimization terminated successfully.
    Best point: %s
    Best value: %g
    Iterations: %d
    Function evaluations: %d"""%(hist.point[0],hist.value[0],
                                 hist.step[0],hist.calls[0])
    if disp: print(msg)
    if not full_output and not retall:
        ret = hist.point[0]
    elif full_output:
        ret = (hist.point[0], hist.value[0],
            hist.step[0], hist.calls[0], status)
        if retall:
            raise NotImplementedError("retall not implemented")
            #ret += (population_monitor.population_points,)
    else:
        raise NotImplementedError("retall not implemented")
        #ret = hist.point[0], population_monitor.population_points
    return ret
