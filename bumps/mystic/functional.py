"""
Defines a functional interface to the optimizers.
"""
import inspect
from copy import copy
from .solver import Minimizer
from .problem import Problem
from . import stop

class Result:
    """Minimization result.

    Attributes::

        *problem* is the function that was minimized
        *steps* is the number iterations of the minimizer
        *calls* is the number of function calls
        *point* is the best point found
        *value* is the value at the best point
        *converged* is true if the fit converged
        *message* gives the reason the fit terminated
        *traces* contains whichever traces the user wished to preserve
        *time* is the amount of time since the fit started
        *cpu_time* is the amount of CPU time used by the fit
    """
    def __init__(self, **kw):
        self.__dict__ = kw
    def __str__(self):
        msg = """\
Minization %(status)s after %(steps)d iterations (%(calls)d function calls)
with a value of %(value)g at %(point)s.\
"""%dict(status="converged" if self.converged else "did not converge",
         steps=self.steps, calls=self.calls,
         value=self.value, points=self.point)
        return msg

def result(minimizer):
    """
    Packages the currrent state as a fit result for return from the
    simplified fitting interface.
    """
    return Result(problem=minimizer.problem,
                  steps=minimizer.history.step[0],
                  calls=minimizer.history.calls[0],
                  point=minimizer.history.point[0],
                  value=minimizer.history.value[0],
                  converged=minimizer.successful,
                  message=str(minimizer.success_cond if minimizer.successful
                              else minimizer.failure_cond),
                  time=minimizer.history.time[0],
                  cpu_time = minimizer.history.cpu_time[0],
                  )


standard_doc="""

Problem definition::

    *f* is the callable function to be minimized [f(p) -> v]
    *x0* is the initial value of the function [vector n]
    *bounds* are the low and high values for each argument [array n x 2]
    *ndim* is the number of dimension [integer]

Note that if the function takes arguments beyond a simple vector then
you can provide use "lambda p: f(p, extra_args)" instead.  If *f* is
a :class:`mystic.Problem` argument which knows about the bounds, or if
there are no bounds, then *bounds* need not be specified.  *ndim* is
only required if there is no *x0* or *bounds* parameter.

Termination conditions::

    *success* is the desired convergence criteria
    *failure* is the abort criteria

Success and failure are expressions composed from :module:`mystic.stop`
stopping conditions.  Default values are provided for each strategy, but
these can be replaced with your own conditions.  For example, the following
stops after 10000 calls, 100 iterations, when the target value is less
than 5+1e-5 or when the optimizer doesn't improve by a factor of 1e-5
for 10 iterations.

    from mystic import stop
    failure=stop.Calls(10000)|stop.Steps(100)
    success=stop.Cf(tol=1e-5, value=5)|stop.Df(1e-5,n=10)


Progress monitoring::

    *verbose* is True if fit progress should be reported throughout the fit
    *callback* is a function monitor(history) to call each iteration

Returns::

    :class:`Result`

Result contains information about the fit such as the final value and the
number of iterations used.
"""

def minimizer_function(strategy=None, **kw):
    """
    Returns a callable optimizer that uses *strategy* for optimization.

    *strategy* should be a subclass of :class:`Strategy`.

    kw arguments can specify defaults to the optimizer such as the
    default success and failure conditions.
    """
    # Default arguments for minimizer
    problem_args = dict(f=None, x0=None, bounds=None)
    stop_args = dict(success=stop.Df(10,1e-5), failure=stop.Steps(100))

    # Get the keyword arguments from the strategy; note that the first
    # argument is self.
    args, varargs, varkw, defaults = inspect.getargspec(strategy.__init__)
    if len(args) != len(defaults)+1:
        raise TypeError("Strategy init has positional arguments")
    if any([(k in problem_args) for k in args[1:]]):
        raise TypeError("Strategy arguments shadow minimizer arguments")
    strategy_args = dict(zip(args[1:],defaults))

    ## Add strategy arguments to the list of optimizer arguments
    #optimizer_args.update(strategy_args)
    #
    ## Override defaults as specified by wrapper
    #if not all([(k in optimizer_args) for k in kw.keys()]):
    #    raise TypeError("Creating minimizer with incorrect default args")
    #optimizer_args.update(kw)

    def wrapper(**kw):
        def _args(t):
            t = copy(t)
            t.update(dict(k,kw[k]) for k in t.keys() if k in kw)
            return t
        s=strategy(**_args(strategy_args))
        p=Problem(**_args(problem_args))
        m=[]
        success = kw.get('success',stop_args['success'])
        failure = kw.get('failure',stop_args['failure'])

        minimize = Minimizer(problem=p, strategy=s, monitors=m,
                             success=success, failure=failure)

        minimize.history.requires(steps=1, calls=1, point=1, value=1,
                                  time=1, cpu_time=1)

        # Run the solver
        minimize()

        return result(minimize)
    wrapper.__doc__ = strategy.__doc__ + standard_doc


    return wrapper
