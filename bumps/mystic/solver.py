"""
Generic minimizers

The general optimization algorithm is as follows::

    fit = Minimizer(problem=Problem(),
                 strategy=Strategy(),
                 monitors=[],
                 success=Df(1e-5) & Dx(1e-5),
                 failure=Calls(10000) | Steps(100))
    population = fit.start()
    while True:
        result = list(map(fit.problem, population))
        fit.update(population, result)
        if fit.isdone(): break
        population = fit.step()

    if fit.successful:
        print "Converged"
    else:
        print "Stopped with",", ".join(str(c) for c in fit.failure_cond)

Variations are possible, such as the multistep process where the
program first generates an initial population, then each time it
is run generates an updated population based on the results of
submitting the previous population to a batch queue.

History traces
==============

Stopping conditions use history traces to evaluate whether the program
should continue.  When adding new stopping conditions, the programmer
must define a config_history method to indicate which values are needed.
For example::

    def config_history(self, history):
        history.requires(points=3)

When checking convergence, the programmer must still check that enough
history is available for the test. For example,
::

    def __call__(self, history):
        from numpy.linalg import norm
        if len(history.value) < 3: return False
        return norm(history.value[0] - history.value[2]) < 0.001

The optimizer will make sure that config_history is called for each
condition before the fit starts::

    for t in success.primitives()|failure.primitives():
        t.config_history()

Each optimizer can control which values it wants to monitor.  For
consistency and ease of use, monitor names should be chosen from
the standard names below.  As new optimizers are created, the
list of standard names may expand.

Fixed properties of history::

    *ndim* (int)
         problem dimension
    *lower_bound*, *upper_bound* ([float])
         bounds constraints

Accumulated properties::

    *step* (int)
         iteration number
    *calls* (int)
        cumulative number of function evaluations
    *time* (seconds)
        cumulative wall clock time
    *cpu_time* (seconds)
        cumulative CPU time
    *value* (float)
        best function value
    *point* (vector)
        parameter values for the best function value
    *gradient* (vector)
        del f: gradient at best point (if available)
    *hessian* (matrix)
        del**2 F: Hessian at best point (if available)
    *population_values* (vector)
        function values of the current population (if available)
    *population_points* (matrix)
        parameter values for the current population (if available)
"""

# TODO: Ordered fits
#
# Want a list of strategies and a parameter subset associated with each
# strategy.  The fit is nested, with the outermost parameters are set
# to a particular value, then the inner parameters are optimized for
# those values before an alternative set of outer parameters is considered.
#
# Will probably want the inner strategy to be some sort of basin hopping
# method with the new value jumping around based on the best value for
# those parameters.  This is complicated because the first fit should be
# done using a more global method.
#
# In the context of simultaneous fitting, it would be natural to partition
# the parameters so that the dependent parameters of the slowest models
# are varied last.  This can be done automatically if each model has
# a cost estimate associated with it.
#
# TODO: Surrogate models
#
# Want to use surrogate models for the expensive models when far from the
# minimum, only calculating the real model when near the minimum.

import os
import time

import numpy as np


def default_mapper(f, x):
    return list(map(f, x))


def cpu_time():
    """Current cpu time for this process"""
    user_time, sys_time, _, _, _ = os.times()
    return user_time + sys_time


class Minimizer:
    """
    Perform a minimization.
    """

    def __init__(self, problem=None, strategy=None, monitors=[], history=None, success=None, failure=None):
        self.problem = problem
        self.strategy = strategy
        # Ask strategy to fill in the default termination conditions
        # in case the user doesn't supply them.
        defsucc, deffail = strategy.default_termination_conditions(problem)
        self.success = success if success is not None else defsucc
        self.failure = failure if failure is not None else deffail
        self.monitors = monitors
        self.history = history
        self.reset()

    def minimize(self, mapper=default_mapper, abort_test=None, resume=False):
        """
        Run the solver to completion, returning the best point.

        Note: only used stand-alone, not within fit service
        """
        self.time = time.time()
        self.remote_time = -cpu_time()
        population = self.step() if resume else self.start()
        try:
            while True:
                result = mapper(self.problem, population)
                # print "map result",result
                self.update(population, result)
                # print self.history.step, self.history.value
                if self.isdone():
                    break  # STOPHERE combine
                if abort_test is not None and abort_test():
                    break
                population = self.step()
        except KeyboardInterrupt:
            pass
        return self.history.point[0]

    __call__ = minimize

    def reset(self):
        """
        Clear the solver history.
        """
        self.history.clear()
        self.history.provides(
            calls=1, time=1, cpu_time=1, step=1, point=1, value=1, population_points=0, population_values=0
        )
        for c in self.success.primitives() | self.failure.primitives():
            c.config_history(self.history)
        for m in self.monitors:
            m.config_history(self.history)
        self.strategy.config_history(self.history)

    def start(self):
        """
        Start the optimization but generating an initial population.
        """
        # Reset the timers so we know how long the fit takes.
        # We are cheating by initializing remote_time to -cpu_time, then
        # later adding cpu_time back to remote_time to get the total cost
        # of local and remote computation.

        if len(self.problem.getp()) == 0:
            raise ValueError("Problem has no fittable parameters")

        return self.strategy.start(self.problem)

    def update(self, points, values):
        """
        Collect statistics on time and resources
        """
        if hasattr(values, "__cpu_time__"):
            self.remote_time += values.__cpu_time__

        # Locate the best member of the population
        values = np.asarray(values)
        # print("values",values,file=sys.stderr)

        # Update the history
        self.history.update(
            time=time.time() - self.time,
            cpu_time=cpu_time() + self.remote_time,
            population_points=points,
            population_values=values,
        )
        self.history.accumulate(step=1, calls=len(points))

        self.strategy.update(self.history)

        minidx = np.argmin(values)
        self.history.update(
            point=points[minidx],
            value=values[minidx],
        )

        # Tell all the monitors that the history has been updated
        for m in self.monitors:
            m(self.history)

    def step(self):
        """
        Generate the next population to evaluate.
        """
        return self.strategy.step(self.history)

    def isdone(self):
        """
        Check if the fit has converged according to the criteria proposed
        by the user and/or the optimizer.

        Returns True if either the fit converged or is forced to stop for
        some other reason.

        Sets the following attributes::

            *successful* (boolean)
                True if the fit converged
            *success_cond* ([Condition])
                Reasons for success or lack of success
            *failed* (boolean)
                True if the fit should stop
            *failure_cond* ([Condition])
                Reasons for failure or lack of failure

        Note that success and failure can occur at the same time if
        for example the convergence criteria are met but the resource
        limits were exceeded.
        """
        self.successful, self.success_cond = self.success.status(self.history)
        self.failed, self.failure_cond = self.failure.status(self.history)
        return self.successful or self.failed

    def termination_condition(self):
        if self.successful:
            return "succeeded with " + ", ".join(str(c) for c in self.success_cond)
        else:
            return (
                "failed with "
                + ", ".join(str(c) for c in self.failure_cond)
                + " and "
                + ", ".join(str(c) for c in self.success_cond)
            )


class Strategy:
    """
    Optimization strategy to use.

    The doc string for the strategy will be used in the construction of
    the doc strings for the simple optimizer interfaces, with a description
    of the standard optimizer options at the end.   The following template
    works well::

        Name of optimization strategy

        Brief description of the strategy

        Optimizer parameters::

            *argument* is the description of the argument

        Additional details about the solver.

    The __init__ arguments for the strategy will be passed in through
    the simple optimizer interface.  Be sure to make them all keyword
    arguments.
    """

    def config_history(self, history):
        """
        Specify which information needs to be preserved in history.

        For example, parallel tempering needs to save its level values::

            history.requires(levels=2)
        """
        pass

    def update(self, history):
        """
        Update history with optimizer specific state information.

        Note: standard history items (step, calls, value, point,
        population_values, population_points, time, cpu_time) are
        already recorded.  Additional items should be recorded
        directly in the trace.  For example::

            history.levels.put([1,2,3])
        """
        pass

    def start(self, problem):
        """
        Generate the initial population.

        Returns a matrix *P* of points to be evaluated.
        """
        raise NotImplementedError

    def step(self, history):
        """
        Generate the next population.

        Returns a matrix *P* of points to be evaluated.

        *history* contains the previous history of the computation,
        including any fields placed by :meth:`update`.
        """
        raise NotImplementedError
