# This program is in the public domain
# Author: Paul Kienzle
"""
Termination conditions for solvers.

In order to decide when to stop fitting, the user needs to specify
stop conditions based on the optimizer history.  The test can use
the most recent value, the last n values or the entire computation.
See :mod:`monitor` for details.

Conditions can be composed, creating complicated criterion for
termination.  We may want to stop when we have found a minimum
and know its location.  We may be content just knowing the
minimum, and not worrying about the uncertainty in its location.
For some searches we want to be sure we that we are examining a
broad range of search space.  Here are some examples::

    import mystic.termination as stop

    # Stop when we know the location of the minimum; fail if run too long
    success = stop.Dx(0.001) & stop.Df(5)
    failure = stop.Steps(100)


    # Stop when we know the value of the minimum; fail if run too long
    success = stop.Dx(0.001) | stop.Df(5)
    failure = stop.Steps(100)

    # GA may want to run for a while, but only with a diverse population
    success = stop.Df(0.001,n=5) & stop.Steps(15)
    failure = stop.Steps(100) | stop.Dx(0.001)

When testing a conditional expression, a list of all conditions which
match is returned, or [] if no conditions match.


Predefined Conditions
---------------------

Dx: difference in x for step size test

    || (x_k - x_{k-1})/scale || < tolerance

Df: difference in f for improvement rate test

    | (f_k - f_{k-1})/scale | < tolerance

Rx: range in x for population distribution test

    max || (y - <y>)/ scale || < tolerance
    for y in population

Rf: range in f for value distribution test

    (max f(y) - min f(y))/scale < tolerance
    for y in population

Cx: constant x for target point

    ||(x_k - Z)/scale|| < tolerance

Cf: constant f for target value

    |f_k - A|/scale < tolerance

Steps: specific number of iterations

    k >= steps

Calls: specific number of function calls

    n >= calls

Time: wall clock time

    t_k >= time

CPU: CPU time

    t(CPU)_k >= time

Worse: fit is diverging

    (f_k - f_{k-1})/scale < -tolerance

Grad: fit is flat

    || del f_k || < tolerance

Feasible: value is in the feasible region ** Not implemented **

    f_k satisfies soft constraints

Invalid: values are not well defined ** Not implemented **

    isinf(y) or isinf(f(y)) or isnan(y) or isnan(f(y))
    for y in population


Distances and scaling
=====================

The following distance functions are predefined:

    norm_p(p): (sum |x_i|^p )^(1/p)  (generalized p-norm)
    norm_1:    sum |x_i|             (Manahattan distance)
    norm_2:    sqrt sum |x_i|^2      (Euclidian distance)
    norm_inf:  max |x_i|             (Chebychev distance)
    norm_min:  min |x_i|             (not a true norm)


The predefined scale factors in essence test for
percentage changes rather than absolute changes.
"""

import math

import numpy as np
from numpy import inf, isinf

from .condition import Condition


# ==== Norms ====
def norm_1(x):
    """1-norm: sum(|x_i|)"""
    return np.sum(abs(x))


def norm_2(x):
    """2-norm: sqrt(sum(|x_i|^2))"""
    return math.sqrt(np.sum(abs(x) ** 2))


def norm_inf(x):
    """inf-norm: max(|x_i|)"""
    return max(abs(x))


def norm_min(x):
    """min-norm: min(|x_i|); this is not a true norm"""
    return min(abs(x))


def norm_p(p):
    """p-norm: sum(|x_i|^p)^(1/p)"""
    if isinf(p):
        if p < 0:
            return norm_min
        else:
            return norm_inf
    elif p == 1:
        return norm_1
    elif p == 2:
        return norm_2
    else:
        return lambda x: np.sum(abs(x) ** p) ** (1 / p)


# ==== Conditions ====
class Dx(Condition):
    """
    Improvement in x.

    This condition measures the improvement over the last n iterations
    in terms of how much the value of x has changed::

        norm((x[k]- x[k-n])/scale) < tol

    where x[k] is the best parameter set for iteration step k.

    The scale factor to use if scaled is upper bound - lower bound
    if the parameter is bounded, or 1/2 (|x[k]| + |x[k-n]|)
    if the parameter is unbounded, with protection against a scale
    factor of zero.

    Parameters::

        *tol* (float = 0.001)
            tolerance to test against
        *norm* ( f(vector): float  =  norm_2)
            norm to use to measure the size of x.  Predefined norms
            include norm_1, norm_2, norm_info, norm_min and norm_p(p)
        *n* (int = 1)
            number of steps back in history to compare
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0.001, norm=norm_2, n=1, scaled=True):
        self.tol = tol
        self.norm = norm
        self.n = n
        self.scaled = scaled

    def _scaled_condition(self, history):
        x1, x2 = history.point[0], history.point[self.n]
        scale = history.upper_bound - history.lower_bound
        scale[isinf(scale)] = ((abs(x1) + abs(x2)) / 2)[isinf(scale)]
        scale[scale == 0] = 1
        return self.norm((x2 - x1) / scale)

    def _raw_condition(self, history):
        x1, x2 = history.point[0], history.point[self.n]
        return self.norm(x2 - x1)

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        if self.tol > 0:
            history.requires(point=self.n + 1)

    def _subcall(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.point) < self.n + 1:
            return False  # Cannot succeed until at least n generations
        elif self.scaled:
            return self._scaled_condition(history)
        else:
            return self._raw_condition(history)

    def __call__(self, history):
        return self._subcall(history) < self.tol

    def completeness(self, history):
        return self._subcall(history) / self.tol

    def __str__(self):
        if self.scaled:
            return "||(x[k] - x[k-%d])/range|| < %g" % (self.n, self.tol)
        else:
            return "||x[k] - x[k-%d]|| < %g" % (self.n, self.tol)


class Df(Condition):
    """
    Improvement in F(x)

    This condition measures the improvement over the last n iterations
    in terms of how much the value of the function has changed::

        | (F[k] - F[k-1])/scale | < tol

    where F[k] is the value for the best parameter set for iteration step k.

    The scale factor to use is 1/2 (|F(k)| + |F(k-n)|) with protection
    against zero.

    Parameters::

        *tol* (float = 0.001)
            tolerance to test against
        *n* (int = 1)
            number of steps back in history to compare
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0.001, n=1, scaled=True):
        self.tol = tol
        self.n = n
        self.scaled = scaled

    def _scaled_condition(self, history):
        f1, f2 = history.value[0], history.value[self.n]
        scale = (abs(f1) + abs(f2)) / 2
        if scale == 0:
            scale = 1
        # print "Df",f1,f2,abs(float(f2-f1)/scale),self.tol
        return abs(float(f2 - f1) / scale)

    def _raw_condition(self, history):
        f1, f2 = history.value[0], history.value[self.n]
        return abs(f2 - f1)

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        if self.tol > 0:
            history.requires(value=self.n + 1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.value) < self.n + 1:
            return False  # Cannot succeed until at least n generations
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "|F[k]-F[k-%d]| / (|F[k]|+|F[k-%d]|)/2 < %g" % (self.n, self.n, self.tol)
        else:
            return "|F[k]-F[k-%d]| < %g" % (self.n, self.tol)


class Worse(Condition):
    """
    Worsening of F(x)

    This condition measures whether the fit is diverging.  You may want
    to use this for non-greedy optimizers which can get worse over time::

        (F[k] - F[k-1])/scale < -tol

    where F[k] is the value for the best parameter set for iteration step k.

    The scale factor to use is 1/2 (|F(k)| + |F(k-n)|) with protection
    against zero.

    Parameters::

        *tol* (float = 0)
            tolerance to test against
        *n* (int = 1)
            number of steps back in history to compare
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0, n=1, scaled=True):
        self.tol = tol
        self.n = n
        self.scaled = scaled

    def _scaled_condition(self, history):
        f1, f2 = history.value[0], history.value[self.n]
        scale = (abs(f1) + abs(f2)) / 2
        if scale == 0:
            scale = 1
        return float(f2 - f1) / scale

    def _raw_condition(self, history):
        f1, f2 = history.value[0], history.value[self.n]
        return f2 - f1

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        history.requires(value=self.n + 1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.value) < self.n + 1:
            return False  # Cannot succeed until at least n generations
        elif self.scaled:
            return self._scaled_condition(history) < -self.tol
        else:
            return self._raw_condition(history) < -self.tol

    def __str__(self):
        if self.scaled:
            return "F[k]-F[k-%d] / (|F[k]|+|F[k-%d]|)/2 < -%g" % (self.n, self.n, self.tol)
        else:
            return "F[k]-F[k-%d] < -%g" % (self.n, self.tol)


class Grad:
    """
    Flat function value

    This condition measures whether the fit surface is flat near the best
    value.  This only works for fits which compute the gradient.

        || del F[k]/scale || < tol

    where F[k] is the value for the best parameter set for iteration step k.

    The scale factor to use is |F(k)| with protection against zero.

    Parameters::

        *tol* (float = 0.001)
            tolerance to test against
        *norm* ( f(vector): float  =  norm_2)
            norm to use to measure the size of x.  Predefined norms
            include norm_1, norm_2, norm_info, norm_min and norm_p(p)
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0.001, norm=norm_2, scaled=True):
        self.tol = tol
        self.norm = norm
        self.scaled = scaled

    def _scaled_condition(self, history):
        df = history.gradient[0]
        f = history.value[0]
        scale = abs(f)
        if scale == 0:
            scale = 1
        return self.norm(df / float(scale))

    def _raw_condition(self, history):
        df = history.gradient[0]
        return self.norm(df)

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        if self.tol > 0:
            history.requires(gradient=1, value=1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.gradient) < 1:
            return False  # Cannot succeed until at least n generations
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "|| del F[k]/F[k] || < %g" % (self.tol)
        else:
            return "|| del F[k] || < %g" % (self.tol)


class r_best:
    """
    Measure of population radius based on distance from the best.

        max ||(y - x[k])/scale|| for y in population

    scipy.optimize.fmin uses r_best(norm_inf) as its measure of radius.
    """

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, population, best, scale):
        P = np.asarray(population)
        r = max(self.norm(p - best) / scale for p in P)
        return r


class r_centroid:
    """
    Measure of population radius based on distance from the centroid.

        max ||(y - <y>)/scale|| for y in population
    """

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, population, best, scale):
        P = np.asarray(population)
        c_i = np.mean(P, axis=0)
        r = max(self.norm(p - c_i) / scale for p in P)
        return r


def r_boundingbox(population, best, scale):
    """
    Measure of population radius based on the volume of the bounding box.

        (product (max(y_i) - min(y_i))/scale)**1/k  for i in dimensions-k
    """
    P = np.asarray(population)
    lo = max(P, index=0)
    hi = max(P, index=0)
    r = np.prod((hi - lo) / scale) ** (1 / len(hi))
    return r


class r_hull:
    """
    Measure of population radius based on maximum diameter in convex hull.

        1/2 max || (y1 - y2)/scale || for y1,y2 in population
    """

    def __init__(self, norm):
        self.norm = norm

    def __call__(self, population, best, scale):
        r = 0
        for i, y1 in enumerate(population):
            for y2 in population[i + 1 :]:
                d = self.norm(y2 - y1) / scale
                if d > r:
                    r = d
        return r / 2


class Rx(Condition):
    """
    Domain size

    This condition measures the size of the population domain.  Some
    algorithms are done when the domain size shrinks while others have
    failed if the domain size shrinks.

    There are a number of ways of measuring the domain size::

        r_best(norm) : radius from best point

            max ||(y - x[k])/scale|| for y in population

        r_centroid(norm) : radius from centroid

            max ||(y - <y>)/scale|| for y in population

        r_boundingbox : radius from bounding box

            (product (max(y_i) - min(y_i))/scale)**1/k  for i in dimensions-k

        r_hull(norm) : radius from convex hull

            1/2 max || (y1 - y2)/scale || for y1,y2 in population

    scale is determined from the fit bounds (max-min) or the
    values sum(|y_i|)/n, with protection against zero values.

    Parameters::

        *tol* (float = 0.001)
            tolerance to test against
        *radius* (function(history,best,scale): float = r_centroid(norm_2))
            measure of domain size
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0, radius=r_centroid(norm_2), scaled=False):
        self.tol = tol
        self.radius = radius
        self.scaled = scaled

    def _scaled_condition(self, history):
        P = np.asarray(history.population_points[0])
        scale = history.upper_bound - history.lower_bound
        idx = isinf(scale)
        if any(idx):
            range = np.sum(abs(P), axis=0) / P.shape[0]
            scale[idx] = range[idx]
        scale[scale == 0] = 1
        r = self.radius(P, history.point[0], scale)
        # print "Rx=%g, scale=%g"%(r,scale)
        return r

    def _raw_condition(self, history):
        P = np.asarray(history.population_points[0])
        r = self.radius(P, history.point[0], scale=1.0)
        # print "Rx=%g"%r
        return r

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        if self.tol > 0:
            history.requires(population_points=1, point=1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.population_points) < 1:
            return False
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "radius(population/scale) < %g" % (self.tol)
        else:
            return "radius(population) < %g" % (self.tol)


class Rf(Condition):
    """
    Range size

    This condition measures the size of the population range::

        (max f(y) - min f(y))/scale < tol

    for y in the current population
    scale is mean(|f(y)|)

    Parameters::

        *tol* (float = 0.001)
            tolerance to test against
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0, scaled=True):
        self.tol = tol
        self.scaled = scaled

    def _scaled_condition(self, history):
        Pf = np.asarray(history.population_values)
        scale = np.mean(abs(Pf))
        if scale == 0:
            scale = 1
        r = float(np.max(Pf) - np.min(Pf)) / scale
        # print "Rf = %g, scale=%g"%(r,scale)
        return r

    def _raw_condition(self, history):
        P = np.asarray(history.population_values)
        r = np.max(P) - np.min(P)
        # print "Rf = %g"%r
        return r

    def config_history(self, history):
        """
        Needs the previous n points from history.
        """
        if self.tol > 0:
            history.requires(population_values=1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.population_values) < 1:
            return False
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "(max(F(p)) - min(F(p))/mean(|F(p)|) < %g" % (self.tol)
        else:
            return "max(F(p)) - min(F(p)) < %g" % (self.tol)


class Cx(Condition):
    """
    Target point

    This condition measures the distance from the best point
    to some target point::

       ||(x_k - Z)/scale|| < tol

    scale is fit range if given,  |Z_i|, or 1 if Z_i=0

    Paramaters::

        *tol* (float = 0.001)
            tolerance to test against
        *point* (array = 0)
            target point
        *norm* ( f(vector): float  =  norm_2)
            norm to use to measure the size of x.  Predefined norms
            include norm_1, norm_2, norm_info, norm_min and norm_p(p)
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0.001, point=0, norm=norm_2, scaled=True):
        self.tol = tol
        self.point = point
        self.norm = norm_2
        self.scaled = scaled

    def _scaled_condition(self, history):
        x = history.point[0]
        scale = history.upper_bound - history.lower_bound
        scale[isinf(scale)] = abs(self.point)[isinf(scale)]
        scale[scale == 0] = 1
        return self.norm((x - self.point) / scale)

    def _raw_condition(self, history):
        x = history.point[0]
        return self.norm(x - self.point)

    def config_history(self, history):
        """
        Needs the previous point from history.
        """
        if self.tol > 0:
            history.requires(point=1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.point) < 1:
            return False
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "||(x[k] - Z)/range|| < %g" % (self.tol)
        else:
            return "||x[k] - Z|| < %g" % (self.tol)


class Cf(Condition):
    """
    Target value

    This condition measures the distance from the best value
    to some target value::

       |(f_k - A)/scale| < tol

    scale is |A| or 1 if A=0

    Paramaters::

        *tol* (float = 0.001)
            tolerance to test against
        *value* (float = 0)
            target value
        *scaled* (boolean = True)
            whether to use raw or scaled differences in the norm

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, tol=0.001, value=0, scaled=True):
        self.tol = tol
        self.value = value
        if value == 0:
            scaled = False
        self.scaled = scaled

    def _scaled_condition(self, history):
        value = history.value[0]
        return abs(float(value - self.value) / self.value)

    def _raw_condition(self, history):
        value = history.value[0]
        return abs(value - self.value)

    def config_history(self, history):
        """
        Needs the previous point from history.
        """
        if self.tol > 0:
            history.requires(value=1)

    def __call__(self, history):
        """
        Returns True if the tolerance is met.
        """
        if self.tol == 0 or len(history.value) < 1:
            return False
        elif self.scaled:
            return self._scaled_condition(history) < self.tol
        else:
            return self._raw_condition(history) < self.tol

    def __str__(self):
        if self.scaled:
            return "|(F[k] - A)/A| < %g" % (self.tol)
        else:
            return "|F[k] - A| < %g" % (self.tol)


class Steps(Condition):
    """
    Specific number of iterations

    This condition test the number of iterations of a fit::

        k >= steps

    Parameters::

        *steps* int (1000)
            total number of steps

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, steps=np.inf):
        self.steps = steps

    def __call__(self, history):
        if len(history.step) < 1:
            return False
        return history.step[0] >= self.steps

    def config_history(self, history):
        history.requires(step=1)

    def __str__(self):
        return "steps >= %d" % self.steps


class Calls(Condition):
    """
    Specific number of function calls

    This condition tests the number of function evaluations::

        n_k >= calls

    Parameters::

        *calls* int (inf)
            total number of function calls

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, calls=np.inf):
        self.calls = calls

    def __call__(self, history):
        if len(history.calls) < 1:
            return False
        return history.calls[0] >= self.calls

    def config_history(self, history):
        history.requires(calls=1)

    def __str__(self):
        return "calls >= %d" % self.calls


class Time(Condition):
    """
    Wall clock time.

    This condition tests wall clock time::

        t_k >= time

    Parameters::

        *time* float (inf)
            Time since start of job in seconds

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, time=inf):
        self.time = time

    def __call__(self, history):
        if len(history.time) < 1:
            return False
        return history.time[0] >= self.time

    def config_history(self, history):
        history.requires(time=1)

    def __str__(self):
        return "time >= %g" % self.time


class CPU(Condition):
    """
    CPU time.

    This condition tests CPU time::

        t(CPU)_k >= time

    Parameters::

        *time* float (inf)
            time since start of job in seconds

    Returns::

        *condition* (f(history) : boolean)
            a callable returning true if the condition is met
    """

    def __init__(self, time=np.inf):
        self.time = time

    def __call__(self, history):
        if len(history.cpu_time) < 1:
            return False
        return history.cpu_time[0] >= self.time

    def config_history(self, history):
        history.requires(cpu_time=1)

    def __str__(self):
        return "cpu_time >= %g" % self.time


"""
class Feasible: value can be used ** Not implemented **

    f_k satisfies soft constraints

class Invalid: values are well defined

    isinf(y) or isinf(f(y)) or isnan(y) or isnan(f(y))

    for y in population
"""


def parse_condition(cond):
    import math
    from . import stop

    return eval(cond, stop.__dict__.copy().update(math.__dict__))
