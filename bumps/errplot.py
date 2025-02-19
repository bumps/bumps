"""
Estimate model uncertainty from random sample.

MCMC uncertainty analysis gives the uncertainty on the model parameters
rather than the model itself.  For example, when fitting a line to a set
of data, the uncertainty on the slope and the intercept does not directly
give you the uncertainty in the expected value of *y* for a given value
of *x*.

The routines in bumps.errplot allow you to generate confidence intervals
on  the model using a random sample of MCMC parameters.  After calculating
the model *y* values for each sample, one can generate 68% and 95% contours
for a set of sampling points *x*.  This can apply even to models which
are not directly measured.  For example, in scattering inverse problems
the scattered intensity is the value measured, but the fitting parameters
describe the real space model that is being probed.  It is the uncertainty
in the real space model that is of primary interest.

Since bumps knows only the probability of seeing the measured value given
the input parameters, it is up to the model itself to calculate and display
the confidence intervals on the model and the expected values for the data
points.  This is done using the :mod:`bumps.plugin` architecture, so
application writers can provide the appropriate functions for their data
types.  Eventually this capability will move to the model definition so
that different types of models can be processed in the same fit.

For a completed MCMC run, four steps are required:

#. reload the fitting problem and the MCMC state
#. select a set of sample points
#. evaluate model confidence intervals from sample points
#. show model confidence intervals

:func:`reload_errors` performs steps 1, 2 and 3, returning *errs*.
If the fitting problem and the MCMC state are already loaded, then use
:func:`calc_errors_from_state` to perform steps 2 and 3, returning *errs*.
If alternative sampling is desired, then use :func:`calc_errors` on a
given set of points to perform step 3, returning *errs*.  Once *errs* has
been calculated and returned by one of these methods, call
:func:`show_errors` to perform step 4.
"""

__all__ = ["reload_errors", "calc_errors_from_state", "calc_errors", "show_errors"]
import os
import traceback
import logging

import numpy as np

from .dream.state import load_state
from . import plugin
from .cli import load_model, load_best


def reload_errors(model, store, nshown=50, random=True):
    """
    Reload the MCMC state and compute the model confidence intervals.

    The loaded error data is a sample from the fit space according to the
    fit parameter uncertainty.  This is a subset of the samples returned
    by the DREAM MCMC sampling process.

    *model* is the name of the model python file

    *store* is the name of the store directory containing the dream results

    *nshown* and *random* are as for :func:`calc_errors_from_state`.

    Returns *errs* for :func:`show_errors`.
    """
    problem = load_model(model)
    load_best(problem, os.path.join(store, problem.name + ".par"))
    state = load_state(os.path.join(store, problem.name))
    state.mark_outliers()
    return calc_errors_from_state(problem, state, nshown=nshown, random=random)


def error_points_from_state(state, nshown=50, random=True, portion=1.0):
    """
    Return a set of points from the state for calculating errors.

    *nshown* is the number of samples to include from the state.

    *random* is True if the samples are randomly selected, or False if
    the most recent samples should be used.  Use random if you have
    poor mixing (i.e., the parameters tend to stay fixed from generation
    to generation), but not random if your burn-in was too short, and
    you want to select from the end.

    Returns *points* for :func:`calc_errors`.
    """

    points, _logp = state.sample(portion=portion)
    if points.shape[0] < nshown:
        nshown = points.shape[0]
    # randomize the draw; skip the last point since state.keep_best() put
    # the best point at the end.
    if random:
        points = points[np.random.permutation(len(points) - 1)]
    return points[-nshown:-1]


def calc_errors_from_state(problem, state, nshown=50, random=True, portion=1.0):
    """
    Compute confidence regions for a problem from the
    Align the sample profiles and compute the residual difference from
    the measured data for a set of points returned from DREAM.

    Returns *errs* for :func:`show_errors`.
    """

    points = error_points_from_state(state, nshown=nshown, random=random, portion=portion)
    return calc_errors(problem, points)


def calc_errors(problem, points):
    """
    Align the sample profiles and compute the residual difference from the
    measured data for a set of points.

    The return value is arbitrary.  It is passed to the :func:`show_errors`
    plugin for the application.
    Returns *errs* for :func:`show_errors`.
    """
    original = problem.getp()
    try:
        ret = plugin.calc_errors(problem, points)
    except Exception:
        info = ["error calculating distribution on model", traceback.format_exc()]
        logging.error("\n".join(info))
        ret = None
    finally:
        problem.setp(original)
    return ret


def show_errors(errs, fig=None):
    """
    Display the confidence regions returned by :func:`calc_errors`.

    The content of *errs* depends on the active plugin.
    """
    return plugin.show_errors(errs, fig=fig)
