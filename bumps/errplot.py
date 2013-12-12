"""
Estimate model uncertainty from random sample.
"""
import os
from .dream.state import load_state
from . import plugin
from .cli import load_problem, recall_best


import numpy


def reload_errors(model, store, nshown=50, random=True):
    """
    Reload the error information for a model.

    The loaded error data is a sample from the fit space according to the
    fit parameter uncertainty.  This is a subset of the samples returned
    by the DREAM MCMC sampling process.

    *model* is the name of the model python file

    *store* is the name of the store directory containing the dream results

    *nshown* and *random* are as for :func:`calc_errors_from_state`.

    See :func:`calc_errors` for details on the return values.
    """
    problem = load_problem([model])
    recall_best(problem, os.path.join(store, model[:-3]+".par"))
    state = load_state(os.path.join(store, model[:-3]))
    state.mark_outliers()
    return calc_errors_from_state(problem, state,
                                  nshown=nshown, random=random)


def show_errors(*args, **kw):
    return plugin.show_errors(*args, **kw)

def calc_errors_from_state(problem, state, nshown=50, random=True):
    """
    Align the sample profiles and compute the residual difference from 
    the measured data for a set of points returned from DREAM.

    *nshown* is the number of samples to include from the state.

    *random* is True if the samples are randomly selected, or False if
    the most recent samples should be used.  Use random if you have
    poor mixing (i.e., the parameters tend to stay fixed from generation
    to generation), but not random if your burn-in was too short, and
    you want to select from the end.

    See :func:`calc_errors` for details on the return values.
    """
    points, _logp = state.sample()
    if points.shape[0] < nshown: nshown = points.shape[0]
    # randomize the draw; skip the last point since state.keep_best() put
    # the best point at the end.
    if random: points = points[numpy.random.permutation(len(points)-1)]
    return calc_errors(problem, points[-nshown:-1])

def calc_errors(problem, points):
    """
    Align the sample profiles and compute the residual difference from the
    measured data for a set of points.

    Return value is arbitrary.  It is passed through to show_errors() for
    the application.
    """
    original = problem.getp()
    try:
        ret = plugin.calc_errors(problem, points)
    except:
        import traceback
        print "error calculating distribution on model"
        traceback.print_exc()
        ret = None
    finally:
        problem.setp(original)
    return ret
