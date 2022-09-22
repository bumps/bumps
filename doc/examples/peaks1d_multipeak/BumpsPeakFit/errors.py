"""
Module for calcuating confidence intervals for the BumpsPeakFit package

This can be used in a notebook (use functions with _api suffix)
 by loading in the DREAM state or passing it the state from a DREAM result object
Or by including:

from bumps.cli import install_plugin
sys.path.append('./BumpsPeakFit/BumpsPeakFit')  # <-- relative path between model file and BumpsPeakFit package
import fitplugin
install_plugin(fitplugin)

At the start of the model file loaded into bumps.
This will install a plugin upon initial load of the model file which will output
 the correct plot for the model uncertainty, and include plotting options not in the default bumps GUI
"""
import numpy as np
import sys
import os
# in the model_uncertainty_update_ajc branch, the function below is included in bumps.fitters
# from bumps.fitters import get_points_from_state
from bumps.cli import load_model, load_best
from bumps.dream.state import load_state

CONTOURS = (68, 95)


# Origionally taken from refl1d errors module.
def get_points_from_state(state, nshown=50, random=True, portion=1.0):
    """
    *nshown* is the number of points from the DREAM state to be used for
    the error plot calculations
    *random* is True if the samples are randomly selected, or False if
    the most recent samples should be used.  Use random if you have
    poor mixing (i.e., the parameters tend to stay fixed from generation
    to generation), but not random if your burn-in was too short, and
    you want to select from the end.
    Returns *points* with the best placed at the end i.e. best = points[-1]
    """
    if state is None:
        # Should we include a warning here?
        return

    points, _logp = state.sample(portion=portion)
    if points.shape[0] < nshown:
        nshown = points.shape[0]
    # randomize the draw; skip the last point since state.keep_best() put
    # the best point at the end.
    if random:
        points = points[np.random.permutation(len(points) - 1)]
    # return the origonal nshown passed to the function -
    #  if this has changed we recalculate points, if not use the cached ones
    return points[-nshown:-1]


def run_errors(**kw):
    """
    Command line tool for generating error plots from models.

    Type the following to regenerate the profile contour plots plots:

        $ bumps align <model>.py <store> [0|1|2|n]

    You can plot the profiles and residuals on one plot by setting plots to 1,
    on two separate plots by setting plots to 2, or each curve on its own
    plot by setting plots to n. Plots are saved in <store>/<model>-err#.png.
    If plots is 0, then no plots are created.

    Additional parameters include:

        *nshown*, *random* :

            see :func:`bumps.errplot.calc_errors_from_state`

        *contours*, *plots*, *save* :

            see :func:`show_errors`
    """

    load = {'model': None, 'store': None, 'nshown': 50, 'random': True}
    # TODO: for now we will keep show to a simple plot figure
    #  rather than multiple plots as in refl1d
    # show = {'plots': 2, 'contours': CONTOURS, 'save': None}
    show = {'plots': 0, 'intervals': CONTOURS, 'save': None}

    for k, v in kw.items():
        if k in load:
            load[k] = v
        elif k in show:
            show[k] = v
        else:
            raise TypeError("Unknown argument "+k)

    if len(sys.argv) > 2:
        load['model'], load['store'] = sys.argv[1], sys.argv[2]

        # TODO: reinstante options if multiple plots required.
        # plots_str = sys.argv[3] if len(sys.argv) >= 4 else '2'
        # show['plots'] = int(plots_str) if plots_str != 'n' else plots_str
        #print align, align_str, len(sys.argv), sys.argv

    if not load['store'] or not load['model']:
        _usage()
        sys.exit(0)

    if show['save'] is None:
        name, _ = os.path.splitext(os.path.basename(load['model']))
        show['save'] = os.path.join(load['store'], name)

    print("loading... this may take awhile")
    errors, problem = reload_errors_bpf(**load)
    show_opts = {key: show[key] for key in ["intervals", "save"]}

    if show['plots'] != 0:
        # TODO: place outside of if statement if we require multiple plots
        #  see refl1d.errors.show_errors for details
        print("showing...")
        problem.fitness.plot_forwardmc(errors, **show_opts)
        import matplotlib.pyplot as plt
        plt.show()
    else:
        # Just save out contours
        problem.fitness.save_forwardmc(errors, **show_opts)

    sys.exit(0)  # Force bumps to terminate.


def reload_errors_bpf(model, store, nshown=50, random=True):
    """
    bpf = BumpsPeakFit - this is an interim solution until reload_errors in Bumps
    handles problem.calc_forwardmc method.

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
    points = get_points_from_state(state, nshown=nshown, random=random)
    # TODO: when upgrades to bumps are done
    #  referring to fitness may no longer be done this way
    return problem.fitness.calc_forwardmc(problem, points), problem


def _usage():
    print(run_errors.__doc__)


def select_random_samples(points, nshown=50):
    total_n_of_rows = points.shape[0]
    # select n number of samples in points
    if nshown < total_n_of_rows:
        random_indices = np.random.choice(total_n_of_rows, size=nshown, replace=False)
        return points[random_indices, :]
    else:
        return points


def get_intervals(array, interval):
    interval_low, interval_high = interval_low_high_from_interval(interval)

    quantile_low = np.quantile(array, interval_low, axis=0)
    quantile_high = np.quantile(array, interval_high, axis=0)

    return quantile_low, quantile_high


def interval_low_high_from_interval(interval):
    # we want half the interval above and below the 50% point (the mean)
    # since we are converting from percent to probability e.g. [0, 1]
    # we should also divide by 100 so for 68% interval: interval_low = 0.16 --> in percent is 16% = 50-(interval/2)
    # simplified by putting on the same denominator: 100/2 - interval/2 => (100 - interval)/2
    # converting to probability gives interval_low = (100 - interval)/(2*100) = (100 - interval)/200
    interval_low = (100 - interval) / 200
    # interval_high = (50 + (interval/2))/100 = (100/2 + interval/2)/100 = (100 + interval)/200
    # or alternatively, interval_high = (100 - interval_low*100)/100 = (100 - ((100 - interval)/200)*100)/100
    # --> interval_high = (100 - (100 - interval)/2)/100 =
    #     (200/2 - (100 - interval)/2)/100 = ((200 - 100 + interval)/2)/100
    # --> interval_high = ((100 + interval)/2)/100 = (100 + interval)/200
    interval_high = (100 + interval) / 200

    return interval_low, interval_high


def calc_errors(problem, points):

    model = problem.fitness
    # calculate the best curves
    best = points[-1]
    problem.setp(best)
    best_total = model.theory()
    best_peaks = model.parts_theory()

    # calculate the peaks and total theory for the samples/points
    peak_names = [part.name for part in model.parts]
    peaks = []
    total = []
    for pts in points:
        problem.setp(pts)
        peaks.append(model.parts_theory())
        total.append(model.theory())
    peaks_array = np.transpose(np.array(peaks), axes=(1, 0, 2))
    total_array = np.array(total)

    return total_array, peaks_array, peak_names, model.X, model.data, best_peaks, best_total


def _calc_intervals(errs, intervals):

    total_array, peaks_array, peak_names = errs
    # caclulate intervals
    total_intv = []
    peaks_dict = {}
    for intv in intervals:
        total_intv.append(get_intervals(total_array, intv))
    for peak, peak_name in zip(peaks_array, peak_names):
        peaks_intv = []
        for intv in intervals:
            peaks_intv.append(get_intervals(peak, intv))
        peaks_dict[peak_name] = peaks_intv

    return total_intv, peaks_dict


def show_errors(errors, intervals=(68, 95), save=None):

    import matplotlib.pyplot as plt
    from itertools import cycle
    import pylab

    # fig, ax = plt.subplots()
    plt.subplot(111)
    colours = cycle(['r', 'b', 'g', 'c', 'm'])
    # total
    # p_68, p_95 = total
    # ax.fill_between(M.X,*p_68, color='r')
    # ax.fill_between(M.X,*p_95, alpha=0.5, color='r')
    total_array, peaks_array, peak_names, x, data, best_peaks, best_total = errors

    total, peaks_dict = _calc_intervals((total_array, peaks_array, peak_names), intervals)

    # peak 1
    for i, (peak_name, peak) in enumerate(peaks_dict.items()):
        p_68, p_95 = peak
        colour = next(colours)
        pylab.fill_between(x, *p_68, alpha=0.5, color=colour)
        pylab.fill_between(x, *p_95, alpha=0.25, color=colour)

        pylab.plot(x, best_peaks[i], color=colour, label=f"{peak_name} Best")

    plt.plot(x, best_total, color='k', label="Best")
    plt.scatter(x, data, s=3, color='k', label="Data")
    plt.legend()
    # plt.draw()

    if save:
        print("trying to save model uncertainty")
        pylab.savefig(save + "-err.png")


def calc_errors_api(results_object, model_object, problem_object, nshown=50, intervals=(68, 95)):
    points = results_object.state.draw().points
    # best = points[-1]
    samples = select_random_samples(points, nshown)
    peak_names = [part.name for part in model_object.parts]
    peaks = []
    total = []
    for pts in samples:
        problem_object.setp(pts)
        peaks.append(model_object.parts_theory())
        total.append(model_object.theory())
    peaks_array = np.transpose(np.array(peaks), axes=(1, 0, 2))
    total_array = np.array(total)
    # caclulate intervals
    total_intv = []
    peaks_dict = {}
    for intv in intervals:
        total_intv.append(get_intervals(total_array, intv))
    for peak, peak_name in zip(peaks_array, peak_names):
        peaks_intv = []
        for intv in intervals:
            peaks_intv.append(get_intervals(peak, intv))
        peaks_dict[peak_name] = peaks_intv

    return total_intv, peaks_dict
