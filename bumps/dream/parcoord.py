import numpy as np


def plot(draw, nlines=150, control_var=None):
    """
    Plot parallel coordinates from a draw from the distribution.

    *draw* is the draw from the sample

    *nlines* is the number of lines to draw

    If *control_var* is provided, then the best value from each bin in the
    histogram for that variable will be chosen as a line to draw on the
    coordination plot. This will will produce a roughly equally spaced set of
    coordination points for that variable. If *control_var* is None, then
    lines will be picked at random.
    """
    data = draw.points
    labels = draw.labels
    logp = draw.logp

    # TODO: confirm that plot is called with 95% interval draw
    # Note: the 95% interval draw is based on the range of the data?
    # Drop points outside the 95% interval of chisq
    # index = np.argsort(logp)
    # npoints = data.shape[0]
    # maxn = int(npoints*0.95)
    # index = index[:maxn]
    # logp = logp[index]
    # data = data[index]

    # TODO: order data by strength of correlation
    # Some options:
    #    Pearson: only works for linear
    #    Kendal, Spearman: only works for monotonic
    #    distance (Szekely, 2005) doi:10.1214/009053607000000505
    #    HGG (Heller, Heller, Gorfine, 2013) doi:10.1093/biomet/ass070
    # estimate correlations
    # corr = matrix_corr(data)
    # order = list(range(data.shape[1]))
    # data = data[order]

    if control_var is None:
        index = np.random.choice(data.shape[0], size=nlines, replace=False)
    else:
        index = best_in_bin(data[:, control_var], -logp, bins=nlines)
    # color_value, color_label = data[index, var], labels[var]
    color_value, color_label = -logp[index], "-log(P)"
    parallel_coordinates(data[index], labels=labels, value=color_value, value_label=color_label)


def best_in_bin(x, value, bins=50, range=None, keep_empty=False):
    """
    Find the index of the minimum *value* in each bin for the data in *x*.

    *x* are the coordinates to be binned.

    *value* are the objective to be minimized in each bin, such as chisq.
    There is one value for each x coordinate.

    *bins* are the number of bins within *range* or an array of bin edges
    if the bins are not uniform.

    *range* is the data range for the bins. The default is (x.min(), x.max()).

    *keep_empty* is True if empty bins will return an index of -1.  When
    False (the default), empty bins are removed.

    Returns indices of the elements which are the minimum within each bin.
    This list may be shorter than the number of bins if *keep_empty* is False.

    The algorithm works by assigning a bin number to each point then adding
    an offset in [0, 1) according to the scaled *value*.  These point values
    are then sorted, and searched by bin number.  The returned index will
    correspond to the first value in each bin, and therefore, the best value
    in that bin.  From this the index into the original list can be returned.
    """

    # Find bin number for each value.
    if isinstance(bins, int):
        # Find intervals for x and use integer division to assign bin number.
        # Don't worry that this might create bin numbers such as -3 or nbins+5
        # since these will be ignored during bin lookup.
        xmin, xmax = range if range is not None else (np.min(x), np.max(x))
        dx = (xmax - xmin) / bins
        value_in_bin = (x - xmin) // dx if dx != 0.0 else np.full_like(x, bins // 2)
        nbins = bins
    else:
        # Lookup x in bin edges. searchsorted returns index 0 for elements
        # before the first edge so we need to put an extra start edge and
        # subtract one from the index so that bin number is -1 before first.
        value_in_bin = np.searchsorted(np.hstack((-np.inf, bins)), x) - 1
        nbins = len(bins) - 1

    # Increment bin number offset using scaled value.  Limit the scaled
    # value to 0.99 rather 1.0 so that the worst value isn't 1.0, which
    # will be the best value in the next bin.
    value_in_bin += scale(value) * 0.99

    # Sort values, preserving original indices
    sort_index = np.argsort(value_in_bin)
    sorted_value_in_bin = value_in_bin[sort_index]

    # Lookup first point in each bin, which will now by the min chisq in bin.
    bin_numbers = np.arange(nbins)
    min_index = np.searchsorted(sorted_value_in_bin, bin_numbers)

    # Find the point number for the best points in the original list.
    index = sort_index[min_index]

    # Indentify empty bins. This will occur if base value returned by
    # search sorted does not match the bin number.
    empty = np.floor(sorted_value_in_bin[min_index]) != bin_numbers

    # Process empty bins.
    if keep_empty:
        index[empty] = -1
    else:
        index = index[~empty]

    return index


def parallel_coordinates(data, labels=None, value=None, value_label=""):
    """
    Produce a parallel coordinates plot.

    *data* is a set of points to draw, one row per point.

    *labels* is the label to assign to the dimensions, one per column in *data*.

    *value* is an optional value to use to color the different lines. This
    could be the value of the control variable, or some additional dimension
    such as overall quality for the individual points.

    *value_label* is the label to put on the colorbar for the line colors.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ndim = data.shape[1]
    if labels is None:
        labels = ["p%d" % k for k in range(ndim)]

    x = np.arange(ndim)
    data = scale(data, axis=0)

    # We need to set the plot limits, they will not autoscale
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlim(0, ndim - 1)
    ax.set_ylim(0, 1)
    plt.xticks(x, labels, rotation=30)
    lines = LineCollection([np.column_stack([x, y]) for y in data], linewidths=1, linestyles="solid")
    ax.add_collection(lines)
    ax.set_yticklabels([])

    if value is not None:
        lines.set_array(value)
        cbar = fig.colorbar(lines)
        cbar.set_label(value_label)
        plt.sci(lines)


def scale(x, axis=None):
    """
    Returns *x* with values scaled to [0, 1].

    If *axis* is not None, then scale values within each axis independently,
    otherwise use the min/max value across all dimensions.
    """
    low = x.min(axis=axis, keepdims=True)
    high = x.max(axis=axis, keepdims=True)
    dx = high - low
    dx[dx == 0.0] = 1.0
    scaled = (x - low) / dx
    return scaled
