"""
Build layout for histogram plots
"""

__all__ = ["var_plot_size", "plot_vars", "plot_var"]

from math import ceil, sqrt

import numpy as np
from matplotlib import pyplot as plt

# Set space between plots in horiz and vert.
H_SPACE = 0.2
V_SPACE = 0.3

# Set top, bottom, left margins.
T_MARGIN = 0.2
B_MARGIN = 0.1
L_MARGIN = 0.2
R_MARGIN = 0.2

# Set desired plot sizes.
TILE_W = 3.0
TILE_H = 2.0
CBAR_WIDTH = 0.75


def var_plot_size(n):
    ncol, nrow = tile_axes_square(n)

    # Calculate total width and figure size
    plots_width = (TILE_W + H_SPACE) * ncol
    figwidth = plots_width + CBAR_WIDTH + L_MARGIN + R_MARGIN
    figheight = (TILE_H + V_SPACE) * nrow + T_MARGIN + B_MARGIN
    return figwidth, figheight


def _make_var_axes(n, fig=None):
    """
    Build a figure with one axis per parameter,
    and one axis (the last one) to contain the colorbar.
    Use to make the vars histogram figure.
    """
    if fig is None:
        fig = plt.gcf()
    fig.clf()
    total_width, total_height = fig.get_size_inches()

    ncol, nrow = tile_axes_square(n)

    # Calculate dimensions as a faction of figure size.
    v_space_f = V_SPACE / total_height
    h_space_f = H_SPACE / total_width
    t_margin_f = T_MARGIN / total_height
    b_margin_f = B_MARGIN / total_height
    l_margin_f = L_MARGIN / total_width
    top = 1 - t_margin_f + v_space_f
    left = l_margin_f

    tile_h = (total_height - T_MARGIN - B_MARGIN) / nrow - V_SPACE
    tile_w = (total_width - L_MARGIN - R_MARGIN - CBAR_WIDTH) / ncol - H_SPACE
    tile_h_f = tile_h / total_height
    tile_w_f = tile_w / total_width

    # Calculate colorbar location (left, bottom) and colorbar height.
    l_cbar_f = l_margin_f + ncol * (tile_w_f + h_space_f)
    b_cbar_f = b_margin_f + v_space_f
    cbar_w_f = CBAR_WIDTH / total_width
    cbar_h_f = 1 - t_margin_f - b_margin_f - v_space_f
    cbar_box = [l_cbar_f, b_cbar_f, cbar_w_f, cbar_h_f]

    k = 0
    for j in range(1, nrow + 1):
        for i in range(0, ncol):
            if k >= n:
                break
            dims = [left + i * (tile_w_f + h_space_f), top - j * (tile_h_f + v_space_f), tile_w_f, tile_h_f]
            ax = fig.add_axes(dims)
            ax.set_facecolor("none")
            k += 1

    fig.add_axes(cbar_box)
    # fig.set_size_inches(total_width, total_height)
    return fig


def tile_axes_square(n):
    """
    Determine number of columns by finding the
    next greatest square, then determine number
    of rows needed.
    """
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / float(cols)))
    return cols, rows


def plot_vars(draw, all_vstats, fig=None, nbins: int = 30, full: bool = False):
    n = len(all_vstats)
    fig = _make_var_axes(n, fig=fig)
    cbar = _make_fig_colorbar(draw.logp, fig=fig)
    for k, vstats in enumerate(all_vstats):
        axes = fig.axes[k]
        plot_var(draw, vstats, k, cbar, axes=axes, nbins=nbins, full=full)
        fig.canvas.draw()


def plot_var(draw, vstats, var, cbar, axes=None, nbins=30, full=False):
    import matplotlib.pyplot as plt

    values = draw.points[:, var].flatten()
    if full:
        bin_range = np.min(values), np.max(values)
    else:
        bin_range = vstats.p95_range

    if axes is None:
        axes = plt.gca()

    make_logp_histogram(values, draw.logp, nbins, bin_range, draw.weights, cbar, axes)
    decorate_histogram(vstats, axes)


def decorate_histogram(vstats, axes):
    from matplotlib.transforms import blended_transform_factory as blend

    l95, h95 = vstats.p95_range
    l68, h68 = vstats.p68_range

    # Shade things inside 1-sigma
    axes.axvspan(l68, h68, color="gold", alpha=0.5, zorder=-2)
    # Mark the median with a vertical line
    axes.axvline(x=vstats.median, color="g", ls=":", alpha=0.7, zorder=-1)
    # build transform with x=data, y=axes(0,1)
    transform = blend(axes.transData, axes.transAxes)

    # Mark the mean and best with symbols
    def marker(symbol: str, position: float) -> None:
        if position < l95:
            text, x, ha = f"{symbol}←", l95, "left"
        elif position > h95:
            text, x, ha = f"→{symbol}", h95, "right"
        else:
            text, x, ha = symbol, position, "center"
        y, va = -0.01, "top"
        axes.text(x, 1 + y, text, va=va, ha=ha, transform=transform, zorder=3, color="g")
        # axes.axvline(v)

    marker("▽", vstats.mean)
    marker("∗", vstats.best)

    # Put the parameter label on the line with mean and best markers. Use the side without
    # the mean/best marker so that they don't overwrite each other.
    if (vstats.mean - l95) / (h95 - l95) > 0.4 or (vstats.best - l95) / (h95 - l95) > 0.4:
        x, ha = 0.02, "left"
    else:
        x, ha = 0.98, "right"
    axes.text(
        x,
        0.99,
        vstats.label,
        zorder=2,
        # backgroundcolor=(0.6, 1.0, 0.6, 0.6),
        verticalalignment="top",
        horizontalalignment=ha,
        transform=axes.transAxes,
    )
    axes.set_yticklabels([])


def _make_fig_colorbar(logp, fig=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Option 1: min to min + 4
    # vmin=-max(logp); vmax=vmin+4
    # Option 1b: min to min log10(num samples)
    # vmin=-max(logp); vmax=vmin+log10(len(logp))
    # Option 2: full range of best 98%
    snllf = np.sort(-logp)
    vmin, vmax = snllf[0], snllf[int(0.98 * (len(snllf) - 1))]  # robust range
    # Option 3: full range
    # vmin,vmax = -max(logp),-min(logp)

    if fig is None:
        fig = plt.gcf()
    ax = fig.axes[-1]
    cmap = mpl.cm.copper

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    class MinDigitsFormatter(mpl.ticker.Formatter):
        def __init__(self, low, high):
            self.delta = high - low

        def __call__(self, x, pos=None):
            # TODO: where did format_value come from?
            # it does not exist anywhere in the project.
            # return format_value(x, self.delta)
            return "{:.3G}".format(x)

    ticks = ()  # (vmin, vmax)
    formatter = MinDigitsFormatter(vmin, vmax)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks, format=formatter, orientation="vertical")
    # cb.set_ticks(ticks)
    # cb.set_ticklabels(labels)
    # cb.set_label('negative log likelihood')

    cbar_box = ax.get_position().bounds
    fig.text(cbar_box[0], cbar_box[1], "{:.3G}".format(vmin), va="top")
    fig.text(cbar_box[0], cbar_box[1] + cbar_box[3], "{:.3G}".format(vmax), va="bottom")

    return cbar


def make_logp_histogram(values, logp, nbins, ci, weights, cbar, ax):
    from numpy import ones_like, searchsorted, linspace, cumsum, unique, argsort, array, hstack, exp

    if weights is None:
        weights = ones_like(logp)
    # TODO: values are being sorted to collect stats and again to plot
    idx = argsort(values)
    values, weights, logp = values[idx], weights[idx], logp[idx]
    # open('/tmp/out','a').write("ci=%s, range=%s\n"
    #                           % (ci,(min(values),max(values))))
    edges = linspace(ci[0], ci[1], nbins + 1)
    idx = searchsorted(values[1:-1], edges)
    # weightsum = cumsum(weights)
    # heights = diff(weightsum[idx])/weightsum[-1]  # normalized weights

    edgecolors = None
    cmap = cbar.cmap
    cmap_edges = linspace(0, 1, cmap.N + 1)[1:-1]
    bins = []  # marginalized maximum likelihood
    for s, e, xlo, xhi in zip(idx[:-1], idx[1:], edges[:-1], edges[1:]):
        if s == e:
            continue
        # parameter interval endpoints
        x = array([xlo, xhi], "d")
        # -logp values within interval, with sort index from low to high
        pv = -logp[s:e]
        pidx = argsort(pv)
        pv = pv[pidx]
        # weights for samples within interval, sorted
        pw = weights[s:e][pidx]
        # vertical colorbar top edges is the cumulative sum of the weights
        y_top = cumsum(pw)

        # For debugging compare with one rectangle per sample
        if False:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            cmap = mpl.cm.flag
            edgecolors = "k"
            xmid = (xlo + xhi) / 2
            x = [xlo, xmid]
            y = hstack((0, y_top))
            z = pv[:, None]
            plt.pcolormesh(x, y, z, norm=cbar.norm, cmap=cmap)
            x = [xmid, xhi]

        # Possibly millions of samples, so group those which have the
        # same colour instead of drawing each as its own rectangle.
        #
        # Norm the values then look up the colormap edges in the sorted
        # normed negative log probabilities.  Drop duplicates, which
        # represent zero-width bars. Assign the value for each interval
        # according to the value at the change point.
        #
        # The indexing logic is very ugly. The searchsorted() function
        # returns 0 if before the first or N if after the last, so the
        # end points of the range are dropped so that there is and implicit
        # [-inf, ... interior points ..., inf] range. Similarly, colours
        # below vmin go to vmin and above vmax go to vmax, so drop those
        # end points as well. Then put the end points back on in the
        # found indices [0, ... interior edges ..., N-1]. Use the value
        # at the end of the boundary to colour the section.
        # Something is not quite right: with this algorithm the first
        # block appears to always be one element long and is often the
        # same colour as the next block. This is only visible if edges
        # are drawn so ignore it for now.
        change_point = searchsorted(cbar.norm(pv[1:-1]), cmap_edges)
        tops = unique(hstack((change_point, len(pv) - 1)))
        y = hstack((0, y_top[tops]))
        z = pv[tops][:, None]
        ax.pcolormesh(x, y, z, norm=cbar.norm, cmap=cmap, edgecolors=edgecolors)

        # centerpoint, histogram height, maximum likelihood for each bin
        bins.append(((xlo + xhi) / 2, y_top[-1], exp(cbar.norm.vmin - pv[0])))
    # Check for broken distribution
    if not bins:
        return
    centers, height, maxlikelihood = array(bins).T
    # Normalize maximum likelihood plot so it contains the same area as the
    # histogram, unless it is really spikey, in which case make sure it has
    # about the same height as the histogram.
    maxlikelihood *= np.sum(height) / np.sum(maxlikelihood)
    hist_peak = np.max(height)
    ml_peak = np.max(maxlikelihood)
    if ml_peak > hist_peak * 1.3:
        maxlikelihood *= hist_peak * 1.3 / ml_peak
        ml_peak = hist_peak * 1.3
    ax.plot(centers, maxlikelihood, "-b")

    # Leave space for parameter label and statistics markers at the top of the plot
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_ylim(0, 1.1 * max(hist_peak, ml_peak))
    # ax.autoscale(enable=True, axis='y', tight=False)

    ## plot marginal gaussian approximation along with histogram
    # def G(x, mean, std):
    #    return np.exp(-((x-mean)/std)**2/2)/np.sqrt(2*np.pi*std**2)
    ## TODO: use weighted average for standard deviation
    # mean, std = np.average(values, weights=weights), np.std(values, ddof=1)
    # pdf = G(centers, mean, std)
    # plt.plot(centers, pdf*np.sum(height)/np.sum(pdf), '-b')


def make_var_histogram(values, logp, nbins, ci, weights):
    # Produce a histogram
    hist, bins = np.histogram(
        values,
        bins=nbins,
        range=ci,
        # new=True,
        density=True,
        weights=weights,
    )

    # Find the max likelihood for values in each bin
    edges = np.searchsorted(values, bins)
    histbest = [np.max(logp[edges[i] : edges[i + 1]]) if edges[i] < edges[i + 1] else -np.inf for i in range(nbins)]

    # scale to marginalized probability with peak the same height as hist
    histbest = np.exp(np.asarray(histbest) - max(logp)) * np.max(hist)

    import matplotlib.pyplot as plt

    # Plot the histogram
    plt.bar(bins[:-1], hist, width=bins[1] - bins[0])

    # Plot the kernel density estimate
    # density = KDE1D(values)
    # x = linspace(bins[0],bins[-1],100)
    # plt.plot(x, density(x), '-k')

    # Plot the marginal maximum likelihood
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, histbest, "-g")
