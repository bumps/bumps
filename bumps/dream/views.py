"""
MCMC plotting methods.
"""
from __future__ import division, print_function

__all__ = ['plot_all', 'plot_corr', 'plot_corrmatrix',
           'plot_trace', 'plot_vars', 'plot_var',
           'plot_R', 'plot_logp', 'format_vars']

import math

import numpy as np
from numpy import arange, squeeze, linspace, meshgrid, vstack, inf
from scipy.stats import kde

from . import corrplot
from .formatnum import format_value
from .stats import var_stats, format_vars, save_vars

def plot_all(state, portion=1.0, figfile=None):
    from pylab import figure, savefig, suptitle, rcParams

    figext = '.'+rcParams.get('savefig.format', 'png')

    draw = state.draw(portion=portion)
    all_vstats = var_stats(draw)
    figure()
    plot_vars(draw, all_vstats)
    if state.title:
        suptitle(state.title)
    print(format_vars(all_vstats))
    if figfile is not None:
        savefig(figfile+"-vars"+figext)
    if figfile is not None:
        save_vars(all_vstats, figfile+"-err.json")
    figure()
    plot_traces(state, portion=portion)
    suptitle("Parameter history" + (" for " + state.title if state.title else ""))
    if figfile is not None:
        savefig(figfile+"-trace"+figext)
    # Suppress R stat for now
    #figure()
    #plot_R(state, portion=portion)
    #if state.title:
    #    suptitle(state.title)
    #if figfile is not None:
    #    savefig(figfile+"-R"+format)
    figure()
    plot_logp(state, portion=portion)
    if state.title:
        suptitle(state.title)
    if figfile is not None:
        savefig(figfile+"-logp"+figext)
    if draw.num_vars <= 25:
        figure()
        plot_corrmatrix(draw)
        if state.title:
            suptitle(state.title)
        if figfile is not None:
            savefig(figfile+"-corr"+figext)


def plot_vars(draw, all_vstats, **kw):
    from pylab import subplot, clf

    clf()
    nw, nh = tile_axes(len(all_vstats))
    cbar = _make_fig_colorbar(draw.logp)
    for k, vstats in enumerate(all_vstats):
        subplot(nw, nh, k+1)
        plot_var(draw, vstats, k, cbar, **kw)


def tile_axes(n, size=None):
    """
    Creates a tile for the axes which covers as much area of the graph as
    possible while keeping the plot shape near the golden ratio.
    """
    from pylab import gcf
    if size is None:
        size = gcf().get_size_inches()
    figwidth, figheight = size
    # Golden ratio phi is the preferred dimension
    #    phi = sqrt(5)/2
    #
    # nw, nh is the number of tiles across and down respectively
    # w, h are the sizes of the tiles
    #
    # w,h = figwidth/nw, figheight/nh
    #
    # To achieve the golden ratio, set w/h to phi:
    #     w/h = phi  => figwidth/figheight*nh/nw = phi
    #                => nh/nw = phi * figheight/figwidth
    # Must have enough tiles:
    #     nh*nw > n  => nw > n/nh
    #                => nh**2 > n * phi * figheight/figwidth
    #                => nh = floor(sqrt(n*phi*figheight/figwidth))
    #                => nw = ceil(n/nh)
    phi = math.sqrt(5)/2
    nh = int(math.floor(math.sqrt(n*phi*figheight/figwidth)))
    if nh < 1:
        nh = 1
    nw = int(math.ceil(n/nh))
    return nw, nh


def plot_var(draw, vstats, var, cbar, nbins=30):
    values = draw.points[:, var].flatten()
    _make_logp_histogram(values, draw.logp, nbins, vstats.p95_range,
                         draw.weights, cbar)
    _decorate_histogram(vstats)


def _decorate_histogram(vstats):
    import pylab
    from matplotlib.transforms import blended_transform_factory as blend

    l95, h95 = vstats.p95_range
    l68, h68 = vstats.p68_range

    # Shade things inside 1-sigma
    pylab.axvspan(l68, h68, color='gold', alpha=0.5, zorder=-1)
    # build transform with x=data, y=axes(0,1)
    ax = pylab.gca()
    transform = blend(ax.transData, ax.transAxes)

    def marker(symbol, position):
        if position < l95:
            symbol, position, ha = '<'+symbol, l95, 'left'
        elif position > h95:
            symbol, position, ha = '>'+symbol, h95, 'right'
        else:
            symbol, position, ha = symbol, position, 'center'
        pylab.text(position, 0.95, symbol, va='top', ha=ha,
                   transform=transform, zorder=3, color='g')
        #pylab.axvline(v)

    marker('|', vstats.median)
    marker('E', vstats.mean)
    marker('*', vstats.best)

    pylab.text(0.01, 0.95, vstats.label, zorder=2,
               backgroundcolor=(1, 1, 0, 0.2),
               verticalalignment='top',
               horizontalalignment='left',
               transform=pylab.gca().transAxes)
    pylab.setp([pylab.gca().get_yticklabels()], visible=False)
    ticks = (l95, l68, vstats.median, h68, h95)
    labels = [format_value(v, h95-l95) for v in ticks]
    if len(labels[2]) > 5:
        # Drop 68% values if too many digits
        ticks, labels = ticks[0::2], labels[0::2]
    pylab.xticks(ticks, labels)


def _make_fig_colorbar(logp):
    import matplotlib as mpl
    import pylab

    # Option 1: min to min + 4
    #vmin=-max(logp); vmax=vmin+4
    # Option 1b: min to min log10(num samples)
    #vmin=-max(logp); vmax=vmin+log10(len(logp))
    # Option 2: full range of best 98%
    snllf = pylab.sort(-logp)
    vmin, vmax = snllf[0], snllf[int(0.98*(len(snllf)-1))]  # robust range
    # Option 3: full range
    #vmin,vmax = -max(logp),-min(logp)

    fig = pylab.gcf()
    ax = fig.add_axes([0.60, 0.95, 0.35, 0.05])
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
            return format_value(x, self.delta)

    ticks = (vmin, vmax)
    formatter = MinDigitsFormatter(vmin, vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   ticks=ticks, format=formatter,
                                   orientation='horizontal')
    #cb.set_ticks(ticks)
    #cb.set_ticklabels(labels)
    #cb.set_label('negative log likelihood')

    return vmin, vmax, cmap


def _make_logp_histogram(values, logp, nbins, ci, weights, cbar):
    from numpy import (ones_like, searchsorted, linspace, cumsum, diff,
                       argsort, array, hstack, exp)
    if weights is None:
        weights = ones_like(logp)
    # TODO: values are being sorted to collect stats and again to plot
    idx = argsort(values)
    values, weights, logp = values[idx], weights[idx], logp[idx]
    #open('/tmp/out','a').write("ci=%s, range=%s\n"
    #                           % (ci,(min(values),max(values))))
    edges = linspace(ci[0], ci[1], nbins+1)
    idx = searchsorted(values[1:-1], edges)
    weightsum = cumsum(weights)
    heights = diff(weightsum[idx])/weightsum[-1]  # normalized weights

    import pylab
    vmin, vmax, cmap = cbar
    cmap_steps = linspace(vmin, vmax, cmap.N+1)
    bins = []  # marginalized maximum likelihood
    for h, s, e, xlo, xhi \
            in zip(heights, idx[:-1], idx[1:], edges[:-1], edges[1:]):
        if s == e:
            continue
        pv = -logp[s:e]
        pidx = argsort(pv)
        pw = weights[s:e][pidx]
        x = array([xlo, xhi], 'd')
        y = hstack((0, cumsum(pw)))
        z = pv[pidx][:, None]
        # centerpoint, histogram height, maximum likelihood for each bin
        bins.append(((xlo+xhi)/2, y[-1], exp(vmin-z[0])))
        if len(z) > cmap.N:
            # downsample histogram bar according to number of colors
            pidx = searchsorted(z[1:-1].flatten(), cmap_steps)
            if pidx[-1] < len(z)-1:
                pidx = hstack((pidx, -1))
            y, z = y[pidx], z[pidx]
        pylab.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap)
    # Check for broken distribution
    if not bins:
        return
    centers, height, maxlikelihood = array(bins).T
    # Normalize maximum likelihood plot so it contains the same area as the
    # histogram, unless it is really spikey, in which case make sure it has
    # about the same height as the histogram.
    maxlikelihood *= np.sum(height)/np.sum(maxlikelihood)
    hist_peak = np.max(height)
    ml_peak = np.max(maxlikelihood)
    if ml_peak > hist_peak*1.3:
        maxlikelihood *= hist_peak*1.3/ml_peak
    pylab.plot(centers, maxlikelihood, '-g')


def _make_var_histogram(values, logp, nbins, ci, weights):
    # Produce a histogram
    hist, bins = np.histogram(values, bins=nbins, range=ci,
                              #new=True,
                              normed=True, weights=weights)

    # Find the max likelihood for values in each bin
    edges = np.searchsorted(values, bins)
    histbest = [np.max(logp[edges[i]:edges[i+1]])
                if edges[i] < edges[i+1] else -inf
                for i in range(nbins)]

    # scale to marginalized probability with peak the same height as hist
    histbest = np.exp(np.asarray(histbest) - max(logp)) * np.max(hist)

    import pylab
    # Plot the histogram
    pylab.bar(bins[:-1], hist, width=bins[1]-bins[0])

    # Plot the kernel density estimate
    #density = KDE1D(values)
    #x = linspace(bins[0],bins[-1],100)
    #pylab.plot(x, density(x), '-k')

    # Plot the marginal maximum likelihood
    centers = (bins[:-1]+bins[1:])/2
    pylab.plot(centers, histbest, '-g')


def plot_corrmatrix(draw):
    c = corrplot.Corr2d(draw.points.T, bins=50, labels=draw.labels)
    c.plot()
    #print "Correlation matrix\n",c.R()


class KDE1D(kde.gaussian_kde):
    covariance_factor = lambda self: 2*self.silverman_factor()


class KDE2D(kde.gaussian_kde):
    covariance_factor = kde.gaussian_kde.silverman_factor

    def __init__(self, dataset):
        kde.gaussian_kde.__init__(self, dataset.T)

    def evalxy(self, x, y):
        grid_x, grid_y = meshgrid(x, y)
        dxy = self.evaluate(vstack([grid_x.flatten(), grid_y.flatten()]))
        return dxy.reshape(grid_x.shape)

    __call__ = evalxy


def plot_corr(draw, vars=(0, 1)):
    from pylab import axes, setp, MaxNLocator

    _, _ = vars  # Make sure vars is length 2
    labels = [draw.labels[v] for v in vars]
    values = [draw.points[:, v] for v in vars]

    # Form kernel density estimates of the parameters
    xmin, xmax = min(values[0]), max(values[0])
    density_x = KDE1D(values[0])
    x = linspace(xmin, xmax, 100)
    px = density_x(x)

    density_y = KDE1D(values[1])
    ymin, ymax = min(values[1]), max(values[1])
    y = linspace(ymin, ymax, 100)
    py = density_y(y)

    nbins = 50
    ax_data = axes([0.1, 0.1, 0.63, 0.63])  # x,y,w,h

    #density_xy = KDE2D(values[vars])
    #dxy = density_xy(x,y)*points.shape[0]
    #ax_data.pcolorfast(x,y,dxy,cmap=cm.gist_earth_r) #@UndefinedVariable

    ax_data.plot(values[0], values[1], 'k.', markersize=1)
    ax_data.set_xlabel(labels[0])
    ax_data.set_ylabel(labels[1])
    ax_hist_x = axes([0.1, 0.75, 0.63, 0.2], sharex=ax_data)
    ax_hist_x.hist(values[0], nbins, orientation='vertical', normed=1)
    ax_hist_x.plot(x, px, 'k-')
    ax_hist_x.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
    setp(ax_hist_x.get_xticklabels(), visible=False,)
    ax_hist_y = axes([0.75, 0.1, 0.2, 0.63], sharey=ax_data)
    ax_hist_y.hist(values[1], nbins, orientation='horizontal', normed=1)
    ax_hist_y.plot(py, y, 'k-')
    ax_hist_y.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
    setp(ax_hist_y.get_yticklabels(), visible=False)


def plot_traces(state, vars=None, portion=None):
    from pylab import subplot, clf, subplots_adjust

    if vars is None:
        vars = list(range(min(state.Nvar, 6)))
    clf()
    nw, nh = tile_axes(len(vars))
    subplots_adjust(hspace=0.0)
    for k, var in enumerate(vars):
        subplot(nw, nh, k+1)
        plot_trace(state, var, portion)


def plot_trace(state, var=0, portion=None):
    from pylab import plot, title, xlabel, ylabel

    draw, points, _ = state.chains()
    label = state.labels[var]
    start = int((1-portion)*len(draw)) if portion else 0
    plot(arange(start, len(points))*state.thinning,
         squeeze(points[start:, state._good_chains, var]))
    xlabel('Generation number')
    ylabel(label)


def plot_R(state, portion=None):
    from pylab import plot, title, legend, xlabel, ylabel

    draw, R = state.R_stat()
    start = int((1-portion)*len(draw)) if portion else 0
    plot(arange(start, len(R)), R[start:])
    title('Convergence history')
    legend(['P%d' % i for i in range(1, R.shape[1]+1)])
    xlabel('Generation number')
    ylabel('R')


def plot_logp(state, portion=None):
    from pylab import plot, title, xlabel, ylabel

    draw, logp = state.logp()
    start = int((1-portion)*len(draw)) if portion else 0
    plot(arange(start, len(logp)), logp[start:], ',', markersize=1)
    title('Log Likelihood History')
    xlabel('Generation number')
    ylabel('Log likelihood at x[k]')
