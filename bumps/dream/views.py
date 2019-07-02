"""
MCMC plotting methods.
"""
from __future__ import division, print_function

__all__ = ['plot_all', 'plot_corr', 'plot_corrmatrix',
           'plot_trace', 'plot_R', 'plot_logp', 'format_vars']

import math

import numpy as np
from numpy import arange, squeeze, linspace, meshgrid, vstack, inf
from scipy.stats import kde

from . import corrplot
from . import varplot
from .formatnum import format_value
from .stats import var_stats, format_vars, save_vars

def plot_all(state, portion=1.0, figfile=None):
    # Print/save uncertainty report before loading pylab or creating plots
    draw = state.draw(portion=portion)
    all_vstats = var_stats(draw)
    print(format_vars(all_vstats))
    if figfile is not None:
        save_vars(all_vstats, figfile+"-err.json")

    from pylab import figure, savefig, suptitle, rcParams
    figext = '.'+rcParams.get('savefig.format', 'png')

    # histograms
    figure(figsize=varplot.var_plot_size(len(all_vstats)))
    varplot.plot_vars(draw, all_vstats)
    if state.title:
        suptitle(state.title, x=0, y=1, va='top', ha='left')
    if figfile is not None:
        savefig(figfile+"-vars"+figext)

    # parameter traces
    figure()
    plot_traces(state, portion=portion)
    suptitle("Parameter history" + (" for " + state.title if state.title else ""))
    if figfile is not None:
        savefig(figfile+"-trace"+figext)

    # R stat plot
    #figure()
    #plot_R(state, portion=portion)
    #if state.title:
    #    suptitle(state.title)
    #if figfile is not None:
    #    savefig(figfile+"-R"+format)

    # convergence plot
    figure()
    plot_logp(state, portion=portion)
    if state.title:
        suptitle(state.title)
    if figfile is not None:
        savefig(figfile+"-logp"+figext)

    # correlation plot
    if draw.num_vars <= 25:
        figure()
        plot_corrmatrix(draw)
        if state.title:
            suptitle(state.title)
        if figfile is not None:
            savefig(figfile+"-corr"+figext)


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
