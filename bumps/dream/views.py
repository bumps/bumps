"""
MCMC plotting methods.
"""
from __future__ import division, print_function

__all__ = ['plot_all', 'plot_corr', 'plot_corrmatrix',
           'plot_trace', 'plot_logp', 'format_vars']

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
    print("\nStatistics and plots based on {nsamp:d} samples "
          "({psamp:.1%} of total samples drawn)".format( \
          nsamp=len(draw.points), psamp=portion))
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

    # parallel coordinates plot
    if draw.num_vars > 1:
        from . import parcoord
        figure()
        parcoord.plot(draw, control_var=0)
        if state.title:
            suptitle(state.title)
        if figfile is not None:
            savefig(figfile+"-parcor"+figext)

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
    ax_hist_x.hist(values[0], nbins, orientation='vertical', density=1)
    ax_hist_x.plot(x, px, 'k-')
    ax_hist_x.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
    setp(ax_hist_x.get_xticklabels(), visible=False,)
    ax_hist_y = axes([0.75, 0.1, 0.2, 0.63], sharey=ax_data)
    ax_hist_y.hist(values[1], nbins, orientation='horizontal', density=1)
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
    genid = arange(state.generation-len(draw)+start, state.generation)+1
    plot(genid*state.thinning,
         squeeze(points[start:, state._good_chains, var]))
    xlabel('Generation number')
    ylabel(label)


def plot_logp(state, portion=None):
    from pylab import axes, title
    from scipy.stats import chi2, kstest
    from matplotlib.ticker import NullFormatter

    # Plot log likelihoods
    draw, logp = state.logp()
    start = int((1-portion)*len(draw)) if portion else 0
    genid = arange(state.generation-len(draw)+start, state.generation)+1
    width, height, margin, delta = 0.7, 0.75, 0.1, 0.01
    trace = axes([margin, 0.1, width, height])
    trace.plot(genid, logp[start:], ',', markersize=1)
    trace.set_xlabel('Generation number')
    trace.set_ylabel('Log likelihood at x[k]')
    title('Log Likelihood History')

    # Plot log likelihood trend line
    from bumps.wsolve import wpolyfit
    from .formatnum import format_uncertainty
    x = np.arange(start, logp.shape[0]) + state.generation - state.Ngen + 1
    y = np.mean(logp[start:], axis=1)
    dy = np.std(logp[start:], axis=1, ddof=1)
    p = wpolyfit(x, y, dy=dy, degree=1)
    px, dpx = p.ci(x, 1.)
    trace.plot(x, px, 'k-', x, px + dpx, 'k-.', x, px - dpx, 'k-.')
    trace.text(x[0], y[0], "slope="+format_uncertainty(p.coeff[0], p.std[0]),
               va='top', ha='left')

    # Plot long likelihood histogram
    data = logp[start:].flatten()
    hist = axes([margin+width+delta, 0.1, 1-2*margin-width-delta, height])
    hist.hist(data, bins=40, orientation='horizontal', density=True)
    hist.set_ylim(trace.get_ylim())
    null_formatter = NullFormatter()
    hist.xaxis.set_major_formatter(null_formatter)
    hist.yaxis.set_major_formatter(null_formatter)

    # Plot chisq fit to log likelihood histogram
    float_df, loc, scale = chi2.fit(-data, f0=state.Nvar)
    df = int(float_df + 0.5)
    pval = kstest(-data, lambda x: chi2.cdf(x, df, loc, scale))
    #with open("/tmp/chi", "a") as fd:
    #    print("chi2 pars for llf", float_df, loc, scale, pval, file=fd)
    xmin, xmax = trace.get_ylim()
    x = np.linspace(xmin, xmax, 200)
    hist.plot(chi2.pdf(-x, df, loc, scale), x, 'r')


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

