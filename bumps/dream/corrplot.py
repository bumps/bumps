# This program is public domain
# Author Paul Kienzle
"""
2-D correlation histograms

Generate 2-D correlation histograms and display them in a figure.

Uses false color plots of density.
"""
__all__ = ['Corr2d']

import numpy as np
from numpy import inf

from matplotlib import cm, colors, image, artist
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

try:
    COLORMAP = colors.LinearSegmentedColormap.from_list(
        'density', ('w', 'y', 'g', 'b', 'r'))
except Exception:
    COLORMAP = cm.gist_earth_r


class Corr2d(object):
    """
    Generate and manage 2D correlation histograms.
    """
    def __init__(self, data, labels=None, **kw):
        if labels is None:
            labels = ["P"+str(i+1) for i, _ in enumerate(data)]
        self.N = len(data)
        self.labels = labels
        self.data = data
        self.hists = _hists(data, **kw)
        #for k, v in self.hists.items():
        #    print k, (v[1][0], v[1][-1]), (v[2][0], v[2][-1])
        self.ax = None  # will be set on plot

    def R(self):
        return np.corrcoef(self.data)

    def __getitem__(self, i, j):
        """
        Retrieve correlation histogram for data[i] X data[j].

        Returns bin i edges, bin j edges, and histogram
        """
        return self.hists[i, j]

    def plot(self, title=None):
        """
        Plot the correlation histograms on the specified figure
        """
        import pylab

        pylab.clf()
        fig = pylab.gcf()
        if title is not None:
            fig.text(0.5, 0.95, title,
                     horizontalalignment='center',
                     fontproperties=FontProperties(size=16))
        self.ax = _plot(fig, self.hists, self.labels, self.N)


def _hists(data, ranges=None, **kw):
    """
    Generate pair-wise correlation histograms
    """
    n = len(data)
    if ranges is None:
        low, high = np.min(data, axis=1), np.max(data, axis=1)
        ranges = [(l, h) for l, h in zip(low, high)]
    return dict(((i, j), np.histogram2d(data[i], data[j],
                                        range=[ranges[i], ranges[j]], **kw))
                for i in range(0, n)
                for j in range(i+1, n))


def _plot(fig, hists, labels, n, show_ticks=None):
    """
    Plot pair-wise correlation histograms
    """
    if n <= 1:
        fig.text(0.5, 0.5, "No correlation plots when only one variable",
                 ha="center", va="center")
        return
    vmin, vmax = inf, -inf
    for data, _, _ in hists.values():
        positive = data[data > 0]
        if len(positive) > 0:
            vmin = min(vmin, np.amin(positive))
            vmax = max(vmax, np.amax(positive))
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
    #norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = image.FigureImage(fig)
    mapper.set_array(np.zeros(0))
    mapper.set_cmap(cmap=COLORMAP)
    mapper.set_norm(norm)

    if show_ticks is None:
        show_ticks = n < 3
    ax = {}
    Nr = Nc = n-1
    for i in range(0, n-1):
        for j in range(i+1, n):
            sharex = ax.get((0, j), None)
            sharey = ax.get((i, i+1), None)
            a = fig.add_subplot(Nr, Nc, (Nr-i-1)*Nc + j,
                                sharex=sharex, sharey=sharey)
            ax[(i, j)] = a
            a.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 4, 5, 10]))
            a.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 4, 5, 10]))
            data, x, y = hists[(i, j)]
            data = np.clip(data, vmin, vmax)
            a.pcolorfast(y, x, data, cmap=COLORMAP, norm=norm)
            # Show labels or hide ticks
            if i != 0:
                artist.setp(a.get_xticklabels(), visible=False)
            if i == n-2 and j == n-1:
                a.set_xlabel(labels[j])
                #a.xaxis.set_label_position("top")
                #a.xaxis.set_offset_position("top")
            if not show_ticks:
                a.xaxis.set_ticks([])
            if j == i+1:
                a.set_ylabel(labels[i])
            else:
                artist.setp(a.get_yticklabels(), visible=False)
            if not show_ticks:
                a.yaxis.set_ticks([])

            a.zoomable = True

    # Adjust subplots and add the colorbar
    fig.subplots_adjust(left=0.07, bottom=0.07, top=0.9, right=0.85,
                        wspace=0.0, hspace=0.0)
    cax = fig.add_axes([0.88, 0.2, 0.04, 0.6])
    fig.colorbar(mapper, cax=cax, orientation='vertical')
    return ax


def zoom(event, step):
    ax = event.inaxes
    if not hasattr(ax, 'zoomable'):
        return

    # TODO: test logscale
    step *= 3

    if ax.zoomable is not True and 'mapper' in ax.zoomable:
        mapper = ax.zoomable['mapper']
        if event.ydata is not None:
            lo, hi = mapper.get_clim()
            pt = event.ydata*(hi-lo)+lo
            lo, hi = _rescale(lo, hi, pt, step)
            mapper.set_clim((lo, hi))
    if ax.zoomable is True and event.xdata is not None:
        lo, hi = ax.get_xlim()
        lo, hi = _rescale(lo, hi, event.xdata, step)
        ax.set_xlim((lo, hi))
    if ax.zoomable is True and event.ydata is not None:
        lo, hi = ax.get_ylim()
        lo, hi = _rescale(lo, hi, event.ydata, step)
        ax.set_ylim((lo, hi))
    ax.figure.canvas.draw_idle()


def _rescale(lo, hi, pt, step):
    scale = float(hi-lo)*step/(100 if step > 0 else 100-step)
    bal = float(pt-lo)/(hi-lo)
    new_lo = lo - bal*scale
    new_hi = hi + (1-bal)*scale
    return new_lo, new_hi
