# This program is public domain
# Authors Paul Kienzle, Brian Maranville
"""
2-D correlation histograms

Generate 2-D correlation histograms and display them in a figure.

Uses false color plots of density.
"""
__all__ = ['Corr2d']

import numpy as np
from numpy import inf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        self.fig = None
        #for k, v in self.hists.items():
        #    print k, (v[1][0], v[1][-1]), (v[2][0], v[2][-1])

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

        fig = make_subplots(
            rows=self.N-1,
            cols=self.N-1,
            horizontal_spacing=0,
            vertical_spacing=0,
            shared_yaxes=True,
            shared_xaxes=True
        )

        # if title is not None:
        #     fig.text(0.5, 0.95, title,
        #              horizontalalignment='center',
        #              fontproperties=FontProperties(size=16))
        return _plot(fig, self.hists, self.labels, self.N)


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
    # if n <= 1:
    #     fig.text(0.5, 0.5, "No correlation plots when only one variable",
    #              ha="center", va="center")
    #     return
    
    vmin, vmax = float('inf'), float('-inf')
    for data, _, _ in hists.values():
        positive = data[data > 0]
        if len(positive) > 0:
            vmin = min(vmin, np.amin(positive))
            vmax = max(vmax, np.amax(positive))
    
    fig = make_subplots(rows=n-1, cols=n-1, horizontal_spacing=0, vertical_spacing=0, shared_yaxes=True, shared_xaxes=True)
    COLORSCALE = ["white", "yellow", "green", "blue", "red"]

    for i in range(0, n-1):
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="right",
            x=-0.1,
            y=0.1,
            showarrow=False,
            col=i+1,
            row=n-i-1,
            text=labels[i],
            textangle=-90
        )
        for j in range(i+1, n):
            data, x, y = hists[(i, j)]
            data = np.clip(data, vmin, vmax)
            trace = go.Heatmap(z=np.log10(data), coloraxis='coloraxis', hoverinfo='skip')
            fig.add_traces([trace], rows=n-i-1, cols=j)
    
    log_cbar = dict(
        tickvals=np.arange(int(np.log10(vmax)) + 1),
        ticktext=10 ** np.arange(int(np.log10(vmax)) + 1),
    )
    fig.update_layout(coloraxis={'colorscale': COLORSCALE, "cmin": np.log10(vmin), "cmax": np.log10(vmax), 'colorbar': log_cbar})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    # fig.update_layout(height=600, width=800)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

### NOT USED AT THE MOMENT: all below
###
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
