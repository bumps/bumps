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
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# if more than this many variables are to be plotted, put them all
# on a single axis for efficiency (no linked axes)
MAKE_SINGLE_BREAKPOINT = 9

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

    def plot(self, title=None, sort=True, max_rows=25, indices=None):
        """
        Plot the correlation histograms on the specified figure

        Use supplied indices to select parameters by index, else
        generate indices (optionally sorted by max correlation coeff.)
        """
        num_to_show = min(max_rows, self.N - 1)
        if indices is None:
            if sort:
                coeffs = (self.R() - np.eye(self.N))
                max_corr = np.max(coeffs**2, axis=0)
                indices = np.argsort(max_corr)[:-max_rows-2:-1]
                labels = _disambiguated(self.labels)
            else:
                indices = np.arange(num_to_show + 1, dtype=np.int32)
                labels = self.labels
        if num_to_show > MAKE_SINGLE_BREAKPOINT:
            fig = _plot_single_heatmap(self.hists, labels, indices=indices)
        else:
            fig = _plot(self.hists, labels, indices=indices)
        if title is not None:
            fig.update_layout(title=dict(text=title, xanchor="center", x=0.5))

        return fig


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


def _plot(hists, labels, indices, show_ticks=None):
    """
    Plot pair-wise correlation histograms
    """

    n = len(indices)
    vmin, vmax = float('inf'), float('-inf')
    for i, index in enumerate(indices[:-1]):
        for cross_index in indices[i+1:]:
            ii, jj = sorted((index, cross_index))
            data, _, _ = hists[(ii, jj)]
            positive = data[data > 0]
            if len(positive) > 0:
                vmin = min(vmin, np.amin(positive))
                vmax = max(vmax, np.amax(positive))
    
    fig = make_subplots(rows=n-1, cols=n-1, horizontal_spacing=0, vertical_spacing=0, shared_yaxes=True, shared_xaxes=True)
    COLORSCALE = ["white", "yellow", "green", "blue", "red"]

    for i, index in enumerate(indices[:-1]):
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            xanchor="right",
            yanchor="bottom",
            x=-0.05,
            y=0.05,
            showarrow=False,
            col=i+1,
            row=n-i-1,
            text=labels[index],
            textangle=-90
        )
        for j, cross_index in enumerate(indices[i+1:], start=i+1):
            ii, jj = sorted((index, cross_index))
            data, x, y = hists[(ii, jj)]
            data = np.clip(data, vmin, vmax)
            hovertemplate = f"{labels[index]}<br>{labels[cross_index]}<extra></extra>"
            trace = go.Heatmap(z=np.log10(data), coloraxis='coloraxis', hovertemplate=hovertemplate, customdata=[ii,jj])
            fig.add_traces([trace], rows=n-i-1, cols=j)
    
    # Add annotation for last parameter:
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        xanchor="left",
        yanchor="bottom",
        x=0.05,
        y=1.05,
        showarrow=False,
        col=i+1,
        row=n-i-1,
        text=labels[indices[-1]],
        textangle=0
    )


    log_cbar = dict(
        tickvals=np.arange(int(np.log10(vmax)) + 1),
        ticktext=10 ** np.arange(int(np.log10(vmax)) + 1),
    )
    fig.update_layout(coloraxis={'colorscale': COLORSCALE, "cmin": np.log10(vmin), "cmax": np.log10(vmax), 'colorbar': log_cbar})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(hoverlabel=dict(bgcolor='white', font_size=16))
    # fig.update_layout(height=600, width=800)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def _plot_single_heatmap(hists, labels, indices, show_ticks=None):
    """
    Plot pair-wise correlation histograms
    """

    n = len(indices)
    vmin, vmax = float('inf'), float('-inf')
    for i, index in enumerate(indices[:-1]):
        for cross_index in indices[i+1:]:
            ii, jj = sorted((index, cross_index))
            data, _, _ = hists[(ii, jj)]
            positive = data[data > 0]
            if len(positive) > 0:
                vmin = min(vmin, np.amin(positive))
                vmax = max(vmax, np.amax(positive))

    fig = go.Figure()
    COLORSCALE = ["white", "yellow", "green", "blue", "red"]

    for i, index in enumerate(indices[:-1]):
        fig.add_annotation(
            xanchor="right",
            yanchor="bottom",
            x=i+1,
            y=i,
            showarrow=False,
            text=labels[index],
            textangle=-90
        )
        for j, cross_index in enumerate(indices[i+1:], start=i+1):
            ii, jj = sorted((index, cross_index))
            data, x, y = hists[(ii, jj)]
            data = np.clip(data, vmin, vmax)
            sx, sy = data.shape
            dx = 1.0 / sx
            dy = 1.0 / sy
            hovertemplate = f"{labels[index]}<br>{labels[cross_index]}<extra></extra>"
            trace = go.Heatmap(z=np.log10(data), y=[i, i+dx], x=[j,j+dy], coloraxis='coloraxis', hovertemplate=hovertemplate, customdata=[ii,jj])
            fig.add_traces([trace])

    # Add annotation for last parameter:
    fig.add_annotation(
        xanchor="left",
        yanchor="bottom",
        x=i+1,
        y=i+1,
        showarrow=False,
        text=labels[indices[-1]],
        textangle=0
    )

    log_cbar = dict(
        tickvals=np.arange(int(np.log10(vmax)) + 1),
        ticktext=10 ** np.arange(int(np.log10(vmax)) + 1),
    )
    fig.update_layout(coloraxis={'colorscale': COLORSCALE, "cmin": np.log10(vmin), "cmax": np.log10(vmax), 'colorbar': log_cbar})
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_layout(hoverlabel=dict(bgcolor='white', font_size=16))
    # fig.update_layout(height=600, width=800)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _disambiguated(labels: List[str]):
    label_count = {}
    output = []
    for label in labels:
        label_count.setdefault(label, 0)
        count = label_count[label]
        l = f"{label} ({count})" if count > 0 else label
        output.append(l)
        label_count[label] += 1
    return output


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
