"""
Build layout for histogram plots
"""

__all__ = ['var_plot_size', 'plot_vars', 'plot_var']

from math import ceil, sqrt
from typing import Dict

import numpy as np

# Set space between plots in horiz and vert.
H_SPACE = 0.2
V_SPACE = 0.2

# Set top, bottom, left margins.
T_MARGIN = 0.2
B_MARGIN = 0.2
L_MARGIN = 0.2
R_MARGIN = 0.4

# Set desired plot sizes.
TILE_W = 3.0
TILE_H = 2.0
CBAR_WIDTH = 0.75
CBAR_COLORS = 64

LINE_COLOR = "red"
ANNOTATION_COLOR = "blue"
HISTOGRAM_COLORMAP = "Greens_r"


def tile_axes_square(n):
    """
    Determine number of columns by finding the
    next greatest square, then determine number
    of rows needed.
    """
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n/float(cols)))
    return cols, rows


def plot_vars(draw, all_vstats, **kw):
    from plotly.subplots import make_subplots

    n = len(all_vstats)
    ncol, nrow = tile_axes_square(n)

    snllf = np.sort(-draw.logp)
    vmin, vmax = snllf[0], snllf[int(0.98*(len(snllf)-1))]  # robust range
    cbar_edges = np.linspace(vmin, vmax, CBAR_COLORS)
    titles = [vstats.label for vstats in all_vstats]

    # fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles)
    fig = make_subplots(rows=nrow, cols=ncol)
    fig.update_yaxes(secondary_y = True)

    for k, vstats in enumerate(all_vstats):
        row = (k // ncol) + 1
        col = (k % ncol) + 1
        plot_var(fig, draw, vstats, k, cbar_edges=cbar_edges, row=row, col=col, **kw)
    
    fig.update_xaxes(dict(exponentformat='e'), overwrite=True)
    fig.update_yaxes(dict(showticklabels=False))
    fig.update_layout(height=600, width=800, coloraxis_colorbar_title="-logP")
    fig.update_layout(coloraxis = {'colorscale': HISTOGRAM_COLORMAP, "cmin": vmin, "cmax": vmax})
    
    return fig


def plot_var(fig, draw, vstats, var, cbar_edges, nbins=30, row=None, col=None):
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    values = draw.points[:, var].flatten()
    bin_range = vstats.p95_range
    #bin_range = np.min(values), np.max(values)
    showscale = (row == 0 and col == 0)
    showscale = True
    traces = _make_logp_histogram(values, draw.logp, nbins, bin_range,
                         draw.weights, cbar_edges=cbar_edges, showscale=showscale)

    fig.add_traces(traces, rows=row, cols=col)
    _decorate_histogram(vstats, fig, row=row, col=col)


def _decorate_histogram(vstats, fig, col=None, row=None):
    import plotly.graph_objects as go

    l95, h95 = vstats.p95_range
    l68, h68 = vstats.p68_range
    # Shade things inside 1-sigma
    assert isinstance(fig, go.Figure)
    fig.add_vrect(x0=l68, x1=h68, fillcolor='gold', opacity=0.5, layer='below', line={"width": 0}, col=col, row=row)

    def marker(symbol, position, info_template):
        if position < l95:
            symbol, position, ha = '<'+symbol, l95, 'left'
        elif position > h95:
            symbol, position, ha = '>'+symbol, h95, 'right'
        else:
            symbol, position, ha = symbol, position, 'center'
        fig.add_annotation(
            xref="x",
            yref="y domain",
            x=position,
            y=0.95,
            text=symbol,
            showarrow=False,
            font=dict(color=ANNOTATION_COLOR),
            col=col,
            row=row,
            hovertext=info_template.format(position=position)
        )

        #pylab.axvline(v)

    marker('|', vstats.median, "median: {position}")
    marker('E', vstats.mean, "mean: {position}")
    marker('*', vstats.best, "best: {position}")
    
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x = 0.0,
        y = 1.1,
        text = vstats.label,
        col=col,
        row=row,
        showarrow=False
    )

def _make_logp_histogram(values, logp, nbins, ci, weights, cbar_edges, showscale=False):
    from numpy import (ones_like, searchsorted, linspace, cumsum, diff,
                       unique, argsort, array, hstack, exp)
    import plotly.graph_objects as go

    if weights is None:
        weights = ones_like(logp)
    # TODO: values are being sorted to collect stats and again to plot
    idx = argsort(values)
    values, weights, logp = values[idx], weights[idx], logp[idx]
    #open('/tmp/out','a').write("ci=%s, range=%s\n"
    #                           % (ci,(min(values),max(values))))
    edges = linspace(ci[0], ci[1], nbins+1)
    bin_index = range(nbins)
    idx = searchsorted(values[1:-1], edges)


    bins = []  # marginalized maximum likelihood

    # H, xedges, yedges = np.histogram2d(values, -logp, bins=(edges, cbar_edges), weights=weights)
    # ybins = len(yedges) - 1
    # x = (xedges[:-1] + xedges[1:]) / 2.0
    # cbar_values = ((cbar_edges[:-1] + cbar_edges[1:]) / 2.0)[::-1]
    # x = x.repeat(ybins) # match length of y array
    # y = H.flatten(order="C")
    # color = np.tile(cbar_values, nbins)

    # # filter out empty y values (speeds up drawing, omitting empty rectangles):
    # y_nonzero = (y > 0)
    # # print(f"y_zeros: {y_nonzero}, {len([z for z in y_nonzero if z])} out of {len(y)}")
    # x = x[y_nonzero]
    # color = color[y_nonzero]
    # y = y[y_nonzero]
    # traces.append(go.Bar(x=x, y=y, showlegend=False, marker=dict(color=color, coloraxis='coloraxis', line=dict(width=0))))
    # # traces.append(go.Bar(x=x, y=y, showlegend=False, hoverinfo='skip', marker=dict(color=color, coloraxis='coloraxis', line=dict(width=0))))

    xscatt = []
    yscatt = []
    zscatt = []

    for i, s, e, xlo, xhi \
            in zip(bin_index, idx[:-1], idx[1:], edges[:-1], edges[1:]):
        if s == e:
            continue

        pv = -logp[s:e]
        # weights for samples within interval
        pw = weights[s:e]
        # vertical colorbar top edges is the cumulative sum of the weights
        bin_height = pw.sum()

         # parameter interval endpoints
        x = array([xlo, xhi], 'd')
        xav = (xlo + xhi)/2
        # -logp values within interval, with sort index from low to high
        pv = -logp[s:e]
        pidx = argsort(pv)
        pv = pv[pidx]
        # weights for samples within interval, sorted
        pw = weights[s:e][pidx]
        # vertical colorbar top edges is the cumulative sum of the weights
        y_top = cumsum(pw)
        bin_height = y_top[-1]

        change_point = searchsorted(pv[1:-1], cbar_edges)
        tops = unique(hstack((change_point, len(pv)-1)))
        # For plotly Bar plot, y values should be [size along y] (stacked)
        yy = np.diff(hstack((0, y_top[tops])))
        zz = pv[tops][:, None].flatten()
        xx = np.ones(tops.shape, dtype=float) * xav
        
        xscatt.extend(xx.tolist())
        yscatt.extend(yy.tolist())
        zscatt.extend(zz.tolist())
        
        # centerpoint, histogram height, maximum likelihood for each bin
        bin_max_likelihood = exp(cbar_edges[0] - pv[0])
        bins.append(((xlo+xhi)/2, bin_height, bin_max_likelihood))

    bar_trace = go.Bar(x=xscatt, y=yscatt, showlegend=False, hoverinfo='skip', marker=dict(color=zscatt, coloraxis='coloraxis', line=dict(width=0)))
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
    
    scatter_trace = go.Scatter(x=centers, y=maxlikelihood, hoverinfo='skip', showlegend=False, line={"color": LINE_COLOR})
    # return dict(bar_trace=bar_trace, scatter_trace=scatter_trace)
    return [bar_trace, scatter_trace]

    ## plot marginal gaussian approximation along with histogram
    #def G(x, mean, std):
    #    return np.exp(-((x-mean)/std)**2/2)/np.sqrt(2*np.pi*std**2)
    ## TODO: use weighted average for standard deviation
    #mean, std = np.average(values, weights=weights), np.std(values, ddof=1)
    #pdf = G(centers, mean, std)
    #pylab.plot(centers, pdf*np.sum(height)/np.sum(pdf), '-b')


def test():
    import pickle
    test_data = pickle.loads(open("./varplot.pickle", 'rb').read())
    draw = test_data['draw']
    stats = test_data['stats']
    import time
    start_time = time.time()
    fig = plot_vars(draw, stats)
    print(f"generating plot took: {time.time() - start_time} seconds")
    # print(fig)
    # print(test_data.keys())
    fig.write_json(open("varplot.json", "w"))
    fig.write_html(open("varplot.html", "w"))
    fig.show()

if __name__ == '__main__':
    test()