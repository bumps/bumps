"""
Build layout for histogram plots (plotly)
"""

__all__ = ["var_plot_size", "plot_vars", "plot_var"]

from math import ceil, sqrt
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from bumps.dream.state import Draw
    from bumps.dream.stats import VarStats
    import plotly.graph_objects as go

# TODO: improve type hinting on VarStats (dataclass with optional fields?)

# # Set space between plots in horiz and vert.
# H_SPACE = 0.2
# V_SPACE = 0.2

# # Set top, bottom, left margins.
# T_MARGIN = 0.2
# B_MARGIN = 0.2
# L_MARGIN = 0.2
# R_MARGIN = 0.4

# # Set desired plot sizes.
# TILE_W = 3.0
# TILE_H = 2.0
# CBAR_WIDTH = 0.75

CBAR_COLORS = 64
LINE_COLOR = "red"
ANNOTATION_COLOR = "blue"
HISTOGRAM_COLORMAP = "Cividis"


def tile_axes_square(n):
    """
    Determine number of columns by finding the
    next greatest square, then determine number
    of rows needed.
    """
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / float(cols)))
    return cols, rows


def plot_vars(draw: "Draw", all_vstats, cbar_colors=CBAR_COLORS, **kw):
    from plotly.subplots import make_subplots

    n = len(all_vstats)
    ncol, nrow = tile_axes_square(n)

    snllf = np.sort(-draw.logp)
    vmin, vmax = snllf[0], snllf[int(0.98 * (len(snllf) - 1))]  # robust range
    cbar_edges = np.linspace(vmin, vmax, cbar_colors)
    # titles = [vstats.label for vstats in all_vstats]

    # fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles)
    fig = make_subplots(rows=nrow, cols=ncol)
    fig.update_yaxes(secondary_y=True)
    fig.update_xaxes(dict(exponentformat="e"), overwrite=True)
    fig.update_xaxes(showticklabels=True, showline=True, mirror=True, linewidth=1, linecolor="black", showgrid=False)
    fig.update_yaxes(showticklabels=False, showline=True, mirror=True, linewidth=1, linecolor="black", showgrid=False)
    fig.update_layout(coloraxis={"colorscale": HISTOGRAM_COLORMAP, "cmin": vmin, "cmax": vmax})
    fig.update_layout(height=nrow * 300, width=ncol * 400)
    fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)")
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
    fig.update_layout(bargap=0)

    fig = fig.to_dict()
    fig["layout"].update(shapes=[], annotations=[])

    for k, vstats in enumerate(all_vstats):
        row = (k // ncol) + 1
        col = (k % ncol) + 1
        subplot = k + 1
        plot_var(fig, draw, vstats, k, cbar_edges=cbar_edges, subplot=subplot, **kw)

    return fig


def plot_var(
    fig: "go.Figure", draw: "Draw", vstats: "VarStats", var: int, cbar_edges: np.ndarray, nbins=30, subplot=None
):
    values = draw.points[:, var].flatten()
    bin_range = vstats.p95_range
    sort_index = draw.get_argsort_indices(var)
    # bin_range = np.min(values), np.max(values)
    # showscale = (subplot == 1)
    showscale = True
    traces = _make_logp_histogram(
        values,
        draw.logp,
        nbins,
        bin_range,
        draw.weights,
        sort_index,
        cbar_edges=cbar_edges,
        showscale=showscale,
        subplot=subplot,
    )

    fig["data"].extend(traces)
    _decorate_histogram(vstats, fig, subplot=subplot)


def _decorate_histogram(vstats: "VarStats", fig: dict, subplot: int = 1):
    import plotly.graph_objects as go

    xaxis, yaxis = subplot_axis_names(subplot)
    l95, h95 = vstats.p95_range
    l68, h68 = vstats.p68_range
    # Shade things inside 1-sigma

    shading_rect = dict(
        type="rect",
        x0=l68,
        x1=h68,
        y0=0,
        y1=1,
        fillcolor="lightblue",
        opacity=0.5,
        layer="below",
        line={"width": 0},
        xref=xaxis,
        yref=f"{yaxis} domain",
    )
    fig["layout"]["shapes"].append(shading_rect)

    def marker(symbol, position, info_template):
        if position < l95:
            symbol, position, ha = "<" + symbol, l95, "left"
        elif position > h95:
            symbol, position, ha = ">" + symbol, h95, "right"
        else:
            symbol, position, ha = symbol, position, "center"
        new_marker = dict(
            xref=xaxis,
            yref=f"{yaxis} domain",
            x=position,
            y=0.95,
            text=symbol,
            showarrow=False,
            font=dict(color=ANNOTATION_COLOR),
            hovertext=info_template.format(position=position, label=vstats.label),
        )
        fig["layout"]["annotations"].append(new_marker)
        # pylab.axvline(v)

    marker("|", vstats.median, "{label}<br>median: {position}")
    marker("E", vstats.mean, "{label}<br>mean: {position}")
    marker("*", vstats.best, "{label}<br>best: {position}")

    label_annotation = dict(
        xref=f"{xaxis} domain",
        yref=f"{yaxis} domain",
        x=0.0,
        y=1.1,
        text=vstats.label,
        showarrow=False,
        name="label",
    )
    fig["layout"]["annotations"].append(label_annotation)


def _make_logp_histogram(values, logp, nbins, ci, weights, idx, cbar_edges, showscale=False, subplot=None):
    from numpy import ones_like, searchsorted, linspace, cumsum, diff, unique, argsort, array, hstack, exp

    if weights is None:
        weights = ones_like(logp)
    # use sorting index calculated during stats collection:
    values, weights, logp = values[idx], weights[idx], logp[idx]
    # open('/tmp/out','a').write("ci=%s, range=%s\n"
    #                           % (ci,(min(values),max(values))))
    edges = linspace(ci[0], ci[1], nbins + 1)
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

    for i, s, e, xlo, xhi in zip(bin_index, idx[:-1], idx[1:], edges[:-1], edges[1:]):
        if s == e:
            continue

        pv = -logp[s:e]
        # weights for samples within interval
        pw = weights[s:e]
        # vertical colorbar top edges is the cumulative sum of the weights
        bin_height = pw.sum()

        # parameter interval endpoints
        x = array([xlo, xhi], "d")
        xav = (xlo + xhi) / 2
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
        tops = unique(hstack((change_point, len(pv) - 1)))
        # For plotly Bar plot, y values should be [size along y] (stacked)
        yy = np.diff(hstack((0, y_top[tops])))
        zz = pv[tops][:, None].flatten()
        xx = np.ones(tops.shape, dtype=float) * xav

        xscatt.extend(xx.tolist())
        yscatt.extend(yy.tolist())
        zscatt.extend(zz.tolist())

        # centerpoint, histogram height, maximum likelihood for each bin
        bin_max_likelihood = exp(cbar_edges[0] - pv[0])
        bins.append(((xlo + xhi) / 2, bin_height, bin_max_likelihood))

    xaxis, yaxis = subplot_axis_names(subplot)
    bar_trace = dict(
        type="bar",
        x=xscatt,
        y=yscatt,
        showlegend=False,
        hoverinfo="skip",
        marker=dict(color=zscatt, coloraxis="coloraxis", line=dict(width=0)),
        xaxis=xaxis,
        yaxis=yaxis,
    )
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

    scatter_trace = dict(
        type="scatter",
        x=centers,
        y=maxlikelihood,
        hoverinfo="skip",
        showlegend=False,
        line={"color": LINE_COLOR},
        xaxis=xaxis,
        yaxis=yaxis,
    )
    # return dict(bar_trace=bar_trace, scatter_trace=scatter_trace)
    return [bar_trace, scatter_trace]

    ## plot marginal gaussian approximation along with histogram
    # def G(x, mean, std):
    #    return np.exp(-((x-mean)/std)**2/2)/np.sqrt(2*np.pi*std**2)
    ## TODO: use weighted average for standard deviation
    # mean, std = np.average(values, weights=weights), np.std(values, ddof=1)
    # pdf = G(centers, mean, std)
    # pylab.plot(centers, pdf*np.sum(height)/np.sum(pdf), '-b')


def subplot_axis_names(subplot: int):
    if subplot == 1:
        return "x", "y"
    else:
        return f"x{subplot}", f"y{subplot}"
