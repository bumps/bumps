"""
Build nested trace plots (plotly)
Adapted from https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/plotting.py
"""

__all__ = ["plot_nested_trace"]

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.ndimage import gaussian_filter as norm_kde
from scipy.stats import gaussian_kde
from dynesty.utils import quantile as _quantile
from dynesty.utils import get_random_generator, resample_equal


def plot_truth(fig,
                axes_indices,
                truths,
                truth_color,
                vertical=None,
                horizontal=None):
    """
    Plot the truth line (horizontal or vertical).
    truths can be None or one value or a list
    """

    if truths is not None:
        try:
            curt = iter(truths)
        except TypeError:
            curt = [truths]
        for t in curt:
            if vertical:
                func = fig.add_vline(x=t, line_color=truth_color, **axes_indices)
            elif horizontal:
                func = fig.add_hline(y=t, line_color=truth_color, **axes_indices)
            else:
                raise ValueError('vertical or horizontal option must be specified')


def check_span(span, samples, weights):
    """
    If span is a list of scalars, replace it by the list of bounds.
    If the input is list of pairs, it is kept intact
    """
    for i, _ in enumerate(span):
        try:
            iter(span[i])
            if len(span[i]) != 2:
                raise ValueError('Incorrect span value')
        except TypeError:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)


def plot_nested_trace(results,
                    span=None,
                    quantiles=(0.025, 0.5, 0.975),
                    smooth=0.02,
                    thin=1,
                    dims=None,
                    post_color='blue',
                    kde=True,
                    nkde=1000,
                    trace_cmap='plasma',
                    trace_color=None,
                    connect=False,
                    connect_highlight=10,
                    connect_color='red',
                    show_titles=True,
                    title_quantiles=(0.025, 0.5, 0.975),
                    truths=None,
                    truth_color='black'):
    """
    Plot traces and marginalized posteriors for each parameter.

    Args:
        results (dict) : The results from a nested sampling run.
        span (iterable, optional) : A list where each element is either a length-2 tuple containing
            lower and upper bounds or a float from `(0., 1.]` giving the
            fraction of (weighted) samples to include. If a fraction is provided,
            the bounds are chosen to be equal-tailed. An example would be::

                span = [(0., 10.), 0.95, (5., 6.)]

            Default is `0.999999426697` (5-sigma credible interval) for each parameter.
        quantiles (iterable, optional) : A list of fractional quantiles to overplot on the 
            1-D marginalized posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
            (the 95%/2-sigma credible interval).
        smooth (float or iterable, optional) : The standard deviation (either a single value or 
            a different value for each subplot) for the Gaussian kernel used to smooth the 1-D
            marginalized posteriors, expressed as a fraction of the span.
            Default is `0.02` (2% smoothing). If an integer is provided instead,
            this will instead default to a simple (weighted) histogram with `bins=smooth`.
        thin (int, optional) : Thin the samples so that only each `thin`-th sample is plotted.
            Default is `1` (no thinning).
        dims (iterable, optional) : The subset of dimensions that should be plotted. 
            If not provided, all dimensions will be shown.
        post_color (str, optional) : A color (either a single color or a different
            value for each subplot) used when plotting the histograms. Default is `'blue'`.
        kde (bool, optional) : Whether to use kernel density estimation to estimate and plot
            the PDF of the importance weights as a function of log-volume (as opposed to the 
            importance weights themselves). Default is `True`.
        nkde (int, optional) : The number of grid points used when plotting the kernel density estimate.
            Default is `1000`.
        trace_cmap (str, optional) : A colormap used when plotting the traces,
            Default is `'plasma'`.
        trace_color (str, optional) : A color used when plotting the traces. This overrides the `trace_cmap` 
            option by giving all points the same color. 
            Default is `None` (not used).
        connect (bool, optional) : Whether to draw lines connecting the paths of unique particles.
            Default is `False`.
        connect_highlight (int or iterable, optional) : If `connect=True`, highlights the paths 
            of a specific set of particles. If an integer is passed, :data:`connect_highlight`
            random particle paths will be highlighted. If an iterable is passed,
            then the particle paths corresponding to the provided indices will be highlighted.
        connect_color (str, optional) : The color of the highlighted particle paths. Default is `'red'`.
        show_titles (bool, optional) : Whether to display a title above each 1-D marginalized posterior
            showing the 0.5 quantile along with the upper/lower bounds associated
            with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
            Default is `True`.
        title_quantiles (iterable, optional) : A list of fractional quantiles to use in the title. 
            Default is `[0.025, 0.5, 0.975]` (median plus 95%/2-sigma credible interval).
        truths (iterable, optional) : A list of reference values that will be overplotted on the traces and
            marginalized 1-D posteriors as solid horizontal/vertical lines. Individual values can be exempt 
            using `None`. 
            Default is `None`.
        truth_color (str, optional) : A color used when plotting `truths`. Default is `'black'`.

    Returns:
        plotly.graph_objects.Figure : Output trace plot.

    """

    # Set defaults.
    rstate = get_random_generator()
    # Extract weighted samples.
    samples = results['samples']
    logvol = results['logvol']
    weights = results['importance_weights']

    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, weights, rstate=rstate))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
        wts = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate
    else:
        wts = weights

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Check ln(volume).
    if logvol.ndim != 1:
        raise ValueError("Ln(volume)'s must be 1-D.")
    if nsamps != logvol.shape[0]:
        raise ValueError("The number of ln(volume)'s and samples disagree!")

    # Check sample IDs.
    if connect:
        if 'samples_id' in results.keys():
            samples_id = results['samples_id']
            uid = np.unique(samples_id)
        else:
            raise ValueError("Sample IDs are not defined!")
        try:
            ids = connect_highlight[0]
            ids = connect_highlight
        except Exception:
            ids = rstate.choice(uid, size=connect_highlight, replace=False)

    # Determine plotting bounds for marginalized 1-D posteriors.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    check_span(span, samples, weights)

    # Setting up labels.
    labels = [r"$x_{" + str(i + 1) + "}$" for i in range(ndim)]

    # Setting up smoothing.
    if isinstance(smooth, (int, float)):
        smooth = [smooth for i in range(ndim)]

    # Setting up default plot layout.
    titles = [" "] * (ndim * 2)
    fig = make_subplots(rows=ndim, cols=2, column_widths=[0.7, 0.3], subplot_titles=titles)
    fig.update_layout(showlegend=False)

    max_n_ticks = 5
    # Plotting.
    for i, x in enumerate(samples):
        # Set trace color(s)
        if trace_color is not None:
            color = trace_color
        else:
            color = wts[::thin]
        # Setup axes.
        fig.update_xaxes(title_text=r"$-\ln X$", range=[0., -min(logvol)], nticks=max_n_ticks, row=i+1, col=1)
        fig.update_yaxes(title_text=labels[i], range=[min(x), max(x)], nticks=max_n_ticks, row=i+1, col=1)
        
        # Plot trace.
        fig.add_trace(
            go.Scatter(x=-logvol[::thin], y=x[::thin], mode='markers', marker={'color': color, 'colorscale': trace_cmap}),
            row=i+1, col=1
        )
        
        if connect:
            # Add lines highlighting specific particle paths.
            for j in ids:
                sel = samples_id[::thin] == j
                fig.add_trace(go.Scatter(x=-logvol[::thin][sel], y=x[::thin][sel], mode='lines', line=dict(width=1, color=connect_color)),
                              row=i+1, col=1
                        )
        
        # Add truth value(s).
        if truths is not None:
            plot_truth(fig, 
                        {'row':i+1, 'col':1},
                        truths[i],
                        truth_color,
                        horizontal=True)
        
        # Set color(s).
        if isinstance(post_color, str):
            color = post_color
        else:
            color = post_color[i]
        
        # Setup axes
        fig.update_xaxes(title_text=labels[i], range=span[i], nticks=max_n_ticks, row=i+1, col=2)
        fig.update_yaxes(showticklabels=False, row=i+1, col=2)
        
        # Generate distribution.
        s = smooth[i]
        # Plot marginalized 1-D posterior.
        if isinstance(s, int):
            # If `s` is an integer, plot a weighted histogram with
            # `s` bins within the provided bounds.
   
            n, b = np.histogram(x, bins=s, weights=weights, range=np.sort(span[i])) 
            fig.add_trace(
                    go.Bar(x=b[:-1], y=n, width=b[1]-b[0], offset=0, color=color, opacity=0.6),
                    row=i+1, col=2
                    )

            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / s))
            n, b = np.histogram(x,
                                bins=bins,
                                weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            x0 = 0.5 * (b[1:] + b[:-1])
            y0 = n
            fig.add_trace(
            go.Scatter(x=x0, y=y0, mode='lines', line=dict(width=1, color=color), fill='tonexty'),
            row=i+1, col=2
            )
            fig.update_yaxes(range=[0., max(y0) * 1.05], row=i+1, col=2)
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                fig.add_vline(x=q, line_width=2, line_dash="dash", line_color=color, row=i+1, col=2)
        # Add truth value(s).
        if truths is not None:
            plot_truth(fig, 
                        {'row':i+1, 'col':2},
                        truths[i],
                        truth_color,
                        vertical=True)
        # Set titles.
        if show_titles:
            ql, qm, qh = _quantile(x, title_quantiles, weights=weights)
            q_minus, q_plus = qm - ql, qh - qm
            fmt = "{0:.2f}".format
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
            title = "{0} = {1}".format(labels[i][:-1], title[1:])
            fig.layout.annotations[2*i + 1].text = title
    return fig
