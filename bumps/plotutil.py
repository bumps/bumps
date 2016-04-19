"""
Pylab plotting utilities.
"""
from __future__ import division

__all__ = ["auto_shift",
           "coordinated_colors", "dhsv", "next_color",
           "plot_quantiles", "form_quantiles"]


def auto_shift(offset):
    """
    Return a y-offset coordinate transform for the current axes.

    Each call to auto_shift increases the y-offset for the next line by
    the given number of points (with 72 points per inch).

    Example::

        from matplotlib import pyplot as plt
        from bumps.plotutil import auto_shift
        trans = auto_shift(plt.gca())
        plot(x, y, hold=True, trans=trans)
    """
    from matplotlib.transforms import ScaledTranslation
    import pylab
    ax = pylab.gca()
    if ax.lines and hasattr(ax, '_auto_shift'):
        ax._auto_shift += offset
    else:
        ax._auto_shift = 0
    trans = pylab.gca().transData
    if ax._auto_shift:
        trans += ScaledTranslation(0, ax._auto_shift/72.,
                                   pylab.gcf().dpi_scale_trans)
    return trans


# ======== Color functions ========

def next_color():
    """
    Return the next color in the plot color cycle.

    Example::

        from matplotlib import pyplot as plt
        from bumps.plotutil import next_color, dhsv
        color = next_color()
        plt.errorbar(x, y, yerr=dy, fmt='.', color=color)
        # Draw the theory line with the same color as the data, but darker
        plt.plot(x, y, '-', color=dhsv(color, dv=-0.2))
    """
    import pylab
    lines = pylab.gca()._get_lines
    try:
        base = next(lines.prop_cycler)['color']
    except Exception:
        try: # Cruft 1.4-1.6?
            base = next(lines.color_cycle)
        except Exception:  # Cruft 1.3 and earlier
            base = lines._get_next_cycle_color()
    return base


def coordinated_colors(base=None):
    """
    Return a set of coordinated colors as c['base|light|dark'].

    If *base* is not provided, use the next color in the color cycle as
    the base.  Light is bright and pale, dark is dull and saturated.
    """
    if base is None:
        base = next_color()
    return dict(base=base,
                light=dhsv(base, dv=+0.3, ds=-0.2),
                dark=dhsv(base, dv=-0.25, ds=+0.35),
                )


def dhsv(color, dh=0., ds=0., dv=0., da=0.):
    """
    Modify color on hsv scale.

    *dv* change intensity, e.g., +0.1 to brighten, -0.1 to darken.
    *dh* change hue
    *ds* change saturation
    *da* change transparency

    Color can be any valid matplotlib color.  The hsv scale is [0,1] in
    each dimension.  Saturation, value and alpha scales are clipped to [0,1]
    after changing.  The hue scale wraps between red to violet.

    :Example:

    Make sea green 10% darker:

        >>> from bumps.plotutil import dhsv
        >>> darker = dhsv('seagreen', dv=-0.1)
        >>> print([int(v*255) for v in darker])
        [37, 113, 71, 255]
    """
    from matplotlib.colors import colorConverter
    from colorsys import rgb_to_hsv, hsv_to_rgb
    from numpy import clip, array, fmod
    r, g, b, a = colorConverter.to_rgba(color)
    # print "from color",r,g,b,a
    h, s, v = rgb_to_hsv(r, g, b)
    s, v, a = [clip(val, 0., 1.) for val in (s + ds, v + dv, a + da)]
    h = fmod(h + dh, 1.)
    r, g, b = hsv_to_rgb(h, s, v)
    # print "to color",r,g,b,a
    return array((r, g, b, a))


# ==== Specialized plotters =====

def plot_quantiles(x, y, contours, color, alpha=None):
    """
    Plot quantile curves for a set of lines.

    *x* is the x coordinates for all lines.

    *y* is the y coordinates, one row for each line.

    *contours* is a list of confidence intervals expressed as percents.

    *color* is the color to use for the quantiles.  Quantiles are draw as
    a filled region with alpha transparency.  Higher probability regions
    will be covered with multiple contours, which will make them lighter
    and more saturated.

    *alpha* is the transparency level to use for all fill regions.  The
    default value, alpha=2./(#contours+1), works pretty well.
    """
    _, q = form_quantiles(y, contours)
    _plot_quantiles(x, q,  color, alpha)

def _plot_quantiles(x, q, color, alpha):
    import pylab
    # print "p",p
    # print "q",q[:,:,0]
    # print "y",y[:,0]
    if alpha is None:
        alpha = 2. / (len(q) + 1)
    edgecolor = dhsv(color, ds=-(1 - alpha), dv=(1 - alpha))
    for lo, hi in q:
        pylab.fill_between(x, lo, hi,
                           facecolor=color, edgecolor=edgecolor,
                           alpha=alpha, hold=True)

def form_quantiles(y, contours):
    """
    Return quantiles and values for a list of confidence intervals.

    *contours* is a list of confidence interfaces [a, b,...] expressed as
    percents.

    Returns:

    *quantiles* is a list of intervals [[a_low, a_high], [b_low, b_high], ...]
    in [0,1].

    *values* is a list of intervals [[A_low, A_high], ...] with one entry in
    A for each row in y.
    """
    from numpy import reshape
    from scipy.stats.mstats import mquantiles
    p = _convert_contours_to_probabilities(reversed(sorted(contours)))
    q = mquantiles(y, prob=p, axis=0)
    p = reshape(p, (2, -1))
    q = reshape(q, (-1, 2, len(y[0])))
    return p, q

def _convert_contours_to_probabilities(contours):
    """
    Given confidence intervals [a, b,...] as percents, return probability
    in [0,1] for each interval as [a_low, a_high, b_low, b_high, ...].
    """
    from numpy import hstack
    # lower quantile for ci in percent = (100 - ci)/2
    # upper quantile = 100 - lower quantile = 100 - (100-ci)/2 = (100 + ci)/2
    # divide by an additional 100 to get proportion from 0 to 1
    return hstack([(100.0 - p, 100.0 + p) for p in contours]) / 200.0
