"""
Pylab plotting utilities.
"""

__all__ = [
    "config_matplotlib",
    "auto_shift",
    "coordinated_colors",
    "dhsv",
    "next_color",
    "plot_quantiles",
    "form_quantiles",
]


def set_mplconfig(appdatadir):
    r"""
    Point the matplotlib config dir to %LOCALAPPDATA%\{appdatadir}\mplconfig.
    """
    import os
    import sys

    if hasattr(sys, "frozen"):
        if os.name == "nt":
            mplconfigdir = os.path.join(os.environ["LOCALAPPDATA"], appdatadir, "mplconfig")
        elif sys.platform == "darwin":
            mplconfigdir = os.path.join(os.path.expanduser("~/Library/Caches"), appdatadir, "mplconfig")
        else:
            return  # do nothing on linux
        mplconfigdir = os.environ.setdefault("MPLCONFIGDIR", mplconfigdir)
        if not os.path.exists(mplconfigdir):
            os.makedirs(mplconfigdir)


def config_matplotlib(backend=None):
    """
    Setup matplotlib to use a particular backend.

    The backend should be 'WXAgg' for interactive use, or 'Agg' for batch.
    This distinction allows us to run in environments such as cluster computers
    which do not have wx installed on the compute nodes.

    This function must be called before any imports to matplotlib.  To allow
    this, modules should not import matplotlib at the module level, but instead
    import it for each function/method that uses it.  Exceptions can be made
    for modules which are completely dedicated to plotting, but these modules
    should never be imported at the module level.
    """
    import os
    import sys
    import matplotlib as mpl

    # When running from a frozen environment created by py2exe, we will not
    # have a range of backends available, and must set the default to WXAgg.
    # With a full matplotlib distribution we can use whatever the user prefers.
    if hasattr(sys, "frozen"):
        if "MPLCONFIGDIR" not in os.environ:
            raise RuntimeError(r"MPLCONFIGDIR should be set to e.g., %LOCALAPPDATA%\YourApp\mplconfig")
        if backend is None:
            backend = "WXAgg"

    ## CRUFT: check that backend is valid, trying alternates if an import fails
    # if backend is None:
    #    backend = os.environ.get('MPLBACKEND', mpl.rcParams['backend'])
    # import importlib
    # for name in (backend, 'MacOSX', 'Qt5Agg', 'Qt4Agg', 'Gtk3Agg', 'TkAgg', 'WXAgg'):
    #    path = 'matplotlib.backends.backend_' + name.lower()
    #    try:
    #        importlib.import_module(path)
    #        backend = name
    #        break
    #    except ImportError:
    #        backend = None

    # Specify the backend to use for plotting and import backend dependent
    # classes.  This must be done before importing pyplot to have an
    # effect.  If no backend is given, let pyplot use the default.
    if backend is not None:
        mpl.use(backend)

    # Disable interactive mode so that plots are only updated on show() or
    # draw(). The interactive function must be called before importing pyplot,
    # otherwise it will have no effect.
    mpl.interactive(False)

    # configure the plot style
    line_width = 1
    pad = 2
    font_family = "Arial" if os.name == "nt" else "sans-serif"
    font_size = 12
    plot_style = {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": line_width,
        "axes.linewidth": line_width,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": line_width,
        "ytick.major.width": line_width,
        "xtick.minor.width": line_width,
        "ytick.minor.width": line_width,
        "xtick.major.pad": pad,
        "ytick.major.pad": pad,
        "xtick.top": True,
        "ytick.right": True,
        "font.size": font_size,
        "font.family": font_family,
        "svg.fonttype": "none",
        "savefig.dpi": 100,
    }
    mpl.rcParams.update(plot_style)


def auto_shift(offset):
    """
    Return a y-offset coordinate transform for the current axes.

    Each call to auto_shift increases the y-offset for the next line by
    the given number of points (with 72 points per inch).

    Example::

        from matplotlib import pyplot as plt
        from bumps.plotutil import auto_shift
        trans = auto_shift(plt.gca())
        plot(x, y, trans=trans)
    """
    from matplotlib.transforms import ScaledTranslation
    import matplotlib.pyplot as plt

    ax = plt.gca()
    if ax.lines and hasattr(ax, "_auto_shift"):
        ax._auto_shift += offset
    else:
        ax._auto_shift = 0
    trans = plt.gca().transData
    if ax._auto_shift:
        trans += ScaledTranslation(0, ax._auto_shift / 72.0, plt.gcf().dpi_scale_trans)
    return trans


# ======== Color functions ========


def next_color(axes=None):
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
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    lines = axes._get_lines
    try:
        base = lines.get_next_color()
    except Exception:
        try:  # Cruft 1.7 - 3.7?
            base = next(lines.prop_cycler)["color"]
        except Exception:
            try:  # Cruft 1.4-1.6?
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
    return dict(
        base=base,
        light=dhsv(base, dv=+0.3, ds=-0.2),
        dark=dhsv(base, dv=-0.25, ds=+0.35),
    )


def dhsv(color, dh=0.0, ds=0.0, dv=0.0, da=0.0):
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
    s, v, a = [clip(val, 0.0, 1.0) for val in (s + ds, v + dv, a + da)]
    h = fmod(h + dh, 1.0)
    r, g, b = hsv_to_rgb(h, s, v)
    # print "to color",r,g,b,a
    return array((r, g, b, a))


# ==== Specialized plotters =====


def plot_quantiles(x, y, contours, color, alpha=None, axes=None):
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
    _plot_quantiles(x, q, color, alpha, axes=axes)


def _plot_quantiles(x, q, color, alpha, axes=None):
    import matplotlib.pyplot as plt

    # print "p",p
    # print "q",q[:,:,0]
    # print "y",y[:,0]
    if axes is None:
        axes = plt.gca()
    if alpha is None:
        alpha = 2.0 / (len(q) + 1)
    edgecolor = dhsv(color, ds=-(1 - alpha), dv=(1 - alpha))
    for lo, hi in q:
        axes.fill_between(x, lo, hi, facecolor=color, edgecolor=edgecolor, alpha=alpha)


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

    p = _convert_contours_to_probabilities(contours)
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
