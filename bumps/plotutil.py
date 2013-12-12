"""
Pylab plotting utilities.
"""
from __future__ import division

__all__ = ["auto_shift", "next_color", "coordinated_colors", "dhsv",
           "plot_quantiles"]

def auto_shift(offset):
    from matplotlib.transforms import ScaledTranslation
    import pylab
    ax = pylab.gca()
    if ax.lines and hasattr(ax, '_auto_shift'):
        ax._auto_shift += offset
    else:
        ax._auto_shift = 0
    trans = pylab.gca().transData
    if ax._auto_shift:
        trans += ScaledTranslation(0,ax._auto_shift/72,
                                   pylab.gcf().dpi_scale_trans)
    return trans

def next_color():
    import pylab
    try:
        base = pylab.gca()._get_lines.color_cycle.next()
    except: # Cruft 1.3 and earlier
        base = pylab.gca()._get_lines._get_next_cycle_color()
    return base

def coordinated_colors(base=None):
    if base is None: base = next_color()
    return dict(base=base,
                light = dhsv(base, dv=+0.3, ds=-0.2),
                dark = dhsv(base, dv=-0.25, ds=+0.35),
                )

# Color functions
def dhsv(color, dh=0, ds=0, dv=0, da=0):
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

        >>> darker = dhsv('seagreen', dv=-0.1)
        >>> print [int(v*255) for v in darker]
        [37, 113, 71, 255]
    """
    from matplotlib.colors import colorConverter
    from colorsys import rgb_to_hsv, hsv_to_rgb
    from numpy import clip, array, fmod
    r,g,b,a = colorConverter.to_rgba(color)
    #print "from color",r,g,b,a
    h,s,v = rgb_to_hsv(r,g,b)
    s,v,a = [clip(val,0.,1.) for val in s+ds,v+dv,a+da]
    h = fmod(h+dh,1.)
    r,g,b = hsv_to_rgb(h,s,v)
    #print "to color",r,g,b,a
    return array((r,g,b,a))


# ==== Quantiles plotter =====

def plot_quantiles(x, y, contours, color, alpha=None):
    import pylab
    import numpy
    from scipy.stats.mstats import mquantiles
    p = _convert_contours_to_probabilities(reversed(sorted(contours)))
    q = mquantiles(y, prob = p, axis=0)
    q = numpy.reshape(q, (-1, 2, len(x)))
    #print "p",p
    #print "q",q[:,:,0]
    #print "y",y[:,0]
    if alpha is None: alpha = 2./(len(contours) + 1)
    edgecolor = dhsv(color, ds = -(1-alpha), dv = (1-alpha))
    for lo,hi in q:
        pylab.fill_between(x, lo, hi,
                           facecolor=color, edgecolor=edgecolor,
                           alpha=alpha, hold=True)

def _convert_contours_to_probabilities(contours):
    """
    given [a,b,c] return [100-a, a, 100-b, b, 100-c, c]/100
    """
    import numpy
    return numpy.hstack( [(100.-v, v) for v in contours] )/100
