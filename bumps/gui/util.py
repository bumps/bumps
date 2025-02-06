"""
Wx-Pylab magic for displaying plots within an application's window.
"""

from math import log10, floor
import string

import wx

import numpy as np


class EmbeddedPylab(object):
    """
    Define a 'with' context manager that lets you use pylab commands to
    plot on an embedded canvas.  This is useful for wrapping existing
    scripts in a GUI, and benefits from being more familiar than the
    underlying object oriented interface.

    As a convenience, the pylab module is returned on entry.

    Example
    -------

    The following example shows how to use the WxAgg backend in a wx panel::

        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
        from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as Toolbar
        from matplotlib.figure import Figure

        class PlotPanel(wx.Panel):
            def __init__(self, *args, **kw):
                wx.Panel.__init__(self, *args, **kw)

                figure = Figure(figsize=(1,1), dpi=72)
                canvas = FigureCanvas(self, wx.ID_ANY, figure)
                self.pylab_interface = EmbeddedPylab(canvas)

                # Instantiate the matplotlib navigation toolbar and explicitly show it.
                mpl_toolbar = Toolbar(canvas)
                mpl_toolbar.Realize()

                # Create a vertical box sizer to manage the widgets in the main panel.
                sizer = wx.BoxSizer(wx.VERTICAL)
                sizer.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=0)
                sizer.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=0)

                # Associate the sizer with its container.
                self.SetSizer(sizer)
                sizer.Fit(self)

            def plot(self, *args, **kw):
                with self.pylab_interface as pylab:
                    pylab.clf()
                    pylab.plot(*args, **kw)

    Similar patterns should work for the other backends.  Check the source code
    in matplotlib.backend_bases for examples showing how to use matplotlib
    with other GUI toolkits.
    """

    def __init__(self, canvas):
        # delay loading pylab until matplotlib.use() is called
        from matplotlib.backend_bases import FigureManagerBase

        self.fm = FigureManagerBase(canvas, -1)

    def __enter__(self):
        # delay loading pylab until matplotlib.use() is called
        import pylab
        from matplotlib._pylab_helpers import Gcf

        # Note: don't need to track and restore the current active since it
        # will automatically be restored when we pop the current figure.
        Gcf.set_active(self.fm)
        return pylab

    def __exit__(self, *args, **kw):
        # delay loading pylab until matplotlib.use() is called
        from matplotlib._pylab_helpers import Gcf

        if hasattr(Gcf, "_activeQue"):  # CRUFT: MPL < 3.3.1
            Gcf._activeQue = [f for f in Gcf._activeQue if f is not self.fm]
            try:
                del Gcf.figs[-1]
            except KeyError:
                pass
        else:
            Gcf.figs.pop(self.fm.num, None)


class Validator(wx.PyValidator):
    def __init__(self, flag):
        wx.PyValidator.__init__(self)
        self.flag = flag
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def Clone(self):
        return Validator(self.flag)

    def Validate(self, win):
        return True

    def TransferToWindow(self):
        return True

    def TransferFromWindow(self):
        return True

    def OnChar(self, evt):
        key = chr(evt.GetKeyCode())
        if self.flag == "no-alpha" and key in string.letters:
            return
        if self.flag == "no-digit" and key in string.digits:
            return
        evt.Skip()


def nice(v, digits=4):
    """Fix v to a value with a given number of digits of precision"""
    if v == 0.0 or not np.isfinite(v):
        return v
    else:
        sign = v / abs(v)
        place = floor(log10(abs(v)))
        scale = 10 ** (place - (digits - 1))
        return sign * floor(abs(v) / scale + 0.5) * scale
