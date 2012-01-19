import wx
import string

# Wx-Pylab magic for displaying plots within an application's window.
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase

class EmbeddedPylab(object):
    """
    Define a 'with' context manager that lets you use pylab commands to
    plot on an embedded canvas.  This is useful for wrapping existing
    scripts in a GUI, and benefits from being more familiar than the
    underlying object oriented interface.
    """
    def __init__(self, canvas):
        self.fm = FigureManagerBase(canvas, -1)
    def __enter__(self):
        Gcf.set_active(self.fm)
    def __exit__(self, *args, **kw):
        Gcf._activeQue = [f for f in Gcf._activeQue if f is not self.fm]
        try: del Gcf.figs[-1]
        except: pass

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

def nice(v, digits = 4):
    from math import log10, floor
    """Fix v to a value with a given number of digits of precision"""
    try:
        sign = v/abs(v)
        place = floor(log10(abs(v)))
        scale = 10**(place-(digits-1))
        return sign*floor(abs(v)/scale+0.5)*scale
    except:  # Too many possible math errors to struggle through
        return v
