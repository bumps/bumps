"""
Signals changes to the model that need to be reflected in the views.

In practice, the main window is the only listener, and it forwards the
messages to the appropriate views.
"""
import wx
from wx.py.dispatcher import send, connect

def model_new(model):
    """
    Inform all views that a new model is available.
    """
    wx.CallAfter(send,'model.new',model=model)

def update_model(model, dirty=True):
    """
    Inform all views that the model structure has changed.  This calls
    model.model_reset() to reset the fit parameters and constraints.
    """
    model.model_reset()  #
    if dirty: model.model_update()
    wx.CallAfter(send,'model.update_structure',model=model)

_DELAYED_SIGNAL = {}
def update_parameters(model, delay=100):
    """
    Inform all views that the model has changed.  Note that if the model
    is changing rapidly, then the signal will be delayed for a time.  This
    calls model.model_update() to let the model know that it needs to be
    recalculated.
    """
    # signaller is responsible for marking the model as needing recalculation
    model.model_update()
    try:
        _DELAYED_SIGNAL[model].Restart(delay)
    except:
        def _send_signal():
            #print "sending update parameters",model
            del _DELAYED_SIGNAL[model]
            wx.CallAfter(send,'model.update_parameters',model=model)
        _DELAYED_SIGNAL[model] = wx.FutureCall(delay, _send_signal)

def log_message(message):
    wx.CallAfter(send,'log',message=message)
