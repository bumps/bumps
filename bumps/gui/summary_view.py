# Copyright (C) 2006-2011, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Nikunj Patel

"""
This module implements the Summary View panel.
"""

# ==============================================================================

import wx
import wx.lib.scrolledpanel as scrolled

from . import signal
from .util import nice

IS_MAC = wx.Platform == "__WXMAC__"

NUMPIX = 400
NUMTICKS = NUMPIX * 5 - 1
COMPACTIFY_VERTICAL = {
    "__WXMAC__": 2,
    "__WXGTK__": 6,
    "__WXMSW__": 4,
}.get(wx.Platform, 6)


class SummaryView(scrolled.ScrolledPanel):
    """
    Model view showing summary of fit (only fittable parameters).
    """

    title = "Summary"
    default_size = (600, 500)

    def __init__(self, *args, **kw):
        scrolled.ScrolledPanel.__init__(self, *args, **kw)

        self.display_list = []
        self.parameters = []

        self.sizer = wx.GridBagSizer(hgap=0, vgap=-COMPACTIFY_VERTICAL)
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        self.SetAutoLayout(True)
        self.SetupScrolling()

        self._need_update_parameters = self._need_update_model = False
        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnShow(self, event):
        # print "show event"
        if not event.Show:
            return
        # print "showing summary"
        if self._need_update_model:
            # print "-update_model"
            self.update_model(self.model)
        elif self._need_update_parameters:
            # print "-update_parameters"
            self.update_parameters(self.model)
        event.Skip()

    # ============ Operations on the model  ===============

    def get_state(self):
        return self.model

    def set_state(self, state):
        self.set_model(state)

    def set_model(self, model):
        self.model = model
        self.update_model(model)

    def update_model(self, model):
        if self.model != model:
            return

        if not IS_MAC and not self.IsShown():
            # print "summary not shown config"
            self._need_update_model = True
        else:
            # print "summary shown config"
            self._update_model()
            self._need_update_parameters = self._need_update_model = False

    def update_parameters(self, model):
        if self.model != model:
            return
        if not IS_MAC and not self.IsShown():
            # print "summary not shown update"
            self._need_update_parameters = True
        else:
            # print "summary shown upate"
            self._need_update_parameters = False
            self._update_parameters()

    def _update_model(self):
        # print "drawing"
        self.parameters = self.model._parameters if self.model is not None else []
        self.sizer.Clear(True)
        # self.sizer.Clear()
        self.display_list = []

        self.layer_label = wx.StaticText(self, wx.ID_ANY, "Fit Parameter", size=(160, -1))
        self.slider_label = wx.StaticText(self, wx.ID_ANY, "", size=(NUMPIX, -1))
        self.value_label = wx.StaticText(self, wx.ID_ANY, "Value", size=(100, -1))
        self.low_label = wx.StaticText(self, wx.ID_ANY, "Minimum", size=(100, -1))
        self.high_label = wx.StaticText(self, wx.ID_ANY, "Maximum", size=(100, -1))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.layer_label, 0, wx.LEFT, 1)
        hbox.Add(self.slider_label, 0, wx.LEFT, 1)
        hbox.Add(self.value_label, 0, wx.LEFT, 21)
        hbox.Add(self.low_label, 0, wx.LEFT, 1)
        hbox.Add(self.high_label, 0, wx.LEFT, 1)

        # Note that row at pos=(0,0) is not used to add a blank row.
        self.sizer.Add(hbox, pos=(1, 0), flag=wx.BOTTOM, border=COMPACTIFY_VERTICAL)

        line = wx.StaticLine(self, wx.ID_ANY)
        self.sizer.Add(line, pos=(2, 0), flag=wx.EXPAND | wx.RIGHT | wx.BOTTOM, border=COMPACTIFY_VERTICAL)

        # TODO: better interface to fittable parameters
        if self.model is not None:
            pars = self.model._parameters
            # pars = sorted(pars, cmp=lambda x,y: cmp(x.name, y.name))
            for p in pars:
                self.display_list.append(ParameterSummary(self, p, self.model))

        for index, item in enumerate(self.display_list):
            self.sizer.Add(item, pos=(index + 3, 0))

        self.SetupScrolling()
        self.Layout()

    def _update_parameters(self):
        # print "updating"
        if self.parameters is not self.model._parameters:
            self._update_model()
        else:
            for p in self.display_list:
                p.update_slider()


VALUE_PRECISION = 6
VALUE_FORMAT = "{{:.{:d}g}}".format(VALUE_PRECISION)


class ParameterSummary(wx.Panel):
    """Build one parameter line for display."""

    def __init__(self, parent, parameter, model):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.parameter = parameter
        self.model = model

        self.low, self.high = (v for v in self.parameter.prior.limits)

        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.layer_name = wx.StaticText(self, wx.ID_ANY, str(self.parameter.name), size=(160, -1), style=wx.TE_LEFT)
        self.slider = wx.Slider(
            self, wx.ID_ANY, value=0, minValue=0, maxValue=NUMPIX * 5 - 1, size=(NUMPIX, -1), style=wx.SL_HORIZONTAL
        )
        self.value = wx.StaticText(
            self, wx.ID_ANY, VALUE_FORMAT.format(self.parameter.value), size=(100, -1), style=wx.TE_LEFT
        )
        self.min_range = wx.StaticText(self, wx.ID_ANY, VALUE_FORMAT.format(self.low), size=(100, -1), style=wx.TE_LEFT)
        self.max_range = wx.StaticText(
            self, wx.ID_ANY, VALUE_FORMAT.format(self.high), size=(100, -1), style=wx.TE_LEFT
        )

        # Add text strings and slider to sizer.
        text_hbox_flags = wx.LEFT | wx.ALIGN_CENTER_VERTICAL
        text_hbox.Add(self.layer_name, 0, text_hbox_flags, 1)
        text_hbox.Add(self.slider, 0, text_hbox_flags, 1)
        text_hbox.Add(self.value, 0, text_hbox_flags, 21)
        text_hbox.Add(self.min_range, 0, text_hbox_flags, 1)
        text_hbox.Add(self.max_range, 0, text_hbox_flags, 1)

        self.SetSizer(text_hbox)

        self.slider.Bind(wx.EVT_SCROLL, self.OnScroll)
        self.update_slider()

    def update_slider(self):
        slider_pos = int(self.parameter.prior.get01(self.parameter.value) * NUMTICKS)
        # Add line below if get01 doesn't protect against values out of range.
        # slider_pos = min(max(slider_pos,0),100)
        self.slider.SetValue(slider_pos)
        self.value.SetLabel(VALUE_FORMAT.format(nice(self.parameter.value, digits=VALUE_PRECISION)))

        # Update new min and max range of values if changed.
        newlow, newhigh = (v for v in self.parameter.prior.limits)
        if newlow != self.low:
            self.min_range.SetLabel(VALUE_FORMAT.format(newlow))

        # if newhigh != self.high:
        self.max_range.SetLabel(VALUE_FORMAT.format(newhigh))

    def OnScroll(self, event):
        value = self.slider.GetValue()
        new_value = self.parameter.prior.put01(value / NUMTICKS)
        nice_new_value = nice(new_value, digits=VALUE_PRECISION)
        self.parameter.value = nice_new_value
        self.value.SetLabel(VALUE_FORMAT.format(nice_new_value))
        signal.update_parameters(model=self.model, delay=1)
