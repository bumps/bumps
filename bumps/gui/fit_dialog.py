#!/usr/bin/python
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
# Author: Nikunj Patel, James Krycka, Paul Kienzle

"""
This module implements the FitControl class which presents a pop-up dialog box
for the user to control fitting options.
"""

# TODO: reset button sets values to factory settings for current optimizer

#==============================================================================
import wx

from .. import fitters
from .input_list import InputListPanel

class FitControl(wx.Dialog):
    """
    FitControl lets the user set fitting options from a pop-up dialog box.
    """
    def __init__(self,
                 parent = None,
                 id     = wx.ID_ANY,
                 title  = "Fit Parameters",
                 pos    = wx.DefaultPosition,
                 size   = wx.DefaultSize, # dialog box size will be calculated
                 style  = wx.DEFAULT_DIALOG_STYLE,
                 name   = "",
                 plist  = None,
                 default_algo = None,
                 fontsize = None
                ):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)

        self.plist = plist
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        # Set the font for this window and all child windows (widgets) from the
        # parent window, or from the system defaults if no parent is given.
        # A dialog box does not inherit font info from its parent, so we will
        # explicitly get it from the parent and apply it to the dialog box.
        if parent is not None:
            font = parent.GetFont()
            self.SetFont(font)

        # If the caller specifies a font size, override the default value.
        if fontsize is not None:
            font = self.GetFont()
            font.SetPointSize(fontsize)
            self.SetFont(font)

        # Section 1
        self.panel1 = wx.Panel(self, -1)
        static_box1 = wx.StaticBox(self.panel1, -1, "Fit Algorithms")

        self.algorithms = list(sorted(self.plist.keys()))

        self.radio_list = []
        rows = (len(self.algorithms)+1)/2

        flexsizer = wx.FlexGridSizer(rows, 2, hgap=20, vgap=10)

        for algo in self.algorithms:
            self.radio = wx.RadioButton(self.panel1, -1, algo)
            self.radio_list.append(self.radio)
            self.Bind(wx.EVT_RADIOBUTTON, self.OnRadio, id=self.radio.GetId())
            flexsizer.Add(self.radio, 0, 0)

        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.VERTICAL)
        fit_hsizer.Add(flexsizer, 0, wx.ALL, 5)

        self.panel1.SetSizer(fit_hsizer)
        self.vbox.Add(self.panel1, 0, wx.ALL, 10)

        # Section 2
        # Create list of all panels for later use in hiding and showing panels.
        self.panel_list = []

        for idx, algo in enumerate(self.algorithms):
            parameters = self.plist[algo]
            self.algo_panel = AlgorithmParameter(self, parameters, algo)
            self.panel_list.append(self.algo_panel)
            self.vbox.Add(self.algo_panel, 1, wx.EXPAND|wx.ALL, 10)
            self.algo_panel.Hide()

            if algo == default_algo:
                self.radio_list[idx].SetValue(True)
                self.panel_list[idx].Show()

        # Section 3
        # Create the button controls (Reset, OK, Cancel) and bind their events.
        reset_btn = wx.Button(self, wx.ID_ANY, "Reset")
        ok_btn = wx.Button(self, wx.ID_OK, "OK")
        ok_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")

        self.Bind(wx.EVT_BUTTON, self.OnReset, reset_btn)
        self.Bind(wx.EVT_BUTTON, self.OnOk, ok_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        # Create the button sizer that will put the buttons in a row, right
        # justified, and with a fixed amount of space between them.  This
        # emulates the Windows convention for placing a set of buttons at the
        # bottom right of the window.
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        #btn_sizer.Add((10,20), 1)  # stretchable whitespace
        btn_sizer.Add(reset_btn, 0)
        btn_sizer.Add((10,20), 1)  # stretchable whitespace
        btn_sizer.Add(ok_btn, 0)
        btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
        btn_sizer.Add(cancel_btn, 0)

        # Add the button sizer to the main sizer.
        self.vbox.Add(btn_sizer, 0, wx.EXPAND|wx.ALL, 10)

        # Finalize the sizer and establish the dimensions of the dialog box.
        # The minimum width is explicitly set because the sizer is not able to
        # take into consideration the width of the enclosing frame's title.
        self.SetSizer(self.vbox)
        #self.vbox.SetMinSize((size[0], -1))
        self.vbox.Fit(self)

        self.Centre()

    def OnRadio(self, event):

        radio = event.GetEventObject()

        for btn_idx, btn_instance in enumerate(self.radio_list):
            if radio is btn_instance:
                break

        for panel in self.panel_list:
            if panel.IsShown():
                panel.Hide()
                self.panel_list[btn_idx].Show()
                self.vbox.Layout()
                break

    def OnReset(self, event):
        """
        Reset parameter values for the currently selected fit algorithm to its
        default values when the application was started.
        """

        reset_values = []
        for idx, panel in enumerate(self.panel_list):
            if panel.IsShown():
                parameters = self.plist[ self.algorithms[idx] ]
                for parameter in parameters:
                    label, default_value, curr_value, datatype = parameter
                    reset_values.append(default_value)
                panel.fit_params.update_items_in_panel(reset_values)
                break

    def OnOk(self, event):
        event.Skip()

    def OnCancel(self, event):
        event.Skip()

    def get_results(self):
        self.fit_option={}
        for idx, algo in enumerate(self.algorithms):
            result = self.panel_list[idx].fit_params.GetResults()
            self.fit_option[algo] = result
            if self.radio_list[idx].GetValue():
                active_algo = algo

        return active_algo, self.fit_option


class AlgorithmParameter(wx.Panel):

    def __init__(self, parent, parameters, algo):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        fields = []
        sbox = wx.StaticBox(self, wx.ID_ANY, algo+" Fitting Parameters")

        for parameter in parameters:
            label, default_value, curr_value, datatype = parameter
            if not label.endswith(':'):
                label += ':'
            if hasattr(datatype, 'choices'):
                extra = [str(v) for v in datatype.choices]
                mode = 'CRE'
            else:
                mode = 'RE'
                extra = None
            sub_list = [label, curr_value, datatype, mode, extra]
            fields.append(sub_list)

        # Set the same minimum height for each panel.  The y-size value should
        # be sufficient to display at least 6 input fields without the need for
        # a scroll bar.  Adjust the size.y value if the maximum number of
        # number of input parameters across fitters changes.
        self.fit_params = InputListPanel(parent=self, itemlist=fields,
                                         align=True, size=(-1,220))

        sbox_sizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)
        sbox_sizer.Add(self.fit_params, 1, wx.EXPAND|wx.ALL, 5)
        self.SetSizer(sbox_sizer)
        sbox_sizer.Fit(self)


def OpenFitOptions():
    # Gather together information about option x from three sources:
    #   name and type come from the FIELDS - name,type mapping
    #   algorithm and factory settings comes from fitter.name, fitter.settings
    #   current value comes from FitOptions.options[x]
    # The fields are displayed in the order of the factory settings.
    FIELD = fitters.FitOptions.FIELDS
    plist = {}
    for fit in sorted(fitters.FIT_OPTIONS.values()):
        items = [(FIELD[name][0],
                  setting,
                  fit.options[name],
                  FIELD[name][1])
                 for name,setting in fit.fitclass.settings]
        plist[fit.fitclass.name] = items
    #print "****** plist =\n", plist

    # Pass in the frame object as the parent window so that the dialog box
    # will inherit font info from it instead of using system defaults.
    frame = wx.FindWindowByName("AppFrame")
    algorithm_name = fitters.FIT_OPTIONS[fitters.FIT_DEFAULT].fitclass.name
    fit_dlg = FitControl(parent=frame, id=wx.ID_ANY, title="Fit Control",
                         plist=plist,
                         default_algo=algorithm_name)

    if fit_dlg.ShowModal() == wx.ID_OK:
        algorithm_name, results = fit_dlg.get_results()
        #print 'results', algorithm_name, results

        # Find the new default fitter from the algorithm name.
        for id, record in fitters.FIT_OPTIONS.items():
            if record.fitclass.name == algorithm_name:
                fitters.FIT_DEFAULT = id
                break
        else:
            raise ValueError("No algorithm selected")

        # Update all algorithm values
        for algorithm_name, pars in results.items():
            # Find algorithm record given the name of the optimizer.
            for record in fitters.FIT_OPTIONS.values():
                if record.fitclass.name == algorithm_name:
                    break
            # Update all values in factory settings order.
            for (field, _), value in zip(record.fitclass.settings, pars):
                #print "parse",field,_,value,type(value),FIELD[field]
                parse = FIELD[field][1]
                record.options[field] = parse(value)

    fit_dlg.Destroy()


if __name__=="__main__":
    app = wx.App()
    FitControl(None, -1, 'Fit Control')
    app.MainLoop()
