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

import wx.lib.newevent

from .. import options
from .input_list import InputListPanel
from .utilities import phoenix

(FitterChangedEvent, EVT_FITTER_CHANGED) = wx.lib.newevent.NewCommandEvent()

class FitConfig(wx.Frame):
    """
    FitControl lets the user set fitting options from a pop-up dialog box.
    """
    def __init__(self,
                 parent = None,
                 id     = wx.ID_ANY,
                 title  = "Fit Options",
                 pos    = wx.DefaultPosition,
                 size   = wx.DefaultSize, # dialog box size will be calculated
                 style  = wx.DEFAULT_DIALOG_STYLE,
                 name   = "",
                 config = None,
                 help = None,
                 fontsize = None,
                ):
        wx.Frame.__init__(self, parent, id, title, pos, size, style, name)

        self.config = config
        self.help = help

        pairs = [(config.names[id],id) for id in config.active_ids]
        self.active_ids = [id for _,id in sorted(pairs)]

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

        rows = (len(self.active_ids)+1)/2

        flexsizer = wx.FlexGridSizer(rows, 2, hgap=20, vgap=10)

        self.fitter_button = {}
        for fitter in self.active_ids:
            button = wx.RadioButton(self.panel1, -1,
                    label=config.names[fitter], name=fitter)
            self.fitter_button[fitter] = button
            self.Bind(wx.EVT_RADIOBUTTON, self.OnRadio, id=button.GetId())
            flexsizer.Add(button, 0, 0)

        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.VERTICAL)
        fit_hsizer.Add(flexsizer, 0, wx.ALL, 5)

        self.panel1.SetSizer(fit_hsizer)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.panel1, 0, wx.ALL, 10)

        # Section 2
        # Create list of all panels for later use in hiding and showing panels.
        self.fitter_panel = {}
        for fitter in self.active_ids:
            items = [(options.FIT_FIELDS[field][0],
                      field,
                      config.values[fitter][field],
                      options.FIT_FIELDS[field][1])
                     for field, default in config.settings[fitter]]
            #print fitter, items
            panel = ParameterPanel(self, items, config.names[fitter])
            self.fitter_panel[fitter] = panel
            self.vbox.Add(panel, 1, wx.EXPAND|wx.ALL, 10)
            panel.Hide()

        # Make the current panel active
        self.fitter_button[config.selected_id].SetValue(True)
        self.fitter_panel[config.selected_id].Show()

        # Section 3
        # Create the button controls (Reset, Apply) and bind their events.
        #apply_btn = wx.Button(self, wx.ID_APPLY, "Apply")
        #apply_btn.SetToolTip(wx.ToolTip("Accept new options for the optimizer"))
        #apply_btn.SetDefault()
        #reset_btn = wx.Button(self, wx.ID_ANY, "Reset")
        #reset_btn.SetToolTip(wx.ToolTip("Restore default options for the optimizer"))
        accept_btn = wx.Button(self, wx.ID_OK)
        accept_btn.SetToolTip(wx.ToolTip("Accept new options for the optimizer"))
        accept_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL)
        cancel_btn.SetToolTip(wx.ToolTip("Restore default options for the optimizer"))
        if help is not None:
            help_btn = wx.Button(self, wx.ID_HELP, 'Help')
            #help_btn = wx.Button(self, wx.ID_ANY, 'Help')
            help_btn.SetToolTip(wx.ToolTip("Help on the options for the optimizer"))


        self.Bind(wx.EVT_BUTTON, self.OnAccept, accept_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)
        if help is not None:
            self.Bind(wx.EVT_BUTTON, self.OnHelp, help_btn)

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Create the button sizer that will put the buttons in a row, right
        # justified, and with a fixed amount of space between them.  This
        # emulates the Windows convention for placing a set of buttons at the
        # bottom right of the window.
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add((10,20), 1)  # stretchable whitespace
        btn_sizer.Add(accept_btn, 0)
        btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
        btn_sizer.Add(cancel_btn, 0)
        btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
        if help is not None:
            btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
            btn_sizer.Add(help_btn, 0)

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

        button = event.GetEventObject()
        for panel in self.fitter_panel.values():
            panel.Hide()
        self.fitter_panel[button.Name].Show()
        self.vbox.Layout()

    def OnCancel(self, event):
        """
        Restore options for the selected fitter to the default values.
        """
        fitter = self._get_fitter()
        panel = self.fitter_panel[fitter]
        panel.Parameters = dict(self.config.settings[fitter])
        self.Hide()

    def OnAccept(self, event):
        """
        Save the current fitter and options to the fit config.
        """
        fitter = self._get_fitter()
        options = self.fitter_panel[fitter].Parameters
        self.config.selected_id = fitter
        self.config.values[fitter] = options
        self.Hide()

        # Signal a change in fitter
        event = FitterChangedEvent(self.Id, config=self.config)
        wx.PostEvent(self, event)

    def OnHelp(self, event):
        """
        Provide help on the selected fitter.
        """
        if self.help is not None:
            self.help(self._get_fitter())

    def OnClose(self, event):
        """
        Don't close the window, just hide it.
        """
        if event.CanVeto():
            self.Hide()
            event.Veto()
        else:
            event.Skip()

    def _get_fitter(self):
        """
        Returns the currently selected algorithm, or None if no algorithm is
        selected.
        """
        for button in self.fitter_button.values():
            if button.Value:
                return button.Name
        else:
            return None

    def _get_options(self):
        fitter = self._get_fitter()
        options = self.fitter_panel[fitter].Parameters

        return fitter, options


class ParameterPanel(wx.Panel):

    def __init__(self, parent, parameters, fitter_name):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.fields = []
        itemlist = []
        sbox = wx.StaticBox(self, wx.ID_ANY, fitter_name+" Fitting Parameters")

        for parameter in parameters:
            label, field, curr_value, datatype = parameter
            if not label.endswith(':'):
                label += ':'
            if hasattr(datatype, 'choices'):
                extra = [str(v) for v in datatype.choices]
                mode = 'CRE'
            else:
                mode = 'RE'
                extra = None
            sub_list = [label, curr_value, datatype, mode, extra]
            itemlist.append(sub_list)
            self.fields.append(field)

        # Set the same minimum height for each panel.  The y-size value should
        # be sufficient to display at least 6 input fields without the need for
        # a scroll bar.  Adjust the size.y value if the maximum number of
        # number of input parameters across fitters changes.
        self.fit_params = InputListPanel(parent=self, itemlist=itemlist,
                                         align=True, size=(-1,220))

        sbox_sizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)
        sbox_sizer.Add(self.fit_params, 1, wx.EXPAND|wx.ALL, 5)
        self.SetSizer(sbox_sizer)
        sbox_sizer.Fit(self)

    @property
    def Parameters(self):
        values = self.fit_params.GetResults()
        return dict(zip(self.fields, values))

    @Parameters.setter
    def Parameters(self, parameters):
        values = [parameters[k] for k in self.fields]
        self.fit_params.update_items_in_panel(values)

_fit_config_frame = None
def show_fit_config(parent, help=None):
    global _fit_config_frame
    if _fit_config_frame is None:
        _fit_config_frame = FitConfig(parent=parent,
                                      config=options.FIT_CONFIG, help=help)
    _fit_config_frame.Show()
    _fit_config_frame.Raise()
    return _fit_config_frame

if __name__=="__main__":
    opts = options.getopts()
    def _help(algo):
        print("asking for help with "+algo)

    app = wx.App()
    top = wx.Frame(None)
    text = wx.TextCtrl(top, wx.ID_ANY, "some text")
    button = wx.Button(top, wx.ID_ANY, "Options...")
    button.Bind(wx.EVT_BUTTON,
                lambda ev: show_fit_config(top, help=_help))

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(text)
    sizer.Add(button)
    sizer.Fit(top)
    top.SetSizer(sizer)
    top.Show()
    app.MainLoop()
