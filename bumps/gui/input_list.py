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

# Author: James Krycka

"""
This module implements InputListPanel, InputListDialog, and InputListValidator
classes to provide general purpose mechanisms for obtaining and validating user
input from a structured list of input fields.
"""
from __future__ import print_function

import wx
from wx.lib.scrolledpanel import ScrolledPanel

from .utilities import phoenix

Validator = wx.Validator if phoenix else wx.PyValidator

WINDOW_BKGD_COLOUR = "#ECE9D8"
PALE_YELLOW = "#FFFFB0"

DATA_ENTRY_ERRMSG = """\
Please correct any highlighted field in error.
Yellow means an input value is required.
Pink indicates a syntax error."""


class ItemListValidator(Validator):
    """
    This class implements a custom input field validator.  Each instance of
    this class services one data entry field (typically implemented as
    wx.TextCtrl or a wx.ComboBox widget).  Parameters are:

    - datatype of the field (used when validating user input) as follows:
      o int       => signed or unsigned integer value
      o float     => floating point value
      o str       => string of characters
      o 'str_alpha' => string of alphabetic characters {A-Z, a-z}
      o 'str_alnum' => string of alphanumeric characters {A-Z, a-z, 0-9}
      o 'str_id'    => string identifier consisting of {A-Z, a-z, 0-9, _, -}
      o '' or any unknown datatype is treated the same as 'str'

    - flag to indicate whether user input is required (True) or optional (False)
    """

    def __init__(self, datatype='str', required=False):
        Validator.__init__(self)
        self.datatype = datatype
        self.required = required


    def Clone(self):
        # Every validator must implement the Clone() method that returns a
        # instance of the class as follows:
        return ItemListValidator(self.datatype, self.required)


    def Validate(self, win):
        """
        Verify user input according to the expected datatype.  Leading and
        trailing whitespace is always stripped before evaluation.  Floating and
        integer values are returned as normalized float or int objects; thus
        conversion can generate an error.  On error, the field is highlighted
        and the cursor is placed there.  Note that all string datatypes are
        returned stripped of leading and trailing whitespace.
        """

        text_ctrl = self.GetWindow()
        text = text_ctrl.GetValue().strip()

        try:
            if callable(self.datatype):
                self.value = self.value_alt = self.datatype(text)
            elif self.datatype == int:
                if len(text) == 0:
                    self.value = 0
                    self.value_alt = None
                else:
                    float_value = float(text)
                    if float_value != int(float_value):
                        raise ValueError("input must be an integer")
                    self.value = self.value_alt = int(float_value)
            elif self.datatype == float:
                if len(text) == 0:
                    self.value = 0.0
                    self.value_alt = None
                else:
                    self.value = self.value_alt = float(text)
            elif self.datatype == 'str_alpha':
                if len(text) == 0:
                    self.value = ''
                    self.value_alt = None
                    if self.required:
                        raise RuntimeError("input required")
                else:
                    if text.isalpha():
                        self.value = self.value_alt = str(text)
                    else:
                        raise ValueError("input must be alphabetic")
            elif self.datatype == 'str_alnum':
                if len(text) == 0:
                    self.value = ''
                    self.value_alt = None
                else:
                    if text.isalnum():
                        self.value = self.value_alt = str(text)
                    else:
                        raise ValueError("input must be alphanumeric")
            elif self.datatype == 'str_id':
                if len(text) == 0:
                    self.value = ''
                    self.value_alt = None
                else:
                    temp = text.replace('_', 'a').replace('-','a')
                    if temp.isalnum():
                        self.value = self.value_alt = str(text)
                    else:
                        raise ValueError("input must be alphanumeric, _, or -")
            else:  # For self.datatype of "str", "", or any unrecognized type.
                if len(text) == 0:
                    self.value = ''
                    self.value_alt = None
                else:
                    self.value = self.value_alt = str(text)

            if len(text) == 0 and self.required:
                raise RuntimeError("input required")

        except RuntimeError:
            from traceback import print_exc; print_exc()
            text_ctrl.SetBackgroundColour(PALE_YELLOW)
            text_ctrl.SetFocus()
            text_ctrl.Refresh()
            return False

        except Exception:
            from traceback import print_exc; print_exc()
            text_ctrl.SetBackgroundColour("PINK")
            text_ctrl.SetFocus()
            text_ctrl.Refresh()
            return False

        else:
            text_ctrl.SetBackgroundColour(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            text_ctrl.Refresh()
            self.TransferFromWindow()
            return True


    def TransferToWindow(self):
        # The parent of this class is responsible for setting the default value
        # for the field (e.g., by calling wx.TextCtrl() or wx.ComboBox() or
        # instance.SetValue(), etc.).
        return True  # Default is False for failure


    def TransferFromWindow(self):
        # Data has already been transferred from the window and validated
        # in Validate(), so there is nothing useful to do here.
        return True  # Default is False for failure


    def GetValidatedInput(self):
        # Return the validated value or zero or blank for a null input.
        return self.value


    def GetValidatedInputAlt(self):
        # Return the validated value or None for a null input.
        return self.value_alt

#==============================================================================

class InputListPanel(ScrolledPanel):
    """
    This class implements a general purpose mechanism for obtaining and
    validating user input from several fields in a window with scroll bars.
    (See InputListDialog that uses a dialog box instead of a scrolled window.)

    It creates a scrolled window in which to display one or more input fields
    each preceded by a label.  The input fields can be a combination of simple
    data entry boxes or drop down combo boxes.  Automatic validation of user
    input is performed.  The caller can use the GetResults() method to obtain
    the final results from all fields in the form of a list of values.

    The scrolled window object is created as a child of the parent panel passed
    in.  Normally the caller of this class puts this returned object in a sizer
    attached to the parent panel to allow it to expand or contract based on the
    size constraints imposed by its parent.

    The layout is:

    +-------------------------------------+-+
    |                                     |v|
    |  Label-1:   [<drop down list>  |V]  |e|
    |                                     |r|   Note that drop down lists and
    |  Label-2:   [<data entry field-2>]  |t|   simple data entry fields can
    |  ...                                |||   be specified in any order.
    |  Label-n:   [<data entry field-n>]  |||
    |                                     |v|
    +-------------------------------------+-+   Note that scroll bars are
    |      horizontal scroll bar -->      | |   visible only when needed.
    +-------------------------------------+-+

    The itemlist parameter controls the display.  It is a list of input field
    description lists where each description list contains 5 or 6 elements and
    the 6th element is optional.  The items in the description list are:

    [0] Label string prefix for the input field
    [1] Default value
    [2] Datatype for validation (see ItemListValidator docstring for details)
    [3] Flags parameter in the form of a string of characters as follows:
        R - input is required; otherwise input is optional and can be blank
        E - field is editable by the user; otherwise it is non-editable and box
            is grayed-out; a non-editable field has its default value returned
        C - field is a combobox; otherwise it is a simple data entry box
        L - field is preceded by a divider line; 'L' takes precedent over 'H'
        H - field is preceded by a header given in the 6th element of the list;
            the following header sub-options are valid only if 'H' is specified:
            0 - header text size is same as label text size (default)
            1 - header text size is label text size + 1 point (large)
            2 - header text size is label text size + 2 points (x-large)
            3 - header text size is label text size + 3 points (2x-large)
            B - header text is bolded
            U - header text is underlined
        Options can be combined in the flags string such as 'REHB2' which means
        field is required, editable, and preceeded by a bold, extra-large header
    [4] List of values for a combo box or None for a simple data entry field
    [5] Header string to be displayed above the label string of the input field;
        if 'H' is not specified, this list element can be omitted or can be None

    The align parameter determines whether input fields are aligned across when
    the input fields are grouped into sections.  If True, the widest text label
    determines the space allocated for all labels; if False, the text label
    width is determined separately for each section.

    The fontsize parameter allows the caller to specify a font point size to
    override the default point size.

    See the AppTestFrame class for a comprehensive example.
    """

    def __init__(self,
                 parent,
                 id       = wx.ID_ANY,
                 pos      = wx.DefaultPosition,
                 size     = wx.DefaultSize,
                 style    = wx.TAB_TRAVERSAL,
                 name     = "",
                 itemlist = [],
                 align    = False,
                 fontsize = None
                ):
        ScrolledPanel.__init__(self, parent, id, pos, size, style, name)

        #self.SetBackgroundColour(WINDOW_BKGD_COLOUR)
        self.align = align
        self.itemlist = itemlist
        self.item_cnt = len(self.itemlist)
        if self.item_cnt == 0:
            return

        # Set the default font for this and all child windows (widgets) if the
        # caller specifies a size; otherwise let it default from the parent.
        if fontsize is not None:
            font = self.GetFont()
            font.SetPointSize(fontsize)
            self.SetFont(font)
        #print "Input List Panel font ptsize =", self.GetFont().GetPointSize()

        # Specify the widget layout using sizers.
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Create the text controls for labels and associated input fields
        # and any optional headers.
        self.add_items_to_panel()

        # Divide the input items into sections prefaced by header text (except
        # that the first section is not required to have a header).  A section
        # list is created that contains the index of the item that starts a new
        # section plus a final entry that is one beyond the last item.
        sect = [0]  # declare item 0 to be start of a new section
        for i in range(self.item_cnt):
            if i > 0 and self.headers[i] is not None:
                sect.append(i)
        sect.append(self.item_cnt)

        # Place the items for each section in its own flex grid sizer.
        for i in range(len(sect)-1):
            j = sect[i]; k = sect[i+1] - 1
            fg_sizer = self.add_items_to_sizer(j, k)

            # Add the flex grid sizer to the main sizer.
            if self.headers[j] is not None:  # self.headers[0] could be None
                main_sizer.Add(self.headers[j], 0, border=10,
                               flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT)
            main_sizer.Add(fg_sizer, 0, border=10,
                           flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT)

        # Finalize the sizer and establish the dimensions of the input box.
        self.SetSizer(main_sizer)
        main_sizer.Fit(self)

        # Enable scrolling and initialize the validators (required when
        # validators are not used in the context of a dialog box).
        self.SetupScrolling(scroll_x=True, scroll_y=True)
        self.InitDialog()


    def add_items_to_panel(self):
        """
        For each input item, create a header (optional), label, and input box
        widget to instantiate it.  Put the handles for these widgets in the
        headers, labels, and inputs lists where the length of each list is the
        same as the number of input boxes.
        """

        self.headers = []; self.labels = []; self.inputs = []
        self.widest = 0

        for x in range(self.item_cnt):
            params = len(self.itemlist[x])
            if params == 6:
                text, default, datatype, flags, plist, header = self.itemlist[x]
            elif params == 5:
                text, default, datatype, flags, plist = self.itemlist[x]
                header = None
            if default is None: default = ""  # display None as a null string

            # Process the flags parameter.
            required = False
            if 'R' in flags: required = True
            editable = False
            if 'E' in flags: editable = True
            combo = False
            if 'C' in flags: combo = True
            line = False
            if 'L' in flags: line = True
            hdr = False
            if 'H' in flags and header is not None: hdr = True
            if hdr:
                delta_pts = 0
                if '1' in flags: delta_pts = 1  # large
                if '2' in flags: delta_pts = 2  # X-large
                if '3' in flags: delta_pts = 3  # 2X-large
                weight = wx.NORMAL
                if 'B' in flags: weight = wx.BOLD
                underlined = False
                if 'U' in flags: underlined = True

            # Optionally, create a header widget to display above the input box.
            # A dividing line is treated as a special case header.
            if line:
                lin = wx.StaticLine(self, wx.ID_ANY, style=wx.LI_HORIZONTAL)
                self.headers.append(lin)
            elif hdr:
                hdr = wx.StaticText(self, wx.ID_ANY, label=header,
                                    style=wx.ALIGN_CENTER)
                font = hdr.GetFont()
                ptsize = font.GetPointSize() + delta_pts
                font.SetPointSize(ptsize)
                font.SetWeight(weight)
                font.SetUnderlined(underlined)
                hdr.SetFont(font)
                hdr.SetForegroundColour("BLUE")
                self.headers.append(hdr)
            else:
                self.headers.append(None)

            # Create the text label widget.
            self.labels.append(wx.StaticText(self, wx.ID_ANY, label=text,
                               style=wx.ALIGN_LEFT))
            w, h = self.labels[x].GetSize()
            if w > self.widest: self.widest = w

            # Create the input box widget (combo box or simple data entry box)
            if combo:              # it is a drop down combo box list
                self.inputs.append(wx.ComboBox(self, wx.ID_ANY,
                                   value=str(default),
                                   validator=ItemListValidator(datatype, required),
                                   choices=plist,
                                   style=wx.CB_DROPDOWN|wx.CB_READONLY))
                self.Bind(wx.EVT_COMBOBOX, self.OnComboBoxSelect, self.inputs[x])
            else:                  # it is a simple data entry field
                self.inputs.append(wx.TextCtrl(self, wx.ID_ANY,
                                   value=str(default),
                                   validator=ItemListValidator(datatype, required)))
                self.Bind(wx.EVT_TEXT, self.OnText, self.inputs[x])

            # Verfiy that field is editable, otherwise don't allow user to edit
            if not editable:
                self.inputs[x].Enable(False)

            # Validate the default value and highlight the field if the value is
            # in error or if input is required and the value is a null string.
            self.inputs[x].GetValidator().Validate(self.inputs[x])

        # Determine if all input boxes should be aligned across sections.
        if self.align:
            for x in range(self.item_cnt):
                self.labels[x].SetMinSize((self.widest, -1))


    def add_items_to_sizer(self, start, end):
        sizer = wx.FlexGridSizer(cols=2, hgap=5, vgap=10)
        for x in range(start, end+1):
            sizer.Add(self.labels[x], 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.inputs[x], 0, wx.EXPAND)
        sizer.AddGrowableCol(1)
        return sizer


    def update_items_in_panel(self, new_values):
        for x in range(len(self.inputs)):
            if new_values[x] is not None:
                self.inputs[x].SetValue(str(new_values[x]))


    def GetResults(self):
        """
        Returns a list of values, one for each input field.  The value for
        a field is either its initial (default) value or the last value
        entered by the user that has been successfully validated.  An input
        that fails validation is not returned by the validator from the
        window.  For a non-editable field, its initial value is returned.

        Blank input is converted to 0 for int, 0.0 for float, or a 0-length
        string for a string datatype.
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(self.inputs[x].GetValidator().GetValidatedInput())
        return ret


    def GetResultsAltFormat(self):
        """
        Returns a list of values, one for each input field.  The value for
        a field is either its initial (default) value or the last value
        entered by the user that has been successfully validated.  An input
        that fails validation is not returned by the validator from the
        window.  For a non-editable field, its initial value is returned.

        Blank input is returned as a value of None.
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(self.inputs[x].GetValidator().GetValidatedInputAlt())
        return ret


    def GetResultsRawInput(self):
        """
        Returns a list of strings corresponding to each input field.  These
        are the current values from the text control widgets whether or not
        they have passed validation.  All values are returned as raw strings
        (i.e., they are not converted to floats or ints and leading and
        trailing whitespace is not stripped).
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(str(self.inputs[x].GetValue()))
        return ret


    def OnText(self, event):
        """
        This method is called each time a key stroke is entered in any text
        control box.  It should be subclassed if special processing is needed.
        The sample code below shows how to obtain the index of the box and its
        value.  Note that the box's index is 0 to n, where n is the number of
        input and combo boxes, not just the number of input boxes.

        # Get index of the input box that triggered the event.
        text_ctrl = event.GetEventObject()
        for box_idx, box_instance in enumerate(self.inputs):
            if text_ctrl is box_instance:
                break
        # Get the edited string.
        text = text_ctrl.GetValue()
        print "Field:", box_idx, text
        """

        # Run the validator bound to the text control box that has been edited.
        # If the validation fails, the validator will highlight the input field
        # to alert the user of the error.
        text_ctrl = event.GetEventObject()
        text_ctrl.GetValidator().Validate(text_ctrl)
        event.Skip()


    def OnComboBoxSelect(self, event):
        """
        This method is called each time a selection is made in any combo box.
        It should be subclassed if the caller wants to perform some action in
        response to a selection event.  The sample code below shows how to
        obtain the index of the box, the index of the item selected, and the
        value.  Note that the box's index is 0 to n, where n is the number of
        combo and input boxes, not just the number of combo boxes.

        # Get index of selected item in combo box dropdown list.
        item_idx = event.GetSelection()
        # Get index of combo box that triggered the event.
        current_box = event.GetEventObject()
        for box_idx, box_instance in enumerate(self.inputs):
            if current_box is box_instance:
                break
        print "Combo:", box_idx, item_idx, self.itemlist[box_idx][3][item_idx]
        """

        # Run the validator bound to the combo box that has a selection event.
        # This should not fail unless the combo options were setup incorrectly.
        # If the validation fails, the validator will highlight the input field
        # to alert the user of the error.
        combo_box = event.GetEventObject()
        combo_box.GetValidator().Validate(combo_box)
        event.Skip()

#==============================================================================

class InputListDialog(wx.Dialog):
    """
    This class implements a general purpose mechanism for obtaining and
    validating user input from several fields in a pop-up dialog box.
    (See InputListPanel that uses a scrolled window instead of a dialog box.)

    It creates a pop-up dialog box in which to display one or more input fields
    each preceded by a label.  The input fields can be a combination of simple
    data entry boxes or drop down combo boxes.  Automatic validation of user
    input is performed.  OK and Cancel buttons are provided at the bottom of
    the dialog box for the user to signal completion of data entry whereupon
    the caller can use the GetResults() method to obtain the final results from
    all fields in the form of a list of values.  As with any dialog box, when
    the user presses OK or Cancel the dialog disappears from the screen, but
    the caller of this class is responsible for destroying the dialog box.

    The dialog box is automatically sized to fit the fields and buttons with
    reasonable spacing between the widgets.  The layout is:

    +-------------------------------------+
    |  Title                          [X] |
    +-------------------------------------+
    |                                     |
    |  Label-1:   [<drop down list>  |V]  |
    |                                     |     Note that drop down lists and
    |  Label-2:   [<data entry field-2>]  |     simple data entry fields can
    |  ...                                |     be specified in any order.
    |  Label-n:   [<data entry field-n>]  |
    |                                     |
    |       [  OK  ]      [Cancel]        |
    |                                     |
    +-------------------------------------+

    The itemlist parameter controls the display.  It is a list of input field
    description lists where each description list contains 5 or 6 elements and
    the 6th element is optional.  The items in the description list are:

    [0] Label string prefix for the input field
    [1] Default value
    [2] Datatype for validation (see ItemListValidator docstring for details)
    [3] Flags parameter in the form of a string of characters as follows:
        R - input is required; otherwise input is optional and can be blank
        E - field is editable by the user; otherwise it is non-editable and box
            is grayed-out; a non-editable field has its default value returned
        C - field is a combobox; otherwise it is a simple data entry box
        L - field is preceded by a divider line; 'L' takes precedent over 'H'
        H - field is preceded by a header given in the 6th element of the list;
            the following header sub-options are valid only if 'H' is specified:
            0 - header text size is same as label text size (default)
            1 - header text size is label text size + 1 point (large)
            2 - header text size is label text size + 2 points (x-large)
            3 - header text size is label text size + 3 points (2x-large)
            B - header text is bolded
            U - header text is underlined
        Options can be combined in the flags string such as 'REHB2' which means
        field is required, editable, and preceeded by a bold, extra-large header
    [4] List of values for a combo box or None for a simple data entry field
    [5] Header string to be displayed above the label string of the input field;
        if 'H' is not specified, this list element can be omitted or can be None

    The align parameter determines whether input fields are aligned across when
    the input fields are grouped into sections.  If True, the widest text label
    determines the space allocated for all labels; if False, the text label
    width is determined separately for each section.

    The fontsize parameter allows the caller to specify a font point size to
    override the default point size.

    See the AppTestFrame class for a comprehensive example.
    """

    def __init__(self,
                 parent   = None,
                 id       = wx.ID_ANY,
                 title    = "Enter Data",
                 pos      = wx.DefaultPosition,
                 size     = (300, -1),  # x is min_width; y will be calculated
                 style    = wx.DEFAULT_DIALOG_STYLE,
                 name     = "",
                 itemlist = [],
                 align    = False,
                 fontsize = None
                ):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)

        self.align = align
        self.itemlist = itemlist
        self.item_cnt = len(self.itemlist)
        if self.item_cnt == 0:
            return

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
        #print "Input Dialog box font ptsize =", self.GetFont().GetPointSize()

        # Create the button controls (OK and Cancel) and bind their events.
        ok_button = wx.Button(self, wx.ID_OK, "OK")
        ok_button.SetDefault()
        cancel_button = wx.Button(self, wx.ID_CANCEL, "Cancel")

        self.Bind(wx.EVT_BUTTON, self.OnOk, ok_button)

        # Specify the widget layout using sizers.
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Create the text controls for labels and associated input fields
        # and any optional headers.
        self.add_items_to_dialog_box()

        # Divide the input items into sections prefaced by header text (except
        # that the first section is not required to have a header).  A section
        # list is created that contains the index of the item that starts a new
        # section plus a final entry that is one beyond the last item.
        sect = [0]  # declare item 0 to be start of a new section
        for i in range(self.item_cnt):
            if i > 0 and self.headers[i] is not None:
                sect.append(i)
        sect.append(self.item_cnt)
        #print "Section index list:", sect

        # Place the items for each section in its own flex grid sizer.
        for i in range(len(sect)-1):
            j = sect[i]; k = sect[i+1] - 1
            #print "Items per section:", j, "to", k
            fg_sizer = self.add_items_to_sizer(j, k)

            # Add the flex grid sizer to the main sizer.
            if self.headers[j] is not None:  # self.headers[0] could be None
                main_sizer.Add(self.headers[j], 0, border=10,
                               flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT)
            main_sizer.Add(fg_sizer, 0, border=10,
                           flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT)

        # Create the button sizer that will put the buttons in a row, right
        # justified, and with a fixed amount of space between them.  This
        # emulates the Windows convention for placing a set of buttons at the
        # bottom right of the window.
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add((10,20), 1)  # stretchable whitespace
        button_sizer.Add(ok_button, 0)
        button_sizer.Add((10,20), 0)  # non-stretchable whitespace
        button_sizer.Add(cancel_button, 0)

        # Add a separator line before the buttons.
        separator = wx.StaticLine(self, wx.ID_ANY, style=wx.LI_HORIZONTAL)
        main_sizer.Add(separator, 0 , border=10,
                       flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT)

        # Add the button sizer to the main sizer.
        main_sizer.Add(button_sizer, 0, border=10,
                       flag=wx.EXPAND|wx.TOP|wx.BOTTOM|wx.RIGHT)

        # Finalize the sizer and establish the dimensions of the dialog box.
        # The minimum width is explicitly set because the sizer is not able to
        # take into consideration the width of the enclosing frame's title.
        self.SetSizer(main_sizer)
        main_sizer.SetMinSize((size[0], -1))
        main_sizer.Fit(self)


    def add_items_to_dialog_box(self):
        """
        For each input item, create a header (optional), label, and input box
        widget to instantiate it.  Put the handles for these widgets in the
        headers, labels, and inputs lists where the length of each list is the
        same as the number of input boxes.
        """

        self.headers = []; self.labels = []; self.inputs = []
        self.widest = 0
        first_error_idx = None

        for x in range(self.item_cnt):
            params = len(self.itemlist[x])
            if params == 6:
                text, default, datatype, flags, plist, header = self.itemlist[x]
            elif params == 5:
                text, default, datatype, flags, plist = self.itemlist[x]
                header = None
            if default is None: default = ""  # display None as a null string

            # Process the flags parameter.
            required = False
            if flags.find('R') >= 0: required = True
            editable = False
            if flags.find('E') >= 0: editable = True
            combo = False
            if flags.find('C') >= 0: combo = True
            line = False
            if flags.find('L') >= 0: line = True
            hdr = False
            if flags.find('H') >= 0 and header is not None: hdr = True
            if hdr:
                delta_pts = 0
                if flags.find('1') >= 0: delta_pts = 1  # large
                if flags.find('2') >= 0: delta_pts = 2  # X-large
                if flags.find('3') >= 0: delta_pts = 3  # 2X-large
                weight = wx.NORMAL
                if flags.find('B') >= 0: weight = wx.BOLD
                underlined = False
                if flags.find('U') >= 0: underlined = True

            # Optionally, create a header widget to display above the input box.
            # A dividing line is treated as a special case header.
            if line:
                lin = wx.StaticLine(self, wx.ID_ANY, style=wx.LI_HORIZONTAL)
                self.headers.append(lin)
            elif hdr:
                hdr = wx.StaticText(self, wx.ID_ANY, label=header,
                                    style=wx.ALIGN_CENTER)
                font = hdr.GetFont()
                ptsize = font.GetPointSize() + delta_pts
                font.SetPointSize(ptsize)
                font.SetWeight(weight)
                font.SetUnderlined(underlined)
                hdr.SetFont(font)
                hdr.SetForegroundColour("BLUE")
                self.headers.append(hdr)
            else:
                self.headers.append(None)

            # Create the text label widget.
            self.labels.append(wx.StaticText(self, wx.ID_ANY, label=text,
                               style=wx.ALIGN_LEFT))
            w, h = self.labels[x].GetSize()
            if w > self.widest: self.widest = w

            # Create the input box widget (combo box or simple data entry box)
            if combo:              # it is a drop down combo box list
                self.inputs.append(wx.ComboBox(self, wx.ID_ANY,
                                   value=str(default),
                                   validator=ItemListValidator(datatype, required),
                                   choices=plist,
                                   style=wx.CB_DROPDOWN|wx.CB_READONLY))
                self.Bind(wx.EVT_COMBOBOX, self.OnComboBoxSelect, self.inputs[x])
            else:                  # it is a simple data entry field
                self.inputs.append(wx.TextCtrl(self, wx.ID_ANY,
                                   value=str(default),
                                   validator=ItemListValidator(datatype, required)))
                self.Bind(wx.EVT_TEXT, self.OnText, self.inputs[x])

            # Verfiy that field is editable, otherwise don't allow user to edit
            if not editable:
                self.inputs[x].Enable(False)

            # Validate the default value and highlight the field if the value is
            # in error or if input is required and the value is a null string.
            # Also, save index of the first field to fail validation, if any.
            ret = self.inputs[x].GetValidator().Validate(self.inputs[x])
            if not ret and first_error_idx is None: first_error_idx = x

        # If any fields failed validation, set focus to the first failed one.
        if first_error_idx is not None: self.inputs[first_error_idx].SetFocus()

        # Determine if all input boxes should be aligned across sections.
        if self.align:
            for x in range(self.item_cnt):
                self.labels[x].SetMinSize((self.widest, -1))


    def add_items_to_sizer(self, start, end):
        sizer = wx.FlexGridSizer(cols=2, hgap=5, vgap=10)
        for x in range(start, end+1):
            sizer.Add(self.labels[x], 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.inputs[x], 0, wx.EXPAND)
        sizer.AddGrowableCol(1)
        return sizer


    def update_items_in_dialog_box(self, new_values):
        for x in range(len(self.inputs)):
            if new_values[x] is not None:
                self.inputs[x].SetValue(str(new_values[x]))


    def GetResults(self):
        """
        Returns a list of values, one for each input field.  The value for
        a field is either its initial (default) value or the last value
        entered by the user that has been successfully validated.  An input
        that fails validation is not returned by the validator from the
        window.  For a non-editable field, its initial value is returned.

        Blank input is converted to 0 for int, 0.0 for float, or a 0-length
        string for a string datatype.
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(self.inputs[x].GetValidator().GetValidatedInput())
        return ret


    def GetResultsAltFormat(self):
        """
        Returns a list of values, one for each input field.  The value for
        a field is either its initial (default) value or the last value
        entered by the user that has been successfully validated.  An input
        that fails validation is not returned by the validator from the
        window.  For a non-editable field, its initial value is returned.

        Blank input is returned as a value of None.
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(self.inputs[x].GetValidator().GetValidatedInputAlt())
        return ret


    def GetResultsRawInput(self):
        """
        Returns a list of strings corresponding to each input field.  These
        are the current values from the text control widgets whether or not
        they have passed validation.  All values are returned as raw strings
        (i.e., they are not converted to floats or ints and leading and
        trailing whitespace is not stripped).
        """

        ret = []
        for x in range(self.item_cnt):
            ret.append(str(self.inputs[x].GetValue()))
        return ret


    def OnOk(self, event):
        """
        This method gets called when the user presses the OK button.
        It is intended to be subclassed if special processing is needed.
        """

        # Explicitly validate all input values before proceeding.  Although
        # char-by-char validation would have warned the user about any invalid
        # entries, the user could have pressed the OK button without making
        # the corrections, so we'll do a full validation pass now.  The only
        # purpose is to display an explicit error if any input fails validation.
        if not self.Validate():
            wx.MessageBox(caption="Data Entry Error",
                          message=DATA_ENTRY_ERRMSG,
                          style=wx.ICON_ERROR|wx.OK)
            return  # keep the dialog box open

        # When the wx.ID_OK event is skipped (to allow handlers up the chain to
        # run), the Validate methods for all text control boxes will be called.
        # If all report success, the TransferFromWindow methods will be called
        # and the dialog box will close.  However, if any Validate method fails
        # this process will stop and the dialog box will remain open allowing
        # the user to either correct the problem(s) or cancel the dialog.
        event.Skip()


    def OnText(self, event):
        """
        This method is called each time a key stroke is entered in any text
        control box.  It should be subclassed if special processing is needed.
        The sample code below shows how to obtain the index of the box and its
        value.  Note that the box's index is 0 to n, where n is the number of
        input and combo boxes, not just the number of input boxes.

        # Get index of the input box that triggered the event.
        text_ctrl = event.GetEventObject()
        for box_idx, box_instance in enumerate(self.inputs):
            if text_ctrl is box_instance:
                break
        # Get the edited string.
        text = text_ctrl.GetValue()
        print "Field:", box_idx, text
        """

        # Run the validator bound to the text control box that has been edited.
        # If the validation fails, the validator will highlight the input field
        # to alert the user of the error.
        text_ctrl = event.GetEventObject()
        text_ctrl.GetValidator().Validate(text_ctrl)
        event.Skip()


    def OnComboBoxSelect(self, event):
        """
        This method is called each time a selection is made in any combo box.
        It should be subclassed if the caller wants to perform some action in
        response to a selection event.  The sample code below shows how to
        obtain the index of the box, the index of the item selected, and the
        value.  Note that the box's index is 0 to n, where n is the number of
        combo and input boxes, not just the number of combo boxes.

        # Get index of selected item in combo box dropdown list.
        item_idx = event.GetSelection()
        # Get index of combo box that triggered the event.
        current_box = event.GetEventObject()
        for box_idx, box_instance in enumerate(self.inputs):
            if current_box is box_instance:
                break
        print "Combo:", box_idx, item_idx, self.itemlist[box_idx][3][item_idx]
        """

        # Run the validator bound to the combo box that has a selection event.
        # This should not fail unless the combo options were setup incorrectly.
        # If the validation fails, the validator will highlight the input field
        # to alert the user of the error.
        combo_box = event.GetEventObject()
        combo_box.GetValidator().Validate(combo_box)
        event.Skip()

#==============================================================================

class AppTestFrame(wx.Frame):
    """
    Interactively test both the InputListPanel and the InputListDialog classes.
    Both will display the same input fields.  Enter invalid data to verify
    char-by-char error processing.  Press the Submit and OK buttons with an
    uncorrected highlighted field in error to generate a pop-up error box.
    Resize the main window to see scroll bars disappear and reappear.
    """

    # Establish efault font and point size for test.
    FONTNAME = "Arial"
    if wx.Platform == "__WXMSW__":
        FONTSIZE = 9
    elif wx.Platform == "__WXMAC__":
        FONTSIZE = 12
    elif wx.Platform == "__WXGTK__":
        FONTSIZE = 11
    else:
        FONTSIZE = 10

    def __init__(self):
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY,
                          title="InputListPanel Test", size=(300, 600))
        panel = wx.Panel(self, wx.ID_ANY, style=wx.RAISED_BORDER)
        panel.SetBackgroundColour("PALE GREEN")

        pt_size = panel.GetFont().GetPointSize()

        # Define fields for both InputListPanel and InputListDialog to display.
        self.fields = [
            ["Integer (int, optional):", 12345, int, 'EH3', None,
                "Test Header (2X-large)"],
            # Test specification of integer default value as a string
            ["Integer (int, optional):", "-60", int, 'E', None],
            # Default value is null, so the required field should be highlighted
            ["Integer (int, required):", "", int, 'RE', None],
            ["Floating Point (float, optional):", 2.34567e-5, float, 'EHB1', None,
                "Test Header (large, bold)"],
            ["Floating Point (float, optional):", "", float, 'E', None],
            ["Floating Point (float, required):", 1.0, float, 'RE', None],
            # Test unknown datatype which should be treated as 'str'
            ["String (str, optional):", "DANSE", "foo", 'EHU', None,
                "Test Header (%dpt font, underlined)"%pt_size],
            ["String (str, reqiured):", "delete me", str, 'RE', None],
            ["Non-editable field:", "Cannot be changed!", str, '', None],
            ["ComboBox String:", "Two", str, 'CREL', ("One", "Two", "Three")],
            # ComboBox items must be specified as strings
            ["ComboBox Integer:", "", int, 'CE', ("100", "200", "300")],
            ["String (alphabetic):", "Aa", "str_alpha", 'E', None],
            ["String (alphanumeric):", "Aa1", "str_alnum", 'E', None],
            ["String (A-Z, a-z, 0-9, _, -):", "A-1_a", "str_id", 'E', None],
                      ]

        # Create the scrolled window with input boxes.  Due to the intentionally
        # small size of the parent panel, both scroll bars should be displayed.
        self.scrolled = InputListPanel(parent=panel, itemlist=self.fields,
                                       align=True)

        # Create a button to request the popup dialog box.
        show_button = wx.Button(panel, wx.ID_ANY, "Show Pop-up Dialog Box")
        self.Bind(wx.EVT_BUTTON, self.OnShow, show_button)

        # Create a button to signal end of user edits and one to exit program.
        submit_button = wx.Button(panel, wx.ID_ANY, "Submit")
        self.Bind(wx.EVT_BUTTON, self.OnSubmit, submit_button)
        exit_button = wx.Button(panel, wx.ID_ANY, "Exit")
        self.Bind(wx.EVT_BUTTON, self.OnExit, exit_button)

        # Create a horizontal sizer for the buttons.
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add((10,20), 1)  # stretchable whitespace
        button_sizer.Add(submit_button, 0)
        button_sizer.Add((10,20), 0)  # non-stretchable whitespace
        button_sizer.Add(exit_button, 0)

        # Create a vertical box sizer for the panel and layout widgets in it.
        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.Add(show_button, 0, wx.ALIGN_CENTER|wx.ALL, border=10)
        box_sizer.Add(self.scrolled, 1, wx.EXPAND|wx.ALL, border=10)
        box_sizer.Add(button_sizer, 0, wx.EXPAND|wx.BOTTOM|wx.ALL, border=10)

        # Associate the sizer with its container.
        panel.SetSizer(box_sizer)
        box_sizer.Fit(panel)


    def OnShow(self, event):
        # Display the same fields shown in the frame in a pop-up dialog box.
        pt_size = self.FONTSIZE
        self.fields[6][5] = "Test Header (%dpt font, underlined)"%pt_size
        dlg = InputListDialog(parent=self,
                              title="InputListDialog Test",
                              itemlist=self.fields,
                              align=True,
                              fontsize=self.FONTSIZE)
        if dlg.ShowModal() == wx.ID_OK:
            print("****** Dialog Box results from validated input fields:")
            print("  ", dlg.GetResults())
            print("****** Dialog Box results from validated input fields" +\
                  " (None if no input):")
            print("  ", dlg.GetResultsAltFormat())
            print("****** Dialog Box results from raw input fields:")
            print("  ", dlg.GetResultsRawInput())
        dlg.Destroy()


    def OnSubmit(self, event):
        # Explicitly validate all input parameters before proceeding.  Even
        # though char-by-char validation would have warned the user about any
        # invalid entries, the user could have pressed the Done button without
        # making the corrections, so a full validation pass is necessary.
        if not self.scrolled.Validate():
            wx.MessageBox(caption="Data Entry Error",
                message="Please correct the highlighted fields in error.",
                style=wx.ICON_ERROR|wx.OK)
            return  # keep the dialog box open
        print("****** Scrolled Panel results from validated input fields:")
        print("  ", self.scrolled.GetResults())
        print("****** Scrolled Panel results from validated input fields" +\
              " (None if no input):")
        print("  ", self.scrolled.GetResultsAltFormat())
        print("****** Scrolled Panel results from raw input fields:")
        print("  ", self.scrolled.GetResultsRawInput())


    def OnExit(self, event):
        # Terminate the program.
        self.Close()

#==============================================================================

if __name__ == '__main__':
    # Interactively test both the InputListPanel and the InputListDialog classes.
    app = wx.PySimpleApp()
    frame = AppTestFrame()
    frame.Show(True)
    app.MainLoop()
