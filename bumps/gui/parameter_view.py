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
This module implements the Parameter View panel.
"""

#==============================================================================

import wx

import wx.gizmos as gizmos

from ..parameter import BaseParameter
from .util import nice
from . import signal


IS_MAC = (wx.Platform == '__WXMAC__')

class ParameterView(wx.Panel):
    title = 'Parameters'
    default_size = (640,500)
    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        #sizers
        vbox = wx.BoxSizer(wx.VERTICAL)
        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.tree = gizmos.TreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        | wx.TR_HAS_BUTTONS
                                        | wx.TR_TWIST_BUTTONS
                                        | wx.TR_ROW_LINES
                                        #| wx.TR_COLUMN_LINES
                                        | wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                       )

        # Create columns.
        self.tree.AddColumn("Model")
        self.tree.AddColumn("Parameter")
        self.tree.AddColumn("Value")
        self.tree.AddColumn("Minimum")
        self.tree.AddColumn("Maximum")
        self.tree.AddColumn("Fit?")

        # Align the textctrl box with treelistctrl.
        self.tree.SetMainColumn(0) # the one with the tree in it...
        self.tree.SetColumnWidth(0, 180)
        self.tree.SetColumnWidth(1, 150)
        self.tree.SetColumnWidth(2, 73)
        self.tree.SetColumnWidth(3, 73)
        self.tree.SetColumnWidth(4, 73)
        self.tree.SetColumnWidth(5, 40)

        # Determine which colunms are editable.
        self.tree.SetColumnEditable(0, False)
        self.tree.SetColumnEditable(1, False)
        self.tree.SetColumnEditable(2, True)
        self.tree.SetColumnEditable(3, True)
        self.tree.SetColumnEditable(4, True)
        self.tree.SetColumnEditable(5, False)

        self.tree.GetMainWindow().Bind(wx.EVT_RIGHT_UP, self.OnRightUp)
        self.tree.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit)
        '''
        self.tree.Bind(wx.EVT_TREE_ITEM_GETTOOLTIP,self.OnTreeTooltip)
        wx.EVT_MOTION(self.tree, self.OnMouseMotion)
        '''

        vbox.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.SetAutoLayout(True)

        self._need_update_parameters = self._need_update_model = False
        self.Bind(wx.EVT_SHOW, self.OnShow)

    # ============= Signal bindings =========================

    '''
    def OnTreeTooltip(self, event):
         itemtext = self.tree.GetItemText(event.GetItem())
         event.SetToolTip("This is a ToolTip for %s!" % itemtext)
         event.Skip()

    def OnMouseMotion(self, event):
        pos = event.GetPosition()
        item, flags, col = self.tree.HitTest(pos)

        if wx.TREE_HITTEST_ONITEMLABEL:
            self.tree.SetToolTipString("tool tip")
        else:
            self.tree.SetToolTipString("")

        event.Skip()
    '''

    def OnShow(self, event):
        if not event.Show: return
        #print "showing parameter"
        if self._need_update_model:
            #print "-model update"
            self.update_model(self.model)
        elif self._need_update_parameters:
            #print "-parameter update"
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
        if self.model != model: return

        if not IS_MAC and not self.IsShown():
            self._need_update_model = True
        else:
            self._need_update_model = self._need_update_parameters = False
            self._update_model()

    def update_parameters(self, model):
        if self.model != model: return
        if not IS_MAC and not self.IsShown():
            self._need_update_parameters = True
        else:
            self._need_update_parameters = False
            self._update_tree_nodes()

    def _update_model(self):
        # Delete the previous tree (if any).
        self.tree.DeleteAllItems()
        if self.model is None: return
        parameters = self.model.model_parameters()
        # Add a root node.
        self.root = self.tree.AddRoot("Model")
        # Add nodes from our data set .
        self._add_tree_nodes(self.root, parameters)
        self._update_tree_nodes()
        self.tree.ExpandAll(self.root)

    def _add_tree_nodes(self, branch, nodes):
        if isinstance(nodes,dict) and nodes != {}:
            for k in sorted(nodes.keys()):
                child = self.tree.AppendItem(branch, k)
                self._add_tree_nodes(child,nodes[k])
        elif ( ( isinstance(nodes, tuple) and nodes != () ) or
              ( isinstance(nodes, list) and nodes != [] ) ):
            for i,v in enumerate(nodes):
                child = self.tree.AppendItem(branch, '[%d]'%i)
                self._add_tree_nodes(child,v)

        elif isinstance(nodes, BaseParameter):
            self.tree.SetItemPyData(branch, nodes)

    def _update_tree_nodes(self):
        node = self.tree.GetRootItem()
        while node.IsOk():
            self._set_leaf(node)
            node = self.tree.GetNext(node)

    def _set_leaf(self, branch):
        par = self.tree.GetItemPyData(branch)
        if par is None: return

        if par.fittable:
            if par.fixed:
                fitting_parameter = 'No'
                low, high = '', ''
            else:
                fitting_parameter = 'Yes'
                low, high = (str(v) for v in par.bounds.limits)
        else:
            fitting_parameter = ''
            low, high = '', ''

        self.tree.SetItemText(branch, str(par.name), 1)
        self.tree.SetItemText(branch, str(nice(par.value)), 2)
        self.tree.SetItemText(branch, low, 3)
        self.tree.SetItemText(branch, high, 4)
        self.tree.SetItemText(branch, fitting_parameter, 5)

    def OnRightUp(self, evt):
        pos = evt.GetPosition()
        branch, flags, column = self.tree.HitTest(pos)
        if column == 5:
            par = self.tree.GetItemPyData(branch)
            if par is None: return

            if par.fittable:
                fitting_parameter = self.tree.GetItemText(branch, column)
                if fitting_parameter == 'No':
                    par.fixed = False
                    fitting_parameter = 'Yes'
                    low, high = (str(v) for v in par.bounds.limits)
                elif fitting_parameter == 'Yes':
                    par.fixed = True
                    fitting_parameter = 'No'
                    low, high = '', ''

                self.tree.SetItemText(branch, low, 3)
                self.tree.SetItemText(branch, high, 4)
                self.tree.SetItemText(branch, fitting_parameter, 5)
                signal.update_model(model=self.model, dirty=False)

    def OnEndEdit(self, evt):
        item = self.tree.GetSelection()
        self.node_object = self.tree.GetItemPyData(evt.GetItem())
        # TODO: Not an efficient way of updating values of Parameters
        # but it is hard to find out which column changed during edit
        # operation. This may be fixed in the future.
        wx.CallAfter(self.get_new_name, item, 1)
        wx.CallAfter(self.get_new_value, item, 2)
        wx.CallAfter(self.get_new_min, item, 3)
        wx.CallAfter(self.get_new_max, item, 4)

    def get_new_value(self, item, column):
        new_value = self.tree.GetItemText(item, column)

        # Send update message to other tabs/panels only if parameter value
        # is updated .
        if new_value != str(self.node_object.value):
            self.node_object.clip_set(float(new_value))
            signal.update_parameters(model=self.model)

    def get_new_name(self, item, column):
        new_name = self.tree.GetItemText(item, column)

        # Send update message to other tabs/panels only if parameter name
        # is updated.
        if new_name != str(self.node_object.name):
            self.node_object.name = new_name
            signal.update_model(model=self.model, dirty=False)

    def get_new_min(self, item, column):
        low = self.tree.GetItemText(item, column)
        if low == '': return
        low = float(low)
        high = self.node_object.bounds.limits[1]

        # Send update message to other tabs/panels only if parameter min range
        # value is updated.
        if low != self.node_object.bounds.limits[0]:
            self.node_object.range(low, high)
            signal.update_model(model=self.model, dirty=False)

    def get_new_max(self, item, column):
        high = self.tree.GetItemText(item, column)
        if high == '': return
        low = self.node_object.bounds.limits[0]
        high = float(high)
        # Send update message to other tabs/panels only if parameter max range
        # value is updated.
        if high != self.node_object.bounds.limits[1]:
            self.node_object.range(low, high)
            signal.update_model(model=self.model, dirty=False)
