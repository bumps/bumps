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

# ==============================================================================

import wx
import wx.dataview as dv
from uuid import uuid4 as uuid_generate
import sys

from ..parameter import Parameter
from .util import nice
from . import signal

IS_MAC = wx.Platform == "__WXMAC__"


class ParameterCategory(object):
    def __init__(self, name):
        self.name = name


class ParametersModel(dv.PyDataViewModel):
    _columns = [
        {"label": "parameter", "type": "string"},
        {"label": "value", "type": "string"},
        {"label": "low", "type": "string"},
        {"label": "high", "type": "string"},
        {"label": "fittable", "type": "bool"},
        {"label": "path", "type": "str"},
        {"label": "link", "type": "str"},
    ]

    def __init__(self, log):
        dv.PyDataViewModel.__init__(self)
        self.SetParameters(None)
        self.log = log

        # The PyDataViewModel derives from both DataViewModel and from
        # DataViewItemObjectMapper, which has methods that help associate
        # data view items with Python objects. Normally a dictionary is used
        # so any Python object can be used as data nodes. If the data nodes
        # are weak-referencable then the objmapper can use a
        # WeakValueDictionary instead.
        # self.UseWeakRefs(True)

    def SetParameters(self, model):
        self.model = model
        self.data = params_to_list(model.model_parameters()) if model is not None else []
        self.Cleared()
        # self.log.write("data is %s"%str(self.data))
        # for obj in self.data:
        #    self.ItemAdded(self.ObjectToItem(obj["parent"]), self.ObjectToItem(obj))

    def UpdateParameters(self):
        for obj in self.data:
            self.ItemChanged(self.ObjectToItem(obj))

    def GetColumnCount(self):
        """5 data columns plus (name + 4)"""
        return len(self._columns)

    def GetColumnType(self, col):
        return self._columns[col]["type"]

    def GetChildren(self, parent, children):
        # The view calls this method to find the children of any node in the
        # control. There is an implicit hidden root node, and the top level
        # item(s) should be reported as children of this node. A List view
        # simply provides all items as children of this hidden root. A Tree
        # view adds additional items as children of the other items, as needed,
        # to provide the tree hierachy.

        # If the parent item is invalid then it represents the hidden root
        # item, so we'll use the genre objects as its children and they will
        # end up being the collection of visible roots in our tree.

        # Otherwise we'll fetch the python object associated with the parent
        # item and make DV items for each of it's child objects.
        if not parent:
            parent_id = None
        else:
            parent_id = self.ItemToObject(parent)["id"]
        child_list = [item for item in self.data if item["parent"] == parent_id]
        for child in child_list:
            children.append(self.ObjectToItem(child))
        return len(child_list)

    def IsContainer(self, item):
        # The hidden root is a container
        if not item:
            return True

        # Return False if it is a leaf
        node = self.ItemToObject(item)
        return not isinstance(node["value"], Parameter)

    def GetParent(self, item):
        # Return the item which is this item's parent.
        ##self.log.write("GetParent\n")

        if not item:
            return dv.NullDataViewItem

        node = self.ItemToObject(item)

        if node["parent"] is None:
            return dv.NullDataViewItem
        else:
            parent = node["parent"]
            for d in self.data:
                if d["id"] == parent:
                    return self.ObjectToItem(d)

    def GetAttr(self, item, col, attr):
        ##self.log.write('GetAttr')
        node = self.ItemToObject(item)
        if isinstance(node, ParameterCategory):
            attr.SetColour("blue")
            attr.SetBold(True)
            return True
        return False

    def GetValue(self, item, col):
        # Return the value to be displayed for this item and column. For this
        # example we'll just pull the values from the data objects we
        # associated with the items in GetChildren.

        # Fetch the data object for this item.
        node = self.ItemToObject(item)

        par = node["value"]

        if isinstance(par, ParameterCategory):
            # We'll only use the first column for the Genre objects,
            # for the other columns lets just return empty values
            mapper = [False if c["type"] == "bool" else "" for c in self._columns]
            mapper[0] = str(par.name)
            return mapper[col]

        elif isinstance(par, Parameter):
            if par.fittable:
                if par.fixed:
                    fitted = False
                    low, high = "", ""
                else:
                    fitted = True
                    low, high = (str(v) for v in par.prior.limits)
            else:
                fitted = False
                low, high = "", ""
            mapper = {
                0: fitted,
                1: str(par.name),
                2: str(nice(par.value)),
                3: low,
                4: high,
                5: str(node["path"]),
                6: str(node["link"]),
            }
            return mapper[col]

        else:
            raise RuntimeError("unknown node type")

    def SetValue(self, value, item, col):
        # self.log.write("SetValue: col %d,  %s\n" % (col, value))

        # We're not allowing edits in column zero (see below) so we just need
        # to deal with Song objects and cols 1 - 5

        node = self.ItemToObject(item)
        par = node["value"]
        if isinstance(par, Parameter):
            if col == 0:
                if par.fittable:
                    par.fixed = not value
            elif col == 2:
                par.clip_set(float(value))
            elif col == 3:
                if value == "":
                    return
                low = float(value)
                high = par.prior.limits[1]
                if low != par.prior.limits[0]:
                    par.range(low, high)
            elif col == 4:
                if value == "":
                    return
                high = float(value)
                low = par.prior.limits[0]
                if high != par.prior.limits[1]:
                    par.range(low, high)

        if col == 0:
            # if the number of fitting parameters changes then we have to
            # call model_reset in order to recalculate the varying parameters
            # (needed for dof calculations and SummaryView)
            self.model.model_reset()

        signal.update_parameters(model=self.model, delay=1)
        return True


class ParameterView(wx.Panel):
    title = "Parameters"
    default_size = (640, 500)

    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # sizers
        vbox = wx.BoxSizer(wx.VERTICAL)
        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.tree = dv.DataViewCtrl(
            self,
            style=wx.BORDER_THEME
            | dv.DV_ROW_LINES  # nice alternating bg colors
            | dv.DV_HORIZ_RULES
            | dv.DV_VERT_RULES
            | dv.DV_MULTIPLE,
        )
        self.dvModel = ParametersModel(sys.stdout)
        self.tree.AssociateModel(self.dvModel)
        # self.dvModel.DecRef()  # avoid memory leak !!

        self.tree.AppendToggleColumn("Fit?", 0, width=40, mode=dv.DATAVIEW_CELL_ACTIVATABLE)
        self.tree.AppendTextColumn("Parameter", 1, width=170)
        self.tree.AppendTextColumn("Value", 2, width=170, mode=dv.DATAVIEW_CELL_EDITABLE)
        self.tree.AppendTextColumn("Minimum", 3, width=100, mode=dv.DATAVIEW_CELL_EDITABLE)
        self.tree.AppendTextColumn("Maximum", 4, width=100, mode=dv.DATAVIEW_CELL_EDITABLE)
        self.tree.AppendTextColumn("Path", 5, width=300)
        self.tree.AppendTextColumn("Link", 6, width=300)

        vbox.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.SetAutoLayout(True)

        self.Bind(wx.EVT_SHOW, self.OnShow)

    # ============= Signal bindings =========================

    def OnShow(self, event):
        if not event.Show:
            return
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
        self.dvModel.SetParameters(self.model)
        if not IS_MAC:
            # Required for Linux; Windows doesn't care; causes mac to crash
            self.tree.AssociateModel(self.dvModel)
        # self.dvModel.DecRef()  # avoid memory leak !!
        self.expandAll()

    def update_parameters(self, model):
        if self.model != model:
            return
        # self.dvModel.Cleared()
        self.dvModel.UpdateParameters()

    def expandAll(self, max_depth=20):
        # print("calling expandAll")
        num_selected = -1
        depth = 0
        while True and depth < max_depth:
            self.tree.SelectAll()
            items = self.tree.GetSelections()
            if len(items) == num_selected:
                # then we've already selected everything before
                break
            else:
                num_selected = len(items)
                for item in items:
                    self.tree.Expand(item)
        self.tree.UnselectAll()


def params_to_dict(params):
    if isinstance(params, dict):
        ref = {}
        for k in sorted(params.keys()):
            ref[k] = params_to_dict(params[k])
    elif isinstance(params, tuple) or isinstance(params, list):
        ref = [params_to_dict(v) for v in params]
    elif isinstance(params, Parameter):
        if params.fittable:
            if params.fixed:
                fitted = "No"
                low, high = "", ""
            else:
                fitted = "Yes"
                low, high = (str(v) for v in params.prior.limits)
        else:
            fitted = ""
            low, high = "", ""

        ref = [str(params.name), str(nice(params.value)), low, high, fitted]
    return ref


def params_to_list(params, parent_uuid=None, output=None, path="M", links=None):
    output = [] if output is None else output
    links = {} if links is None else links
    if isinstance(params, dict):
        for k in sorted(params.keys()):
            # new_id = uuid_generate()
            # new_item = {"parent": parent_uuid, "id": new_id, "value": ParameterCategory(k)}
            # output.append(new_item)
            new_id = None
            params_to_list(params[k], parent_uuid=new_id, output=output, path=path + "." + k, links=links)
    elif isinstance(params, tuple) or isinstance(params, list):
        for i, v in enumerate(params):
            # new_id = uuid_generate()
            # new_item = {"parent": parent_uuid, "id": new_id, "value": ParameterCategory('[%d]' % (i,))}
            # output.append(new_item)
            new_id = None
            params_to_list(v, parent_uuid=new_id, output=output, path=path + "[%d]" % (i,), links=links)
    elif isinstance(params, Parameter):
        link_path = links.get(id(params), "")
        if link_path == "":
            links[id(params)] = path
        new_id = uuid_generate()
        new_item = {"parent": parent_uuid, "id": new_id, "value": params, "path": path, "link": link_path}
        output.append(new_item)

    return output
