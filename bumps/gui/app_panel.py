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
# Author: James Krycka, Nikunj Patel

"""
This module implements the AppPanel class which creates the main panel on top
of the frame of the GUI for the Bumps application.
"""

# ==============================================================================

import os
import threading
from copy import deepcopy

import wx
import wx.aui

from .. import plugin
from ..cli import load_best, load_model
from ..dream import stats as dream_stats
from ..util import redirect_console
from . import signal
from .convergence_view import ConvergenceView
from .fit_dialog import show_fit_config
from .fit_thread import EVT_FIT_COMPLETE, EVT_FIT_PROGRESS, FitThread
from .log_view import LogView
from .parameter_view import ParameterView
from .plot_view import PlotView
from .summary_view import SummaryView
from .uncertainty_view import CorrelationView, ModelErrorView, TraceView, UncertaintyView
from .util import nice
from .utilities import get_bitmap, phoenix

# File selection strings.
MODEL_EXT = ".pickle"
MODEL_FILES = "Model files (*%s)|*%s" % (MODEL_EXT, MODEL_EXT)
PYTHON_FILES = "Script files (*.py)|*.py"
DATA_FILES = "Data files (*.dat)|*.dat"
TEXT_FILES = "Text files (*.txt)|*.txt"
ALL_FILES = "All files (*.*)|*"
PARS_FILES = "Parameter files (*.par)|*.par"

# Custom colors.
WINDOW_BKGD_COLOUR = "#ECE9D8"

# ==============================================================================


class AppPanel(wx.Panel):
    """
    This class builds the GUI for the application on a panel and attaches it
    to the frame.
    """

    def __init__(self, *args, **kw):
        # Create a panel on the frame.  This will be the only child panel of
        # the frame and it inherits its size from the frame which is useful
        # during resize operations (as it provides a minimal size to sizers).

        wx.Panel.__init__(self, *args, **kw)

        self.SetBackgroundColour("WHITE")

        # Modify the tool bar.
        frame = self.GetTopLevelParent()
        self.init_toolbar(frame)
        self.init_menubar(frame)

        # Reconfigure the status bar.
        self.init_statusbar(frame, [-34, -50, -16, -16])

        # Create the model views
        self.init_views()

        # Add data menu
        mb = frame.GetMenuBar()
        data_view = self.view["data"]
        if hasattr(data_view, "menu"):
            mb.Append(data_view.menu(), data_view.title)

        # Create a PubSub receiver.
        signal.connect(self.OnLogMessage, "log")
        signal.connect(self.OnModelNew, "model.new")
        signal.connect(self.OnModelChange, "model.update_structure")
        signal.connect(self.OnModelSetpar, "model.update_parameters")

        EVT_FIT_PROGRESS.Bind(self, wx.ID_ANY, wx.ID_ANY, self.OnFitProgress)
        EVT_FIT_COMPLETE.Bind(self, wx.ID_ANY, wx.ID_ANY, self.OnFitComplete)

        # _fit_abort and maybe _uncertainty_state are managed by fit_lock
        self.fit_lock = threading.Lock()
        self._fit_abort = False
        # self._uncertainty_state = None
        self.fit_thread = None
        self.fit_config = None

    def init_menubar(self, frame):
        """
        Adds items to the menu bar, menus, and menu options.
        The menu bar should already have a simple File menu and a Help menu.
        """
        mb = frame.GetMenuBar()

        file_menu_id = mb.FindMenu("File")
        file_menu = mb.GetMenu(file_menu_id)
        # help_menu = mb.GetMenu(mb.FindMenu("Help"))

        # Add items to the "File" menu (prepending them in reverse order).
        # Grey out items that are not currently implemented.
        file_menu.PrependSeparator()

        _item = file_menu.Prepend(wx.ID_ANY, "E&xport Results ...", "Save theory, data and parameters")
        frame.Bind(wx.EVT_MENU, self.OnFileExportResults, _item)

        _item = file_menu.Prepend(wx.ID_ANY, "&Reload", "Reload the existing model")
        frame.Bind(wx.EVT_MENU, self.OnFileReload, _item)

        _item = file_menu.Prepend(wx.ID_SAVEAS, "Save &As", "Save model as another name")
        frame.Bind(wx.EVT_MENU, self.OnFileSaveAs, _item)
        # file_menu.Enable(id=wx.ID_SAVEAS, enable=False)
        _item = file_menu.Prepend(wx.ID_SAVE, "&Save", "Save model")
        frame.Bind(wx.EVT_MENU, self.OnFileSave, _item)
        # file_menu.Enable(id=wx.ID_SAVE, enable=False)
        _item = file_menu.Prepend(wx.ID_OPEN, "&Open", "Open existing model")
        frame.Bind(wx.EVT_MENU, self.OnFileOpen, _item)
        # file_menu.Enable(id=wx.ID_OPEN, enable=False)
        _item = file_menu.Prepend(wx.ID_ANY, "Apply &Pararameters", "Apply parameters from .par file to loaded model")
        frame.Bind(wx.EVT_MENU, self.OnParsFileOpen, _item)
        _item = file_menu.Prepend(wx.ID_NEW, "&New", "Create new model")
        frame.Bind(wx.EVT_MENU, self.OnFileNew, _item)
        # file_menu.Enable(id=wx.ID_NEW, enable=False)

        # Add 'Fitting' menu to the menu bar and define its options.
        # Items are initially greyed out, but will be enabled after a script
        # is loaded.
        fit_menu = self.fit_menu = wx.Menu()

        _item = fit_menu.Append(wx.ID_ANY, "Start", "Start fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnFitStart, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_start = _item

        _item = fit_menu.Append(wx.ID_ANY, "Stop", "Stop fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnFitStop, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_stop = _item

        _item = fit_menu.Append(wx.ID_ANY, "&Options ...", "Edit fitting options")
        frame.Bind(wx.EVT_MENU, self.OnFitOptions, _item)
        self.fit_menu_options = _item

        # _item = fit_menu.Append(wx.ID_ANY,
        #                        "&Save ...",
        #                        "Save fit results")
        # frame.Bind(wx.EVT_MENU, self.OnFitSave, _item)
        # fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_options = _item

        mb.Append(fit_menu, "&Fitting")

    def init_toolbar(self, frame):
        """Populates the tool bar."""
        self.tb = frame.GetToolBar()

        script_bmp = get_bitmap("import_script.png", wx.BITMAP_TYPE_PNG)
        reload_bmp = get_bitmap("reload.png", wx.BITMAP_TYPE_PNG)
        start_bmp = get_bitmap("start_fit.png", wx.BITMAP_TYPE_PNG)
        stop_bmp = get_bitmap("stop_fit.png", wx.BITMAP_TYPE_PNG)

        _tool = self._add_tool(script_bmp, "Open model", "Load model from script")
        frame.Bind(wx.EVT_TOOL, self.OnFileOpen, _tool)
        _tool = self._add_tool(reload_bmp, "Reload model", "Reload model from script")
        frame.Bind(wx.EVT_TOOL, self.OnFileReload, _tool)
        # TODO: add reload

        self.tb.AddSeparator()

        _tool = self._add_tool(start_bmp, "Start Fit", "Start fitting operation")
        frame.Bind(wx.EVT_TOOL, self.OnFitStart, _tool)
        self.tb.EnableTool(_tool.GetId(), False)
        self.tb_start = _tool

        _tool = self._add_tool(stop_bmp, "Stop Fit", "Stop fitting operation")
        frame.Bind(wx.EVT_TOOL, self.OnFitStop, _tool)
        self.tb.EnableTool(_tool.GetId(), False)
        self.tb_stop = _tool

        self.tb.Realize()
        frame.SetToolBar(self.tb)

    def _add_tool(self, bitmap, label, help):
        if phoenix:
            return self.tb.AddTool(wx.ID_ANY, label, bitmap, shortHelp=help)
        else:
            return self.tb.AddSimpleTool(wx.ID_ANY, bitmap, label, help)

    def init_statusbar(self, frame, subbars):
        """Divides the status bar into multiple segments."""

        self.sb = frame.GetStatusBar()
        self.sb.SetFieldsCount(len(subbars))
        self.sb.SetStatusWidths(subbars)

    def init_views(self):
        # initial view
        self.aui = wx.aui.AuiNotebook(self)
        self.aui.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSE, self.OnViewTabClose)
        self.view_constructor = {
            "data": plugin.data_view(),
            "model": plugin.model_view(),
            "parameter": ParameterView,
            "summary": SummaryView,
            "log": LogView,
            "convergence": ConvergenceView,
            "uncertainty": UncertaintyView,
            "correlation": CorrelationView,
            "trace": TraceView,
            "error": ModelErrorView,
        }
        self.view_list = [
            "data",
            "model",
            "parameter",
            "summary",
            "log",
            "convergence",
            "uncertainty",
            "correlation",
            "trace",
            "error",
        ]
        self.view = {}
        for v in self.view_list:
            if self.view_constructor[v]:
                self.view[v] = self.view_constructor[v](self.aui, size=(600, 600))
                self.aui.AddPage(self.view[v], self.view_constructor[v].title)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.aui, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def show_view(self, tag):
        if self.view[tag].Parent == self.aui:
            self.aui.SetSelection(self.aui.GetPageIndex(self.view[tag]))
        else:
            self.view[tag].Raise()
            self.view[tag].SetFocus()

    def OnViewTabClose(self, evt):
        win = self.aui.GetPage(evt.GetSelection())
        # print "Closing tab",win.GetId()
        for k, w in self.view.items():
            if w == win:
                tag = k
                break
        else:
            raise RuntimeError("Lost track of view")
        # print "creating external frame"
        state = self.view[tag].get_state()
        constructor = self.view_constructor[tag]
        frame = wx.Frame(self, title=constructor.title, size=constructor.default_size)
        panel = constructor(frame)
        self.view[tag] = panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel, 1, wx.EXPAND)
        frame.SetSizer(sizer)
        frame.Bind(wx.EVT_CLOSE, self.OnViewFrameClose)
        frame.Show()
        panel.set_state(state)
        evt.Skip()

    def OnViewFrameClose(self, evt):
        win = evt.GetEventObject()
        # print "Closing frame",win.GetId()
        for k, w in self.view.items():
            if w.GetParent() == win:
                tag = k
                break
        else:
            raise RuntimeError("Lost track of view!")
        state = self.view[tag].get_state()
        constructor = self.view_constructor[tag]
        panel = constructor(self.aui)
        self.view[tag] = panel
        self.aui.AddPage(panel, constructor.title)
        panel.set_state(state)
        evt.Skip()

    # model viewer interface
    def OnLogMessage(self, message):
        for v in self.view.values():
            if hasattr(v, "log_message"):
                v.log_message(message)

    def OnModelNew(self, model):
        self.set_model(model)

    def OnModelChange(self, model):
        for v in self.view.values():
            if hasattr(v, "update_model"):
                v.update_model(model)

    def OnModelSetpar(self, model):
        for v in self.view.values():
            if hasattr(v, "update_parameters"):
                v.update_parameters(model)

    def OnFileNew(self, event):
        self.new_model()

    def OnFileOpen(self, event):
        # Load the script which will contain model definition and data.
        dlg = wx.FileDialog(
            self,
            message="Select File",
            # defaultDir=os.getcwd(),
            # defaultFile="",
            wildcard=(ALL_FILES),
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
        )

        # Wait for user to close the dialog.
        status = dlg.ShowModal()
        path = dlg.GetPath()
        dlg.Destroy()

        # Process file if user clicked okay.
        if status == wx.ID_OK:
            self.load_model(path)

    def OnParsFileOpen(self, event):
        # Load a parameters (.pars) file to apply to current problem.
        dlg = wx.FileDialog(
            self,
            message="Select Parameters File",
            # defaultDir=os.getcwd(),
            # defaultFile="",
            wildcard=PARS_FILES,
            style=wx.FD_OPEN,
        )

        # Wait for user to close the dialog.
        status = dlg.ShowModal()
        path = dlg.GetPath()
        dlg.Destroy()

        # Process file if user clicked okay.
        if status == wx.ID_OK:
            self.apply_parameters(path)

    def OnFileReload(self, event):
        path = getattr(self, "_reload_path", self.model.path)
        self.load_model(path)

    def OnFileSave(self, event):
        if self.model is not None:
            # Force file extension to be MODEL_EXT
            self.model.path = os.path.splitext(self.model.path)[0] + MODEL_EXT
            self.save_model(self.model.path)
        else:
            self.OnFileSaveAs(event)

    def OnFileSaveAs(self, event):
        dlg = wx.FileDialog(
            self,
            message="Select File",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=(MODEL_FILES + "|" + ALL_FILES),
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR | wx.FD_OVERWRITE_PROMPT,
        )
        # Wait for user to close the dialog.
        status = dlg.ShowModal()
        path = dlg.GetPath()
        dlg.Destroy()

        # Process file if user clicked okay.
        if status == wx.ID_OK:
            self.model.path = os.path.splitext(path)[0] + MODEL_EXT
            self.save_model(self.model.path)

    def OnFileExportResults(self, event):
        dlg = wx.DirDialog(self, message="Export results", defaultPath=os.getcwd(), style=wx.DD_DEFAULT_STYLE)
        # Wait for user to close the dialog.
        status = dlg.ShowModal()
        path = dlg.GetPath()
        dlg.Destroy()

        # Process file if user clicked okay.
        if status == wx.ID_OK:
            self.save_results(path)

    def set_fit_config(self, fit_config):
        self.fit_config = fit_config

    def OnFitOptions(self, event):
        # TODO: send fit_config to dialog rather than using options.FIT_CONFIG.
        # Wait until sasview wx is abandoned before making this change.
        show_fit_config(self)

    def OnFitStart(self, event):
        if self.fit_thread:
            self.sb.SetStatusText("Error: Fit already running")
            return
        # TODO: better access to model parameters
        if len(self.model.getp()) == 0:
            raise ValueError("Problem has no fittable parameters")

        # Start a new thread worker and give fit problem to the worker.
        fitclass = self.fit_config.selected_fitter
        options = self.fit_config.selected_values

        # Clear abort and uncertainty state
        self.abort = False
        self.uncertainty_state = None
        self.fit_thread = FitThread(
            win=self,
            abort_test=self._is_abort,
            problem=self.model,
            fitclass=fitclass,
            options=options,
            # Number of seconds between updates to the GUI, or 0 for no updates
            convergence_update=5,
            uncertainty_update=3600,
        )
        self.fit_thread.start()
        self.sb.SetStatusText("Fit status: Running", 3)

    def _is_abort(self):
        return self.abort

    @property
    def abort(self):
        with self.fit_lock:
            return self._fit_abort

    @abort.setter
    def abort(self, state):
        with self.fit_lock:
            self._fit_abort = state

    @property
    def uncertainty_state(self):
        """
        MCMC chain created by the DREAM optimizer.

        This is a thread-locked attribute maintained by the fitting thread.
        A deep copy of the attribute is made before setting so that state
        can continue to update within the fit thread while the user interface
        accesses the last state that was set.
        """
        with self.fit_lock:
            return self._uncertainty_state

    @uncertainty_state.setter
    def uncertainty_state(self, state):
        state_copy = deepcopy(state)
        with self.fit_lock:
            self._uncertainty_state = state_copy

    def OnFitStop(self, event):
        self.abort = True

    def OnFitComplete(self, event):
        if self.fit_thread is not None:
            self.fit_thread.join(1)  # 1 second timeout on join
            if self.fit_thread.is_alive():
                signal.log_message(message="fit thread failed to complete")
        self.fit_thread = None
        chisq = event.problem.chisq_str(nllf=event.value)
        event.problem.setp(event.point)
        signal.update_parameters(model=event.problem)
        signal.log_message(message="done with chisq %s" % chisq)
        signal.log_message(message=event.info)
        self.sb.SetStatusText("Fit status: Complete", 3)
        beep()

    def OnFitSave(self, event):
        raise NotImplementedError()
        dlg = wx.FileDialog(
            self,
            message="Fit results",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=(MODEL_FILES + "|" + ALL_FILES),
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR | wx.FD_OVERWRITE_PROMPT,
        )
        # Wait for user to close the dialog.
        status = dlg.ShowModal()
        path = dlg.GetPath()
        dlg.Destroy()

        # Process file if user clicked okay.
        if status == wx.ID_OK:
            self.save_fit(path)

    def OnFitProgress(self, event):
        self._fit_progress(event)
        return

        # Note: The following code was used to debugging plotting memory
        # and speed issues. It is here in case it might be useful again.
        if event.message == "uncertainty_final":
            event.message = "uncertainty_update"  # don't do model uncertainty
            n = 0
            import gc
            import time

            import psutil

            pid = os.getpid()
            proc = psutil.Process()
            t0 = time.perf_counter()
            signal.log_message("****** entering cycle ******")
            while True:
                self._fit_progress(event)
                t1 = time.perf_counter()
                msg = "========== cycle %d pid %d time %.3f s ===========" % (n, pid, t1 - t0)
                t0 = t1
                # signal.log_message(msg)
                # signal.log_message(f"memory {proc.memory_full_info()}")
                gc.collect()
                print(msg)
                print("memory", proc.memory_full_info())
                n += 1
                wx.Yield()
                # if n > 100: break

    def _fit_progress(self, event):
        if event.message == "progress":
            chisq = event.problem.chisq_str(nllf=event.value)
            message = "step %5d chisq %s" % (event.step, chisq)
            signal.log_message(message=message)
        elif event.message == "improvement":
            event.problem.setp(event.point)
            event.problem.model_update()
            signal.update_parameters(model=event.problem)
        elif event.message == "convergence_update":
            self.view["convergence"].OnFitProgress(event)
        elif event.message in ("uncertainty_update", "uncertainty_final"):
            # TODO: revert to event.uncertainty_state if memory leak isn't fixed
            # Note: uncertainty_state is updated directly by the fit thread.
            # It's a copy protected by fit_lock so it should be self-consistent.
            state = self.uncertainty_state
            stats = dream_stats.var_stats(state.draw())
            self.console["state"] = self.uncertainty_state
            self.view["uncertainty"].fit_progress(event.problem, state, stats)
            self.view["correlation"].fit_progress(event.problem, state)
            self.view["trace"].fit_progress(event.problem, state)
            # TODO: send parameter stats to a view
            # It would be nice to have a "log scale" log of parameter history,
            # where the density of entries decreases exponentially over time.
            # So for example, the last 5, then 10 20 30 40 50, then 100, 200,
            # 300, 400, 500, ... Maybe have a slider so that you can drag it
            # forward and backward and see changes over time. Use box plots
            # to represent mode, median, mean, 68% and 95% intervals. Export
            # latest or all to CSV or text for pasting into word or excel.
            if event.message == "uncertainty_final":
                self.view["error"].fit_progress(event.problem, state)
                # Format the final variable nicely and show them on the console.
                # Note: only done at the end because GUI gets overwhelmed when
                # the text control has too much text.
                signal.log_message(dream_stats.format_vars(stats))
        else:
            raise ValueError("Unknown fit progress message " + event.message)

    def new_model(self):
        from ..plugin import new_model as gen

        self.set_model(gen())

    def load_model(self, path):
        self._reload_path = path
        model = load_model(path)
        signal.model_new(model=model)

    def apply_parameters(self, path):
        load_best(self.model, path)
        # TODO: remove this once LHS randomization is controlled from the command line. Ticket #51
        if hasattr(self.model, "undefined"):
            del self.model.undefined
        signal.update_parameters(model=self.model)

    def save_model(self, path):
        try:
            import dill

            dump = lambda obj, fid: dill.dump(obj, fid, recurse=True)
        except ImportError:
            from pickle import dump
        try:
            if hasattr(plugin, "save_json"):
                plugin.save_json(self.model, path)
            elif hasattr(self.model, "save_json"):
                self.model.save_json(path)
            with open(path, "wb") as fid:
                dump(self.model, fid)
        except Exception:
            import traceback

            signal.log_message(message=traceback.format_exc())

    def save_results(self, path):
        output_path = os.path.join(path, self.model.name)

        # Storage directory
        if not os.path.exists(path):
            os.mkdir(path)

        # Ask model to save its information
        self.model.save(output_path)

        # Save a snapshot of the model that can (hopefully) be reloaded
        self.save_model(output_path + MODEL_EXT)

        # Save the current state of the parameters
        with redirect_console(output_path + ".out"):
            self.model.show()
        pardata = "".join("%s %.15g\n" % (name, value) for name, value in zip(self.model.labels(), self.model.getp()))
        open(output_path + ".par", "wt").write(pardata)

        # Produce model plots
        self.model.plot(figfile=output_path)

        # Produce uncertainty plots
        if self.uncertainty_state is not None:
            with redirect_console(output_path + ".err"):
                self.uncertainty_state.show(figfile=output_path)
            self.uncertainty_state.save(output_path)

    def _add_measurement_type(self, type):
        """
        Add the panels needed to view a measurement of the given type.

        *type* is fitness.__class__, where fitness is the measurement cost function.
        """
        name = type.__name__
        if type not in self.data_tabs:
            tab = self.data_notebook.add_tab(type, name + " Data")
            constructor = getattr(type, "data_panel", PlotView)
            constructor(tab)
        if type not in self.model_notebook and hasattr(type, "model_panel"):
            tab = self.model_notebook.add_tab(type, name + " Model")
            type.model_panel(tab)

    def _view_problem(self, problem):
        """
        Set the model and data views to those necessary to display the problem.
        """
        # What types of measurements do we have?
        models = problem.models if hasattr(problem, "models") else [problem]
        types = set(p.fitness.__class__ for p in models)
        for p in types:
            self._add_measurement_type(p)

        # Show only the relevant views
        for p, tab in self.data_notebook.tabs():
            tab.Show(p in types)
        for p, tab in self.model_notebook.tabs():
            tab.Show(p in types)

    def set_model(self, model):
        # Inform the various tabs that the model they are viewing has changed.
        self.model = model

        # Point all of our views at the new model
        for v in self.view.values():
            if hasattr(v, "set_model"):
                v.set_model(model)
        self.console["model"] = model

        # Enable appropriate menu items.
        self.fit_menu.Enable(id=self.fit_menu_start.GetId(), enable=True)
        self.fit_menu.Enable(id=self.fit_menu_stop.GetId(), enable=True)
        self.fit_menu.Enable(id=self.fit_menu_options.GetId(), enable=True)

        # Enable appropriate toolbar items.
        self.tb.EnableTool(self.tb_start.GetId(), True)
        self.tb.EnableTool(self.tb_stop.GetId(), True)
        if hasattr(model, "broken_constraints"):
            signal.log_message(message="Unsatisfied constraints: \n\t%s" % (",\n\t".join(model.broken_constraints)))
        if hasattr(model, "path"):
            signal.log_message(message="loaded " + model.path)
            self.GetTopLevelParent().SetTitle("Bumps: %s" % model.name)
        else:
            signal.log_message(message="new model")
            self.GetTopLevelParent().SetTitle("Bumps")


SOUND = None


def beep():
    """
    Play fit completion sound.
    """
    wx.Bell()
    ## FIXME why doesn't sound work?
    # global SOUND
    # if SOUND is None:
    #    SOUND = wx.Sound(resource('done.wav'))
    # if SOUND.IsOk():
    #    SOUND.Play(wx.SOUND_ASYNC)
