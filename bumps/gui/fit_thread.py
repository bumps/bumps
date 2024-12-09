from copy import deepcopy
from threading import Thread
import os

import wx.lib.newevent

from .. import monitor
from ..fitters import FitDriver
from ..mapper import MPMapper, SerialMapper, can_pickle
from ..util import redirect_console

from .convergence_view import ConvergenceMonitor
# ==============================================================================

PROGRESS_DELAY = 5
IMPROVEMENT_DELAY = 5

(FitProgressEvent, EVT_FIT_PROGRESS) = wx.lib.newevent.NewEvent()
(FitCompleteEvent, EVT_FIT_COMPLETE) = wx.lib.newevent.NewEvent()


# NOTE: GUI monitors are running in a separate thread.  They should not
# touch the problem internals.
class GUIProgressMonitor(monitor.TimedUpdate):
    def __init__(self, win, problem, progress=None, improvement=None):
        monitor.TimedUpdate.__init__(
            self, progress=progress or PROGRESS_DELAY, improvement=improvement or IMPROVEMENT_DELAY
        )
        self.win = win
        self.problem = problem

    def show_progress(self, history):
        evt = FitProgressEvent(
            problem=self.problem,
            message="progress",
            step=history.step[0],
            value=history.value[0],
            point=history.point[0] + 0,
        )  # avoid race
        wx.PostEvent(self.win, evt)

    def show_improvement(self, history):
        evt = FitProgressEvent(
            problem=self.problem,
            message="improvement",
            step=history.step[0],
            value=history.value[0],
            point=history.point[0] + 0,
        )  # avoid race
        wx.PostEvent(self.win, evt)


class GUIMonitor(monitor.Monitor):
    """
    Generic GUI monitor for fitting.

    Sends a fit progress event to the selected window every *rate*
    seconds. The *monitor* is a bumps monitor which processes fit
    history as it arrives. Instead of the usual *show_progress* method
    to update the managing process it has a *progress* method which
    returns a dictionary of progress attributes.  These attributes are
    added to a *FitProgressEvent* along with *problem* and *message* which
    is then posted to *win*.

    *problem* should be the fit problem handed to the fit thread, and not
    a copy. This is because it is used for direct comparison with the current
    fit object in the progress panels so that stray messages don't cause
    confusion when making graphs.

    *message* is a dispatch string used by the OnFitProgress event processor
    in the app to determine which progress panel should receive the event.

    TODO: this is a pretty silly design, especially since GUI monitor is only
    used for the convergence plot, and a separate monitor class was created
    for DREAM progress updates.
    """

    def __init__(self, win, problem, message, monitor, rate=None):
        self.time = 0
        self.rate = rate  # rate=0 for no progress update, only final
        self.win = win
        self.problem = problem
        self.message = message
        self.monitor = monitor

    def config_history(self, history):
        self.monitor.config_history(history)
        history.requires(time=1)

    def __call__(self, history):
        self.monitor(history)
        if self.rate > 0 and history.time[0] >= self.time + self.rate:
            evt = FitProgressEvent(problem=self.problem, message=self.message, **self.monitor.progress())
            wx.PostEvent(self.win, evt)
            self.time = history.time[0]

    def final(self):
        """
        Close out the monitor
        """
        evt = FitProgressEvent(problem=self.problem, message=self.message, **self.monitor.progress())
        wx.PostEvent(self.win, evt)


# Horrible hacks:
# (1) We are grabbing uncertainty_state from the history on monitor update
# and holding onto it for the monitor final call. Need to restructure the
# history/monitor interaction so that object lifetimes are better controlled.
# (2) We set the uncertainty state directly in the GUI window. This is an
# attempt to work around a memory leak which causes problems in the GUI when
# it is updated too frequently. If the problem is that the event structure
# is too large and wx isn't cleaning up properly then this should address it,
# but if the problem lies within the DREAM plot functions or within matplotlib
# then this change won't do anything and should be reverted.
class DreamMonitor(monitor.Monitor):
    def __init__(self, win, problem, message, fitter, rate=None):
        self.time = 0
        self.rate = rate  # rate=0 for no progress update, only final
        self.win = win
        self.problem = problem
        self.fitter = fitter
        self.message = message
        self.uncertainty_state = None

    def config_history(self, history):
        history.requires(time=1)

    def __call__(self, history):
        self.uncertainty_state = getattr(history, "uncertainty_state", None)
        if self.rate > 0 and history.time[0] >= self.time + self.rate and self.uncertainty_state is not None:
            # Note: win.uncertainty_state protected by win.fit_lock
            self.time = history.time[0]
            self.win.uncertainty_state = self.uncertainty_state
            evt = FitProgressEvent(
                problem=self.problem,
                message="uncertainty_update",
                # uncertainty_state=deepcopy(self.uncertainty_state),
            )
            wx.PostEvent(self.win, evt)

    def final(self):
        """
        Close out the monitor
        """
        if self.uncertainty_state is not None:
            # Note: win.uncertainty_state protected by win.fit_lock
            self.win.uncertainty_state = self.uncertainty_state
            evt = FitProgressEvent(
                problem=self.problem,
                message="uncertainty_final",
                # uncertainty_state=deepcopy(self.uncertainty_state),
            )
            wx.PostEvent(self.win, evt)


# ==============================================================================


class FitThread(Thread):
    """Run the fit in a separate thread from the GUI thread."""

    def __init__(
        self,
        win,
        abort_test=None,
        problem=None,
        fitclass=None,
        options=None,
        mapper=None,
        convergence_update=5,
        uncertainty_update=300,
    ):
        # base class initialization
        # Process.__init__(self)

        Thread.__init__(self)
        self.win = win
        self.abort_test = abort_test
        self.problem = problem
        self.fitclass = fitclass
        self.options = options
        self.mapper = mapper
        self.convergence_update = convergence_update
        self.uncertainty_update = uncertainty_update

    def run(self):
        # TODO: we have no interlocks on changes in problem state.  What
        # happens when the user changes the problem while a fit is being run?
        # May want to keep a history of changes to the problem definition,
        # along with a function to reverse them so we can handle undo.

        # NOTE: Problem must be the original problem (not a copy) when used
        # inside the GUI monitor otherwise AppPanel will not be able to
        # recognize that it is the same problem when updating views.
        monitors = [
            GUIProgressMonitor(self.win, self.problem),
            GUIMonitor(
                self.win,
                self.problem,
                message="convergence_update",
                monitor=ConvergenceMonitor(),
                rate=self.convergence_update,
            ),
            DreamMonitor(
                self.win, self.problem, fitter=self.fitclass, message="uncertainty_update", rate=self.uncertainty_update
            ),
        ]
        # Only use parallel if the problem can be pickled
        mapper = MPMapper if can_pickle(self.problem) else SerialMapper

        # Be safe and send a private copy of the problem to the fitting engine
        # print "fitclass",self.fitclass
        problem = deepcopy(self.problem)
        # print "fitclass id",id(self.fitclass),self.fitclass,threading.current_thread()
        driver = FitDriver(
            self.fitclass,
            problem=problem,
            monitors=monitors,
            abort_test=self.abort_test,
            mapper=mapper.start_mapper(problem, []),
            **self.options,
        )

        x, fx = driver.fit()
        # Give final state message from monitors
        for M in monitors:
            if hasattr(M, "final"):
                M.final()

        with redirect_console() as fid:
            driver.show()
            captured_output = fid.getvalue()

        evt = FitCompleteEvent(problem=self.problem, point=x, value=fx, info=captured_output)
        wx.PostEvent(self.win, evt)
