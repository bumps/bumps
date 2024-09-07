
from copy import deepcopy
from threading import Thread
from threading import Event
import traceback

from blinker import Signal

import numpy as np
from bumps import monitor
from bumps.fitters import FitDriver, nllf_scale, format_uncertainty
from bumps.mapper import MPMapper, SerialMapper, can_pickle
from bumps.util import redirect_console
from bumps.history import History

#from .convergence_view import ConvergenceMonitor
# ==============================================================================

PROGRESS_DELAY = 5
IMPROVEMENT_DELAY = 5

EVT_FIT_PROGRESS = Signal()
EVT_FIT_COMPLETE = Signal()

# NOTE: GUI monitors are running in a separate thread.  They should not
# touch the problem internals.


class GUIProgressMonitor(monitor.TimedUpdate):
    def __init__(self, problem, progress=None, improvement=None):
        monitor.TimedUpdate.__init__(
            self, progress=progress or PROGRESS_DELAY,
            improvement=improvement or IMPROVEMENT_DELAY)
        self.problem = problem

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = format_uncertainty(scale*history.value[0], err)
        evt = dict(
            # problem=self.problem,
            message="progress",
            step=history.step[0],
            value=history.value[0],
            chisq=chisq,
            point=history.point[0]+0)  # avoid race
        EVT_FIT_PROGRESS.send(evt)

    def show_improvement(self, history):
        evt = dict(
            message="improvement",
            step=history.step[0],
            value=history.value[0],
            point=history.point[0]+0)  # avoid race
        EVT_FIT_PROGRESS.send(evt)


class ConvergenceMonitor(monitor.Monitor):
    """
    Generic GUI monitor for fitting.

    Sends a convergence_update event every *rate*
    seconds.  Gathers statistics about the best, worst, median and +/- 1 interquartile
    range.  This will be the input for the convergence plot.

    *problem* should be the fit problem handed to the fit thread, and not
    a copy. This is because it is used for direct comparison with the current
    fit object in the progress panels so that stray messages don't cause
    confusion when making graphs.

    *message* is a dispatch string used by the OnFitProgress event processor
    in the app to determine which progress panel should receive the event.
    """

    def __init__(self, problem, message="convergence_update", rate=0):
        self.time = 0
        self.rate = rate  # rate=0 for no progress update, only final
        self.problem = problem
        self.message = message
        self.pop = []


    def config_history(self, history):
        history.requires(population_values=1, value=1)
        history.requires(time=1)
        # history.requires(time=1, population_values=1, value=1)

    def __call__(self, history):
        # from old ConvergenceMonitor:
        best = history.value[0]
        try:
            pop = history.population_values[0]
            n = len(pop)
            p = np.sort(pop)
            QI,Qmid, = int(0.2*n),int(0.5*n)
            self.pop.append((best, p[0],p[QI],p[Qmid],p[-1-QI],p[-1]))
        except AttributeError:
            self.pop.append((best, ))

        if self.rate > 0 and history.time[0] >= self.time+self.rate:
            evt = dict(
                message=self.message,
                pop=self.progress())
            EVT_FIT_PROGRESS.send(evt)
            self.time = history.time[0]

    def progress(self):
        return np.empty((0,1),'d') if not self.pop else np.array(self.pop)

    def final(self):
        """
        Close out the monitor
        """
        evt = dict(
            message=self.message,
            pop=self.progress())
        EVT_FIT_PROGRESS.send(evt)

# Horrible hacks:
# (1) We are grabbing uncertainty_state from the history on monitor update
# and holding onto it for the monitor final call. Need to restructure the
# history/monitor interaction so that object lifetimes are better controlled.


class DreamMonitor(monitor.Monitor):
    def __init__(self, problem, message, fitter, rate=0):
        self.time = 0
        self.rate = rate  # rate=0 for no progress update, only final
        self.update_counter = 0
        self.problem = problem
        self.fitter = fitter
        self.message = message
        self.uncertainty_state = None
        # emit None uncertainty state to start with
        evt = dict(
            message=message,
            uncertainty_state=None,
        )
        EVT_FIT_PROGRESS.send(evt)

    def config_history(self, history):
        history.requires(time=1)

    def __call__(self, history):
        self.uncertainty_state = getattr(history, 'uncertainty_state', None)
        self.time = history.time[0]
        if (self.rate > 0):
            update_counter = history.time[0] // self.rate
            if update_counter > self.update_counter:
                self.update_counter = update_counter
                evt = dict(
                    message=self.message,
                    time = self.time,
                    uncertainty_state=deepcopy(self.uncertainty_state),
                )
                EVT_FIT_PROGRESS.send(evt)

    def final(self):
        """
        Close out the monitor
        """
        if self.uncertainty_state is not None:
            # Note: win.uncertainty_state protected by win.fit_lock
            # self.win.uncertainty_state = self.uncertainty_state
            evt = dict(
                message="uncertainty_final",
                time=self.time,
                uncertainty_state=deepcopy(self.uncertainty_state),
            )
            EVT_FIT_PROGRESS.send(evt)

# ==============================================================================


class FitThread(Thread):
    """Run the fit in a separate thread from the GUI thread."""

    def __init__(self, abort_event: Event, problem=None,
                 fitclass=None, options=None, mapper=None, parallel=0,
                 convergence_update=5, uncertainty_update=300,
                 terminate_on_finish=False):
        # base class initialization
        # Process.__init__(self)

        Thread.__init__(self)
        self.abort_event = abort_event
        self.problem = problem
        self.fitclass = fitclass
        self.options = options if isinstance(options, dict) else dict()
        self.mapper = mapper
        self.parallel = parallel
        self.convergence_update = convergence_update
        self.uncertainty_update = uncertainty_update
        self.terminate_on_finish = terminate_on_finish

    def abort_test(self):
        return self.abort_event.is_set()

    def run(self):
        # TODO: we have no interlocks on changes in problem state.  What
        # happens when the user changes the problem while a fit is being run?
        # May want to keep a history of changes to the problem definition,
        # along with a function to reverse them so we can handle undo.

        # NOTE: Problem must be the original problem (not a copy) when used
        # inside the GUI monitor otherwise AppPanel will not be able to
        # recognize that it is the same problem when updating views.
        try:
            monitors = [GUIProgressMonitor(self.problem),
                        ConvergenceMonitor(self.problem,
                                        rate=self.convergence_update),
                        # GUIMonitor(self.problem,
                        #            message="convergence_update",
                        #            monitor=ConvergenceMonitor(),
                        #            rate=self.convergence_update),
                        DreamMonitor(self.problem,
                                    fitter=self.fitclass,
                                    message="uncertainty_update",
                                    rate=self.uncertainty_update),
                        ]

            mapper = self.mapper
            if mapper is None:
                # fallback options if no mapper is specified
                mp_available = True
                try:
                    import _multiprocessing
                except ImportError:
                    mp_available = False
                # Only use MPMapper if the problem can be pickled
                # and multiprocessing is available, and parallel != 1
                mapper = MPMapper if mp_available and self.parallel != 1 and can_pickle(self.problem) else SerialMapper

            # Be safe and send a private copy of the problem to the fitting engine
            # print "fitclass",self.fitclass
            problem = deepcopy(self.problem)
            # print "fitclass id",id(self.fitclass),self.fitclass,threading.current_thread()
            driver = FitDriver(
                self.fitclass, problem=problem,
                monitors=monitors, abort_test=self.abort_test,
                mapper=mapper.start_mapper(problem, [], cpus=self.parallel),
                **self.options)

            x, fx = driver.fit()
            # Give final state message from monitors
            for M in monitors:
                if hasattr(M, 'final'):
                    M.final()

            with redirect_console() as fid:
                driver.show()
                captured_output = fid.getvalue()

            evt = dict(message="complete", problem=self.problem,
                    point=x, value=fx, info=captured_output, fitter_id = self.fitclass.id)
            EVT_FIT_COMPLETE.send(evt)
            self.result = evt

        except Exception as exc:
            tb = "".join(traceback.TracebackException.from_exception(exc).format())
            evt = dict(message="error", error_string=str(exc), traceback=tb)
            EVT_FIT_COMPLETE.send(evt)
