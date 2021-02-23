from __future__ import with_statement

from ..dream import views as dream_views
from ..dream import stats as dream_stats
from ..dream import varplot as dream_varplot
from .. import errplot
from .plot_view import PlotView


class UncertaintyView(PlotView):
    title = "Uncertainty"

    def plot(self):
        if not self.plot_state:
            return
        history, stats = self.plot_state
        with self.pylab_interface as pylab:
            pylab.clf()
            dream_varplot.plot_vars(history.draw(), stats)
            pylab.draw()

    def update(self, state):
        self.plot_state = state
        self.plot()

    def fit_progress(self, problem, uncertainty_state, stats):
        if problem != self.model:
            return
        self.update((uncertainty_state, stats))


class CorrelationView(PlotView):
    title = "Correlations"

    def plot(self):
        if not self.plot_state:
            return
        # suppress correlation plot if too many variables
        if self.plot_state.Nvar > 15:
            return
        history = self.plot_state
        with self.pylab_interface as pylab:
            pylab.clf()
            dream_views.plot_corrmatrix(history.draw())
            pylab.draw()

    def update(self, state):
        self.plot_state = state
        self.plot()

    def fit_progress(self, problem, uncertainty_state):
        if problem != self.model:
            return
        self.update(uncertainty_state)


class TraceView(PlotView):
    title = "Parameter Trace"

    def plot(self):
        if not self.plot_state:
            return
        history = self.plot_state
        with self.pylab_interface as pylab:
            pylab.clf()
            dream_views.plot_trace(history)
            pylab.draw()

    def update(self, state):
        self.plot_state = state
        self.plot()

    def fit_progress(self, problem, uncertainty_state):
        if problem != self.model:
            return
        self.plot_state = uncertainty_state
        self.plot()


class ModelErrorView(PlotView):
    title = "Model Uncertainty"

    def plot(self):
        if not self.plot_state:
            return
        with self.pylab_interface as pylab:
            pylab.clf()
            # Won't get here if plot_state is None
            errplot.show_errors(self.plot_state)
            pylab.draw()

    def update(self, problem, state):
        # TODO: Should happen in a separate process
        self.plot_state = errplot.calc_errors_from_state(problem, state)
        self.plot()

    def fit_progress(self, problem, uncertainty_state):
        if problem != self.model:
            return
        self.update(problem, uncertainty_state)
