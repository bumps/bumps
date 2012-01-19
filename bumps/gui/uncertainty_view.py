from __future__ import with_statement

from ..dream import views as dream_views
from .. import errplot
from .plot_view import PlotView
from .signal import log_message

class UncertaintyView(PlotView):
    title = "Uncertainty"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            stats = dream_views.plot_vars(history)
            pylab.draw()
            # TODO: separate calculation of parameter uncertainty from plotting
            self.model.parameter_uncertainty = stats
            log_message(dream_views.format_vars(stats))
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.uncertainty_state
        self.plot()

class CorrelationView(PlotView):
    title = "Correlations"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            dream_views.plot_corrmatrix(history)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.uncertainty_state
        self.plot()


class TraceView(PlotView):
    title = "Parameter Trace"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            dream_views.plot_trace(history)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.uncertainty_state
        self.plot()

class ModelErrorView(PlotView):
    title = "Model Uncertainty"
    def plot(self):
        if not self.plot_state: return
        import pylab
        with self.pylab_interface:
            pylab.clf()
            # Won't get here if plot_state is None
            errplot.show_errors(self.plot_state)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.new_state(event.problem, event.uncertainty_state)
    def new_state(self, problem, state):
        # Should happen in a separate process
        self.plot_state = errplot.calc_errors_from_state(problem, state)
        self.plot()
