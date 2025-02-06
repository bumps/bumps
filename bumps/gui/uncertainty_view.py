from .. import errplot

# from ..dream import stats as dream_stats
from ..dream import varplot as dream_varplot
from ..dream import views as dream_views
from .plot_view import PlotView


class UncertaintyView(PlotView):
    title = "Uncertainty"

    def update_parameters(self, *args, **kw):
        pass

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
    """
    CorrelationView has a maximum number of correlations that it will show.
    Change this by setting CorrelationView.MAX_CORR, either in the individual
    view or in the class.
    """

    MAX_CORR = 15
    title = "Correlations"

    def update_parameters(self, *args, **kw):
        pass

    def plot(self):
        if not self.plot_state:
            return
        with self.pylab_interface as pylab:
            pylab.clf()
            if self.plot_state.Nvar > self.MAX_CORR:
                return
            history = self.plot_state
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

    def update_parameters(self, *args, **kw):
        pass

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

    def update_parameters(self, *args, **kw):
        pass

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
