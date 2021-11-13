from __future__ import with_statement

from ..dream import views as dream_views
from ..dream import stats as dream_stats
from ..dream import varplot as dream_varplot
from .. import errplot
from .. import fitters
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
    title = "Correlations"

    def update_parameters(self, *args, **kw):
        pass

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

        if self.plot_state is None:
            return
        with self.pylab_interface as pylab:
            pylab.clf()
            # Won't get here if plot_state is None
            if self.problem is not None:
                self.problem.plot_forwardmc(self.plot_state)
            else:
                errplot.show_errors(self.plot_state)
            pylab.draw()

    def update(self, problem, state):
        # TODO: Should happen in a separate process
        # TODO: require correct handling of multifit problem:
        #  How much of the model lists and iterations should be done
        #  outside of the problem.fitness.calc_forwardmc?
        #  should we include a method in MultiFitProblem?
        #  In refl1d, this would mean the removal of the creation of model lists in fitness
        #  see code below - we could just pass the model lists to
        #  the equivilent of refl1d.errors.calc_errors?
        # TODO: is there a better way of identifying and handling a MultiFitProblem
        #  object to retreve fitness?
        if hasattr(problem, 'models'):
            model = problem.active_model.fitness
        else:
            model = problem.fitness

        # TODO: Crude check for now, just to see if BaseFitProblem implementation is working.
        #  Not sure if we will need to iterate over all elements of the list to confirm?
        #  Would we expect different models to be derived from a different version of fitness?
        #  Probably not.
        if hasattr(model, 'plot_forwardmc'):
            # Shim included for deprecation of plugin (show_error)
            self.problem = problem
            self.plot_state = fitters.get_points_from_state(state)
        else:
            import warnings
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("Plugin usage for model uncertainty and error plot is Deprecated, \n"
                          "in future include a plot_forwardmc method in the fitness object", DeprecationWarning)
            self.plot_state = errplot.calc_errors_from_state(problem, state)
        self.plot()

    def fit_progress(self, problem, uncertainty_state):
        if problem != self.model:
            return
        self.update(problem, uncertainty_state)
