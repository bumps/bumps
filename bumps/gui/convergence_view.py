import numpy as np

from .. import monitor
from ..plotutil import coordinated_colors
from .plot_view import PlotView


class ConvergenceMonitor(monitor.Monitor):
    """
    Gather statistics about the best, worst, median and +/- 1 interquartile
    range.  This will be the input for the convergence plot.
    """

    def __init__(self):
        self.pop = []

    def config_history(self, history):
        history.requires(population_values=1, value=1)

    def __call__(self, history):
        best = history.value[0]
        try:
            pop = history.population_values[0]
            n = len(pop)
            p = np.sort(pop)
            (
                QI,
                Qmid,
            ) = int(0.2 * n), int(0.5 * n)
            self.pop.append((best, p[0], p[QI], p[Qmid], p[-1 - QI], p[-1]))
        except (AttributeError, TypeError):
            self.pop.append((best,))

    def progress(self):
        if not self.pop:
            return dict(pop=np.empty((0, 1), "d"))
        else:
            return dict(pop=np.array(self.pop))


class ConvergenceView(PlotView):
    title = "Convergence"

    def plot(self):
        if not self.plot_state:
            return
        pop, best = self.plot_state
        with self.pylab_interface as pylab:
            pylab.clf()
            ni, npop = pop.shape
            iternum = np.arange(1, ni + 1)
            tail = int(0.25 * ni)
            c = coordinated_colors(base=(0.4, 0.8, 0.2))
            if npop == 5:
                pylab.fill_between(iternum[tail:], pop[tail:, 1], pop[tail:, 3], color=c["light"], label="_nolegend_")
                pylab.plot(iternum[tail:], pop[tail:, 2], label="80% range", color=c["base"])
                pylab.plot(iternum[tail:], pop[tail:, 0], label="_nolegend_", color=c["base"])
            pylab.plot(iternum[tail:], best[tail:], label="best", color=c["dark"])
            pylab.xlabel("iteration number")
            pylab.ylabel("chisq")
            pylab.legend()
            # pylab.gca().set_yscale('log')
            pylab.draw()

    def update(self, best, pop):
        self.plot_state = pop, best
        self.plot()

    def OnFitProgress(self, event):
        if event.problem != self.model:
            return
        pop = 2 * event.pop / self.model.dof
        self.update(pop[:, 0], pop[:, 1:])
