from __future__ import with_statement

import numpy

from .. import monitor
from .plot_view import PlotView
from ..plotutil import coordinated_colors


class ConvergenceMonitor(monitor.Monitor):
    """
    Gather statistics about the best, worst, median and +/- 1 interquartile
    range.  This will be the input for the convergence plot.
    """
    def __init__(self):
        self.best = []
        self.pop = []
    def config_history(self, history):
        history.requires(population_values=1, value=1)
    def __call__(self, history):
        self.best.append(history.value[0])
        try: pop = history.population_values[0]
        except: pop = [history.value[0]]
        n = len(pop)
        p = numpy.sort(pop)
        if n > 5:
            QI,Qmid, = int(0.2*n),int(0.5*n)
            self.pop.append((p[0],p[QI],p[Qmid],p[-1-QI],p[-1]))
        else:
            self.pop.append(p)
    def progress(self):
        if not self.best:
            return dict(best=numpy.empty(0), pop=numpy.empty((0,1)))
        else:
            return dict(best=numpy.array(self.best),
                        pop=numpy.array(self.pop))


class ConvergenceView(PlotView):
    title = "Convergence"
    def plot(self):
        if not self.plot_state: return
        pop,best = self.plot_state
        with self.pylab_interface:
            import pylab
            pylab.clf()
            n,p = pop.shape
            iternum = numpy.arange(1,n+1)
            tail = int(0.25*n)
            pylab.hold(True)
            c = coordinated_colors(base=(0.4,0.8,0.2))
            if p==5:
                pylab.fill_between(iternum[tail:], pop[tail:,1], pop[tail:,3],
                                   color=c['light'], label='_nolegend_')
                pylab.plot(iternum[tail:],pop[tail:,2],
                           label="80% range", color=c['base'])
                pylab.plot(iternum[tail:],pop[tail:,0],
                           label="_nolegend_", color=c['base'])
            else:
                pylab.plot(iternum,pop, label="population",
                           color=c['base'])
            pylab.plot(iternum[tail:], best[tail:], label="best",
                       color=c['dark'])
            pylab.xlabel('iteration number')
            pylab.ylabel('chisq')
            pylab.legend()
            #pylab.gca().set_yscale('log')
            pylab.hold(False)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        dof = event.problem.dof
        pop = 2*event.pop/dof
        best = 2*event.best/dof
        self.plot_state = pop,best
        self.plot()
