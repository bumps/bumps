"""
Fit job definition for the distributed job queue.
"""

import os
import sys
import json

try:
    import dill as pickle
except ImportError:
    import pickle

from . import cli
from . import __version__

# Site configuration determines what kind of mapper to use
# This should be true in cli.py as well
from .mapper import MPMapper as Mapper
from . import monitor
from .fitters import FitDriver


def fitservice(request):
    import matplotlib

    matplotlib.use("Agg")

    path = os.getcwd()

    service_version = __version__
    request_version = str(request["version"])
    if service_version != request_version:
        raise ValueError("fitter version %s does not match request %s" % (service_version, request_version))

    data = request["data"]
    model = str(data["package"])

    service_model_version = __version__
    request_model_version = str(data["version"])
    if service_model_version != request_model_version:
        raise ValueError(
            "%s version %s does not match request %s" % (model, service_model_version, request_model_version)
        )
    options = pickle.loads(str(data["options"]))
    problem = pickle.loads(str(data["problem"]))
    problem.store = path
    problem.output_path = os.path.join(path, "model")

    fitdriver = FitDriver(options.fit, problem=problem, **options)

    fitdriver.mapper = Mapper.start_mapper(problem, options.args)
    problem.show()
    print("#", " ".join(sys.argv))
    best, fbest = fitdriver.fit()
    cli.save_best(fitdriver, problem, best)
    matplotlib.pyplot.show()
    return list(best), fbest


class ServiceMonitor(monitor.TimedUpdate):
    """
    Display fit progress on the console
    """

    def __init__(self, problem, path, progress=60, improvement=60):
        monitor.TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.path = path
        self.problem = problem
        self.images = []

    def show_progress(self, history):
        p = self.problem.getp()
        try:
            self.problem.setp(history.point[0])
            dof = self.problem.dof
            summary = self.problem.summarize()
        finally:
            self.problem.setp(p)

        status = {
            "step": history.step[0],
            "cost": history.value[0] / dof,
            "pars": history.point[0],
        }
        json_status = json.dumps(status)
        open(os.path.join(self.path, "status.json"), "wt").write(json_status)
        status["table"] = summary
        status["images"] = "\n".join('<img file="%s" alt="%s" />' % (f, f) for f in self.images)
        html_status = (
            """\
<html><body>
Generation %(step)d, chisq %(cost)g
<pre>
%(table)s
</pre>
%(images)s
</body></html>
"""
            % status
        )
        open(os.path.join(self.path, "status.html"), "wt").write(html_status)

    def show_improvement(self, history):
        import pylab

        # print "step",history.step[0],"chisq",history.value[0]
        self.problem.setp(history.point[0])
        pylab.cla()
        self.problem.plot(figfile=os.path.join(self.path, "K"))
        pylab.gcf().canvas.draw()
