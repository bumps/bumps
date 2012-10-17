"""
Fit job definition for the distributed job queue.
"""
import os
import sys
import json
import cPickle as pickle
from copy import deepcopy

import matplotlib

from . import cli
from . import __version__

# Site configurate determines what kind of mapper to use
# This should be true in cli.py as well
from . import parameter
from .mapper import MPMapper as mapper
from . import monitor
from .fitters import FitDriver

def fitservice(request):
    matplotlib.use('Agg')

    path = os.getcwd()

    service_version = __version__
    request_version = str(request['version'])
    if service_version != request_version:
        raise ValueError('fitter version %s does not match request %s'
                         % (service_version, request_version))

    data = request['data']
    model = str(data['package'])

    service_model_version = __version__
    request_model_version = str(data['version'])
    if service_model_version != request_model_version:
        raise ValueError('%s version %s does not match request %s'
                         % (model, service_model_version, request_model_version))
    options = pickle.loads(str(data['options']))
    problem = pickle.loads(str(data['problem']))
    problem.store = path
    problem.output_path = os.path.join(path,'model')


    fitdriver = FitDriver(options.fit, problem=problem, **options)

    fitdriver.mapper = mapper.start_mapper(problem, options.args)
    problem.show()
    print "#", " ".join(sys.argv)
    best, fbest = fitdriver.fit()
    cli.remember_best(fitdriver, problem, best)
    matplotlib.pyplot.show()
    return list(best), fbest

class ServiceMonitor(monitor.TimedUpdate):
    """
    Display fit progress on the console
    """
    def __init__(self, problem, path, progress=60, improvement=60):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.path = path
        self.problem = deepcopy(problem)
        self.images = []
    def show_progress(self, history):
        self.problem.setp(history.point[0])
        status = {
            "step":  history.step[0],
            "cost":  history.value[0]/self.problem.dof,
            "pars":  history.point[0],
        }
        open(os.path.join(self.path,'status.json'),"wt").write(json.dumps(status))
        status['table'] = self.problem.summarize()
        status['images'] = "\n".join('<img file="%s" alt="%s" />'%(f,f)
                                     for f in self.images)
        html = """\
<html><body>
Generation %(step)d, chisq %(cost)g
<pre>
%(table)
</pre>
<img file="K-model.png" alt="model plot"/>
</body></html>
"""%status

    def show_improvement(self, history):
        #print "step",history.step[0],"chisq",history.value[0]
        self.problem.setp(history.point[0])
        try:
            import pylab
            pylab.hold(False)
            self.problem.plot(figfile=os.path.join(self.path,'K'))
            pylab.gcf().canvas.draw()
        except:
            raise
