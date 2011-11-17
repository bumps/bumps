# Dead code
print "mystic.fitservice is untested"

_ = '''
from numpy import inf

#from park Request, Service
class Service: pass
class Request: pass

from .solver import Minimizer
from .optimizer.de import DifferentialEvolution
from .problem import Function

class FitService(Service):
    def prepare(self, request):
        self._build_solver(request)
        self.population = self.solver.start()
        self.best = inf

    def onSignalModifyProblem(self, handler, msg):
        """
        Steering mechanism from user
        """
        #process the msg in terms of bounds changing, variables floated
        #or fixed, etc.

        #modify the optimizer state to reflect these changes

        #update the mapper with a set of changes that need to happen
        #on the individual workers to bring them up to date with the
        #new problem
        self.map.update(msg)

    def run(self, handler):
        F = self.solver
        while True:
            popvals = handler.map(F.problem, self.population)
            F.update(self.population, popvals)
            if F.history.value[0] < self.best:
                value = F.history.value[0]
                handler.improved(value)
                self.best = value
            if F.isdone(): break
            self.population = F.next()
            handler.ready()
        return F.history.value[0],F.history.point[0]

    def checkpoint(self):
        return dict(request=self.request,
                    population=self.population,
                    best=self.best,
                    history=self.solver.history)

    def restore(self, state):
        self.request = state['request']
        self.population = state['population']
        self.best = state['best']
        self._build_solver(self.request)
        self.solver.history = state['history']
        return

    def progress(self):
        """
        Report on the stopping condition which is nearest to completion.

        Note that this only reports on failure conditions.  Success
        conditions are reported as improvements to the fit.
        """
        failure = self.solver.failure.progress(self.history)
        return failure.k, failure.n, failure.units

    def cleanup(self):
        pass

    def _build_solver(self, request):
        self.request = request
        if request.version != FitRequest.version:
            raise ValueError('expected FitRequest %s but got %s'
                             %(FitRequest.version, request.version))
        self.solver = Minimizer(
                                problem = request.problem,
                                strategy = request.strategy,
                                monitors = request.monitors,
                                success = request.success,
                                failure = request.failure
                                )
        self.solver.history.requires(value=1, point=1)

class FitRequest(Request):
    version = '0.9'
    service = "mystic.FitService"
    requires = [('mystic',1.2)]
    def __init__(self,
                 problem = None,
                 strategy = DifferentialEvolution(),
                 monitors = [],
                 success = None,
                 failure = None):
        fn, po, bounds = problem
        self.problem = Function(f=fn, args=(), po=po, bounds=bounds)
        self.strategy = strategy
        self.monitors = monitors
        self.success = success
        self.failure = failure
        #self.requires  = strategy.requires + problem.requires

@park.service_wrapper
def fit(*args, **kw):
    return FitRequest(*args, **kw)
'''
