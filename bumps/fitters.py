import time
from copy import deepcopy

import numpy

from . import monitor, parameter
from .history import History
from . import initpop

class ConsoleMonitor(monitor.TimedUpdate):
    """
    Display fit progress on the console
    """
    def __init__(self, problem, progress=1, improvement=30):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = deepcopy(problem)
    def show_progress(self, history):
        print "step", history.step[0], \
            "cost", 2*history.value[0]/self.problem.dof
    def show_improvement(self, history):
        #print "step",history.step[0],"chisq",history.value[0]
        self.problem.setp(history.point[0])
        print parameter.summarize(self.problem.parameters)

class StepMonitor(monitor.Monitor):
    """
    Collect information at every step of the fit and save it to a file.

    *fid* is the file to save the information to
    *fields* is the list of "step|time|value|point" fields to save

    The point field should be last in the list.
    """
    FIELDS = ['step', 'time', 'value', 'point']
    def __init__(self, problem, fid, fields=FIELDS):
        if any(f not in self.FIELDS for f in fields):
            raise ValueError("invalid monitor field")
        self.dof = problem.dof
        self.fid = fid
        self.fields = fields
        self._pattern = "%%(%s)s\n" % (")s %(".join(fields))
        fid.write("# "+' '.join(fields)+'\n')
    def config_history(self, history):
        history.requires(time=1, value=1, point=1, step=1)
    def __call__(self, history):
        point = " ".join("%.15g"%v for v in history.point[0])
        time = "%g"%history.time[0]
        step = "%d"%history.step[0]
        value = "%.15g"%(2*history.value[0]/self.dof)
        out = self._pattern%dict(point=point, time=time,
                                 value=value, step=step)
        self.fid.write(out)


class MonitorRunner(object):
    """
    Adaptor which allows solvers to accept progress monitors.
    """
    def __init__(self, monitors, problem):
        if monitors == None:
            monitors = [ConsoleMonitor(problem)]
        self.monitors = monitors
        self.history = History(time=1,step=1,point=1,value=1,
                               population_points=1, population_values=1)
        for M in self.monitors:
            M.config_history(self.history)
        self._start = time.time()
    def __call__(self, step, point, value,
                 population_points=None, population_values=None):
        self.history.update(time=time.time()-self._start,
                            step=step, point=point, value=value,
                            population_points=population_points,
                            population_values=population_values)
        for M in self.monitors:
            M(self.history)

class FitBase(object):
    def __init__(self, problem):
        """Fit the models and show the results"""
        self.problem = problem
    def solve(self, monitors=None, mapper=None, **options):
        raise NotImplementedError

class MultiStart(FitBase):
    name = "Multistart Monte Carlo"
    settings = [('starts', 100)]
    def __init__(self, fitter):
        self.fitter = fitter
        self.problem = fitter.problem
    def solve(self, monitors=None, mapper=None, **options):
        starts = options.pop('starts',1)
        reset = not options.pop('keep_best',True)
        f_best = numpy.inf
        for _ in range(max(starts,1)):
            print "round",_
            x,fx = self.fitter.solve(monitors=monitors, mapper=mapper,
                                     **options)
            if fx < f_best:
                x_best, f_best = x,fx
                print x_best, fx
            if reset:
                self.problem.randomize()
            elif 0:
                # Jitter
                self.problem.setp(x_best)
                pop = initpop.eps_init(1, self.problem.parameters,
                                       include_current=False, eps=1e-5)
                self.problem.setp(pop[0])
        return x_best, f_best

class DEFit(FitBase):
    name = "Differential Evolution"
    settings = [('steps',1000), ('pop', 10), ('CR', 0.9), ('F', 2.0) ]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        from .mystic.optimizer import de
        from .mystic.solver import Minimizer
        from .mystic.stop import Steps
        if monitors == None:
            monitors = [ConsoleMonitor(self.problem)]
        if mapper is not None:
            _mapper = lambda p,x: mapper(x)
        else:
            _mapper = lambda p,x: map(self.problem.nllf,x)
        strategy = de.DifferentialEvolution(npop=options['pop'],
                                            CR=options['CR'],
                                            F=options['F'])
        minimize = Minimizer(strategy=strategy, problem=self.problem,
                             monitors=monitors,
                             failure=Steps(options['steps']))
        x = minimize(mapper=_mapper)
        return x, minimize.history.value[0]


class BFGSFit(FitBase):
    name = "Quasi-Newton BFGS"
    settings = [('steps',3000), ('starts',100) ]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        from quasinewton import quasinewton, STATUS
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        result = quasinewton(fn=self.problem.nllf,
                             x0=self.problem.getp(),
                             monitor = self._monitor,
                             itnlimit = options['steps'],
                             )
        code = result['status']
        print "%d: %s" % (code, STATUS[code])
        return result['x'], result['fx']
    def _monitor(self, step, x, fx):
        self._update(step=step, point=x, value=fx,
                     population_points=[x],
                     population_values=[fx])
        return True

class PSFit(FitBase):
    name = "Particle Swarm"
    settings = [('steps',3000), ('pop', 1) ]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: map(self.problem.nllf,x)
        from random_lines import particle_swarm
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        cfo = dict(parallel_cost=mapper,
                   n = len(bounds[0]),
                   x0 = self.problem.getp(),
                   x1 = bounds[0],
                   x2 = bounds[1],
                   f_opt = 0,
                   monitor = self._monitor)
        NP = int(cfo['n']*options['pop'])

        result = particle_swarm(cfo, NP, maxiter=options['steps'])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best

    def _monitor(self, step, x, fx, k):
        self._update(step=step, point=x[:,k], value=fx[k],
                     population_points=x.T, population_values=fx)
        return True

class RLFit(FitBase):
    name = "Random Lines"
    settings = [('steps',3000), ('starts',20), ('pop', 0.5), ('CR', 0.9)]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        if mapper is None:
            mapper = lambda x: map(self.problem.nllf,x)
        from random_lines import random_lines
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        cfo = dict(parallel_cost=mapper,
                   n = len(bounds[0]),
                   x0 = self.problem.getp(),
                   x1 = bounds[0],
                   x2 = bounds[1],
                   f_opt = 0,
                   monitor = self._monitor)
        NP = max(int(cfo['n']*options['pop']),3)

        result = random_lines(cfo, NP, maxiter=options['steps'],
                              CR=options['CR'])
        satisfied_sc, n_feval, f_best, x_best = result

        return x_best, f_best

    def _monitor(self, step, x, fx, k):
        #print "rl best",k, x.shape,fx.shape
        self._update(step=step, point=x[:,k], value=fx[k],
                     population_points=x.T, population_values=fx)
        return True


class PTFit(FitBase):
    name = "Parallel Tempering"
    settings = [('steps',1000), ('nT', 25), ('CR', 0.9),
                ('burn',4000),  ('Tmin', 0.1), ('Tmax', 10)]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from partemp import parallel_tempering
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        T = numpy.logspace(numpy.log10(options['Tmin']),
                           numpy.log10(options['Tmax']),
                           options['nT'])
        history = parallel_tempering(nllf=self.problem.nllf,
                                    p=self.problem.getp(),
                                    bounds=bounds,
                                    #logfile="partemp.dat",
                                    T=T,
                                    CR=options['CR'],
                                    steps=options['steps'],
                                    burn=options['burn'],
                                    monitor=self._monitor)
        return history.best_point, history.best
    def _monitor(self, step, x, fx, P, E):
        self._update(step=step, point=x, value=fx,
                     population_points=P, population_values=E)
        return True

class AmoebaFit(FitBase):
    name = "Nelder-Mead Simplex"
    settings = [ ('steps',1000), ('starts',1), ('radius',0.15) ]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from simplex import simplex
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        result = simplex(f=self.problem.nllf, x0=self.problem.getp(),
                         bounds=bounds,
                         update_handler=self._monitor,
                         maxiter=options['steps'],
                         radius=options['radius'])
        # Let simplex propose the starting point for the next amoeba
        # fit in a multistart amoeba context.  If the best is always
        # used, the fit can get stuck in a local minimum.
        self.problem.setp(result.next_start)
        return result.x, result.fx
    def _monitor(self, k, n, x, fx):
        self._update(step=k, point=x[0], value=fx[0],
                     population_points=x, population_values=fx)
        return True

class SnobFit(FitBase):
    name = "SNOBFIT"
    settings = [('steps',200)]
    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        # TODO: no mapper??
        from snobfit.snobfit import snobfit #@UnresolvedImport snobfit is optional
        self._update = MonitorRunner(problem=self.problem,
                                     monitors=monitors)
        bounds = numpy.array([p.bounds.limits
                              for p in self.problem.parameters]).T
        x, fx, _ = snobfit(self.problem, self.problem.getp(), bounds,
                          fglob=0, callback=self._monitor)
        return x, fx
    def _monitor(self, k, x, fx, improved):
        # TODO: snobfit does have a population...
        self._update(step=k, point=x, value=fx,
                     population_points=[x], population_values=[fx])

try:
    from .dream import MCMCModel
except:
    MCMCModel = object
class DreamModel(MCMCModel):
    """
    DREAM wrapper for fit problems.
    """
    def __init__(self, problem=None, mapper=None):
        """
        Create a sampling from the multidimensional likelihood function
        represented by the problem set using dream.
        """
        self.problem = problem
        self.bounds = zip(*[p.bounds.limits for p in problem.parameters])
        self.labels = [p.name for p in problem.parameters]

        self.mapper = mapper if mapper else lambda p: map(self.nllf,p)

    def log_density(self, x):
        return -self.nllf(x)

    def nllf(self, x):
        """Negative log likelihood of seeing models given parameters *x*"""
        #print "eval",x; sys.stdout.flush()
        return self.problem.nllf(x)

    def map(self, pop):
        #print "calling mapper",self.mapper
        return -numpy.array(self.mapper(pop))

class DreamFit(FitBase):
    name = "DREAM"
    settings = [('steps',500),  ('burn', 0), ('pop', 10), ('init', 'eps')]
    def __init__(self, problem):
        self.dream_model = DreamModel(problem)

    def solve(self, monitors=None, mapper=None, **options):
        _fill_defaults(options, self.settings)
        from . import dream

        if mapper: self.dream_model.mapper = mapper
        self._update = MonitorRunner(problem=self.dream_model.problem,
                                     monitors=monitors)

        population = initpop.generate(self.dream_model.problem, **options)
        pop_size = population.shape[0]
        population = population[None,:,:]
        sampler = dream.Dream(model=self.dream_model, population=population,
                              draws = pop_size*options['steps'],
                              burn = pop_size*options['burn'],
                              monitor = self._monitor,
                              DE_noise = 1e-6)

        self.state = sampler.sample()
        self.state.mark_outliers()
        self.state.keep_best()
        self.state.title = self.dream_model.problem.name

        x,fx = self.state.best()

        points,logp = self.state.sample()
        assert logp[-1] == fx
        assert all(points[-1,i]==xi for i,xi in enumerate(x))

        return x,-fx

    def _monitor(self, state, pop, logp):
        self._update.history.uncertainty_state = state # Get an early copy of the state
        step = state.generation
        x,fx = state.best()
        self._update(step=step, point=x, value=-fx,
                     population_points=pop, population_values=-logp)
        return True

    def save(self, output_path):
        self.state.save(output_path)

    def plot(self, output_path):
        self.state.show(figfile=output_path)
        self.error_plot(figfile=output_path)

    def show(self):
        pass

    def error_plot(self, figfile):
        # Produce error plot
        import errplot, pylab
        # TODO: shouldn't mix calc and display!
        res = errplot.calc_errors_from_state(self.dream_model.problem,
                                            self.state)
        if res is not None:
            pylab.figure()
            errplot.show_errors(res)
            pylab.savefig(figfile+"-errors.png", format='png')

class Resampler(FitBase):
    #TODO: why isn't cli.resynth using this?
    def __init__(self, fitter):
        raise NotImplementedError
        self.fitter = fitter
    def solve(self, **options):
        starts = options.pop('starts',1)
        restart = options.pop('restart',False)
        x,fx = self.fitter.solve(**options)
        points = _resampler(self.fitter, x, samples=starts,
                            restart=restart, **options)
        return x,fx

def _resampler(fitter, xinit, samples=100, restart=False, **options):
    """
    Refit the result multiple times with resynthesized data, building
    up an array in Result.samples which contains the best fit to the
    resynthesized data.  *samples* is the number of samples to generate.
    *fitter* is the (local) optimizer to use. **kw are the parameters
    for the optimizer.
    """
    x = xinit
    points = []
    try: # TODO: some solvers already catch KeyboardInterrupt
        for _ in range(samples):
            #print "== resynth %d of %d" % (i, samples)
            fitter.problem.resynth_data()
            if restart:
                parameter.randomize(fitter.problem.parameters)
            else:
                fitter.problem.setp(x)
            x, fx = fitter.solve(**options)
            points.append(numpy.hstack((fx,x)))
            #print parameter.summarize(self.problem.parameters)
            #print "[chisq=%g]" % (nllf*2/self.problem.dof)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore the state of the problem
        fitter.problem.restore_data()
        fitter.problem.setp(xinit)
        fitter.problem.model_update()
    return points


class FitDriver(object):
    def __init__(self, fitclass=None, problem=None, monitors=None,
                 mapper=None, **options):
        self.fitclass = fitclass
        self.problem = problem
        self.options = options
        self.monitors = monitors
        self.mapper = mapper if mapper else lambda p: map(problem.nllf,p)

    def fit(self):
        fitter = self.fitclass(self.problem)
        starts = self.options.get('starts', 1)
        if starts > 1:
            fitter = MultiStart(fitter)
        t0 = time.clock()
        x, fx = fitter.solve(monitors=self.monitors,
                             mapper=self.mapper,
                             **self.options)
        self.fitter = fitter
        self.time = time.clock() - t0
        self.result = x, fx
        self.problem.setp(x)
        return x, fx

    def show(self):
        if hasattr(self.problem, 'show'):
            self.problem.show()
        if hasattr(self.fitter, 'show'):
            self.fitter.show()

    def save(self, output_path):
        #print "calling driver save"
        if hasattr(self.problem, 'save'):
            self.problem.save(output_path)
        if hasattr(self.fitter, 'save'):
            self.fitter.save(output_path)

    def plot(self, output_path):
        #print "calling fitter.plot"
        if hasattr(self.problem, 'plot'):
            self.problem.plot(figfile=output_path)
        if hasattr(self.fitter, 'plot'):
            self.fitter.plot(output_path=output_path)


def _fill_defaults(options, settings):
    for field,value in settings:
        if field not in options:
            options[field] = value

class FitOptions(object):
    # Field labels and types for all possible fields
    FIELDS = dict(
        starts = ("Starts",          "int"),
        steps  = ("Steps",           "int"),
        burn   = ("Burn-in Steps",   "int"),
        pop    = ("Population",      "float"),
        init   = ("Initializer",     ("eps","lhs","cov","random")),
        CR     = ("Crossover Ratio", "float"),
        F      = ("Scale",           "float"),
        nT     = ("# Temperatures",  "int"),
        Tmin   = ("Min Temperature", "float"),
        Tmax   = ("Max Temperature", "float"),
        radius = ("Simplex Radius",  "float"),
        )

    def __init__(self, fitclass):
        self.fitclass = fitclass
        self.options = dict(fitclass.settings)
    def set_from_cli(self, opts):
        # Convert supplied options to the correct types and save them in value
        for field,reset_value in self.fitclass.settings:
            value = getattr(opts,field,None)
            dtype = FitOptions.FIELDS[field][1]
            if value is not None:
                if dtype == 'int':
                    self.options[field] = int(value)
                elif dtype == 'float':
                    self.options[field] = float(value)
                else: # string
                    if not value in dtype:
                        raise ValueError('invalid option "%s" for %s: use '
                                         % (value, field)
                                         + '|'.join(dtype))
                    self.options[field] = value

# List of (parameter,factory value) required for each algorithm
FIT_OPTIONS = dict(
    amoeba  = FitOptions(AmoebaFit),
    de      = FitOptions(DEFit),
    dream   = FitOptions(DreamFit),
    newton  = FitOptions(BFGSFit),
    ps      = FitOptions(PSFit),
    pt      = FitOptions(PTFit),
    rl      = FitOptions(RLFit),
    snobfit = FitOptions(SnobFit),
    )

FIT_DEFAULT = 'amoeba'
