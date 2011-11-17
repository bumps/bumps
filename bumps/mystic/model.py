class FitFunction:
    """
    The basic kind of object handed to an optimizer.
    """
    def __call__(self, pvec):
        """
        Evaluate the objective
        """

class LeastSquares(FitFunction):
    def residuals(self):
        """
        """

class Objective:
    def __init__(self, fitness):
        self.fitness = fitness
    def _getp(self):
        pars = fitness.parameterset;
        self.fitpars = [p for k,p in set(pars.flatten()) if p.fittable()]
        #self.constraints = pars.constraints()
    def _setp(self, pvec):
        for v,par in zip(pvec,self.fitpars): par.value = v
        self.constraints()
    def residuals(self, pvec):
        self._setp(pvec)
        return self.fitness.residuals()
    def __call__(self, pvec):
        return numpy.sum(self.residuals(pvec)**2)

class Problem:
    def parameters(self):
        """Returns set of parameters"""
    def constraints(self):
        """returns set of constraints"""
    def __call__(self):
        """returns the function value for the current parameters"""

class Fitness:
    def __init__(self, problem):
        self.problem = problem
        self.pars = problem.parameters()
        self.cons = problem.constraints

    def parameters(self):
        """
        The parameters defined in the model
        """

    def constraints(self):
        """
        The constraint expressions defined in the model
        """

    def residuals(self, pvec):
        """
        The residuals defined for the model
        """
        raise NotImplementedError

    def __call__(self):
        """returns the sumsq residuals"""
        return self.residual(pvec)

class MultiFit(Fitness):
    def __init__(self, fits, weights=None):
        self.parameterset = ParameterSet(f.parameterset for f in fits)
        self.fits = fits
        if weights is None:
            weights = [1 for f in fits]
        self.weights = weights
    def residuals(self):
        resid = numpy.concatenate([w*f.residuals()
                                   for w,f in zip(self.weights,self.fits)])
