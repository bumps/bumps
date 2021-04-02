from bumps.names import Parameter, FitProblem
class Model:
    def __init__(self):
        self.par = Parameter(5, name='par')
        self.constraint = self.par > 1
    def parameters(self):
        return dict(par=self.par, constraint=self.constraint)
    def numpoints(self):
        return 2
    def nllf(self):
        return (self.par.value)**2

problem = FitProblem(Model())

#problem.fitness.par.value = 0
problem.fitness.par.range(-1, 10)
