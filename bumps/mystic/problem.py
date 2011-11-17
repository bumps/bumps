import numpy
from .parameter import Parameter

class Problem:
    """
    Optimization problem.

    Must define the number of dimensions and the bounds.

    Provides a callable which computes the cost function to minimize.
    """
    def __call__(self, x):
        raise NotImplementedError

class Function(Problem):
    def __init__(self, f, ndim=None, po=None, bounds=None, args=()):
        if bounds != None and po != None:
            self.parameters = [Parameter(value=v,bounds=b)
                               for v,b in zip(po,bounds)]
        elif bounds != None:
            self.parameters = [Parameter(b) for b in bounds]
        elif po != None:
            self.parameters = [Parameter(v) for v in po]
        elif ndim != None:
            self.parameters = [Parameter() for _ in range(ndim)]
        else:
            raise TypeError("Need ndim, po or bounds to get problem dimension")
        if ((ndim != None and ndim != len(self.parameters))
            or (po != None and len(po) != len(self.parameters))
            or (bounds != None and len(bounds) != len(self.parameters))):
            raise ValueError("Inconsistent dimensions for ndim, po and bounds")
        if po == None:
            po = [p.start_value() for p in self.parameters]

        self.f = f
        self.bounds = bounds
        self.po = po
        self.args = args

    def guess(self):
        if self.po != None:
            return self.po
        else:
            return [p.start_value() for p in self.parameters]
    def __call__(self, p):
        return self.f(p, *self.args)


class FitProblem(Problem):
    def data(self, x, y, dy=1):
        self.x, self.y, self.dy = x,y,dy
    def curve(self, p, x, deriv=False):
        raise NotImplementedError
    def residuals(self, p, deriv=False):
        if deriv:
            fx, dfx = self.curve(p, self.x, deriv)
            return (fx-self.y)/self.dy, dfx/self.dy
        else:
            fx = self.curve(p, self.x, deriv)
            return (fx-self.y)/self.dy
    def f(self, p):
        return numpy.sum(self.residuals(p, deriv=False)**2)
    def fdf(self, p):
        resid, deriv = self.residuals(p, deriv=True)
        return numpy.sum(resid**2), numpy.sum(2*resid*deriv)
    __call__ = f
