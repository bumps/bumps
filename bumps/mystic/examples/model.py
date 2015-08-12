import numpy as np

class Function(object):
    def __init__(self, f=None, limits=None, start=None):
        """
        f=callable is the function
        limits=(lo,hi) are the limits
        start=vector is the start point
        """
        self.f, self.limits, self.start = f, limits, start
        self.__name__ = f.__name__

    def __call__(self,p):
        return self.f(p)

    def response_surface(self, p=None, dims=[0,1]):
        if p is None: p = self.start
        plot_response_surface(self, p, dims)

class Fitness(object):
    def __init__(self, f=None, data=None, limits=None, start=None):
        self.f, self.limits, self.start = f, limits, start
        if len(data) == 2:
            self.x, self.y = data
            self.dy = None
        else:
            self.x, self.y, self.dy = data

    def profile(self, p):
        return self.x, self.f(self.x, p)

    def residuals(self, p):
        return (self.profile(p) - self.y)/self.dy

    def __call__(self, p):
        return np.sum(self.residuals(p)**2)

    def plot(self, p=None):
        """
        Plot a profile for the given p
        """
        import pylab
        if self.dy is not None:
            pylab.errorbar(self.x, self.y, yerr=self.dy, fmt='x')
        else:
            pylab.plot(self.x, self.y, 'x')
        if p is None: p = self.start
        x,y = self.profile(p)
        pylab.plot(x,y)

    def response_surface(self, p=None, dims=[0,1]):
        if p is None: p = self.start
        plot_response_surface(self, p, dims)

def plot_response_surface(f, p, dims=[0,1]):
    """
    Plot a line or a slice around a point in a n-D function
    """
    import pylab
    if len(dims) == 1:
        xi = dims[0]
        x = pylab.linspace(-10,10,40) - p[xi]
        def value(v):
            p[xi] = v
            return f(p)
        z = [value(v) for v in x]
        pylab.plot(x,z)
    else:
        xi,yi = dims
        x = pylab.linspace(-10,10,40) - p[xi]
        y = pylab.linspace(-10,10,40) - p[yi]
        def value(pt):
            p[xi] = pt[0]
            p[yi] = pt[1]
            return f(p)
        z = np.array([[value((v,w)) for v in x] for w in y])
        pylab.pcolor(x,y,z)
