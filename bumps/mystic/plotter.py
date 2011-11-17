import numpy
import pylab

def meshgrid(f, p, dims, steps):
    xi,yi = dims
    if steps.ndim == 1:
        x = y = steps
    else:
        x,y = steps
    def value(pt):
        p[xi] = pt[0]
        p[yi] = pt[1]
        return f(p)
    z = numpy.array([map(value, [(v,w) for v in x]) for w in y])
    return x,y,z


def plot_response_surface(f, p, dims=[0,1], steps=pylab.linspace(-10,10,40)):
    """
    Plot a line or a slice around a point in a n-D function
    """
    import pylab
    if len(dims) == 1:
        xi = dims[0]
        x = steps - p[xi]
        def value(v):
            p[xi] = v
            return f(p)
        z = [value(v) for v in x]
        pylab.plot(x,z)
    else:
        x,y,z = meshgrid(f,p,dims,steps)
        pylab.pcolor(x,y,z)
