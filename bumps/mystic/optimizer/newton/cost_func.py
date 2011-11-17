'''
Created on Nov 23, 2009

@author: Ismet Sahin
'''
import numpy

def rosen(x):
    """
    Rosenbrock's banana function in n-D.
    """
    # The minimizer of this function is ones(n)
    x = numpy.asarray(x)
    a = x[1:] - x[:-1]**2
    b = 1 - x[:-1]
    f = numpy.sum(100*a**2 + b**2)
    g = numpy.zeros_like(x)
    g[:-1] = -400*x[:-1]*a - 2*b
    g[1:] += 200*a
    #f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2;
    #g = [2*(x[0]-1) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)]
    return f,g

def example_call():
    f,g = rosen([2,1])
    print 'function :', f
    print 'gradient :', g


def main():
    example_call();

if __name__ == "__main__":
    main();


#example_call()
