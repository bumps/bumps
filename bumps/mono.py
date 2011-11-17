"""
Monotonic spline modeling
"""


from __future__ import division
from numpy import (diff, hstack, sqrt, searchsorted, asarray, cumsum,
                   nonzero, linspace, isnan)
from . import numpyerrors


@numpyerrors.ignored
def monospline(x, y, xt):
    r"""
    Monotonic cubic hermite interpolation.

    Returns $p(x_t)$ where $p(x_i)= y_i$ and $p(x) \leq p(xi)$
    if $y_i \leq y_{i+1}$ for all $y_i$.  Also works for decreasing
    values $y$, resulting in decreasing $p(x)$.  If $y$ is not monotonic,
    then $p(x)$ may peak higher than any $y$, so this function is not
    suitable for a strict constraint on the interpolated function when
    $y$ values are unconstrained.

    http://en.wikipedia.org/wiki/Monotone_cubic_interpolation
    """
    x = hstack((x[0]-1,x,x[-1]+1))
    y = hstack((y[0], y, y[-1]))
    dx = diff(x)
    dy = diff(y)
    delta = dy/dx
    m = (delta[1:]+delta[:-1])/2
    m = hstack( (0, m, 0) )
    alpha, beta = m[:-1]/delta, m[1:]/delta
    d = alpha**2+beta**2

    for i in range(len(m)-1):
        if dy[i] == 0 or alpha[i] == 0 or beta[i] == 0:
            m[i] = m[i+1] = 0
        elif d[i] > 9:
            tau = 3./sqrt(d[i])
            m[i] = tau*alpha[i]*delta[i]
            m[i+1] = tau*beta[i]*delta[i]
            #if numpy.isnan(m[i]) or numpy.isnan(m[i+1]):
            #    print i,"isnan",tau,d[i], alpha[i],beta[i],delta[i]
        #elif numpy.isnan(m[i]):
        #    print i,"isnan",delta[i],dy[i]
    #m[ dy[1:]*dy[:-1]<0 ] = 0
    m[isnan(m)] = 0

    return hermite(x,y,m,xt)


@numpyerrors.ignored
def hermite(x,y,m,xt):
    """
    Computes the cubic hermite polynomial p(xt).

    The polynomial goes through all points (x_i,y_i) with slope
    m_i at the point.
    """
    x,y,m,xt = [asarray(v,'d') for v in x,y,m,xt]
    idx = searchsorted(x[1:-1],xt)
    h = x[idx+1] - x[idx]
    h[h<=1e-10]=1e-10
    s = (y[idx+1] - y[idx])/h
    v = xt-x[idx]
    c3,c2,c1,c0 = ((m[idx]+m[idx+1]-2*s)/h**2,
                   (3*s-2*m[idx]-m[idx+1])/h,
                   m[idx],
                   y[idx])
    return ((c3*v + c2)*v + c1)*v + c0



def inflections(dx,dy):
    x = hstack( (0, cumsum(dx)) )
    y = hstack( (0, cumsum(dy)) )
    return count_inflections(x,y)


@numpyerrors.ignored
def count_inflections(x,y):
    """
    Count the number of inflection points in the spline curve
    """
    m = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    b = y[2:] - m*x[2:]
    delta = y[1:-1] - (m*x[1:-1] + b)
    delta = delta[nonzero(delta)] # ignore points on the line
    sign_change = (delta[1:]*delta[:-1]) < 0
    return sum(sign_change)

def plot_inflection(x,y):
    m = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    b = y[2:] - m*x[2:]
    delta = y[1:-1] - (m*x[1:-1] + b)
    t = linspace(x[0],x[-1],400)
    import pylab
    ax1=pylab.subplot(211)
    pylab.plot(t,monospline(x,y,t),'-b',x,y,'ob')
    pylab.subplot(212, sharex=ax1)
    delta_x = x[1:-1]
    pylab.stem(delta_x,delta)
    pylab.plot(delta_x[delta<0],delta[delta<0],'og')
    pylab.axis([x[0],x[-1],min(min(delta),0),max(max(delta),0)])
