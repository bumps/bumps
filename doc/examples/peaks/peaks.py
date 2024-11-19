
from math import radians, sin, cos, sqrt, pi

import numpy as np

from bumps.parameter import Parameter, varying

def plot(X,Y,theory,data,err):
    import pylab

    #print "theory",theory[1:6,1:6]
    #print "data",data[1:6,1:6]
    #print "delta",(data-theory)[1:6,1:6]
    vmin = np.amin(data)
    vmax = np.amax(data)
    window = 0.2*(vmax - vmin)
    pylab.subplot(131)
    pylab.pcolormesh(X,Y, data, vmin=vmin-window, vmax=vmax+window)
    pylab.subplot(132)
    pylab.pcolormesh(X,Y, theory, vmin=vmin-window, vmax=vmax+window)
    pylab.subplot(133)
    pylab.pcolormesh(X,Y, (data-theory)/(err+1))

class Gaussian(object):
    def __init__(self, A=1, xc=0, yc=0, s1=1, s2=1, theta=0, name=""):
        self.A = Parameter(A,name=name+"A")
        self.xc = Parameter(xc,name=name+"xc")
        self.yc = Parameter(yc,name=name+"yc")
        self.s1 = Parameter(s1,name=name+"s1")
        self.s2 = Parameter(s2,name=name+"s2")
        self.theta = Parameter(theta,name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    s1=self.s1, s2=self.s2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        s1 = self.s1.value
        s2 = self.s2.value
        t  = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        # shift and rotate
        x, y = x-xc, y-yc
        x, y= x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t)
        #Zf = gauss(x, s1) * gauss(y, s2)
        # Slightly faster to do inline
        Zf = np.exp(-0.5*((x/s1)**2 + (y/s2)**2))/(2*pi*s1*s2)
        #return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total>0 else np.zeros_like(x)


class Cauchy(object):
    r"""
    2-D Cauchy

    https://en.wikipedia.org/wiki/Cauchy_distribution#Multivariate_Cauchy_distribution
    """
    def __init__(self, A=1, xc=0, yc=0, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A,name=name+"A")
        self.xc = Parameter(xc,name=name+"xc")
        self.yc = Parameter(yc,name=name+"yc")
        self.g1 = Parameter(g1,name=name+"g1")
        self.g2 = Parameter(g2,name=name+"g2")
        self.theta = Parameter(theta,name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        g1 = self.g1.value
        g2 = self.g2.value
        t = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        xbar,ybar = x-xc,y-yc
        a = cos(t)**2/g1**2 + sin(t)**2/g2**2
        b = sin(2*t)*(-1/g1**2 + 1/g2**2)
        c = sin(t)**2/g1**2 + cos(t)**2/g2**2
        gsq = a*xbar**2 + b*xbar*ybar + c*ybar**2
        Zf = 1./(2*pi*sqrt(g1*g2)*(1 + gsq)**1.5)
        #return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total>0 else np.zeros_like(x)

class Lorentzian(object):
    r"""
    Lorentzian peak.

    Note that this is not equivalent to the multidimensional Cauchy
    distribution which models the sum of parameters as having a cauchy
    distribution.  Instead, it sets the gamma parameter according to
    elliptical direction
    sum
    """
    def __init__(self, A=1, xc=0, yc=0, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A,name=name+"A")
        self.xc = Parameter(xc,name=name+"xc")
        self.yc = Parameter(yc,name=name+"yc")
        self.g1 = Parameter(g1,name=name+"g1")
        self.g2 = Parameter(g2,name=name+"g2")
        self.theta = Parameter(theta,name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        g1 = self.g1.value
        g2 = self.g2.value
        t  = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        # shift and rotate
        x, y = x-xc, y-yc
        x, y= x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t)
        Zf = cauchy(x, g1) * cauchy(y, g2)
        #return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total>0 else np.zeros_like(x)

class Voigt(object):
    r"""
    Voigt peak
    """
    def __init__(self, A=1, xc=0, yc=0, s1=1, s2=1, g1=1, g2=1, theta=0, name=""):
        self.A = Parameter(A,name=name+"A")
        self.xc = Parameter(xc,name=name+"xc")
        self.yc = Parameter(yc,name=name+"yc")
        self.s1 = Parameter(s1,name=name+"s1")
        self.s2 = Parameter(s2,name=name+"s2")
        self.g1 = Parameter(g1,name=name+"g1")
        self.g2 = Parameter(g2,name=name+"g2")
        self.theta = Parameter(theta,name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    s1=self.s1, s2=self.s2,
                    g1=self.g1, g2=self.g2,
                    theta=self.theta)

    def __call__(self, x, y):
        area = self.A.value
        s1 = self.s1.value
        s2 = self.s2.value
        g1 = self.g1.value
        g2 = self.g2.value
        t  = radians(self.theta.value)
        xc = self.xc.value
        yc = self.yc.value
        # shift and rotate
        x, y = x-xc, y-yc
        x, y= x*cos(t) + y*sin(t), -x*sin(t) + y*cos(t)
        Zf = voigt(x, s1, g1) * voigt(y, s2, g2)
        #return Zf*abs(area)
        total = np.sum(Zf)
        return Zf/total*abs(area) if total>0 else np.zeros_like(x)


class Background(object):
    def __init__(self, C=0, name=""):
        self.C = Parameter(C,name=name+"background")
    def parameters(self):
        return dict(C=self.C)
    def __call__(self, x, y):
        return self.C.value

class Peaks(object):
    def __init__(self, parts, X, Y, data, err):
        self.X,self.Y,self.data,self.err = X, Y, data, err
        self.parts = parts

    def numpoints(self):
        return np.prod(self.data.shape)

    def parameters(self):
        return [p.parameters() for p in self.parts]

    def theory(self):
        #return self.parts[0](self.X,self.Y)
        #parts = [M(self.X,self.Y) for M in self.parts]
        #for i,p in enumerate(parts):
        #    if np.any(np.isnan(p)): print "NaN in part",i
        return sum(M(self.X,self.Y) for M in self.parts)

    def residuals(self):
        #if np.any(self.err ==0): print "zeros in err"
        return (self.theory()-self.data)/(self.err+(self.err==0.))

    def nllf(self):
        R = self.residuals()
        #if np.any(np.isnan(R)): print "NaN in residuals"
        return 0.5*np.sum(R**2)

    def __call__(self):
        return 2*self.nllf()/self.dof

    def plot(self, view='linear'):
        plot(self.X, self.Y, self.theory(), self.data, self.err)

    def save(self, basename):
        import json
        pars = [(p.name,p.value) for p in varying(self.parameters())]
        out = json.dumps(dict(theory=self.theory().tolist(),
                              data=self.data.tolist(),
                              err=self.err.tolist(),
                              X = self.X.tolist(),
                              Y = self.Y.tolist(),
                              pars = pars))
        open(basename+".json","w").write(out)

    def update(self):
        pass

def cauchy(x, gamma):
    return gamma/(x**2 + gamma**2)/pi

def gauss(x, sigma):
    return np.exp(-0.5*(x/sigma)**2)/np.sqrt(2*pi*sigma**2)

def voigt(x, sigma, gamma):
    """
    Return the voigt function, which is the convolution of a Lorentz
    function with a Gaussian.

    :Parameters:
     gamma : real
      The half-width half-maximum of the Lorentzian
     sigma : real
      The 1-sigma width of the Gaussian, which is one standard deviation.

    Ref: W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64

    Note: adjusted to use stddev and HWHM rather than FWHM parameters
    """
    # wofz function = w(z) = Fad[d][e][y]eva function = exp(-z**2)erfc(-iz)
    from scipy.special import wofz
    z = (x+1j*gamma)/(sigma*np.sqrt(2))
    V = wofz(z)/(np.sqrt(2*pi)*sigma)
    return V.real
