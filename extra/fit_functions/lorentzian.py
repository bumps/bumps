#!/usr/bin/env python

"""
Lorentzian peak model

References::
    None
"""

from numpy import array, pi, asarray, arange, sqrt
from numpy import random


def lorentzian(E, Io, Eo, Gamma, A=0, B=0, C=0):
    """
    lorentzian with quadratic background::

        I = Io  (Gamma/2*pi) / ( (E-Eo)^2 + (Gamma/2)^2 ) + (A + B E + C E^2)
    """
    E = asarray(E) # force E to be a numpy array
    return (A + (B + C*E)*E + Io * (Gamma/2/pi) / ( (E-Eo)**2 + (Gamma/2)**2 ))


def simulate_events(params,xmin,xmax,npts=4000):
    """Generate a lorentzian dataset of npts between [min,max] from given params"""
    def gensample(F, xmin, xmax):
        a = arange(xmin, xmax, (xmax-xmin)/200.)
        ymin = 0
        ymax = F(a).max()
        while 1:
            t1 = random.random() * (xmax-xmin) + xmin
            t2 = random.random() * (ymax-ymin) + ymin
            t3 = F(t1)
            if t2 < t3:
                return t1
    fwd = lambda x: lorentzian(x, **params)
    return array([gensample(fwd, xmin,xmax) for _ in range(npts)])

def simulate_histogram(pars, Emin, Emax, dE, npts=4000):
    events = simulate_events(pars, Emin, Emax)
    E,I = histogram(events, dE, Emin, Emax)
    #print min(events),max(events)
    dI = sqrt(I)
    data = { 'x': (E[1:]+E[:-1])/2., 'y': I, 'dy': dI }
    return data

def demo_data():
    # integrated intensity = 4000
    # center = 6400
    # width gamma = 180
    # background = -(E-6340)^2/1000 + 10
    bgC = 6340.
    bgW = 10000.
    A = 35 - bgC**2/bgW
    B = 2.*bgC/bgW
    C = -1./bgW
    #A,B,C = 0,0,0
    N = 4000
    pars = { 'Eo': 6500.0, 'Gamma': 180.0, 'Io': 20*N, 'A': A, 'B': B, 'C': C }
    Emin,Emax = 6000, 6700
    dE = 20

    return pars, simulate_histogram(pars, Emin, Emax, dE, N)


# probably shouldn't be in here...
from numpy import histogram as numpyhisto
def histogram(data,binwidth,xmin,xmax):
    """
    generate bin-centered histogram of provided data

    return bins of given binwidth (and histogram) generated between [xmin,xmax]
    """
    edges = arange(xmin,xmax+binwidth*0.9999999, binwidth)
    centers = edges + (0.5 * binwidth)
    histo,_ = numpyhisto(data, bins=edges)
    #print data.size, sum(histo), edges[0], edges[-1], min(data),max(data)
    return centers, histo

coeff, data = demo_data()

def demo():
    import pylab
    x,y,dy = data['x'],data['y'],data['dy']
    A,B,C = coeff['A'],coeff['B'],coeff['C']
    Io,Eo,Gamma = coeff['Io'],coeff['Eo'],coeff['Gamma']
    pylab.errorbar(x, y, yerr=dy, label="data")
    pylab.plot(x, pylab.polyval([C,B,A], x), label="background")
    pylab.plot(x, lorentzian(x,Eo=Eo,Io=Io,Gamma=Gamma), label="peak")
    pylab.plot(x, lorentzian(x,**coeff), label="peak+bkg")
    pylab.legend()
    pylab.show()

if __name__ == "__main__": demo()
