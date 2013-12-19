#!/usr/bin/env python

"""
The Rosenbrock banana function
"""
from dream import *
from pylab import *
import numpy
#numpy.seterr(all='raise')

def rosen(x):
    x = asarray(x)
    return sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

n=6
sampler = Dream(model=LogDensity(rosen),
                population=randn(2*n,5,n),
                thinning=1,
                draws=25000,
                burn=10000,
                #DE_snooker_rate=0,
                #cycles=3,
                )

state = sampler.sample()
state.mark_outliers()
state.title = "Banana function example"
#plot_corr(state); show()
plot_all(state)
show()
