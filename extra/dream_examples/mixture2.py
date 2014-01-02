#!/usr/bin/env python

"""
Multimodal demonstration using gaussian mixture model.

The model is a mixture model representing the probability density from a
product of gaussians.

This example show performance of the algorithm on multimodal densities,
with adjustable number of densities and degree of separation.

The peaks are distributed about the x-y plane so that the marginal densities
in x and y are spaced every 2 units using latin hypercube sampling.  For small
peak widths, this means that the densities will not overlap, and the marginal
maximum likelihood for a given x or y value should match the estimated density.
With overlap, the marginal density will over estimate the marginal maximum
likelihood.

Adjust the width of the peaks, *S*, to see the effect of relative diameter of
the modes on sampling.  Adjust the height of the peaks, *I*, to see the
effects of the relative height of the modes.  Adjust the count *n* to see
the effects of the number of modes.

Note that dream.diffev.de_step adds jitter to the parameters at the 1e-6 level,
so *S* < 1e-4 cannot be modeled reliably.

*draws* is set to 1000 samples per mode.  *burn* is set to 100 samples per mode.
Population size *h* is set to 20 per mode.  A good choice for number of
sequences *k* is not yet determined.
"""
from pylab import *
from dream import *

if 1: # Fixed layout of 5 minima
    n = 5
    S = [0.1]*5
    x = [-4, -2, 0, 2, 4]
    y = [2, -2, -4, 0, 4]
    I = [5, 2.5, 1, 4, 1]
else: # Semirandom layout of n minima
    n = 40
    S = [0.1]*n
    x = linspace(-n+1,n-1,n)
    y = permutation(x)
    I = 2*linspace(-1,1,n)**2 + 1

args = [] # Sequence of density, weight, density, weight, ...
for xi,yi,Si,Ii in zip(x,y,S,I):
    args.extend( (MVNormal([xi,yi],Si*eye(2)), Ii) )
model = Mixture(*args)

k = 20*n
h = int(20*n/k)
sampler = Dream(model=model,
                population=randn(h,k,2),
                #use_delayed_rejection=False,
                DE_snooker_rate=0.5,
                outlier_test='none',
                draws=40000*n,burn=5000*k,
                thinning=1)
mc = sampler.sample()
stats = plot_vars(mc, ci=1, nbins=9*max(x));
print(format_vars(stats))
plot_all(mc)
show()
