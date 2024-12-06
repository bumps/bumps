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
import numpy as np
from bumps.dream.model import MVNormal, Mixture
from bumps.names import *
from bumps.util import push_seed

def make_model(dims=10):
    """
    Adjust S (sigma^2) to change the diameter of the peak

    Small values of sigma lead to stuck fits.

    Adjust relative intensity of peaks with I.

    Choose peak spacing with x. Keep it between -10 and 20

    The peaks are located in a hypercube of dimension d, with coordinates in
    each dimension a random permutation of x. That means the resulting histogram
    which is a projection along that dimension will contain a random permutation
    of peak heights, with centers at the x positions. The 2D histograms will be
    be similar random permutations.
    """
    if 1: # Uneven spacing of 5 peaks
        num_modes = 5
        S = [0.1]*5
        x = [1, 3, 7, 14, 18]
        I = [2.5, 1, 5, 4, 1]
        #S[2] = 0.05; I[2] = 60
    elif 1: # Even spacing of 5 peaks
        num_modes = 5
        S = [0.1]*5
        x = [-4, -2, 0, 2, 4]
        I = [15, 2.5, 1, 4, 1]
    else: # Lots of evenly spaced peaks
        num_modes = 40
        S = [0.1]*num_modes
        x = np.linspace(-9, 9, num_modes)
        I = 2*np.linspace(-1, 1, num_modes)**2 + 1

    centers = [x] + [np.random.permutation(x) for _ in range(1, dims)]
    centers = np.asarray(centers).T
    args = [] # Sequence of density, weight, density, weight, ...
    for mu_i, Si, Ii in zip(centers, S, I):
        args.extend((MVNormal(mu_i, Si*np.eye(dims)), Ii))
    model = Mixture(*args)

    if 1:
        from bumps.dream.entropy import GaussianMixture
        pairs = zip(args[0::2], args[1::2])
        triples = ((M.mu, M.sigma, I) for M, I in pairs)
        mu, sigma, weight = zip(*triples)
        D = GaussianMixture(weight, mu=mu, sigma=sigma)
        print("*** Expected entropy: %s bits"%(D.entropy(N=100000)/np.log(2),))

    return model

# Need reproducible models if we want to be able to resume a fit
dims = 6
with push_seed(1):
    model = make_model(dims=dims)

def plot2d(fn, args=None, range=(-10, 10)):
    """
    Return a mesh plotter for the given function.

    *args* are the function arguments that are to be meshed (usually the
    first two arguments to the function).  *range* is the bounding box
    for the 2D mesh.

    All arguments except the meshed arguments are held fixed.
    """
    if args is None:
        args = [0, 1]
    def plotter(p, view=None):
        import pylab
        if len(p) == 1:
            x = p[0]
            r = np.linspace(range[0], range[1], 400)
            pylab.plot(x+r, [fn(v) for v in x+r])
            pylab.xlabel(args[0])
            pylab.ylabel("-log P(%s)"%args[0])
        else:
            r = np.linspace(range[0], range[1], 20)
            x, y = p[args[0]], p[args[1]]
            data = np.empty((len(r), len(r)), 'd')
            for j, xj in enumerate(x+r):
                for k, yk in enumerate(y+r):
                    p[args[0]], p[args[1]] = xj, yk
                    data[j, k] = fn(p)
            pylab.pcolormesh(x+r, y+r, data)
            pylab.plot(x, y, 'o', markersize=6,
                       markerfacecolor='red', markeredgecolor='black',
                       markeredgewidth=1, alpha=0.7)
            pylab.xlabel(args[0])
            pylab.ylabel(args[1])
    return plotter


M = VectorPDF(model.nllf, p=[1.]*dims, plot=plot2d(model.nllf))
for _, p in M.parameters().items():
    p.range(-10, 20)
problem = FitProblem(M)
