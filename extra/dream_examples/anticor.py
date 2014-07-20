#!/usr/bin/env python

"""
Example model with strong correlations between the fitted parameters.

We use a*x = y + N(0,1) made complicated by defining a=p1+p2.

The expected distribution for p1 and p2 will be uniform, with p2 = a-p1 in
each sample.  Because this distribution is inherently unbounded, artificial
bounds are required on a least one of the parameters for finite duration
simulations.

The expected distribution for p1+p2 can be determined from the linear model
y = a*x.  This is reported along with the values estimated from MCMC.
"""
from __future__ import print_function

from pylab import *  # Numeric functions and plotting
from dream import *  # sampler functions

# Create the correlation function and generate some fake data
x = linspace(-1., 1, 40)
fn = lambda p: sum(p)*x
bounds=(-20,-inf),(40,inf)
sigma = 1
data = fn((1,1)) + randn(*x.shape)*sigma  # Fake data


# Sample from the posterior density function
n=2
model = Simulation(f=fn, data=data, sigma=sigma, bounds=bounds,
                   labels=["x","y"])
sampler = Dream(model=model,
                population=randn(5*n,4,n),
                thinning=1,
                draws=20000,
                )
mc = sampler.sample()
mc.title = 'Strong anti-correlation'

# Create a derived parameter without the correlation
mc.derive_vars(lambda p: (p[0]+p[1]), labels=['x+y'])

# Compare the MCMC estimate for the derived parameter to a least squares fit
from bumps.wsolve import wpolyfit
poly = wpolyfit(x,data,degree=1,origin=True)
print("x+y from linear fit", poly.coeff[0], poly.std[0])
points,logp = mc.sample(portion=0.5)
print("x+y from MCMC",mean(points[:,2]), std(points[:,2],ddof=1))

# Plot the samples
plot_all(mc, portion=0.5)
show()
