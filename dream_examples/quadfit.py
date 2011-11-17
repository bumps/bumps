#!/usr/bin/env python

from pylab import *
from dream import *

x = linspace(-4., 4, 40)
fn = lambda p: polyval(p,x)
bounds=(-20,-20,-20),(40,40,40)
sigma = 1
data = fn((2,-1,3)) + randn(*x.shape)*sigma  # Fake data


n=len(bounds[0])
model = Simulation(f=fn, data=data, sigma=sigma, bounds=bounds)
sampler = Dream(model=model,
                population=randn(5*n,4,n),
                thinning=1,
                draws=20000,
                )

mc = sampler.sample()
figure(6); model.plot(mc.best()[0])
plot_all(mc)
