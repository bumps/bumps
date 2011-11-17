#!/usr/bin/env python

"""
Multivariate gaussian mixture model.

Demonstrates bimodal sampling from a multidimensional space.

Change the relative weights of the modes to see how effectively DREAM samples
from lower probability regions.  In particular, with low *w2*, long chains
(thinning*generations*cycles) and small population, the second mode can
get lost.
"""
from pylab import *
from dream import *
#import numpy; numpy.seterr(all='raise')

n = 4
pop = 10
w1,w2 = 5,3
mu1 = -5 * ones(n)
mu2 = 5 * ones(n)
#mu2[0] = -3  # Watch marginal distribution for p1 overlap between modes
sigma = eye(n)
model = Mixture(MVNormal(mu1,sigma), w1, MVNormal(mu2,sigma), w2)
#model = MVNormal(zeros(n),sigma)

# TODO: with large number of samples, the 1/6 weight peak is lost
sampler = Dream(model=model, population=randn(pop,n,n),
                #use_delayed_rejection=False,
                #outlier_test='IQR',
                thinning=1, draws=20000)
state = sampler.sample()
save_state(filename='mixture',state=state)
state = load_state('mixture')
plot_all(state, portion=1)
show()
