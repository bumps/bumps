# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:21:28 2015

@author: kwj
"""

from scipy.stats import norm
from bumps.names import *

def dist(z):
    return -norm.logpdf(z, 20, 1)
    
M = PDF(dist, z=0)
M.z.range(-200,200)
problem = FitProblem(M)