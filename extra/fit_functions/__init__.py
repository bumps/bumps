"""
models: sample models and functions prepared for use in mystic


Functions
=========

Standard test functions for minimizers:

    rosenbrock         -- Rosenbrock's function
    step               -- De Jong's step function
    quartic            -- De Jong's quartic function
    shekel             -- Shekel's function
    corana1d,2d,3d,4d  -- Corana's function
    fosc3d             -- the fOsc3D Mathematica function
    griewangk          -- Griewangk's function
    zimmermann         -- Zimmermann's function
    wavy1              -- a simple sine-based multi-minima function
    wavy2              -- another simple sine-based multi-minima function


Models
======

Curve fitting tests:

    disk_coverage  -- minimal disk for covering a set of points
    lorentzian     -- Lorentzian peak model
    decay          -- Bevington & Robinson's model of dual exponential decay
    mogi           -- Mogi's model of surface displacements from a point spherical
                      source in an elastic half space

For each model s there will be a sample data set:

   s_data = {'x':x, 'y':y, 'dy':dy}

and generating parameters if they are available:

   s_pars = { ... }

"""

# models
from .mogi import mogi
from .br8 import dual_exponential as decay, data as decay_data
from .lorentzian import lorentzian, data as lorentzian_data, coeff as lorenztian_pars
from .circle import disk_coverage, simulate_circle, simulate_disk

# functions
from .dejong import rosenbrock, step, quartic, shekel
from .corana import corana1d, corana2d, corana3d, corana4d
from .fosc3d import fOsc3D
from .griewangk import griewangk
from .zimmermann import zimmermann
from .wavy import wavy1, wavy2
