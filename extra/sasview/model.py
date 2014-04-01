from __future__ import print_function

# ======
# Put current directory and sasview directory on path.
# This won't be necessary once bumps is in sasview
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: 
    import sans
except ImportError:
    from distutils.util import get_platform
    platform = '.%s-%s'%(get_platform(),sys.version[:3])
    sasview = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','sasview','build','lib'+platform))
    sys.path.insert(0,sasview)
    #print("\n".join(sys.path))
# ======

from sasbumps import *

# Set up the target model
sldSamp = Parameter(2.07, name='sample sld')
sldSolv = Parameter(1.0, name='solvent sld')
sphere = load_model('SphereModel', radius=60, radius_width=0.1,
                    sldSph=1e-6*sldSamp, sldSolv=1e-6*sldSolv,
                    background=0, scale=1.0)
ellipsoid = load_model('EllipsoidModel', radius_a=60, radius_b=160,
                       sldEll=1e-6*sldSamp, sldSolv=1e-6*sldSolv,
                       background=0, scale=1.0)

# Simulate data
# Use seed(n) for reproducible data, or seed() for new data each time.
try: k = int(sys.argv[2])
except: k = 1
with seed(k): data = sim_data(ellipsoid, noise=15)

# Fit to sphere or ellipse, depending on command line
if "sphere" in sys.argv[1:]:
    M = Experiment(model=sphere, data=data)
    M['radius'].range(0,200)
    M['radius.width'].range(0,0.7)
else: # ellipse
    M = Experiment(model=ellipsoid, data=data)
    M['radius_a'].range(0,1e3)
    M['radius_b'].range(0,1e3)
    #M['scale'].range(0,100)
sldSamp.range(0,1e2)
sldSolv.range(1,7)

problem = FitProblem([M])
problem.randomize()

