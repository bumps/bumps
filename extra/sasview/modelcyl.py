from __future__ import print_function

# Look for the peak fitter in the same file as the modeller
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: 
    import sans
except ImportError:
    from distutils.util import get_platform
    platform = '.%s-%s'%(get_platform(),sys.version[:3])
    sasview = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','..','sasview','build','lib'+platform))
    sys.path.insert(0,sasview)
    #raise Exception("\n".join(sys.path))

from sasbumps import *

# Load data
data = load_data('cyl_400_40.txt')

# Set up the target model
sample_sld = Parameter(2.07, name='sample sld')
solvent_sld = Parameter(1.0, name='solvent sld')
model = load_model('CylinderModel', radius=60, radius_width=0,
                   sldCyl=1e-6*sample_sld, sldSolv=1e-6*solvent_sld,
                   background=0, scale=1.0)

M = Experiment(model=model, data=data)
M['length'].range(0,1000)
M['radius'].range(0,200)
#M['length.width'].range(0,0.7)
#M['radius.width'].range(0,0.7)
sample_sld.range(solvent_sld.value,solvent_sld.value+7)

problem = FitProblem([M])
problem.randomize()

