# ======
# Put current directory on path.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ======

from pymcfit import PymcProblem

from pymc.examples import disaster_model
pars = (
    disaster_model.switchpoint,
    disaster_model.early_mean,
    disaster_model.late_mean
    )
conds = (
    disaster_model.disasters,
    )
problem = PymcProblem(pars=pars, conds=conds)
