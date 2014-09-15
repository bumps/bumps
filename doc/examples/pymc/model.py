from bumps.pymcfit import PyMCProblem
from pymc.examples import disaster_model

pars = (
    disaster_model.switchpoint,
    disaster_model.early_mean,
    disaster_model.late_mean
    )
conds = (
    disaster_model.disasters,
    )
problem = PyMCProblem(pars=pars, conds=conds)
