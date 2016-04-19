import sys
from importlib import import_module
from bumps.pymcfit import PyMCProblem

if len(sys.argv) != 2:
    raise ValueError("Expected name of pymc file containing a model")

module =sys.argv[1]
__name__ = module.split('.')[-1]
model = import_module(module)
problem = PyMCProblem(model)
