"""
Test functions for optimizers.

Note that model_lib.py needs to be on your python path to run these tests.

Usage:

    PYTHONPATH=. bumps model.py MODEL DIM

where MODEL is one of:

    ackley              gauss               rastrigin
    beale               griewank            rosenbrock
    cross_in_tray       laplace             sin_plus_quadratic

and DIM is an integer. See model definitions in model_lib.py.
"""

import sys
import numpy as np

import bumps.names as bp
from model_lib import select_function, plot2d


USAGE = "Give the name of the model followed by dimension (default=2)"

try:
    nllf = select_function(sys.argv[1:], vector=False)
except Exception as exc:
    # import traceback; traceback.print_exc()
    print(USAGE, file=sys.stderr)
    print(str(exc), file=sys.stderr)
    sys.exit(1)

plotter = plot2d(nllf, range=(-10, 10))

M = bp.PDF(nllf, plotter=plotter)

for p in M.parameters().values():
    # TODO: really should pull value and range out of the bounds for the
    # function, if any are provided.
    p.value = 400 * (np.random.rand() - 0.5)
    # p.range(-1,1)
    p.range(-200, 200)
    # p.range(-inf,inf)

problem = bp.arccoshFitProblem(M)
