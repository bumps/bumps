# Calling fit from scripts
# ========================
#
# Revisiting our curve fit example, let's call the optimizer directly from
# the script.
#
# Setting up the problem remains the same:

from __future__ import print_function
from bumps.names import *

x = [1, 2, 3, 4, 5, 6]
y = [2.1, 4.0, 6.3, 8.03, 9.6, 11.9]
dy = [0.05, 0.05, 0.2, 0.05, 0.2, 0.2]

def line(x, m, b=0):
    return m*x + b

M = Curve(line, x, y, dy, m=2, b=2)
M.m.range(0, 4)
M.b.range(-5, 5)

problem = FitProblem(M)

# With the problem defined, we can now call the fitter.  The following
# uses the minimalist fit interface defined in bumps, which takes a problem
# definition and returns a results object with x, dx attributes for the
# best value and the estimated uncertainty.  The 'dream' fitter will
# additionally return the dream state, which allows for more detailed
# uncertainty analysis.

from bumps.fitters import fit
from bumps.formatnum import format_uncertainty

# Allow choice of fitter from the command line
method = 'amoeba' if len(sys.argv) < 2 else sys.argv[1]

print("initial chisq", problem.chisq_str())
result = fit(problem, method=method, xtol=1e-6, ftol=1e-8)
print("final chisq", problem.chisq_str())
for k, v, dv in zip(problem.labels(), result.x, result.dx):
    print(k, ":", format_uncertainty(v, dv))
