# Inequality constraints
# ======================
#
# The usual pattern for constraints within bumps is to set the value for one
# parameter to be some function of the other parameters.  This does not
# allow contraints of the form $a < b$ for parameters $a$ and paramter $b$.
#
# Instead, along with the fit problem definition, you can supply your
# own penalty constraints function which adds an artificial value to the
# probability function for points outside the feasible region.  The
# ideal constraints function will incorporate the distance from the
# boundary of the feasible region so that if the fitter is started
# outside forces the fit back into the feasible region.
#
# The *soft_limit* value can be used in conjunction with the penalty to
# avoid evaluating the function outside the feasible region.  For example,
# the function $\log(a-b)$ is only defined for $a > b$, so setting a
# constraint such as $10^6 + (a-b)^2$ for $a <= b$ and $0$ along with a
# soft limit of $10^6$ will keep the function defined everywhere.  With
# the penalty value sufficiently large, the probability of any evaluation
# in the infeasible region will be neglible, and will not skew the
# posterior distribution statistics.

# Define the model as usual

from bumps.names import *


def line(x, m, b):
    return m * x + b


# Simulated data for f(x)=mx+b with m=1.972(20) and b=0.11(5)
x = [1, 2, 3, 4, 5, 6]
y = [2.1, 4.0, 6.3, 8.03, 9.6, 11.9]
dy = [0.05, 0.05, 0.2, 0.05, 0.2, 0.2]

M = Curve(line, x, y, dy, m=2, b=0)
M.m.range(0, 4)
M.b.range(0, 5)

# Attach the constraints to the problem.  In this case we will force m < b
# using a list of inequality constraints.  Note that we are starting deep in
# the infeasible region, so use the default penalty_nllf = 1e6

problem = FitProblem(M, constraints=[M.m < M.b])
