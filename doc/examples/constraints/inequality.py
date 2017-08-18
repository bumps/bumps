# Inequality constraints
# ======================
#
# The usual pattern for constraints within bumps is to set the value for one
# parameter to be some function of the other parameters.  Inequality
# constraints are not directly supported.
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
# soft limit of $10^6$ will keep the function defined everywhere.

# Define the model as usual

from bumps.names import *

def line(x, m, b):
    return m*x + b

x = [1,2,3,4,5,6]
y = [2.1,4.0,6.3,8.03,9.6,11.9]
dy = [0.05,0.05,0.2,0.05,0.2,0.2]

M = Curve(line,x,y,dy,m=2,b=0)
M.m.range(0,4)
M.b.range(0,5)

# Define the constraints as a function which takes no parameters and returns
# a floating point value.  Note the value *1e6* in the penalty condition:
# this is the soft limit value which we will use to avoid evaluating the
# curve in the infeasible region.

def constraints():
    m, b = M.m.value, M.b.value
    return 0 if m < b else 1e6 + (m-b)**6

# Attach the constraints to the problem.  Give the soft limit value that is
# used for the constraints.  Without the soft limit, the fit would stall
# since we started it at a deep local minimum near the true solution without
# constraints.

problem = FitProblem(M, constraints=constraints, soft_limit=1e6)

# The constraint relies on the ability for python to access the parameters
# from the module.  Furthermore, the parameters still "boxed", and so you
# need to reference the value attribute to get the parameter value at the
# time the constraint is evaluated.  Not an elegant solution, but it works.
# Eventually we will add constraint expressions such as *M.m < M.b* or
# *M.m + M.b < 10* using the same infrastructure as equality constraints.
