# Fitting a curve
# ===============
#
# Fitting a curve to a data set and getting uncertainties on the
# parameters was the main reason that bumps was created, so it
# should be very easy to do.  Let's see if it is.
#
# As usual, let's import the standard names:

from bumps.names import *

# Let's start with some data.  Here we have some x values, which
# is the independent variable, and some y values, which represent
# the value measured for condition x.  In this case x is 1-D, but
# it could be a sequence of tuples instead.  We also need the
# uncertainty on each measurement if we want to get a meaningful
# uncertainty on the fitted parameters.

x = [1,2,3,4,5,6]
y = [2.1,4.0,6.3,8.03,9.6,11.9]
dy = [0.05,0.05,0.2,0.05,0.2,0.2]

# In this case we entered the data as a set of lists, but we could
# just as easily have loaded it from a three-column file using
# something like:
#
# .. parsed-literal::
#
#    data = numpy.loadtxt("data.txt").T
#    x,y,dy = data[0,:], data[1,:], data[2,:]
#
# The variations are endless --- cleaning the data so that it is
# in a fit state to model is often the hardest part in the analysis.

# Next We will define the function we want to fit.  The first argument
# to the function names the independent variable, and the remaining
# arguments are the fittable parameters.  The parameter arguments can
# use a bare name, or they can use name=value to indicate the default
# value for each parameter.  Our function defines a straight like of
# slope $m$ with intercept $b$ defaulting to 0.

def line(x, m, b=0):
    return m*x + b

# We can build a curve fitting object from our function and our data.
# This assumes that the measurement uncertainty is normally
# distributed, with a $1-\sigma$ confidence interval *dy* for each point.
# We can specify initial values for $m$ and $b$ when we define the
# model. We are going to constraint the fit so $m \in [0,4]$
# and $b \in [-5,5]$.

M = Curve(line,x,y,dy,m=2,b=2)
M.m.range(0,4)
M.b.range(-5,5)

# Every model file ends with a problem definition including a
# list of all models and datasets which are to be fitted.

problem = FitProblem(M)

# The complete model file :download:`curve.py <curve.py>` looks as follows:
#
# .. literalinclude:: curve.py
#
# It can be used as input to the bumps command:
#
# .. parsed-literal::
#
#    $ bumps curve.py --edit
#
# or it can be loaded into your own python script and used to
# drive bumps directly.
