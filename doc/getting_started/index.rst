.. _getting-started-index:

###############
Getting Started
###############

Bumps is a set of routines for curve fitting and uncertainty analysis from
a Bayesian perspective.  In addition to traditional optimizers which search
for the best minimum they can find in the search space, bumps provides
uncertainty analysis which explores all viable minima and finds confidence
intervals on the parameters based on uncertainty in the measured values.
Bumps has been used for systems of up to 100 parameters with tight
constraints on the parameters.  Full uncertainty analysis requires hundreds
of thousands of function evaluations, which is only feasible for cheap
functions, systems with many processors, or lots of patience.

Bumps includes several traditional local optimizers such as Nelder-Mead
simplex, BFGS and differential evolution. Bumps uncertainty analysis uses
Markov chain Monte Carlo to explore the parameter space. Although
it was created for curve fitting problems, Bumps can explore any probability 
density function, such as those defined by PyMC.  In particular, the
bumps uncertainty analysis works well with correlated parameters.

Bumps can be used as a library within your own applications, or as a framework
for fitting, complete with a graphical user interface to manage your models.

.. toctree::
   :maxdepth: 2

   install.rst
   webview.rst
   server.rst
   contributing.rst
   license.rst
