.. _getting-started-index:

###############
Getting Started
###############

Bumps is a set of routines for curve fitting and uncertainty analysis from
a Bayesian perspective.  Unlike traditional optimizers which search for the
best minimum they can find in the search space, uncertainty analysis tries
to find all viable minima, and find confidence intervals on the parameters
based on uncertainty in the measured values.  Bumps has been used for systems
of up to 100 parameters with tight constraints on the parameters.  

Bumps uses Markov chain Monte Carlo to explore the parameter space.  Although
it was created for curve fitting problems, Bumps can explore any probability 
density function, such as those defined by pyMCMC.  Unlike PyMCMC, Bumps 
includes a number of additional local and global optimizers, and a stepper which
can operate effectively on correlated parameter spaces.

Bumps can be used as a library within your own applications, or as a framework
for fitting, complete with a graphical user interface to manage your models.

.. toctree::
   :maxdepth: 2

   install.rst
   server.rst
   contributing.rst
   license.rst
