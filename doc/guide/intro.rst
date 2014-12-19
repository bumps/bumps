.. _intro-guide:

***********
Using Bumps
***********

.. contents:: :local:

The first step in using Bumps is to define a fit file.  This is python
code defining the function, the fitting parameters and any data that is
being fitted.

A fit file usually starts with an import statement::

    from bumps.names import *

This imports names from :mod:`bumps.names` and makes the available to the
model definition.

Next the fit file should load the data with something like *np.loadtxt*
which loads columnar ASCII data into an array.  This data feeds into a
:class:`Fitness <bumps.fitproblem.Fitness>` function for a particular
model that gives the  probability of seeing the data for a given set of
model parameters.  These model functions can be quite complex, involving
not only the calculation of the theory function, but also simulating
instrumental resolution and background signal.

The fitness function will have :class:`Parameter <bumps.parameter.Parameter>`
objects defining the fittable parameters.  Usually the model is initialized
without any fitted parameters, allowing the user to set a
:meth:`range <bumps.parameter.Parameter.range>` on each parameter that
needs to be fitted.  Although it is a little tedious to set up, keeping the
fitted ranges separate from the model definition works better in the fitting
process, which usually involves multiple iterations with different
configurations. It is convenient to be able to turn on and off fitting for
individual parameter with a simple comment character ('#') at the start of
the line.

Every fit file ends with a :func:`FitProblem <bumps.fitproblem.FitProblem>`
definition::

    problem = FitProblem(model)

In fact, this is the only requirement of the fit file.  The Bumps engine
loads the fit file, retieves the *problem* symbol and feeds it to the selected
:mod:`fitter <bumps.fitter>`.  Some fit files do not even use *FitProblem* to
define *problem*, or use *Parameter* objects for the fitted parameters, so
long as *problem* implements the
:class:`BaseFitProblem <bumps.fitproblem.BaseFitProblem>` interface. which
provides *getp* to get the existing parameter vector, *setp* to set a new
parameter vector, *bounds* to return the parameter bounds, and *nllf* to


Note that the pattern of importing all names from a file using
*from bumps.names import \**, while convenient for simple scripts, can
make the code more difficult to understand later, and can lead to
unexpected results when moving code around to other files.  The alternative
pattern to use is::

    import bumps.names as bmp
    ...
    problem = bmp.FitProblem(model)

This documents to the reader unfamiliar with your code (such as you, dear
reader, when looking at your model files two years from now) exactly where
the name comes from.

The :ref:`tutorial-index` walks through the process for several different
data sets.