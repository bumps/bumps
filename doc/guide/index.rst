.. _users-guide-index:

############
User's Guide
############

Bumps is designed to determine the ideal model parameters for a given
set of measurements, and provide uncertainty on the parameter values.
This is an inverse problem, where measured data can be predicted from
theory, but theory cannot be directly inferred from measured data.  This
means that bumps must search through parameter space, calling the theory
function many times to find the parameter values that are most consistent
with the data.

Unlike traditional Levenburg-Marquardt fitting programs, Bumps does not
require normally distributed measurement uncertainty.  If a measurement
comes from counting statistics, for example, you can define your model with
poisson probability rather than gaussian probability.  Parameter values
can have constraints.  For example, if the size of a sample is known to
within 5%, the size parameter in the model can set to a gaussian distribution
with a standard deviation of 5%.  Simple bounds are also supported.  Parameter
expressions allow you to set the value of a parameter based on other
parameters, which allows simultaneous fitting of multiple datasets to
different models without having to define a specialized fit function.

Bumps includes Markov chain Monte Carlo (MCMC) methods to compute the
joint distribution of parameter probabilities.  These methods require
hundreds of thousand function calls to explore the search space, so
for moderately complex problems, you need to run in parallel.  Bumps
can fully utilize multiple cores on one computer, or through MPI, it
runs on supercomputing clusters.

..

    # Data handling has been removed so that we can ship a pure python package.
    In addition to inverse problem solving, bumps has acquired code for
    theory building and data handling.  For example, many problems have
    measurements in which the instrument resolution plays a role, and
    the theory function must be convolved with a data dependent resolution
    function.

:ref:`intro-guide`

     Model scripts associate a sample description with data and fitting
     options to define the system you wish to refine.

:ref:`data-guide`

     Data is loaded from instrument specific file
     formats into a generic :class:`Probe <bumps.data.Probe>`.  The
     probe object manages the data view and by extension, the view of
     the theory.  The probe object also knows the measurement resolution,
     and controls the set of theory points that must be evaluated
     in order to computed the expected value at each point.

:ref:`experiment-guide`

     Sample descriptions and data sets are combined into an
     :class:`Experiment <bumps.experiment.Experiment>` object,
     allowing the program to compute the expected reflectivity
     from the sample and the probability that reflectivity measured
     could have come from that sample.  For complex cases, where the
     sample varies on a length scale larger than the coherence length
     of the probe, you may need to model your measurement with a
     :class:`CompositeExperiment <bumps.experiment.CompositeExperiment>`.

:ref:`parameter-guide`

     The adjustable values in each component of the system are defined
     by :class:`Parameter <bumps.parameter>` objects.  When you
     set the range on a parameter, the system will be able to automatically
     adjust the value in order to find the best match between theory
     and data.

:ref:`fitting-guide`

     One or more experiments can be combined into a
     :class:`FitProblem <bumps.fitproblem.FitProblem>`.  This is then
     given to one of the many fitters, such as
     :class:`DEFit <refl1d.fitter.PTFit>`, which adjust the fitting
     parameters, trying to find the best fit.  See :ref:`optimizer-guide`
     for a description of available optimizers and :ref:`option-guide` for
     a description of the bumps options.


.. toctree::
    :hidden:

    intro.rst
    data.rst
    experiment.rst
    parameter.rst
    fitting.rst
    optimizer.rst
    options.rst
