.. _users-guide-index:

############
User's Guide
############

The complexity comes from multiple sources:

   * We are solving inverse problems on landscape with multiple minima,
     whose global minimum is small and often in an unpromising region.
   * The solution is not unique:  multiple minima may be equally valid
     solutions to the inversion problem.
   * The measurement is sensitive to nuisance parameters such as sample
     alignment.  That means the analysis program must include data
     reduction steps, making data handling complicated.
   * The models are complex.  Since the ideal profile is not unique and
     is difficult to locate, we often constrain our search to feasible
     physical models to limit the search space, and to account for
     information from other sources.


`Introduction <intro.html>`_

     Model scripts associate a sample description with data and fitting
     options to define the system you wish to refine.

`Parameters <parameter.html>`_

     The adjustable values in each component of the system are defined
     by :class:`Parameter <bumps.parameter>` objects.  When you
     set the range on a parameter, the system will be able to automatically
     adjust the value in order to find the best match between theory
     and data.

`Data <data.html>`_

     Data is loaded from instrument specific file
     formats into a generic :class:`Probe <bumps.data.Probe>`.  The
     probe object manages the data view and by extension, the view of
     the theory.  The probe object also knows the measurement resolution,
     and controls the set of theory points that must be evaluated
     in order to computed the expected value at each point.

`Experiments <experiment.html>`_

     Sample descriptions and data sets are combined into an
     :class:`Experiment <bumps.experiment.Experiment>` object,
     allowing the program to compute the expected reflectivity
     from the sample and the probability that reflectivity measured
     could have come from that sample.  For complex cases, where the
     sample varies on a length scale larger than the coherence length
     of the probe, you may need to model your measurement with a
     :class:`CompositeExperiment <bumps.experiment.CompositeExperiment>`.

`Fitting <fitting.html>`_

     One or more experiments can be combined into a
     :class:`FitProblem <refl1d.fitter.FitProblem>`.  This is then
     given to one of the many fitters, such as
     :class:`PTFit <refl1d.fitter.PTFit>`, which adjust the varying
     parameters, trying to find the best fit.  PTFit can also
     be used for Bayesian analysis in order to estimate the confidence
     in which the parameter values are known.


.. toctree::
   :maxdepth: 2
   :hidden:

   intro.rst
   parameter.rst
   data.rst
   experiment.rst
   fitting.rst
