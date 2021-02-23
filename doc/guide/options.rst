.. :

    Fit option names are defined in bumps/fitters.py  Make sure any changes
    are done both hear and there.

.. _option-guide:

~~~~~~~~~~~~~
Bumps Options
~~~~~~~~~~~~~

*Bumps* has a number of options available to control the fits and the
output.  On the command line, each option is either *--option* if it
is True/False or *--option=value* if the option takes a value.  The
fit control form is used by graphical users interfaces to set the optimizer
and its controls and stopping conditions.  The long form name of the the
option will be used on the form.  Not all controls will appear on the form,
and will be set from the command line.

**Need to describe the array of output files produced by optimizers,
particularly dream.  Some of them (convergence plot, model plot, par file,
model file) are common to all.  Others (mcmc points) are specific to one
optimizer**


Bumps Command Line
==================

Usage::

    bumps [options] modelfile [modelargs]

The modelfile is a Python script (i.e., a series of Python commands)
which sets up the data, the models, and the fittable parameters.
The model arguments are available in the modelfile as sys.argv[1:].
Model arguments may not start with '-'.  The options all start with
'-' and can appear in any order anywhere on the command line.




Problem Setup
=============

.. _option-pars:

``--pars``
----------

Set initial parameter values from a previous fit.  The par file is a list
of lines with parameter name followed by parameter value on each line.
The parameters must appear with the same name and in the same order as
the fitted parameters in the model. Additional parameters are ignored. Missing
parameters are filled using LHS. :ref:`option-preview` will show the
model parameters.

.. _option-shake:

``--shake``
-----------

Set random initial values for the parameters in the model.  Note that
shake happens after :ref:`option-simulate` so that you can simulate a random
model, shake it, then try to recover its initial values.

.. _option-simulate:

``--simulate``
--------------

Simulate a dataset using the initial problem parameters.  This is useful
when setting up a model before an experiment to see what data it might
produce, and for seeing how well the fitting program might recover the
parameters of interest.

.. _option-simrandom:

``--simrandom``
---------------

Simulate a dataset using random initial parameters.  Because :ref:`option-shake`
is applied after :ref:`option-simulate`, we need a separate way to shake the
parameters before simulating the model.

.. _option-noise:

``--noise``
-----------

Set the noise percentage on the simulated data.  The default is 5 for 5%
normally distributed uncertainty in the measured values.  Use ``--noise=data``
to use the uncertainty on a dataset in the simulation.

.. _option-seed:

``--seed``
----------

Set a specific seed to the random number generator.  This happens before
shaking and simulating so that fitting tests, and particularly failures,
can be reliably reproduced.  The numpy random number generator is used
for all values, so any consistency guarantees between versions of bumps
over time and across platforms depends on the consistency of the numpy
generators. If no seed is specified then one will be generated and printed
so that the fit can be rerun with the same random sequence.



Stopping Conditions
===================

.. _option-steps:

``--steps``
-----------

*Steps* is the number of iterations that the algorithm will perform.  The
meaning of iterations will differ from optimizer to optimizer.  In the case
of population based optimizers such as :ref:`fit-de`, each step is an update to
every member of the population.  For local descent optimizers such as
:ref:`fit-amoeba` each step is an iteration of the algorithm.
:ref:`fit-dream` uses steps plus :ref:`option-burn` for the total number
of iterations.


.. _option-samples:

``--samples``
-------------

*Samples* sets the number of function evaluations.  This is an alternative
for setting the number of iterations of the algorithm, used when
:ref:`option-steps` is zero. Population optimizers perform :ref:`option-pop`
times the number of parameters in the fit for each step of the operation,
so given the desired number of samples, you can control the number of steps.
The number of samples is particularly convenient for :ref:`fit-dream`
(the only optimizer for which it is implemented at the moment), where 100,000
samples are needed to estimate the 1-sigma interval to 2 digits of accuracy
(assuming an approximately gaussian distribution), and 1,000,000 samples are
needed for the 95% confidence interval.  Like :ref:`option-steps`, the total
evaluations does not include any :ref:`option-burn` iterations.

.. _option-ftol:

``--ftol``
----------

*f(x) tolerance* uses differences in the function value to decide when the
fit is complete.  The different fitters will interpret this in different
ways.  The Newton descent algorithms (:ref:`fit-newton`, :ref:`fit-lm`) will use
this as the minimum improvement of the function value with each step.  The
population-based algorithms (:ref:`fit-de`, :ref:`fit-amoeba`) will use the
maximum difference between highest and lowest value in the population.
:ref:`fit-dream` does not use this stopping condition.


.. _option-xtol:

``--xtol``
----------

*x tolerance* uses differences in the parameter value to decide when the
fit is complete.  The different fitters will interpret this in different
ways.  The Newton descent algorithms (:ref:`fit-newton`, :ref:`fit-lm`) will use
this as the minimum change in the parameter values with each step.   The
population-based algorithgms (:ref:`fit-de`, :ref:`fit-amoeba`) will use the
maximum difference between highest and lowest parameter in the population.
:ref:`fit-dream` does not use this stopping condition.


.. _option-time:

``--time``
----------

*Max time* is the maximum running time of the optimizer.  This forces
the optimizer to stop even if tolerance or steps conditions are not met.
It is particularly useful for batch jobs run in an environment where the
queuing system stops the job unceremoniously when the time allocation is
complete.  Time is checked between iterations, so be sure to set it well
below the queue allocation so that it does not stop in the middle of an
iteration, and so that it has time to save its state.

.. _option-alpha:

``--alpha``
-----------

*Convergence* is the test criterion to use when deciding if stopping
conditions are met. This is for the variety of stopping tests built into
the DREAM algorithm. Usual values are `--alpha=0.01` or `--alpha=0.05`.
Note that various stopping criteria depend on the the number samples and
the chain length (where chain length x #pars x #pop = #samples), so there
is no definitive value to use for alpha, but larger values will allow the
fit to stop sooner.


Optimizer Controls
==================


.. _option-fit:

``--fit``
---------

*Fit Algorithm* selects the optimizer.  The available optimizers are:

  ====== ================
  amoeba :ref:`fit-amoeba`
  de     :ref:`fit-de`
  dream  :ref:`fit-dream`
  lm     :ref:`fit-lm`
  newton :ref:`fit-newton`
  pt     :ref:`fit-pt`
  ps     :ref:`fit-ps`
  rl     :ref:`fit-rl`
  ====== ================

The default fit method is ``--fit=amoeba``.


.. _option-pop:

``--pop``
---------

*Population* determines the size of the population.  For :ref:`fit-de` and
:ref:`fit-dream` it is a scale factor, where the number of individuals, $k$, is
equal to the number of fitted parameters times pop.  For :ref:`fit-amoeba`
the number of individuals is one plus the number of fitted parameters, as
determined by the size of the simplex.


.. _option-init:

``--init``
----------

*Initializer*  is used by population-based algorithms (:ref:`fit-dream`)
to set the initial population.  The options are as follows:

     *lhs* (latin hypersquare), which chops the bounds within each dimension
     in $k$ equal sized chunks where $k$ is the size of the population and
     makes sure that each parameter has at least one value within each chunk
     across the population.

     *eps* (epsilon ball), in which the entire initial population is chosen
     at random from within a tiny hypersphere centered about the initial point

     *cov* (covariance matrix), in which the uncertainty is estimated using
     the covariance matrix at the initial point, and points are selected
     at random from the corresponding gaussian ellipsoid

     *rand* (uniform random), in which the points are selected at random
     within the bounds of the parameters

:ref:`fit-amoeba` uses :ref:`option-radius` to initialize its simplex.
:ref:`fit-de` uses a random number from the prior distribution for the
parameter, if any.



.. _option-burn:

``--burn``
----------

*Burn-in Steps* is the number of iterations to required for the Markov
chain to converge to the equilibrium distribution.  If the fit ends
early, the tail of the burn will be saved to the start of the steps.
:ref:`fit-dream` uses burn plus steps as the total number of iterations to run.



.. _option-thin:

``--thin``
----------

*Thinning* is used by the Markov chain analysis to give samples time to
wander to different points in parameter space.  In an ideal chain, there
would be no correlation between points in the chain other than that which
is dictated by the equilibrium distribution.  However, if the space has
complicated boundaries and taking a step can easily lead to a highly
improbable point, then the chain may be stuck at the same value for
long periods of time.  If this is observed, then thinning can be used to
only keep every $n^\text{th}$ step, giving the saved chain a better opportunity
for good mixing.


.. _option-CR:

``--CR``
--------

*Crossover ratio* indicates the proportion of mixing which occurs with
each iteration.  This is a value in [0,1] giving the probability that
each individual dimension will be selected for update in the next generation.

.. _options-outliers:

``--outliers``
--------------

*Outliers* is used to identify chains that are stuck in high local minima
during dream burn-in. Options are:

* iqr: Use the interquartile range to determine the width of the distribution
  then exclude all chains whose log likelihood is more that two standard
  deviations below the first quartile.
* grubbs: Use a t-test to determine whether the samples in each chain are
  significantly different from the mean.
* mahal: Use the mahalanobis distance to determine whether the lowest
  probability chain is close to the remaining chain in parameter space.
  Only this chain will be marked as an outlier if the test fails.
* none: Don't do any outlier trimming.

The default is ``--outliers=none``. Outlier removal occurs every $2n$ steps
where $n$ is #samples/(#pars #pop), or when the convergence test indicates
the chains are stable.

Note that outliers are marked at the end of the fit using IQR and not
included in the statistics, though they are saved in the MCMC files. This
is independent of the ``--outliers`` setting.

.. _option-F:

``--F``
-------

*Scale* is a factor applied to the difference vector before adding it to
the parent in differential evolution.


.. _option-radius:

``--radius``
------------

*Simplex radius* is the radius of the initial simplex in :ref:`fit-amoeba`


.. _option-nT:

``--nT``
--------

*# Temperatures*  is the number of temperature chains to run using parallel
tempering.  Default is 25.

.. _option-Tmin:

``--Tmin``
----------

*Min temperature* is the minimum temperature in the log-spaced series of
temperatures to run using parallel tempering.  Default is 0.1.

.. _option-Tmax:

``--Tmax``
----------

*Max temperature* is the maximum temperature in the log-spaced series of
temperatures to run using parallel tempering.  Default is 10.

.. _option-starts:

``--starts``
------------

*Starts* is the number of times to run the fit from random starting points.

.. _option-keep-best:

``--keep_best``
---------------

If *Keep best* is set, then the each subsequent restart for the multi-start
fitter keeps the best value from the previous fit(s).



Execution Controls
==================

.. _option-store:

``--store``
-----------

Directory in which to store the results of the fit.  Fits produce multiple
files and plots.  Rather than cluttering up the current directory, all the
outputs are written to the store directory along with a copy of the model
file.

.. _option-overwrite:

``--overwrite``
---------------

If the store directory already exists then you need to include overwrite on
the command line to reuse it.  While inconvenient, this prevents accidental
overwriting of fits that may have taken hours to generate.

.. _option-checkpoint:

``--checkpoint``
----------------

Save fit state every ``--checkpoint=n`` hours. [dream only]

.. _option-resume:

``--resume``
------------

Continue fit from a previous store directory. Use ``--resume`` or ``--resume=-``
to reuse the existing store directory.

.. _option-parallel:

``--parallel``
--------------

Run fit using multiprocessing for parallelism. Use "--parallel=0" for all
CPUs or "--parallel=n" for only "n" CPUs.

.. _option-mpi:

``--mpi``
---------

Run fit using MPI for parallelism. Use command "mpirun -n cpus ..."
to run bumps for MPI. This will usually be the last line of a queue
submission script. Be sure to include ``--time=...`` to limit the fit
to run within the queue allocation time.

.. _option-batch:

``--batch``
-----------

Run fit in batch mode.  Progress updates are sent to *STORE/MODEL.mon*, and
can be monitored using *tail -f* (unix, mac).  When the fit is complete, the
plot png files are created as usual, but the interactive plots are not shown.
This allows you to set up a sequence of runs in a shell script where the
first run completes before the next run starts.  Batch is also useful for
cluster computing where the cluster nodes do not have access to the outside
network and can't display an interactive window.  Batch is automatic
when running with :ref:`option-mpi`.

.. _option-stepmon:

``--stepmon``
-------------

Create a log file tracking each point examined during the fit.  This does
not provide any real utility except for generating plots of the population
over time, which can be useful for understanding the different fitting
methods.


Output Controls
===============

.. _option-err:

```--err``
----------

Show uncertainties at the end of the fit using the square root of the
diagonals of the covariance matrix. See :ref:`option-cov`.

.. _option-cov:

``--cov``
---------

Compute the covariance matrix for the model at the minimum. With gaussian
uncertainties on the data, bumps is minimizing the sum of squares, so the
Jacobian matrix is used for the covariance, formed from the numerical
derivative of each residual with respect to each parameter. If the
likelihood function is not a simple sum of squared residuals, then
the Hessian matrix is used for the covariance, formed from the numerical
derivative of the likelihood with respect to pairs of parameters.

.. _option-entropy:

``--entropy``
-------------

*Calculate entropy* is a flag which indicates whether entropy should be
computed for the final fit. Entropy an estimate of the number of bits of
information available from the fit. Use "--entropy=method" to specify the
entropy calcualation method. This can be one of:

* gmm: fit sample to a gaussian mixture model (GMM) with $5 \sqrt{d}$
  components where $d$ is the number fitted parameters and estimate
  entropy by sampling from the GMM.

* llf: estimates likelihood scale factor from ratio of density
  estimate to model likelihood, then computes Monte Carlo entropy
  from sample; this does not work for marginal likelihood estimates.
  DOI:10.1109/CCA.2010.5611198

* mvn: fit sample to a multi-variate Gaussian and return the entropy
  of the best fit gaussian; uses bootstrap to estimate uncertainty.
  This method is only valid if the sample distribution is approximately
  Gaussian.

* wnn: estimate entropy from weighted nearest-neighbor distances in sample.
  Note: use with caution. The results from this implementation are not
  consistent with other methods. DOI:10.1214/18-AOS1688


.. _option-plot:

``--plot``
----------

For problems that have different view options for plotting, select the default
option to display.  For example, when fitting a power law to a dataset, you
may want to choose *log* or *linear* as the output plot type.


.. _option-trim:

``--trim``
----------

*Burn-in trim* finds the "burn point" after which the DREAM Markov chains
appear to have converged and ignores all points before it when plotting or
computing covariance and entropy. The trimmed points are still written to
the MCMC output files so they will be available when the fit is resumed.
Use ``--trim=true`` to set trimming.


.. _option-noshow:

``--noshow``
------------

*No show* suppresses the plot window after the fit. This is done automatically
when ``--batch`` is selected.

Bumps Controls
==============

.. _option-preview:

``--preview``
-------------

If the command contains *preview* then display model but do not perform
a fitting operation.  Use this to see the initial model before running a fit.
It will also show the fit range.

.. _option-chisq:

``--chisq``
-----------

If the command contains *chisq* then show $\chi^2$ and exit.  Use this to
check that the model does not have any syntax errors.

.. _option-resynth:

``--resynth``
-------------

Run a resynth uncertainty analysis on the model.  After finding a good
minimum, you can rerun bumps with:

     bumps --store=T1 --pars=T1/model.par --fit=amoeba --resynth=20 model.py

This will generate 20 data simulated datasets using the initial data
values as the mean and the data uncertainty as the standard deviation.
Each of these datasets will be fit with the specified optimizer, and the
resulting parameters saved in *T1/model.rsy*.  On completion, the parameter
values can be loaded into python and averaged or histogrammed.

.. _option-time_model:

``--time_model``
----------------

Run the model :ref:`option-steps` times and find the average run time per step.
If :ref:`option-parallel` is used, then the models will be run in parallel.


.. _option-profile:

``--profile``
-------------

Run the model :ref:`option-steps` times using the python profiler.  This can
be useful for identifying slow parts of your model definition, or
alternatively, finding out that the model runtime is smaller than the
Bumps overhead.  Use a larger value of steps for better statistics.


Special Options
===============

.. _option-edit:

``--edit``
----------

If the command contains *edit* then start the Bumps user interface so that
you can interact with the model, adjusting fitted parameters with a slider
and seeing how they impact the result.

.. _option-help:

``--help``, ``-h``, ``-?``
--------------------------

Use ``-?``, ``-h`` or ``--help`` to show a brief description of each
command line option.


.. _option-python:

``-i``, ``-m``, ``-c``, ``-p``
------------------------------

The bumps program can be used as a python interpreter with numpy, scipy,
matplotlib and bumps packages available.  This is useful if you do not have
python set up on your system, and you are using a bundled executable like
Bumps or Refl1D on windows.  Even if you have python, you may want to run the
bumps post-analysis scripts through the bumps command which already has
the appropriate path set up to bumps on your system.

The options are:

* ``-i``: run an interactive interpreter.

* ``-m package.module``: run a module as main. This is similar to
  ``python -m package.module`` with the python interpreter.

* ``-c expression``: run a python command and quit.

* ``-p script.py``: run a python script.
