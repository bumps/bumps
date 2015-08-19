.. _fitting-guide:

*******
Fitting
*******

.. contents:: :local:


Obtaining a good fit depends foremost on having the correct model to fit.

For example, if you are modeling a curve with spline, you will overfit
the data if you have too many spline points, or underfit it if you do not
have enough.  If the underlying data is ultimately an exponential, then
the spline order required to model it will require many more parameters
than the corresponding exponential.

Even with the correct model, there are systematic errors to address
(see :ref:`data-guide`).  A distorted sample can lead to broader resolution
than expected for the measurement technique, and you will need to adjust your
resolution function.  Imprecise instrument control will lead to uncertainty
in the position of the sample, and corresponding changes to the measured
values.  For high precision experiments, your models will need to incorporate
these instrument effects so that the uncertainty in instrument configuration
can be properly accounted for in the uncertainty in the fitted parameter
values.


Quick Fit
=========

While generating an appropriate model, you will want to perform a number
of quick fits.  The :ref:`fit-amoeba` works well for this.  You will want
to run enough iterations ``--steps=1000`` so the algorithm has a
chance to  converge.  Restarting a number of times ``--starts=10`` gives
a reasonably thorough search of the fit space.  Once the fit converges,
additional starts are very quick.  From the graphical user interface, using
``--starts=1`` and clicking the fit button to improve the fit as needed works
pretty well. From the command line interface, the command line will be
something like::

    bumps --fit=amoeba --steps=1000 --starts=20 --parallel model.py --store=T1

Here, the results are kept in a directory ``--store=T1`` relative to the current
directory, with files containing the current model in *model.py*, the fit
result in *model.par* and a plots in *model-\*.png*.  The parallel option
indicates that multiple cores should be used on the cpu when running the fit.

The fit may be able to be improved by using the current best fit value as
the starting point for a new fit::

    bumps --fit=amoeba --steps=1000 --starts=20 --parallel model.py --store=T1 --pars=T1/model.par

If the fit is well behaved, and a numerical derivative exists, then
switching to :ref:`fit-newton` is useful, in that it will very rapidly
converge to a nearby local minimum.

::

    bumps --fit=newton model.py --pars=T1/model.par --store=T1

:ref:`fit-de` is an alternative to :ref:`fit-amoeba`, perhaps a little
more likely to find the global minimum but somewhat slower.  This is a
population based algorithms in which several points from the current
population are selected, and based on the position and value, a new point
is generated.  The population is specified as a multiplier on the number
of parameters in the model, so for example an 8 parameter model with
DE's default population ``--pop=10`` would create 80 points each generation.
This algorithms can be called from the command line as follows::

    bumps --fit=de --steps=3000 --parallel model.py --store=T1

Some fitters save the complete state of the fitter on termination so that
the fit can be resumed.  Use ``--resume=path/to/previous/store`` to resume.
The resumed fit also needs a ``--store=path/to/store``, which could be the
same as the resume path if you want to update it, or it could be a completely
new path.


See :ref:`optimizer-guide` for a description of the available optimizers, and
:ref:`option-guide` for a description of all the bumps options.

Uncertainty Analysis
====================

More important than the optimal value of the parameters is an estimate
of the uncertainty in those values.  By casting our problem as the
likelihood of seeing the data given the model, we not only give
ourselves the ability to incorporate prior information into the fit
systematically, but we also give ourselves a strong foundation for
assessing the uncertainty of the parameters.

Uncertainty analysis is performed using :ref:`fit-dream`.  This is a
Markov chain Monte Carlo (MCMC) method with a differential evolution
step generator.  Like simulated annealing, the MCMC explores the space
using a random walk, always accepting a better point, but sometimes
accepting a worse point depending on how much worse it is.

DREAM can be started with a variety of initial populations.  The
random population ``--init=random`` distributes the initial points using
a uniform distribution across the space of the parameters.  Latin
hypersquares ``--init=lhs`` improves on random by making sure that
there is on value for each subrange of every variable. The covariance
population ``--init=cov`` selects points from the uncertainty ellipse
computed from the derivative at the initial point.  This method
will fail if the fitting parameters are highly correlated and the
covariance matrix is singular.  The $\epsilon$-ball population ``--init=eps``
starts DREAM from a tiny region near the initial point and lets it
expand from there.  It can be useful to start with an epsilon ball
from the previous best point when DREAM fails to converge using
a more diverse initial population.

The Markov chain will take time to converge on a stable population.
This burn in time needs to be specified at the start of the analysis.
After burn, DREAM will collect all points visited for N iterations
of the algorithm.  If the burn time was long enough, the resulting
points can be used to estimate uncertainty on parameters.

A common command line for running DREAM is::

   bumps --fit=dream --burn=1000 --samples=1e5 --init=cov --parallel --pars=T1/model.par model.py --store=T2


Bayesian uncertainty analysis is described in the GUM Supplement 1,[8]
and is a valid technique for reporting parameter uncertainties in NIST
publications.   Given sufficient burn time, points in the search space
will be visited with probability proportional to the goodness of fit.
The file T1/model.err contains a table showing for each
parameter the mean(std), median and best values, and the 68% and 95%
credible intervals.  The mean and standard deviation are computed from
all the samples in the returned distribution.  These statistics are not
robust: if the Markov process has not yet converged, then outliers will
significantly distort the reported values.  Standard deviation is
reported in compact notation, with the two digits in parentheses
representing uncertainty in the last two digits of the mean.  Thus, for
example, $24.9(28)$ is $24.9 \pm 2.8$.  Median is the best value in the
distribution.  Best is the best value ever seen.  The 68% and 95%
intervals are the shortest intervals that contain 68% and 95% of
the points respectively.  In order to report 2 digits of precision on
the 95% interval, approximately 1000000 samples drawn from the distribution
are required, or steps = 1000000/(#parameters  #pop).  The 68% interval
will require fewer draws, though how many has not yet been determined.

.. image:: var.png
    :scale: 50

Histogramming the set of points visited will gives a picture of the
probability density function for each parameter.  This histogram is
generated automatically and saved in T1/model-var.png.  The histogram
range represents the 95% credible interval, and the shaded region
represents the 68% credible interval.  The green line shows the highest
probability observed given that the parameter value is restricted to
that bin of the histogram.  With enough samples, this will correspond
to the maximum likelihood value of the function given that one parameter
is restricted to that bin.  In practice, the analysis has converged
when the green line follows the general shape of the histogram.

.. image:: corr.png
    :scale: 50

The correlation plots show that the parameters are not uniquely
determined from the data.  For example, the thickness of
lamellae 3 and 4 are strongly anti-correlated, yielding a 95% CI of
about 1 nm for each compared to the bulk nafion thickness CI of 0.2 nm.
Summing lamellae thickness in the sampled points, we see the overall
lamellae thickness has a CI of about 0.3 nm.  The correlation
plot is saved in T1/model-corr.png.


.. image:: error.png
    :scale: 50

To assure ourselves that the uncertainties produced by DREAM do
indeed correspond to the underlying uncertainty in the model, we perform
a Monte Carlo forward uncertainty analysis by selecting 50 samples from
the computed posterior distribution, computing the corresponding
theory function and calculating the normalized residuals.  Assuming that
our measurement uncertainties are approximately normally distributed,
approximately 68% of the normalized residuals should be within +/- 1 of
the residual for the best model, and 98% should be within +/- 2. Note
that our best fit does not capture all the details of the data, and the
underlying systematic bias is not included in the uncertainty estimates.

Plotting the profiles generated from the above sampling method, aligning
them such that the cross correlation with the best profile is maximized,
we see that the precise details of the lamellae are uncertain but the
total thickness of the lamellae structure is well determined.  Bayesian
analysis can also be used to determine relative likelihood of different
number of layers, but we have not yet performed this analysis.  This plot
is stored in *T1/model-errors.png*.

The trace plot, *T1/model-trace.png*, shows the mixing properties of the
first fitting parameter.  If the Markov process is well behaved, the
trace plot will show a lot of mixing.  If it is ill behaved, and each
chain is stuck in its own separate local minimum, then distinct lines
will be visible in this plot.

The convergence plot, *T1/model-logp.png*, shows the log likelihood
values for each member of the population.  When the Markov process
has converged, this plot will be flat with no distinct lines visible.
If it shows a general upward sweep, then the burn time was not
sufficient, and the analysis should be restarted.  The ability to
continue to burn from the current population is not yet implemented.

Just because all the plots are well behaved does not mean that the
Markov process has converged on the best result.  It is practically
impossible to rule out a deep minimum with a narrow acceptance
region in an otherwise unpromising part of the search space.

In order to assess the DREAM algorithm for suitability for our
problem space we did a number of tests.  Given that our fit surface is
multimodal, we need to know that the uncertainty analysis can return
multiple modes.  Because the fit problems may also be ill-conditioned,
with strong correlations or anti-correlations between some parameters,
the uncertainty analysis needs to be able to correctly indicate that
the correlations exist. Simple Metropolis-Hastings sampling does not
work well in these conditions, but we found that DREAM is able to 
handle them.  We are still affected by the curse of dimensionality.
For correlated parameters in high dimensional spaces, even DREAM has
difficulty taking steps which lead to improved likelihood.  For
example, we can recover an eight point spline with generous ranges
on its 14 free parameters close to 100% of the time, but a 10 point
spline is rarely recovered.



Using the posterior distribution
================================

You can load the DREAM output population an perform uncertainty analysis
operations after the fact::

    from bumps.dream.state import load_state
    state = load_state(modelname)
    state.mark_outliers() # ignore outlier chains
    state.show()  # Plot statistics


You can restrict a variable to a certain range when doing plots.
For example, to restrict the third parameter to $[0.8,1.0]$ and the
fifth to $[0.2,0.4]$::

    from bumps.dream import views
    selection={2: (0.8,1.0), 4:(0.2,0.4),...}
    views.plot_vars(state, selection=selection)
    views.plot_corrmatrix(state, selection=selection)

You can also add derived variables using a function to generate the
derived variable.  For example, to add a parameter which is ``p[0]+p[1]``
use::

    state.derive_vars(lambda p: p[0]+p[1], labels=["x+y"])

You can generate multiple derived parameters at a time with a function
that returns a sequence::


    state.derive_vars(lambda p: (p[0]*p[1],p[0]-p[1]), labels=["x*y","x-y"])

These new parameters will show up in your plots::

    state.show()

The plotting code is somewhat complicated, and matplotlib doesn't have a
good way of changing plots interactively.  If you are running directly
from the source tree, you can modify the dream plotting libraries as you
need for a one-off plot, the replot the graph::


    # ... change the plotting code in dream.views/dream.corrplot
    reload(dream.views)
    reload(dream.corrplot)
    state.show()

Be sure to restore the original versions when you are done.  If the change
is so good that everyone should use it, be sure to feed it back to the
community via the bumps source control system at
`github <https://github.com/bumps>`_.

Publication Graphics
====================

The matplotlib package is capable of producing publication quality
graphics for your models and fit results, but it requires you to write
scripts to get the control that you need.  These scripts can be run
from the Bumps application by first loading the model and the fit
results then accessing their data directly to produce the plots that
you need.

The model file (call it *plot.py*) will start with the following::

    import sys
    from bumps.cli import load_problem, load_best

    model, store = sys.argv[1:3]

    problem = load_problem([model])
    load_best(problem, os.path.join(store, model[:-3]+".par"))
    chisq = problem.chisq

    print "chisq",chisq

Assuming your model script is in model.py and you have run a fit with
``--store=X5``, you can run this file using::

    $ bumps plot.py model.py X5

Now *model.py* is loaded and the best fit parameters are set.

To produce plots, you will need access to the data and the theory.  This
can be complex depending on how many models you are fitting and how many
datasets there are per model.  For single experiment models defined
by :func:`FitProblem <bumps.fitproblem.FitProblem>`, your original
experiment object  is referenced by *problem.fitness*.  For simultaneous
refinement defined by *FitProblem* with multiple *Fitness* objects,
use ``problem.models[k].fitness`` to access the experiment for
model *k*.  Your experiment object should provide methods for retrieving
the data and plotting data vs. theory.

How does this work in practice?  Consider the reflectivity modeling
problem where we have a simple model such as nickel film on a silicon
substrate.  We measure the specular reflectivity as various angles and
try to recover the film thickness.  We want to make sure that our
model fits the data within the uncertainty of our measurements, and
we want some graphical representation of the uncertainty in our film
of interest.  The refl1d package provides tools for generating the
sample profile uncertainty plots.  We access the experiment information
as follows::

    experiment = problem.fitness
    z,rho,irho = experiment.smooth_profile(dz=0.2)
    # ... insert profile plotting code here ...
    QR = experiment.reflectivity()
    for p,th in self.parts(QR):
        Q,dQ,R,dR,theory = p.Q, p.dQ, p.R, p.dR, th[1]
        # ... insert reflectivity plotting code here ...

Next we can reload the the error sample data from the DREAM MCMC sequence::

    import dream.state
    from bumps.errplot import calc_errors_from_state, align_profiles

    state = load_state(os.path.join(store, model[:-3]))
    state.mark_outliers()
    # ... insert correlation plots, etc. here ...
    profiles,slabs,Q,residuals = calc_errors_from_state(problem, state)
    aligned_profiles = align_profiles(profiles, slabs, 2.5)
    # ... insert profile and residuals uncertainty plots here ...

The function :func:`bumps.errplot.calc_errors_from_state` calls the
calc_errors function defined by the reflectivity model.  The return value is
arbitrary, but should be suitable for the show_errors function defined
by the reflectivity model.

Putting the pieces together, here is a skeleton for a specialized
plotting script::

    import sys
    import pylab
    from bumps.dream.state import load_state
    from bumps.cli import load_problem, load_best
    from bumps.errplot import calc_errors_from_state
    from refl1d.align import align_profiles

    model, store = sys.argv[1:3]

    problem = load_problem([model])
    load_best(problem, os.path.join(store, model[:-3]+".par"))

    chisq = problem.chisq
    experiment = problem.fitness
    z,rho,irho = experiment.smooth_profile(dz=0.2)
    # ... insert profile plotting code here ...
    QR = experiment.reflectivity()
    for p,th in self.parts(QR):
        Q,dQ,R,dR,theory = p.Q, p.dQ, p.R, p.dR, th[1]
        # ... insert reflectivity plotting code here ...

    if 1:  # Loading errors is expensive; may not want to do so all the time.
        state = load_state(os.path.join(store, model[:-3]))
        state.mark_outliers()
        # ... insert correlation plots, etc. here ...
        profiles,slabs,Q,residuals = calc_errors_from_state(problem, state)
        aligned_profiles = align_profiles(profiles, slabs, 2.5)
        # ... insert profile and residuals uncertainty plots here ...

    pylab.show()
    raise Exception()  # We are just plotting; don't run the model

Tough Problems
==============

.. note::

   DREAM is currently our most robust fitting algorithm.  We are
   exploring other algorithms such as parallel tempering, but they
   are not currently competitive with DREAM.

With the toughest fits, for example freeform models with arbitrary 
control points, DREAM only succeeds if the model is small or the 
control points are constrained.  We have developed a parallel 
tempering (fit=pt) extension to DREAM.  Whereas DREAM runs with a 
constant temperature, $T=1$, parallel tempering runs with multiple
temperatures concurrently.   The high temperature points are able to 
walk up steep hills in the search space, possibly crossing over into a
neighbouring valley.  The low temperature points agressively seek the
nearest local minimum, rejecting any proposed point that is worse than
the current.  Differential evolution helps adapt the steps to the shape
of the search space, increasing the chances that the random step will be
a step in the right direction.  The current implementation uses a fixed
set of temperatures defaulting to ``--Tmin=0.1`` through ``--Tmax=10`` in
``--nT=25`` steps; future versions should adapt the temperature based
on the fitting problem.

Parallel tempering is run like dream, but with optional temperature
controls::

   bumps --fit=dream --burn=1000 --samples=1e5 --init=cov --parallel --pars=T1/model.par model.py --store=T2

Parallel tempering does not yet generate the uncertainty plots provided
by DREAM.  The state is retained along the temperature for each point,
but the code to generate histograms from points weighted by inverse
temperature has not yet been written.

Parallel tempering performance has been disappointing.  In theory it
should be more robust than DREAM, but in practice, we are using a
restricted version of differential evolution with the population
defined by the current chain rather than a set of chains running in
parallel.  When the Markov chain has converged these populations
should be equivalent, but apparently this optimization interferes
with convergence.  Time permitting, we will improve this algorithm
and look for other ways to improve upon the robustness of DREAM.


Command Line
============

The GUI version of Bumps is slower because it frequently updates the graphs
showing the best current fit.

Run multiple models overnight, starting one after the last is complete
by creating a batch file (e.g., run.bat) with one line per model.  Append
the parameter --batch to the end of the command lines so the program
doesn't stop to show interactive graphs::

    bumps model.py ... --parallel --batch

You can view the fitted results in the GUI the next morning using::

    bumps --edit model.py --pars=T1/model.par
