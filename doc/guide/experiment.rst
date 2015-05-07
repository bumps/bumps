.. _experiment-guide:

**********
Experiment
**********

.. contents:: :local:

It is the responsibility of the user to define their own experiment
structure.  The usual definition will describe the sample of interest,
the instrument configuration, and the measured data, and will provide
a theory function which computes the expected data given the sample
and instrument parameters.  The theory function frequently has a
physics component for computing the ideal data given the sample and
an instrument effects component which computes the expected data from
the ideal data.  Together, sample, instrument, and theory function
define the fitting model which needs to match the data.

The curve fitting problem can be expressed as:

.. math::

    P(\text{model}\ |\ \text{data}) =
        {P(\text{data}\ |\ \text{model})P(\text{model}) \over P(\text{data})}

That is, the probability of seeing a particular set of model parameter values
given the observed data depends on the probability of seeing the measured
data given a proposed set of parameter values scaled by the probability of
those parameter values and the probability of that data being measured.
The experiment definition must return the negative log likelihood as
computed using the expression on the right.  Bumps will explore the
space of the sample and instrument parameters in the model, returning the
maximum likelihood and confidence intervals on the parameters.

There is a strong relationship between the usual $\chi^2$ optimization
problem and the maximum likelihood problem. Given Gaussian uncertainty
for data measurements, we find that data $y_i$ measured with
uncertainty $\sigma_i$ will be observed for sample parameters $p$
when the instrument is at position $x_i$ with probability

.. math::

    P(y_i\ |\ f(x_i;p)) = \frac{1}{\sqrt{2\pi\sigma_i^2}}
        \exp\left(-\frac{(y_i-f(x_i;p))^2}{2\sigma_i^2}\right)

The negative log likelihood of observing all points in the data set for
the given set of sample parameters is

.. math::

   -\log \prod_i{P(y_i\ |\ f(x_i;p))} =
       \tfrac12 \sum_i{\frac{(y_i-f(x_i;p))^2}{\sigma_i^2}}
       - \tfrac12 \sum_i{\log 2 \pi \sigma_i^2}
       = \tfrac12 \chi^2 + C

Note that this is the unnormalized $\chi^2$, whose expected value is the 
number of degrees of freedom in the model, not the reduced $\chi^2_R$ whose
expected value is $1$.  The Bumps fitting process is not sensitive to the
constant $C$ and it can be safely ignored.

Casting the problem as a log likelihood problem rather than $\chi^2$
provides several advantages.  We can support a richer set of measurement
techniques whose uncertainties do not follow a Gaussian distribution.
For example, if we have a Poisson process with a low count rate, the
likelihood function will be asymmetric, and a gaussian fit will tend
to overestimate the rate.  Furthermore, we can properly handle
background rates since we can easily compute the probability of seeing
the observed number of counts given the proposed signal plus background
rate.  Gaussian modeling can lead to negative rates for signal or
background, which is fundamentally wrong. See :ref:`poisson-fit` for
a demonstration of this effect.

We can systematically incorporate prior information into our models, such
as uncertainty in instrument configuration.  For example,  if our sample
angle control motor position follows a Gaussian distribution with a target
position of 3\ |deg| and an uncertainty of 0.2\ |deg|, we can set

.. math::

   -\log P(\text{model}) = -\frac{1}{2} \frac{(\theta-3)^2}{0.2^2}

ignoring the scaling constant as before, and add this to $\tfrac12\chi^2$
to get log of the product of the uncertainties.  Similarly, if we
know that our sample should have a thickness of 100 |pm| 3.5 |Ang|
based on how we constructed the sample, we can incorporate this
information into our model in the same way.

Simple experiments
====================

The simplest experiment is defined by a python function which takes
a list of instrument configuration and has arguments defining the 
parameters.  For example, to fit a line you would need::

    def line(x, m, b):
        return m*x + b

Assuming the data was in a 3 column ascii file with x, y and
uncertainty, you would turn this into a bumps model file using::

    # 3 column data file with x, y and uncertainty
    x,y,dy = numpy.loadtxt('line.txt').T  
    M = Curve(line, x, y, dy)

Using the magic of python introspection, 
:class:`Curve <bumps.curve.Curve>` is able to determine
the names of the fittable parameters from the arguments to the
function.  These are converted to 
:class:`Parameter <bumps.parameter.Parameter>` objects, the 
basis of the Bumps modeling system.  For each parameter, we can set
bounds or values::

    M.m.range(0,1)  # limit slope between 0 and 45 degrees
    M.b.value = 1   # the intercept is set to 1.

We could even set a parameter to a probability distribution, using
:meth:`Parameter.dev <bumps.parameter.Parameter.dev>` for Gaussian
distributions or setting parameter.bounds to
:class:`Distribution <bumps.bounds.Distribution>` for other distributions.

Bumps includes code for polynomial interpolation including
:func:`B-splines <bumps.bspline>`,
:func:`monotonic splines <bumps.mono>`,
and :func:`chebyshev polynomials <bumps.cheby>`.

For counts data, :class:`PoissonCurve <bumps.curve.PoissonCurve>` is also
available.

Likelihood functions
====================

If you are already have the negative log likelihood function and you don't
need to manage data, you can use it with :class:`PDF <bumps.pdfwrapper.PDF>`::

    x,y,dy = numpy.loadtxt('line.txt').T
    def nllf(m, b):
        return numpy.sum(((y - (m*x + b))/dy)**2)
    M = PDF(nllf)

You can use *M.m* and *M.b* to the parameter ranges as usual, then return
the model as a fitting problem:

    M.m.range(-inf,inf)
    M.b.range(-inf,inf)
    problem = FitProblem(M)

.. _fitness:

Complex models
==============

More sophisticated models, with routines for data handling and specialized
plotting should define the :class:`Fitness <bumps.fitproblem.Fitness>`
interface.  The :ref:`peaks-example` example sets up a problem for fitting
multiple peaks plus a background against a 2-D data set.

Models are parameterized using :class:`Parameter <bumps.parameter.Parameter>`
objects, which identify the fitted parameters in the model, and the bounds over
which they may vary.  The fitness object must provide a set of fitting
parameters to the fit problem using the
:meth:`parameters <bumps.fitproblem.Fitness.parameters>`  method.
Usually this returns a dictionary, with the key corresponding to the
attribute name for the parameter and the value corresponding to a
parameter object.  This allows the user of the model to guess that
parameter "p1" for example can be referenced using *model.p1*.  If the
model consists of parts, the parameters for each part must be returned.
The usual approach is to define a *parameters* method for each part
and build up the dictionary when needed (the *parameters* function is
only called at the start of the fit, so it does not need to be efficient).
This allows the user to guess that parameter "p1" of part "a" can be
referenced using *model.a.p1*.  A set of related parameters, p1, p2, ...
can be placed in a list and referenced using, e.g., *model.a.p[i]*.

The fitness constructor should accept keyword arguments for each
parameter giving reasonable defaults for the initial value.  The
parameter attribute should be created using
:meth:`Parameter.default <bumps.parameter.Parameter.default>`.
This method allows the user to set an initial parameter value when the
model is defined, or set the value to be another parameter in the fitting
problem, or to a parameter expression. The name given to the *default*
method should include the name of the model.  That way when the same
type of model is used for different data sets, the two sets of parameters
can be distinguished.  Ideally the model name would be based on the
data set name so that you can more easily figure out which parameter
goes with which data.

During an analysis, the optimizer will ask to evaluate a series of
points in parameter space.  Once the parameters have been set, the
:meth:`update <bumps.fitproblem.Fitness.update>` method will be called,
if there is one.  This method should clear any cached results from the
last fit point.  Next the :meth:`nllf <bumps.fitproblem.Fitness.nllf>`
method will be called to compute the negative log likelihood of observing
the data given the current values of the parameters.   This is usually
just $\sum{(y_i - f(x_i))^2 / (2 \sigma_i^2)}$ for data measured with
Gaussian uncertainty, but any probability  distribution can be used.

For the Levenberg-Marquardt optimizer, the
:meth:`residuals <bumps.fitproblem.Fitness.residuals>` method will be
called instead of *nllf*.  If residuals are unavailable, then the L-M
method cannot be used.

The :meth:`numpoints <bumps.fitproblem.Fitness.numpoints>` method is used
to report fitting progress.  With Gaussian measurement uncertainty, the
*nllf* return value is $\chi^2/2$, which has an expected value of
the number of degrees of freedom in the fit.  Since this is an awkward
number, the normalized chi-square,
$\chi^2_N = \chi^2/\text{DoF} = -2 \ln (P)/(n-p)$, is shown
instead, where $-\ln P$ is the *nllf* value, $n$ is the of points
and $p$ is the number of fitted parameters.  $\chi^2_N$ has a value near 1
for a good fit.  The same calculation is used for non-gaussian
distributions even though *nllf* is not returning sum squared residuals.

The :meth:`save <bumps.fitproblem.Fitness.save>` and
:meth:`plot <bumps.fitproblem.Fitness.plot>` methods will  be called at
the end of the fit.  The *save* method should save the model for the
current point.  This may include things such as the calculated scattering
curve and the real space model for scattering inverse problems, or it
may be a save of the model parameters in a format that can be loaded by
other programs.  The *plot* method should use the current matplotlib
figure to draw the model, data, theory and residuals.

The :meth:`resynth_data <bumps.fitproblem.Fitness.resynth_data>` method
is used for an alternative monte carlo error analysis where random
data sets are generated from the measured value and the uncertainty
then fitted.  The resulting fitted parameters can be processed much
like the MCMC datasets, yielding a different estimate on the uncertainties
in the parameters.  The
:meth:`restore_data <bumps.fitproblem.Fitness.restore_data>` method
restores the data to the originally measured values.  These methods
are optional, and only used if the alternative error analysis is
requested.

Linear models
=============

Linear problems with normally distributed measurement error can be
solved directly.  Bumps provides :func:`bumps.wsolve.wsolve`, which weights
values according to the uncertainty.  The corresponding
:func:`bumps.wsolve.wpolyfit` function fits polynomials with measurement
uncertainty.


Foreign models
==============

If your modeling environment already contains a sophisticated parameter
handling system (e.g. sympy or PyMC) you may want to tie into the Bumps
system at a higher level.  In this case you will need to define a
class which implements the :class:`FitProblem <bumps.fitproblem.FitProblem>`
interface.  This has been done already for 
:class:`PyMCProblem <bumps.pymcfit.PyMCProblem`
and interested parties are directed therein for a working example.


External constraints
====================
