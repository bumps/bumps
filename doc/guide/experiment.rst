.. _experiment-guide:

*******************
Experiment
*******************

.. contents:: :local:

It is the responsibility of the user to define their own experiment
structure.  The usual definition will describe the sample of interest,
the instrument configuration, and the measured data, and will provide
a theory function which computes the expected data given the sample
and instrument parameters.  The theory function frequently has an
physics component for computing the ideal data given the sample, and
an instrument effects component which computes the expected data from
the ideal data.  Together, sample, instrument, and theory function
define the fitting model which needs to match the data.

Fundamentally, the curve fitting problem can be expressed as:

.. math::

    P({\rm model}|{\rm data}) = \frac{P({\rm data}|{\rm model})P({\rm model})}{P({\rm data})}

That is, the probability of seeing a particular sample descrition given 
the observed data depends on the probability of seeing the measured
data given a proposed sample parameters scaled by the probability of 
those sample parameters and the probability of that data being measured.  
The experiment definition must return the negative log likelihood as
computed using the expression on the right.  Bumps will explore the
space of the sample and instrument parameters, returning the maximum
likelihood and confidence intervals on the parameters.

There is a strong relationship between the usual $\chi^2$ optimization
problem and the maximum likelihood problem. Given Gaussian uncertainty
for data measurements, we find that data $y_i$ measured with
uncertainty $\sigma_i$ will be observed for sample parameters $p$
when the instrument is at position $x_i$ with probability

.. math::

    P(y_i|f(x_i;p)) = \frac{1}{\sqrt{2\pi\sigma_i^2}} e ^ {-\frac{(y_i-f(x_i;p))^2}{2\sigma_i^2}}

and the log likelihood of observing all points in the data set for
the given set of sample parameters is

.. math::

   -\log \prod_i{P(y_i|f(x_i;p))} = \frac{1}{2}\sum_i{\frac{(y_i-f(x_i;p))^2}{\sigma_i^2}} - \frac{1}{2}\sum_i{\log 2 \pi \sigma_i^2}
                                = \frac{1}{2}\chi^2 + C

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
background, which is fundamentally wrong. See `_possion_example` for
a demonstration of this effect.

We can systematically incorporate prior information into our models. 
For example, if we characterize our instrumental uncertainty parameters 
against a known sample, we can incorporate this uncertainty into our 
models.  So if our sample angle control motor position follows
a Gaussian distribution with a target position of 3\ |deg| and an
uncertainty of 0.2\ |deg| with a Gaussian distribution, we can
set

.. math::

   -log P(model) = -\frac{1}{2} \frac{(\theta-3)^2}{0.2^2}

ignoring the scaling constant as before, and add this to $\chi^2/2$
to get log of the product of the uncertainties.  Similarly, if we
know that our sample should have a thickness of 100\ |pm| 3.5\ |Ang| 
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
*parameter.dev()* for Gaussian distributions or setting
parameter.bounds = :class:`Distribution <bumps.bounds.Distribution>`
for other distributions.

For counts data, :class:`PoissonCurve <bumps.curve.PoissonCurve>` is also
available.

If you are already have the negative log likelihood function, you can use
it with :class:`<bumps.pdfwrapper.PDF>`::

    x,y,dy = numpy.loadtxt('line.txt').T
    def nllf(m, b):
        return numpy.sum(((y - (m*x + b))/dy)**2)
    M = PDF(nllf)

Once you have defined your models and your parameter ranges, your
model file must define the fitting problem::
 
    problem = FitProblem(M)

More sophisticated models, with routines for data handling and specialized
plotting should define the :class:`Fitness <bumps.fitproblem.Fitness>`
interface.  `_example_peaks` sets up a problem for fitting multiple
peaks plus a background against a 2-D data set.  The Fitness object
must provide a method which returns a list of
:class:`Parameter <bumps.parameter.Parameter>` objects.  These
parameters are the basis of the Bumps model
to define models and constraints.  

External constraints
====================

Foreign models
==============

If your modeling environment already contains a sophisticated parameter
handling system (e.g. sympy or pyMCMC) you may want to tie into the Bumps
system at a higher level.  In this case you will need to define a
class which implements the :class:`FitProblem <bumps.fitproblem.FitProblem>`
interface.  This has been done already for 
:class:`PyMCMCProblem <bumps.pymcmc_model.PyMCMCProblem`
and interested parties are directed therein for a working example.

