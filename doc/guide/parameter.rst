.. py:currentmodule:: bumps.parameter

.. _parameter-guide:

**********
Parameters
**********

.. contents:: :local:

Bumps fitting is centered on :class:`Parameter` objects.  Parameters define
the search space, the uncertainty analysis and even the user interface.
Constraints within and between models are implemented through parameters.
Prior probabilities are defined by for parameters.

Model classes for Bumps should make it easy to define the initial
value of fitting parameters and tie parameters together.  When creating
a model, you should be able specify *parameter=value* for each of the
model parameters.  Later, you should be able to reference the parameter
within the model using *M.parameter*.  Parameters can also be tied together
by assigning the same *Parameter* object to two different parameters.
For example, a hollow cylinder can be created using::

    solvent = Parameter("solvent", value=1.2)
    shell = Parameter("shell", value=4.5)
    M = CoreShellCylinder(core=solvent, shell=shell, solvent=solvent,
                          radius=95, thickness=10, length=100)

The model parameter can also be a derived value that is the result of
a parameter expression.  For example, the following creates a cylinder
whose length is twice the radius::

     radius = Parameter("radius", value=3)
     M = Cylinder(radius=radius, length=2*radius)

Any time you ask for *M.length.value* it will compute the result as
*2\*radius.value* and return that.

You can also tie parameters together after the fact.  For example, you
can create the constrained cylinder using::

    M = Cylinder(radius=3, length=6)
    M.length = 2*M.radius

The advantage of this method is that you can easily comment out the
constraint when exploring the model space, and fit *length* and *radius*
freely.

Once you have defined your models and constraints you can set up
you fitting parameters.  There are several parameter methods which
are helpful:

- :meth:`range <Parameter.range>` forces the parameter to lie within
  a fixed range.  The parameter value can take on any value within
  the range with equal probability, and has zero probability outside
  the range.
- :meth:`pm <Parameter.pm>` is a convenient way to set up a range
  based on the initial value of parameter.  For example, *M.thickness.pm(10)*
  will allow the thickness parameter to vary by plus or minus 10.  You
  can do asymmetric ranges by calling *pm* with plus and minus values,
  such as *M.thickness.pm(-3,2)*.  The actual range gets set to a
  :func:`nice_range <bumps.bounds.nice_range>` that includes the bounds.
- :meth:`pmp <Parameter.pmp>` is like *pm* but the range is specified as
  a percent.  For example, to let thickness vary by 10%, use
  *M.thickness.pmp(10)*.  Again, a *nice_range* is used.
- :meth:`dev <Parameter.dev>` sets up a parameter whose prior probability
  is not equal across its range, but instead follows a normal distribution.
  If for example, you have measure the thickness to be $32.1 \pm 0.6$
  by some other technique, you can use this information to constrain your
  model by initializing *thickness* to 32.1 and setting
  *M.thickness.dev(0.6)* as a fitting constraint.  The *dev* method also
  accepts absolute limits, creating a truncated normal distribution.  You
  can set the central value *mu* as well, but you probably want to do this
  in the model initialization so that you are free to turn fitting of the
  parameter on and off by commenting out the *dev* line.
- :meth:`soft_range <Parameter.soft_range>` is a combination of *range*
  and *dev* in that the parameter has equal probability within [*low*,*high*]
  but Gaussian probability of width *std* as it strays outside of the range.

All these methods set the *bounds* attribute on the parameter in one way
or another.  See :mod:`bumps.bounds` for details.  Technically, setting
the parameter to *dev*, *soft_range* or *pdf* is equivalent to creating
a probability distribution model with a single data point and
:meth:`Fitness.nllf <bumps.fitproblem.Fitness.nllf>` equal to the negative
log likelihood of seeing the parameter value in the distribution.  This
*PDF* model would be fit simultaneously with your target model with the
parameter shared between them.  The result is statistically sound (it is
just more prior information), and conveniently, it does not affect the
number of degrees of freedom in the fit.

When defining new model classes, use the static method
:meth:`Parameter.default` to initialize the parameter.  This will
accept the input argument passed in by the user and depending on its
type, either create a new parameter slot and set its initial value,
or link the slot to another parameter.


.. _freevariables:

Free Variables
==============

When fitting multiple datasets, you will undoubtedly have models with
many shared parameters, and some parameters that differ between the models.
Common patterns include:

- different measurements may use the same material but different contrast agents,
- they may use the same contrast agent but different materials,
- the same material and contrast, but different sizes, or
- a cross product with several materials and several sizes.

Often with complex models the parameter of interest is buried within the
model structure.  One approach is to clone the models using a deep copy of
the entire structure, then tie together parameters for the bits
that are changing.  This proves to be confusing and difficult for new python
programmers, so instead :func:`FitProblem <bumps.fitproblem.FitProblem>` was
extended to support :class:`FreeVariables`.  The FreeVariables class allows
you to use the same model structure with different data sets, but have
some parameters that vary between the models.  Each varying parameter
is a slot, and FreeVariables keeps an array of parameters
(actually a :class:`ParameterSet`) to fill that slot, one for each model.


To define the free variables, you need the names of the different
models, a parameter slot to hold the values, and a list of the
different parameter values for each model.  You then define the
free variables as follows::

    free = FreeVariables(names=["model1", "model2", ...],
                     p1=model.p1, p2=model.p2, ...)
    ...
    problem = FitProblem(experiments, freevars=free)

The slots can be referenced by name, with the underlying parameters
referenced by variable number.  In the above, *free.p1[1]* refers to
the parameter p1 when fitting data2.  You can also refer to
the slots by name, such as *free.p1[data2.name]*.  The parameters in the
slots have the usual properties of parameters, such as values and
fit ranges.  Setting the fit range makes the parameter a fitted parameter,
and the fit will give the uncertainty on each parameter independently.
Parameters can be copied, so that a pair of models can share the same value.

The following examples shows a neutron scattering problems with two datasets,
one measured with light water and the other measured with heavy water, you
can share the same material object, but use the light water scattering
factors in the first and the heavy water scattering factors in the
second.  The problem would be composed as follows::

    material = SLD('silicon', rho=2.07)
    solvent = SLD('solvent') # unspecified rho
    model = Sphere(radius=10, material=material, solvent=solvent)
    M1 = ScatteringFitness(model, hydrogenated_data)
    M2 = ScatteringFitness(model, deuterated_data)
    free = FreeVariables(names=['hydrogenated', 'deuterated'],
                         solvent=solvent.sld)
    free.solvent.values = [-0.561, 6.402]
    model.radius.range(1,35)
    problem = FitProblem([M1, M2], freevars=free)

In this particular example, the solvent is fixed for each measurement, and
the sphere radius is allowed to vary between 1 and 35.  Since the radius
is not a free variable, the fitted radius will be chosen such that it minimizes
the combined fitness of both models.   In a more complicated situation, we may
not know either the sphere radius or the solvent densities, but still the
radius is shared between the two models.  In this case we could set::

    fv.solvent.range(-1,7)

and the SLD of the solvent would be fitted independently in the two data sets.
Notice that we did not refer to the individual model index when setting the
range.  This is a convenience---range, pm and pmp can be set on the entire
set as above, or individually using, e.g.,

::

    fv.solvent[0].range(-1,0)
    fv.solvent[1].range(6,7)
