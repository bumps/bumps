.. _parameter-guide:

*****************
Parameters
*****************

.. contents:: :local:

.. _freevariables:

Free Variables
==============

When fitting multiple datasets, you will undoubtedly have models with
many shared parameters, and some parameters that differ between the models.
Common patterns include:

# different measurements may use the same material but different contrast agents,
# or they may use the same contrast agent but different materials,
# or the same material and contrast, but different sizes,
# or a cross product with several materials and several sizes.

Often with complex models the parameter of interest is buried within the
model structure.  One approach is to clone the models using a deep copy of
the entire structure, then create new independent parameters for the bits
that are changing.  This proved to be confusing and difficult for new python
programmers, so instead :func:`bumps.fitproblem.FitProblem` was extended to
support  :class:`FreeVariables`.  The FreeVariables class allows you to use
the same model structure with different data sets, but have some parameters
that vary between the models.  Each varying parameter is a slot, and
FreeVariables keeps an array of parameters (actually a :class:`ParameterSet`)
to fill that slot, one for each model.

To define the free variables, you just need the names of the different models,
the name of the slot, and a reference to the parameter that is controlled
by that slot.  The result will look something like the following::

	model = Model()
	M1 = Fitness(model, data1)
	M2 = Fitness(model, data2)
	fv = FreeVariables(names=[M1.name, M2.name, ...],
			slot1=model.p1, ...)
	problem = FitProblem(M1, M2, ...,  freevars=fv)

The slots can be referenced by name, with the underlying parameters
referenced by variable number.  In the above, fv.slot1[1] refers to
the parameter p1 when fitting the data in M2.  The parameters in the
slots have the usual properties of parameters.  They have values and
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
	fv = FreeVariables(names=['hydrogenated', 'deuterated'],
			solvent=solvent.sld)
	fv.solvent[0].value = -0.561
	fv.solvent[1].value = 6.402
	model.radius.range(1,35)
	problem = FitProblem(M1, M2, freevars=fv)

In this particular example, the solvent is fixed for each measurement, and
the sphere radius is allowed to vary between 1 and 35.  Since the radius
is not a free variable, the fitted radius will be chosen such that it minimizes
the combined fitness of both models.   In a more complicated situation, we may
not know either the sphere radius or the solvent densities, but still the
radius is shared between the two models.  In this case we could set::

	fv.solvent.range(-1,7)

an the SLD of the solvent would be fitted independently in the two data sets.
Notice that we did not refer to the individual model index when setting the
range.  This is a convenience.  Range, pm and pmp can be set on the entire
set, or individually using, e.g.,

::
	fv.solvent[0].range(-1,0)
	fv.solvent[1].range(6,7)


