.. _sample-guide:

*********************
Sample Representation
*********************

.. contents:: :local:

Stacks
================

Reflectometry samples consist of 1-D stacks of layers joined by error
function interfaces. The layers themselves may be uniform slabs, or
the scattering density may vary with depth in the layer.  The first
layer in the stack is the substrate and the final layer is the surface.
Surface and substrate are assumed to be semi-infinite, with any thickness
ignored.

Multilayers
=============

Interfaces
=============

The interface between layers is assumed to smoothly follow and
error function profile to blend the layer above with the layer below.
The interface value is the 1-\ $\sigma$ gaussian roughness.
Adjacent flat layers with zero interface will act like a step function,
while positive values will introduce blending between the layers.

Blending is usually done with the Nevot-Croce formalism, which scales
the index of refraction between two layers by $\exp(-2 k_n k_{n+1} \sigma^2)$.
We show both a step function profile for the interface, as well as the
blended interface.

.. note::

    The blended interface representation is limited to the neighbouring
    layers, and is not an accurate representation of the effective
    reflectivity profile when the interface value is large relative to
    the thickness of the layer.

We will have a mechanism to force the use of the blended profile for
direct calculation of the interfaces rather than using the interface
scale factor.


Slabs
============

Materials can be stacked as slabs, with a thickness for each layer and
roughness at the top of each layer.  Because this is such a common
operation, there is special syntax to do it, using '|' as the layer
separator and `()` to specify thickness and interface.  For example,
the following is a 30 |Ang| gold layer on top of silicon, with a
silicon:gold interface of 5 |Ang| and a gold:air interface of 2 |Ang|::

    >> from refl1d import *
    >> sample = silicon(0,5) | gold(30,2) | air
    >> print sample
    Si | Au(30) | air

Individual layers and stacks can be used in multiple models, with all
parameters shared except those that are explicitly made separate.  The
syntax for doing so is similar to that for lists.  For example, the
following defines two samples, one with Si+Au/30+air and the other with
Si+Au/30+alkanethiol/10+air, with the silicon/gold layers shared::


    >> alkane_thiol = Material('C2H4OHS',bulk_density=0.8,name='thiol')
    >> sample1 = silicon(0,5) | gold(30,2) | air
    >> sample2 = sample1[:-1] | alkane_thiol(10,3) | air
    >> print sample2
    Si | Au(30) | thiol(10) | air

Stacks can be repeated using a simple multiply operation.  For example,
the following gives a cobalt/copper multilayer on silicon::

    >> Cu = Material('Cu')
    >> Co = Material('Co')
    >> sample = Si | [Co(30) | Cu(10)]*20 | Co(30) | air
    >> print sample
    Si | [Co(30) | Cu(10)]*20 | Co(30) | air

Multiple repeat sections can be included, and repeats can contain repeats.
Even freeform layers can be repeated.  By default the interface between
the repeats is the same as the interface between the repeats and the cap.
The cap interface can be set explicitly.  See :class:`model.Repeat` for
details.


Magnetic layers
===============

Polymer layers
==============

Functional layers
=================

Freeform layers
===============

Freeform profiles allow us to adjust the shape of the depth profile using
control parameters.  The profile can directly represent the scattering
length density as a function of depth (a FreeLayer), or the relative
fraction of one material and another (a FreeInterface).  With a freeform
interface you can simultaneously fit two systems which should share the
same volume profile but whose materials have different scattering length
densities.  For example, a polymer in deuterated and undeuterated solvents
can be simultaneously fit with freeform profiles.

We have multiple representations for freeform profiles, each with its
own strengths and weaknesses:

   * `monotone cubic interpolation
     <http://en.wikipedia.org/wiki/Monotone_cubic_interpolation>`_
     (:mod:`refl1d.mono`)
   * `parameteric B-splines
     <http://en.wikipedia.org/wiki/B-spline>`_
     (:mod:`refl1d.freeform`)
   * `Chebyshev interpolating polynomials
      <http://en.wikipedia.org/wiki/Chebyshev_polynomials>`_
      (:mod:`refl1d.cheby`)

At present, monotone cubic interpolation is the most developed, but work
on all representations is in flux.  In particular not every representation
supports all features, and the programming interface may vary. See the
documentation for the individual models for details.

Comparison of models
--------------------

There are a number of issues surrounding the choice of model.

* How easy is it to bound the profile values

  If the you can put reasonable bounds on the control points, then the
  user can bring to bear prior information to limit the search space.
  For example, it is common to add an unknown silicon-oxide profile
  to the surface of silicon, with SLD varying between the values for
  Si and SiO\ :sub:`2`

* How easy is it to edit the profile interactively

  Given a representation of the freeform layer, we want to be able to
  plot control points that you can drag in order to change the shape
  of the profile.

* Is the profile stable or does it oscillate wildly

  Many systems are best described by smoothly varying density profiles.
  If the profile oscillates wildly it makes the search for optimal
  parameters more difficult.

* Can you change the order of interpolation and preserve the profile

  While the current code does not support it, we would like to be
  able to select the freeform profile order automatically, using the
  minimum order we can to achieve $\chi^2 = 1$, and rejecting profiles
  which overfit the data.  For now this is done by hand, performing
  fits with different orders independently, but there are likely to
  be speed gains by first fitting coarse models with low Q then adding
  detail to the profile while adding additional Q values.

* Is the representation unique?  Are the control parameters strongly
  correlated?

  Fitting and uncertainty analysis benefit from unique solutions.  If
  the model representation is matched by a family of parameters it is
  more difficult to interpret the results of the uncertainty analysis
  or to get convergence from the parameter refinement engine.

Monotone cubic interpolation is the easiest to control.  The value of the
interpolating polynomial lies mostly within the range of the control
points, and the profile goes through the control points.  This means
you can set up bounds on the control parameters that limit the profile
to a certain range of scattering length densities in a region of the
profile.  It also leads to a very intuitive interactive profile editor
since the control points can be moved directly on profile view.  However,
although the profile is $C^1$ smooth everywhere, the $C^2$ transitions
can be abrupt at the control points.  Better algorithms for selecting the
gradient exist but have not been implemented, so this may improve in
the future.

Parametric B-splines are commonly used in computer graphics because they
create pleasing curves.  The interpolating polynomial lies within the
convex hull of the control points.  Unfortunately the distance between the
curve and the control point can be large, and this makes it difficult
to set reasonable bounds on the values of the control points.  One can
reformulate the interpolation so that control points lie on the curve
and still preserve the property of pleasing curves, but this can lead
to wild oscillations in the profile when the control points become too
close together.  While the natural representation can be used in an
interactive profile editor, the fact that the control points are sometimes
far away from the profile makes this inconvenient.  The complementary
representation is used in programs such as Microsoft Excel, with the
control point directly on the curve and a secondary control point to
adjust the slope at that control point.

Chebyshev interpolating polynomials are a near optimal representation
for an function over an interval with respect to the maximum norm.  The
interpolating polynomial is a weighted sum $\sigma_{i=0}^n c_i T_i(z)$
of the Chebyshev basis polynomials $T_i$ with Chebyshev coefficients $c_i$.
One very interesting property is that the lower order coefficients remain
the same has higher order interpolation polynomials are constructed.
This makes the Chebyshev polynomials very interesting candidates for
a freeform profile fitter which selects the order of the profile as
part of the fit.  Chebyshev interpolating polynomials can exhibit
wild oscillations if the coefficients become large, so the smoothness
can be somewhat controlled by limiting these higher values, but we have
not explored this in depth. The Chebyshev coefficient values are not
directly tied to the profile, so there is no intuitive way to directly
control the coefficients in an interactive editor. The complementary
representation uses the profile value at the chebyshev nodes for
specific positions $z_i$ on the profile.  This representation is much
more natural for an interactive editor, but some choices of control
values will lead to wild oscillations between the nodes.  Similarly
the complementary representation is unsuitable as a representation
for the fittable parameters since the bounds on the parameters do
not directly limit the range of possible values of the profile.


Future work
-----------

We only have polynomial spline representations for our profiles.  Similar
profiles could be constructed from different basis functions such as
wavelets, the idea being to find a multiscale representation of your
profile and use model selection techniques to determine the most coarse
grained representation that matches your data.

Totally freeform representations as separately controlled microslab
heights would also be interesting in the context of a maximum entropy
fitting engine: find the smoothest profile which matches the data, for
some definition of 'smooth'.  Some possible smoothness measures are the
mean squared distance from zero, the number of sign changes in the second
derivative, the sum of the absolute value of the first derivative, the
maximum flat region, the minimum number of flat slabs, etc.  Given that
reflectometry inversion is not unique, the smoothness measure must
correspond to the likelihood of finding the system in that particularly
state:  that is, don't expect your sample to show zebra stripes unless
you are on an African safari or visiting a zoo.


.. _new_layers:

Subclassing Layer
=================


.. TODO:  add references
