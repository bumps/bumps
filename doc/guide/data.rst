.. _data-guide:

*******************
Data Representation
*******************

.. contents:: :local:

Data is x,y,dy.  Anything more complicated you will need to do yourself.

Modeling
========

Bumps includes code for polynomial interpolation including
:func:`B-splines <bumps.bspline.bspline>`,
:func:`Parametric B-splines <bumps.bspline.pbs>`,
:func:`monotonic splines <bumps.mono.mono>`,
and :func:`chebyshev polynomials <bumps.cheby.cheby>`.

Once a theory function has been calculated, instrumental effects such
as resolution and background may need to be applied.

Linear models
=============

Linear problems with normally distributed measurement error can be
solved directly.  Bumps provides :func:`bumps.wsolve.wsolve`, which weights
values according to the uncertainty.  The corresponding
:func:`bumps.wsolve.wpolyfit` function fits polynomials with measurement
uncertainty.
