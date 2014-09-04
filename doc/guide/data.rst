.. _data-guide:

*******************
Data Representation
*******************

.. contents:: :local:

Data is x,y,dy.  Anything more complicated you will need to do yourself.

Bumps provides a rebinning functions for :func:`1-D <bumps.rebin.rebin>`
and :func:`2-D <bumps.rebin.rebin2d>` which can adjust a matrix of counts
to lie on a different grid.

Modeling
========

Bumps includes code for polynomial interpolation including
:func:`B-splines <bumps.bspline.bspline>`,
:func:`Parametric B-splines <bumps.bspline.pbs>`,
:func:`monotonic splines <bumps.mono.mono>`,
and :func:`chebyshev polynomials <bumps.cheby.cheby>`.

Once a theory function has been calculated, instrumental effects such
as resolution and background may need to be applied.  The
:func:`convolution function <bumps.data.convolve>` can be used for
data in which each point in a 1-D curve has an independent gaussian
resolution width.

Linear models
=============

Linear problems with normally distributed measurement error can be
solved directly.  Bumps provides :func:`bumps.wsolve.wsolve`, which weights
values according to the uncertainty.  The corresponding
:func:`bumps.wsolve.wpolyfit` function fits polynomials with measurement
uncertainty.
