.. _data-guide:

*******************
Data Representation
*******************

.. contents:: :local:

Data is represented using :class:`Probe <refl1d.probe.Probe>` objects.
The probe defines the Q values and the resolution of the individual
measurements, returning the scattering factors associated with the
different materials in the sample.  If the measurement has already
been performed, the probe stores the measured reflectivity and its
estimated uncertainty.

Probe objects are independent of the underlying instrument.  When
data is loaded, it is converted to angle $(\theta, \Delta \theta)$,
wavelength $(\lambda, \Delta \lambda)$ and reflectivity
$(R, \Delta R)$, with :class:`NeutronProbe <refl1d.probe.NeutronProbe>`
used for neutron radiation and :class:`XrayProbe <refl1d.probe.XrayProbe>`
used for X-ray radiation.  Additional properties,


Knowing the angle is necessary to correct for errors in sample alignment.

.. _data_simulation:

Simulated probes
================

.. _data_loading:

Loading data
============

For time-of-flight measurements, each angle should be represented as
a different probe.  This eliminates the 'stitching' problem, where
$Q = 4 \pi \sin(\theta_1)/\lambda_1 = 4 \pi \sin(\theta_2)/\lambda_2$
for some $(\theta_1,\lambda_1)$ and $(\theta_2,\lambda_2)$.
With stitching, it is impossible to account for effects such as
alignment offset since two nominally identical Q values will in
fact be different.  No information is lost treating the two data sets
separately --- each points will contribute to the overall cost function
in accordance with its statistical weight.


.. _data_views:

Viewing data
============

The probe object controls the plotting of theory and data curves.  This
is reasonable since it is only the probe which knows details such as
the original points and the points used in the calculation

.. _data_resolution:

Instrument Resolution
=====================

With the instrument in a given configuration ($\theta_i = \theta_f, \lambda$),
each neutron that is received is assigned to a particular $Q$ based on
the configuration.  However, these vaues are only nominal.  For example,
a monochromator lets in a range of wavelengths, and slits permit a range
of angles.  In effect, the reflectivity measured at the configuration
corresponds to a range of $Q$.

For monochromatic instruments, the wavelength resolution is fixed and
the angular resolution varies.  For polychromatic instruments, the
wavelength resolution varies and the angular resolution is fixed.
Resolution functions are defined in :mod:`refl1d.resolution`.

The angular resolution is determined by the geometry (slit positions,
openings and sample profile) with perhaps an additional contribution
from sample warp.  For monochromatic instruments, measurements are taken
with fixed slits at low angles until the beam falls completely onto the
sample.  Then as the angle increases, slits are opened to preserve full
illumination.  At some point the slit openings exceed the beam width,
and thus they are left fixed for all angles above this threshold.

When the sample is tiny, stray neutrons miss the sample and are not
reflected onto the detector.  This results in a resolution that is
tighter than expected given the slit openings.  If the sample width
is available, we can use that to determine how much of the beam is
intercepted by the sample, which we then use as an alternative second
slit.  This simple calculation isn't quite correct for very low $Q$, but
data in this region will be contaminated by the direct beam, so we
won't be using those points.

When the sample is warped, it may act to either focus or spread the
incident beam.  Some samples are diffuse scatters, which also acts
to spread the beam.  The degree of spread can be estimated from the
full-width at half max (FWHM) of a rocking curve at known slit settings.
The expected FWHM will be $\frac{1}{2}(s_1+s_2)/(d_1-d_2)$.  The difference
between this and the measured FWHM is the sample_broadening value.
A second order effect is that at low angles the warping will cast
shadows, changing the resolution and intensity in very complex ways.

For time of flight instruments, the wavelength dispersion
is determined by the reduction process which usually bins the time
channels in a way that sets a fixed relative resolution
$\Delta \lambda / \lambda$ for each bin.

Resolution in Q is computed from uncertainty in wavelength $\sigma_\lambda$
and angle $\sigma_\theta$ using propagation of errors:

.. math::

    \sigma^2_Q
        &= \left|\frac{\partial Q}{\partial \lambda}\right|^2 \sigma_\lambda^2
         + \left|\frac{\partial Q}{\partial \theta}\right|^2 \sigma_\theta^2
         + 2 \left|\frac{\partial Q}{\partial \lambda}
                   \frac{\partial Q}{\partial \theta}\right|^2
                   \sigma_{\lambda\theta}
         \\
    Q &= 4 \pi \sin(\theta) / \lambda \\
    \frac{\partial Q}{\partial \lambda} &= -4 \pi \sin(\theta)/\lambda^2
         = -Q/\lambda \\
    \frac{\partial Q}{\partial \theta} &= 4 \pi \cos(\theta)/\lambda
         = \cos(\theta) \cdot Q/\sin(\theta) = Q/\tan(\theta)

With no correlation between wavelength dispersion and angular divergence,
$\sigma_{\theta\lambda} = 0$, yielding the traditional form:

.. math::

    \left(\frac{\Delta Q}{Q}\right)^2
         = \left(\frac{\Delta \lambda}{\lambda}\right)^2
         + \left(\frac{\Delta \theta}{\tan(\theta)}\right)^2

Computationally, $1/\tan(\theta) \rightarrow \infty$ at $\theta=0$, so
it is better to use the direct calculation:

.. math::

    \Delta Q = 4 \pi/\lambda \sqrt{\sin(\theta)^2 (\Delta\lambda/\lambda)^2
                                   + \cos(\theta)^2 \Delta \theta^2}

Wavelength dispersion $\Delta \lambda/\lambda$ is usually constant
(e.g., for AND/R it is 2% FWHM), but it can vary on time-of-flight
instruments depending on how the data is binned.

Angular divergence $\delta \theta$ comes primarily from the slit geometry,
but can have broadening or focusing due to a warped sample.  The FWHM
divergence in radians due to slits is:

.. math::

    \Delta\theta_{\rm slits} = \frac{1}{2} \frac{s_1 + s_2}{d_1 - d_2}

where $s_1,s_2$ are slit openings edge to edge and $d_1,d_2$ are the distances
between the sample and the slits.  For tiny samples of width $m$, the sample
itself can act as a slit.  If $s = m \sin(\theta)$ is smaller than $s_2$ for
some $\theta$, then use:

.. math::

    \Delta\theta_{\rm slits} = \frac{1}{2} \frac{s_1 + m \sin(\theta)}{d_1}

The sample broadening can be read off a rocking curve using:

.. math::

    \Delta\theta_{\rm sample} = w - \Delta\theta_{\rm slits}

where $w$ is the measured FWHM of the peak in degrees. Broadening can be
negative for concave samples which have a focusing effect on the beam.  This
constant should be added to the computed $\Delta \theta$ for all angles and
slit geometries.  You will not usually have this information on hand, but
you can leave space for users to enter it if it is available.

FWHM can be converted to 1-\ $\sigma$ resolution using the scale factor of
$1/\sqrt{8 \ln 2}$.

With opening slits we assume $\Delta \theta/\theta$ is held constant, so if
you know $s$ and $\theta_o$ at the start of the opening slits region you
can compute $\Delta \theta/\theta_o$, and later scale that to your
particular $\theta$:

.. math::

    \Delta\theta(Q) = \Delta\theta/\theta_o \cdot \theta(Q)

Because $d$ is fixed, that means
$s_1(\theta) = s_1(\theta_o) \cdot \theta/\theta_o$ and
$s_2(\theta) = s_2(\theta_o) \cdot \theta/\theta_o$.


.. _data_resolution_calculator:

Applying Resolution
===================

The instrument resolution is applied to the theory calculation on
a point by point basis using a value of $\Delta Q$ derived from
$\Delta\lambda$ and $\Delta\theta$.   Assuming the resolution is
well approximated by a Gaussian,
:func:`convolve <refl1d.reflectivity.convolve>` applies it to the
calculated theory function.

The convolution at each point $k$ is computed from the piece-wise linear
function $\bar R_i(q)$ defined by the refectivity $R(Q_i)$ computed
at points $Q_i \in Q_\text{calc}$

.. math::

    \bar R_i(q) &= m_i q + b_i \\
    m_i &= (R_{i+1} - R_i)/(Q_{i+1} - Q_i) \\
    b_i &= R_i - m_i Q_i

and the Gaussian of width $\sigma_k = \Delta Q_k$

.. math::

    G_k(q) = \frac{1}{\sqrt{2 \pi}\sigma_k} e^{(q-Q_k)^2 / (2 \sigma_k^2)}

using the piece-wise integral

.. math::

    \hat R_k = \sum_{i=i_\text{min}}^{i_\text{max}}
        \int_{Q_i}^{Q_{i+1}} \bar R_i(q) G_k(q) dq

The range $i_\text{min}$ to $i_\text{max}$ for point $k$ is defined
to be the first $i$ such that $G_k(Q_i) < 0.001$, which is
about $3 \Delta Q_k$ away from $Q_k$.

By default the calculation points $Q_\text{calc}$ are the same
nominal $Q$ points at which the reflectivity was measured.   If the
data was measured densely enough, then the piece-wise linear function
$\bar R$ will be a good approximation to the underlying reflectivity.
There are two places in particular where this assumption breaks down.
One is near the critical edge for a sample that has sharp interfaces,
where the reflectivity drops precipitously. The other is in thick
samples, where the Kissig fringes are so close together that the
instrument cannot resolve them separately.

The method :meth:`Probe.critical_edge` fills in calculation points
near the critical edge.  Points are added linear around $Q_c$ for
a range of $\pm \delta Q_c$.  Thus, if the backing medium SLD or
the theta offset are allowed to vary a little during the fit, the
region after the critical edge may still be over-sampled.
The method :meth:`Probe.oversample` fills in calculation points
around every point, giving each $\hat R$ a firm basis of support.

While the assumption of Gaussian resolution is reasonable on fixed
wavelength instruments, it is less  so on time of flight instruments,
which have asymmetric wavelength  distributions.  You can explore the
effects of different distributions by subclassing
:class:`Probe <refl1d.probe.Probe>`  and overriding the
``_apply_resolution`` method.  We will happily accept code for
improved resolution calculators and non-gaussian convolution.


.. _data_backrefl:

Back reflectivity
=================

While reflectivity is usually performed from the sample surface,
there are many instances where them comes instead through the
substrate.  For example, when the sample is soaked in water or
${\rm D}_2{\rm O}$, a neutron beam will not penetrate well and
it is better to measure the sample through the substrate.  Rather
than reversing the sample representation, these datasets can
be flagged with the attribute *back_reflectivity=True*, and the
sample constructed from substrate to surface as usual.

When the beam enters the side of the substrate, there is a
small refractive shift in $Q$ based on the angle of the beam relative
to the side of the substrate. The refracted beam reflects off the
the reversed film then exits the substrate on the other side, with an
opposite refractive shift.  Depending on the absorption coefficient
of the substrate, the beam will be attenuated in the process.

The refractive shift and the reversing of the film are automatically
handled by the underlying reflectivity calculation.  You can even
combine measurements through the sample surface and the substrate
into a single measurement, with negative $Q$ values representing
the transition from surface to substrate.  This is not uncommon with
magnetic thin film samples.

Usually the absorption effects of the substrate are accounted for
by measuring the incident beam through the same substrate before
normalizing the reflectivity.  There is a slight difference in path
length through the substrate depending on angle, but it is not
significant.  When this is not the case, particularly for measurements
which cross from the surface to substrate in the same scan, an
additional *back_absorption* parameter can be used to scale the
back reflectivity relative to the surface reflectivity.  There
is an overall *intensity* parameter which scales both the surface
and the back reflectivity.

The interaction between *back_reflectivity*, *back_absorption*,
sample representation and $Q$ value can be somewhat tricky.  It


.. _data_alignment:

Alignment offset
================

It can sometimes be difficult to align the sample, particularly on
X-ray instruments.  Unfortunately, a misaligned sample can lead to
a error in the measured position of the critical edge.  Since the
statistics for the measurement are very good in this region, the
effects on the fit can be large.  By representing the angle directly,
an alignment offset can be incorporated into the reflectivity calculation.
Furthermore, the uncertainty in the alignment can be estimated from
the alignment scans, and this information incorporated directly into
the fit.  Without the theta offset correction you would need to
compensate for the critical edge by allowing the scattering length
density of the substrate to vary during the fit, but this would lead to
incorrectly calculated reflectivity for the remaining points.  For
example, the simulation :download:`toffset.py` shows more than 5% error
in reflectivity for a silicon substrate with a 0.005\ |deg| offset.

The method
:meth:`Probe.alignment_uncertainty <refl1d.probe.Probe.alignment_uncertainty>`
computes the uncertainty in a alignment from the information in a
rocking curve.  The alignment itself comes from the peak position in
the rocking curve, with uncertainty determined from the uncertainty
in the peak position.  Note that this is not the same as the width
of the peak; the peak stays roughly the same width as statistics are
improved, but the uncertainty in position and width will
decrease.\ [#Daymond2002]_ There is an additional uncertainty in
alignment due to motor step size, easily computed from the
variance in a uniform distribution.  Combined, the uncertainty
in *theta_offset* is:

.. math::

    \Delta\theta \approx \sqrt{w^2/I + d^2/12}

where $w$ is the full-width of the peak in radians at half maximum,
$I$ is the integrated intensity under the peak and $d$ is the motor
step size is radians.


.. _data_scattering_factors:

Scattering Factors
==================

The effective scattering length density of the material is dependent
on the composition of the material and on the type and wavelength of
the probe object.  Using the chemical formula,
:meth:`scattering_factors <refl1d.probe.Probe.scattering_factors>`
computes the scattering factors ($\rho$, $\rho_i$, $\rho_{\rm inc}$)
associated with the material.  This means the same sample representation
can be used for X-ray and neutron experiments, with mass density as the
fittable parameter.  For energy dependent materials (e.g., Gd for neutrons),
then scattering factors will be returned for all of the energies in the
probe. (Note: energy dependent neutron scattering factors are not yet
implemented in periodic table.)

The returned scattering factors are normalized to density=1 |g/cm^3|.
To use these values in the calculation of reflectivity, they need to
be scaled by density and volume fraction.  Using normalized density,
the value returned by scattering_factors can be cached so only one
lookup is necessary during the fit even when density is a fitting
parameter.

The material itself can be flagged to use the incoherent scattering
factor $\rho_{\rm inc}$ which is by default ignored.

Magnetic scattering factors for the material are not presently
available in the periodic table.  Interested parties may consider
extending periodic table with magnetic scattering information and
adding support to
:class:`PolarizedNeutronProbe <refl1d.probe.PolarizedNeutronProbe>`


.. [#Daymond2002] M.R. Daymond, P.J. Withers and M.W. Johnson;
   The expected uncertainty of diffraction-peak location",
   Appl. Phys. A 74 [Suppl.], S112 - S114 (2002).
   http://dx.doi.org/10.1007/s003390201392
