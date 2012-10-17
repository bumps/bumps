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


