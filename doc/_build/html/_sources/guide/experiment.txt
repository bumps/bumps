.. _experiment-guide:

*******************
Experiment
*******************

.. contents:: :local:

The :class:`Experiment <refl1d.experiment.Experiment>` object links a
`sample <sample>`_ with an experimental `probe <data>`_.
The probe defines the Q values and the resolution of the individual 
measurements, and returns the scattering factors associated with the 
different materials in the sample.


For the simple case of exploring the reflectivity of new samples,
this means that you must define 

the 
purposes:

  * defining the instrument resolution
  * providing the scattering factors for materials

Because our models allow representation based on composition, it is no
longer trivial to compute the reflectivity from the model.  We now have
to look up the effective scattering density based on the probe type and
probe energy.  You've already seen this in the `new_layers`_ section:
the render method for the layer requires the probe to look up the material
scattering factors.


Direct Calculation
==================

Rather than using :class:`Stack <refl1d.model.Stack`, 
:class:`Probe <refl1d.probe.Probe>` and 
class:`Experiment <refl1d.experiment.Experiment`, 
we  can compute reflectivities directly with the functions in
:mod:`refl1d.reflectivity`.  These routines provide the raw
calculation engines for the optical matrix formalism, converting
microslab models of the sample into complex reflectivity amplitudes,
and convolving the resulting reflectivity with the instrument resolution.

The following performs a complete calculation for a silicon
substrate with 5 |Ang| roughness using neutrons.  The theory is sampled 
at intervals of 0.001, which is convolved with a 1% $\Delta Q/Q$ resolution
function to yield reflectivities at intervals of 0.01.

    >>> from numpy import arange
    >>> from refl1d.reflectivity import reflectivity_amplitude as reflamp
    >>> from refl1d.reflectivity import convolve
    >>> Qin = arange(0,0.21,0.001)
    >>> w,rho,irho,sigma = zip((0,2.07,0,5),(0,0,0,0))
    >>> r = reflamp(kz=Qin/2, depth=w, rho=rho, irho=irho, sigma=sigma)
    >>> Rin = (r*r.conj()).real
    >>> Q = arange(0,0.2,0.01)
    >>> dQ = Q*0.01 # resolution dQ/Q = 0.01
    >>> R = convolve(Qin, Rin, Q, dQ)
    >>> print "\n".join("Q: %.2g  R: %.5g"%(Qi,Ri) for Qi,Ri in zip(Q,R))
