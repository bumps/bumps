.. _materials-guide:

*******************
Materials
*******************

.. contents:: :local:

Because this
is elemental nickel, we already know it's density.  For compounds
such as 'SiO2' we would have to specify an additional
``density=2.634`` parameter.      


Common materials defined in :mod:`materialdb <refl1d.materialdb>`:

    *air*, *water*, *silicon*, *sapphire*, ...

Specific elements, molecules or mixtures can be added using the
classes in :mod:`refl1d.material`:

    *SLD*       unknown material with fittable SLD
    *Material*  known chemical formula and fittable density
    *Mixture*   known alloy or mixture with fittable fractions

