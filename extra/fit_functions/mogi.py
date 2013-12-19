#!/usr/bin/env python

"""
Mogi's model of surface displacements from a point spherical source in an
elastic half space

References::
    [3] Mogi, K. Relations between the eruptions of various
    volcanoes and the deformations of the ground surfaces around them,
    Bull. Earthquake. Res. Inst., 36, 99-134, 1958.
"""

from numpy import array, pi

def mogi(data, x0, y0, z0, dV):
    """
    Computes surface displacements Ux, Uy, Uz in meters from a point spherical
    pressure source in an elastic half space [3].

    evaluate a single Mogi peak over a 2D (2 by N) numpy array of evalpts,
    where coeffs = (x0,y0,z0,dV)
    """
    dx = data[0,:] - x0
    dy = data[1,:] - y0
    dz = 0 - z0
    c = dV * 3. / 4. * pi
    # or equivalently c= (3/4) a^3 dP / rigidity
    # where a = sphere radius, dP = delta Pressure
    r2 = dx*dx + dy*dy + dz*dz
    C = c / pow(r2, 1.5)
    return array((C*dx,C*dy,C*dz))

