"""
Demonstration of alignment compensation using a change in substrate SLD.

Using::

      Qc = 4 pi/lambda sin(Tc)
    Qc^2 = 16 pi rho

we can compute Tc from rho using::

      Tc = arcsin( lambda/(4 pi) sqrt(16 pi rho) )

With Tc we can compute the compensated rho_adj which matches the critical
edge of the true rho measured with an alignment offset T_offset::

    Qc_offset = 4 pi/lambda sin(Tc - T_offset)
      rho_adj = Qc_offset^2 / (16 pi)

We can then plot R(Q_offset, rho) against R(Q, rho_adj) and the critical
angles will be identical, but the curves will differ at high Q.
"""

from pylab import *
from refl1d.reflectivity import reflectivity

rho=2.07 # silicon rho for neutrons
L = 5  # Wavelength 5 angstroms
T = linspace(0,1,100)  # Measured angles
T_offset = 0.005       # alignment offset

# Find location of critical edge and computed the adjusted rho
Tc = degrees(arcsin( L/4./pi * sqrt(16e-6*pi*rho)))
rho_adj = (4*pi/L * sin( radians(Tc-T_offset)))**2/16e-6/pi

# Q where the measurement is supposed to be and
# Qoffset where the measurement actually is
Q = 4*pi/L * sin(radians(T))
Q_offset = 4*pi/L * sin(radians(T+T_offset))

# Reflectivity measurement returns (Q, R(Qoffset))
# Adjusted measurement returns (Q, Radj(Q))
R = reflectivity(kz=-Q_offset/2, depth=[0,0], rho=[rho,0])
Radj = reflectivity(kz=-Q/2, depth=[0,0], rho=[rho_adj,0])

# Plot log reflectivity and relative difference using a shared Q axis.
h = subplot(211)
semilogy(Q,R,'-',label=r'$R$')
semilogy(Q,Radj,'-',label=r'$R_{\rm adj}$', hold=True)
legend()
ylabel(r'$R$')
subplot(212, sharex=h)
plot(Q, (R-Radj)/R, 'o')
xlabel(r'$Q$ ($A^{-1}$)')
ylabel(r'$(R-R_{\rm adj})/R$')
show()
