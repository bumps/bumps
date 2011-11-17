from refl1d.names import *
from refl1d.stajconvert import load_mlayer

# Load neutron model and data from staj file
M = load_mlayer("De2_VATR.staj")

# Set thickness/roughness fitting parameters to +/- 20 %
# Set SLD to +/- 5% for all but the incident medium and the substrate.
for L in M.sample[1:-1]:
    L.thickness.pmp(20)
    L.interface.pmp(20)
    L.material.rho.pmp(5)

# Let the substrate SLD vary by 2%
M.sample[0].material.rho.pmp(2)
M.sample[0].interface.range(0,20)
M.sample[1].interface.range(0,20)

problem = FitProblem(M)
problem.name = "Desorption 2"
