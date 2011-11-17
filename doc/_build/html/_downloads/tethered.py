from refl1d.names import *
from copy import copy

# === Materials ===
SiOx = SLD(name="SiOx",rho=3.47)
D_toluene = SLD(name="D-toluene",rho=5.66)
D_initiator = SLD(name="D-initiator",rho=1.5)
D_polystyrene = SLD(name="D-PS",rho=6.2)
H_toluene = SLD(name="H-toluene",rho=0.94)
H_initiator = SLD(name="H-initiator",rho=0)

# === Sample ===
# Deuterated sample
D_brush = PolymerBrush(polymer=D_polystyrene, solvent=D_toluene,
                       base_vf=70, base=120, length=80, power=2,
                       sigma=10)

D = (silicon(0,5) | SiOx(100,5) | D_initiator(100,20) | D_brush(400,0)
     | D_toluene)

# Undeuterated sample is a copy of the deuterated sample
H_brush = copy(D_brush)       # Share tethered polymer parameters...
H_brush.solvent = H_toluene   # ... but use different solvent
H = silicon | SiOx | H_initiator | H_brush | H_toluene

for i,_ in enumerate(D):
    H[i].thickness = D[i].thickness
    H[i].interface = D[i].interface

# === Fit parameters ===
for i in 0, 1, 2:
    D[i].interface.range(0,100)
D[1].thickness.range(0,200)
D[2].thickness.range(0,200)
D_polystyrene.rho.range(6.2,6.5)
SiOx.rho.range(2.07,4.16) # Si to SiO2
D_toluene.rho.pmp(5)
D_initiator.rho.range(0,1.5)
D_brush.base_vf.range(50,80)
D_brush.base.range(0,200)
D_brush.length.range(0,500)
D_brush.power.range(0,5)
D_brush.sigma.range(0,20)

# Undeuterated system adds two extra parameters
H_toluene.rho.pmp(5)
H_initiator.rho.range(-0.5,0.5)

# === Data files ===
instrument = NCNR.NG7(Qlo=0.005, slits_at_Qlo=0.075)
D_probe = instrument.load('10ndt001.refl', back_reflectivity=True)
H_probe = instrument.load('10nht001.refl', back_reflectivity=True)

# === Problem definition ===
D_model = Experiment(sample=D, probe=D_probe, dz=0.5, dA=1)
H_model = Experiment(sample=H, probe=H_probe, dz=0.5, dA=1)
models = H_model, D_model

problem = MultiFitProblem(models=models)
problem.name = "tethered"

