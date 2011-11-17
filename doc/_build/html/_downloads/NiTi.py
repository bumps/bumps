from refl1d.names import *

nickel = Material('Ni')
titanium = Material('Ti')

# Superlattice description
bilayer = nickel(50,5) | titanium(50,5)
sample = silicon(0,5) | bilayer*10 | air

# Fitting parameters
bilayer[0].thickness.pmp(100)
bilayer[1].thickness.pmp(100)

bilayer[0].interface.range(0,30)
bilayer[1].interface.range(0,30)
sample[0].interface.range(0,30)
sample[1].interface.range(0,30)

sample[1].repeat.range(5,15)

T = numpy.linspace(0, 5, 100)
probe = XrayProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)
problem = FitProblem(M)
