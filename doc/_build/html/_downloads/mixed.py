from refl1d.names import *
nickel = Material('Ni')

plateau = silicon(0,5) | nickel(1000,200) | air
valley = silicon(0,5) | air

T = numpy.linspace(0, 2, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = MixedExperiment(samples=[plateau,valley], probe=probe, ratio=[1,1])
M.simulate_data(5)

valley[0].interface = plateau[0].interface

plateau[0].interface.range(0,200)
plateau[1].interface.range(0,200)
plateau[1].thickness.range(200,1800)

M.ratio[1].range(0,5)

problem = FitProblem(M)
