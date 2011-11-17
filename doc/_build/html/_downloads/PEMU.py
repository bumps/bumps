from refl1d.names import *

# == Sample definition ==
chrome = Material('Cr')
gold = Material('Au')

PDADMA_dPSS = SLD(name ='PDADMA dPSS',rho = 2.77)
PDADMA_PSS = SLD(name = 'PDADMA PSS',rho = 1.15)

bilayer = PDADMA_PSS(178,10) | PDADMA_dPSS(44.3,10)

sample = (silicon(0,5) | chrome(30,3) | gold(120,5)
          | (bilayer)*4 | PDADMA_PSS(178,10) | PDADMA_dPSS(44.3,10) | air)

# == Fit parameters ==
PDADMA_dPSS.rho.range(1.15,2.77)
PDADMA_PSS.rho.range(1.15,2.77)

sample[3][0].interface.range(5,45)
sample[3][1].interface.range(5,45)

sample[3].interface.range(5,45)

sample[4].interface.range(5,45)
sample[5].interface.range(5,45)


# == Data ==
T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)

M = Experiment(probe=probe, sample=sample)
M.simulate_data(5)

problem = FitProblem(M)
