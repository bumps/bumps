from refl1d.names import *

nickel = Material('Ni')
sample = silicon(0,5) | nickel(100,5) | air

instrument = SNS.Liquids()
M = instrument.simulate(sample,
                        T=[0.3,0.7,1.5,3],
                        slits=[0.06, 0.14, 0.3, 0.6],
                        uncertainty = 5,
                        )

problem = FitProblem(M)
