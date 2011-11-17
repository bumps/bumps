from refl1d.names import *

chrome = Material('Cr')
gold = Material('Au')
solvent = Material('H2O', density=1)


wrap = SLD(name="wrap", rho=0)

n1, n2, n3 = 3,9,3

tether = FreeLayer(below=gold, above=wrap, thickness=10,
                   z=numpy.linspace(0,1,n1+2)[1:-1],
                   rho=numpy.random.rand(n1),name="tether")
bilayer = FreeLayer(below=wrap, above=wrap, thickness=80,
                    z=numpy.linspace(0,1,n2+2)[1:-1],
                    rho=5*numpy.random.rand(n2)-1,name="bilayer")
tail = FreeLayer(below=wrap, above=solvent, thickness=10,
                   z=numpy.linspace(0,1,n3+2)[1:-1],
                   rho=numpy.random.rand(n3),name="tail")

sample = (silicon(0,5) | chrome(20,2) | gold(50,5)
          | tether | bilayer*10 | tail | solvent)

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475,
                     back_reflectivity=True)
M = Experiment(probe=probe, sample=sample, dA=5)
M.simulate_data(5)
problem = FitProblem(M)
