"""
1d peak fitting example script
"""
import os
from bumps.names import *
from bumps.cli import install_plugin


sys.path.append('.')
from BumpsPeakFit.peaks_1d import Peaks, Gaussian, Background, Voigt
from BumpsPeakFit.model_builder import ModelBuilder1d
from BumpsPeakFit import fitplugin

install_plugin(fitplugin)

path = os.path.curdir
filename = r'test_data.dat'

x, y, e = np.loadtxt(f"{path}/{filename}").T

G1 = Gaussian(name="G1", A=500, xc=0.8, sig=0.1)
G2 = Gaussian(name="G2", A=250, xc=0.1, sig=0.01)
V1 = Voigt(name="V1", A=100, xc=0.3, sig=0.05, gam=0.01)
B1 = Background(name="B1", C=50)

sample = (G1, G2, V1, B1)

M = Peaks(parts=sample, X=x, data=y, err=e)

# Crude universal setting of all fit parameters
# by using pmp we are using plus minus percent of the initial value.
# we can include 1 or 2 values - e.g. pmp(50) will give plus, minus 50% of the value,
# while pmp(-100, 50) will give the minus range to be 100% of the value 
# and the plus to be 50% of the value
# This will set the fit ranges to be the same pmp for all peaks.

for par in M.parts:
    if isinstance(par, Voigt):
        # we can do something more sensible than using pmp
        par.A.range(0, np.max(M.data))
        par.xc.range(np.min(M.X), np.max(M.X))
        par.sig.pmp(100)
        par.gam.pmp(100)
    elif isinstance(par, Gaussian):
        # we can do something more sensible than using pmp
        par.A.range(0, np.max(M.data))
        par.xc.range(np.min(M.X), np.max(M.X))
        par.sig.pmp(100)
    elif isinstance(par, Background):
        par.C.range(0, np.max(M.data))

problem = FitProblem(M)
