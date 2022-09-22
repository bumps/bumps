"""
Short script to create some test data which can be loaded into the
BumpsPeakFit test script
"""

from BumpsPeakFit.peaks_1d import Gaussian, Voigt, Background, plot
import numpy as np
import os


def make_simulated_data(x, peaks_list, noise=0.1):

    parts = []
    for part in peaks_list:
        model_part = part.pop("peak")
        parts.append(model_part(**part))

    total_theory = sum_parts_theory(parts, x)
    y = simulate_data(total_theory, noise)
    dy = y**0.5
    return y, dy, total_theory


def sum_parts_theory(parts, x):
    return sum(M(x) for M in parts)


def simulate_data(theory, noise=None):

    if noise is not None:
        if noise == 'data':
            return
        elif noise < 0:
            dy = -0.01*noise*theory
        else:
            dy = noise
        y = theory + np.random.randn(*theory.shape)*dy
        return y


if __name__ == "__main__":
    import pylab
    peaks = (dict(name="G1", A=500, xc=0.8, sig=0.1, peak=Gaussian),
             dict(name="G2", A=250, xc=0.1, sig=0.01, peak=Gaussian),
             dict(name="V1", A=100, xc=0.3, sig=0.05, gam=0.01, peak=Voigt),
             dict(name="B1", C=50, peak=Background))
    x = np.linspace(0, 1, 101)
    y, dy, theory = make_simulated_data(x, peaks, noise=10)
    plot(x, theory, y, dy)
    pylab.show()
    np.savetxt(f"{os.getcwd()}/test_data.dat", np.vstack((x, y, dy)).T)
