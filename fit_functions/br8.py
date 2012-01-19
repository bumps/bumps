#!/usr/bin/env python

"""
Bevington & Robinson's model of dual exponential decay

References::
    [5] Bevington & Robinson (1992).
    Data Reduction and Error Analysis for the Physical Sciences,
    Second Edition, McGraw-Hill, Inc., New York.
"""

from numpy import exp, sqrt, vstack, array, asarray

def dual_exponential(t, A, B, C, tauA, tauB):
    """
    Computes dual exponential decay.

        y = A exp(-t/tauA) + B exp(-t/tauB) + C
    """
    t = asarray(t)
    return C + A*exp(-t/tauA) + B*exp(-t/tauB)

# data from Chapter 8 of [5].
data = array([[15, 775], [30, 479], [45, 380], [60, 302],
[75, 185], [90, 157], [105,137], [120, 119], [135, 110],
[150, 89], [165, 74], [180, 61], [195, 66], [210, 68],
[225, 48], [240, 54], [255, 51], [270, 46], [285, 55],
[300, 29], [315, 28], [330, 37], [345, 49], [360, 26],
[375, 35], [390, 29], [405, 31], [420, 24], [435, 25],
[450, 35], [465, 24], [480, 30], [495, 26], [510, 28],
[525, 21], [540, 18], [555, 20], [570, 27], [585, 17],
[600, 17], [615, 14], [630, 17], [645, 24], [660, 11],
[675, 22], [690, 17], [705, 12], [720, 10], [735, 13],
[750, 16], [765, 9], [780, 9], [795, 14], [810, 21],
[825, 17], [840, 13], [855, 12], [870, 18], [885, 10]])

# Set uncertainty to sqrt(counts)
data = { 'x': data[0], 'y': data[1], 'dy': sqrt(data[1]) }

#coeff = {'A': 1, 'B': 1, 'C': 1, 'tauA': 1, 'tauB': 1}
