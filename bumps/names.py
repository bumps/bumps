r"""
Exported names.

In model definition scripts, rather than importing symbols one by one,
you can simply perform::

    from bumps.names import *

This is bad style for library and applications but convenient for
model scripts.

The following symbols are defined:

- *np* for the `numpy <http://docs.scipy.org/doc/numpy/reference>`_ array package
- *sys* for the python `sys <https://docs.python.org/2/library/sys.html>`_ module
- *inf* for infinity
- :mod:`pmath <bumps.pmath>` for parameter expressions like *2\*pmath.sin(M.theta)*
- :class:`Parameter <bumps.parameter.Parameter>` for defining parameters
- :class:`FreeVariables <bumps.parameter.FreeVariables>` for defining shared parameters
- :class:`Distribution <bumps.bounds.Distribution>` for indicating prior
    probability for a model parameter
- :class:`Curve <bumps.curve.Curve>` for defining models from functions
- :class:`PoissonCurve <bumps.curve.PoissonCurve>` for modelling data with Poisson uncertainty
- :class:`PDF <bumps.pdfwrapper.PDF>` for fitting a probability distribution directly
- :func:`FitProblem <bumps.fitproblem.FitProblem>` for defining the fit (see
    :class:`BaseFitProblem <bumps.fitproblem.BaseFitProblem>` or
    :class:`MultiFitProblem <bumps.fitproblem.MultiFitProblem>` for details,
    depending on whether you are fitting a single model or multiple models
    simultaneously).
"""

#__all__ = [ 'sys', 'np', 'inf', 'pmath',
#    'Parameter', 'FreeVariables', 'Distribution', 'PDF', 'Curve', 'PoissonCurve',
#        'FitProblem', 'MultiFitProblem' ]

import sys
import numpy as np
from numpy import inf, pi, e

from numpy import exp, log, log10, sqrt
from numpy import degrees, radians
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2
from numpy import sinh, cosh, tanh, arcsinh, arccosh, arctanh

from . import pmath
from .parameter import Parameter, FreeVariables
from .parameter import sind, cosd, tand, arcsind, arccosd, arctand, arctan2d
from .bounds import Distribution
from .pdfwrapper import PDF, VectorPDF, DirectProblem
from .curve import Curve, PoissonCurve
from .fitproblem import FitProblem, MultiFitProblem
from .fitters import fit
from .util import relative_import
