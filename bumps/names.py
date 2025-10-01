r"""
Exported names.

Usage:

    import bumps.names as bp
    ...
    bp.FitProblem(experiment)

In model definition scripts, rather than importing symbols one by one,
you can simply perform::

    from bumps.names import *

This is bad style for library and applications but convenient for
model scripts.

The following symbols are defined:

- *np* for the `numpy <http://docs.scipy.org/doc/numpy/reference>`_ array package
- *sys* for the python `sys <https://docs.python.org/2/library/sys.html>`_ module

Math functions:

- math constants: *inf*, *pi*, *e*
- exponentials: *exp*, *log*, *log10*, *sqrt*
- *degrees*, *radians*
- radians trig: *sin*, *cos*, *tan*, *arcsin*, *arccos*, *arctan*, *arctan2*
- degrees trig: *sind*, *cosd*, *tand*, *arcsind*, *arccosd*, *arctand*, *arctan2d*
- hyperbolic trig: *sinh*, *cosh*, *tanh*, *arcsinh*, *arccosh*, *arctanh*
- :mod:`pmath <bumps.pmath>` for *pmath.min()* and *pmath.max()*

Problem definition functions:

- :class:`Parameter <bumps.parameter.Parameter>` for defining parameters
- :class:`FreeVariables <bumps.parameter.FreeVariables>` for defining shared parameters
- :class:`Distribution <bumps.bounds.Distribution>` for indicating prior
    probability for a model parameter
- :class:`Curve <bumps.curve.Curve>` for defining models from functions
- :class:`PoissonCurve <bumps.curve.PoissonCurve>` for modelling data with Poisson uncertainty
- :class:`PDF <bumps.pdfwrapper.PDF>` for fitting a probability distribution directly
- :class:`FitProblem <bumps.fitproblem.FitProblem>` for defining the fit.

Jupyter notebook functions:

- Simple fitter: :func:`bumps.fitters.fit`, :func:`bumps.fitters.plot_convergence`
- Webview server: `bumps.webview.server.webserver.start_bumps`,
  `bumps.webview.server.webserver.display_bumps`
- MCMC save/load: :func:`bumps.fitters.load_session`,
  :func:`bumps.fitters.load_fit_from_session`,
  :func:`bumps.fitters.load_fit_from_export`
"""

# __all__ = [ 'sys', 'np', 'inf', 'pmath',
#    'Parameter', 'FreeVariables', 'Distribution', 'PDF', 'Curve', 'PoissonCurve',
#        'FitProblem' ]

import sys
import numpy as np

# === math functions ===
# Useful math functions, with support for delayed evaluation in parameter expressions.
# Because parameters have methods for sin, cos, etc., numpy functions on parameter expressions
# are automatically returned as parameter expressions, but when the arguments are simple
# floats, the floating point value is returned immediately.
from numpy import inf, pi, e
from numpy import exp, log, log10, sqrt
from numpy import degrees, radians
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2
from numpy import sinh, cosh, tanh, arcsinh, arccosh, arctanh

# These functions always return expressions, never immediate values
from .parameter import sind, cosd, tand, arcsind, arccosd, arctand, arctan2d

# Not importing min, max since "from bumps.names import *" shouldn't override builtins
# from .parameter import min, max
from . import pmath

# === problem definition ===
from .parameter import Parameter, FreeVariables
from .bounds import Distribution
from .pdfwrapper import PDF, VectorPDF, DirectProblem
from .curve import Curve, PoissonCurve
from .fitproblem import FitProblem, Fitness

# No longer used in bumps
# from .util import relative_import

# === jupyter notebook support ===
from .webview.server.webserver import start_bumps, display_bumps
from .fitproblem import load_problem
from .fitters import (
    fit,
    plot_convergence,
    show_results,
    load_session,
    load_fit_from_session,
    load_fit_from_export,
    get_fit_from_webview,
    help,
    save_fit,
)
from .webview.server.api import (
    set_problem,
    start_fit_thread,
    wait_for_fit_complete,
    export_fit,
)
