"""
Standard math functions for parameter expressions.

These functions delay evaluation of the expression until the value is requested,
thus allowing the value to change when the parameter is updated. Most numpy functions
will do this automatically, so nothing special is needed.

- *degrees*, *radians*, *min*, *max*
- powers: *exp*, *log*, *log10*, *sqrt*
- radians trig: *sin*, *cos*, *tan*, *arcsin*, *arccos*, *arctan*, *arctan2*
- degrees trig: *sind*, *cosd*, *tand*, *arcsind*, *arccosd*, *arctand*, *arctan2d*
- hyperbolic trig: *sinh*, *cosh*, *tanh*, *arcsinh*, *arccosh*, *arctanh*

arc-function aliases (*asin*, *asinh*, ...) are added to pmath for convenience.
"""

__all__ = []

# Note: the symbols in this module are defined dynamically by parameter.py

# TODO: need a pmath test
# Something like:
#
# import numpy as np
# from bumps.parameter import Parameter, function
#
# # Define a plugin function
# @function
# def square(x): return x*x
#
# # Check all pmath symbols are imported, including plugin functions. This
# # must be done after the plugin functions are registered.
# from bumps.pmath import *
# a = Parameter(value=0.2)
# assert asind(a).value == np.degrees(np.arcsin(a.value))
# assert square(a).value = a.value**2
