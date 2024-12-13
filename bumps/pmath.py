"""
Standard math functions for parameter expressions.
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
