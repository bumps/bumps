"""
Standard math functions for parameter expressions.
"""
import math

from six.moves import reduce, builtins

from .parameter import function
__all__ = [
    'exp', 'log', 'log10', 'sqrt',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand', 'atan2d',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'degrees', 'radians',
    'sum', 'prod',
]

def _cosd(v):
    """Return the cosine of x (measured in in degrees)."""
    return math.cos(math.radians(v))


def _sind(v):
    """Return the sine of x (measured in in degrees)."""
    return math.sin(math.radians(v))


def _tand(v):
    """Return the tangent of x (measured in in degrees)."""
    return math.tan(math.radians(v))


def _acosd(v):
    """Return the arc cosine (measured in in degrees) of x."""
    return math.degrees(math.acos(v))


def _asind(v):
    """Return the arc sine (measured in in degrees) of x."""
    return math.degrees(math.asin(v))


def _atand(v):
    """Return the arc tangent (measured in in degrees) of x."""
    return math.degrees(math.atan(v))


def _atan2d(dy, dx):
    """Return the arc tangent (measured in in degrees) of y/x.
    Unlike atan(y/x), the signs of both x and y are considered."""
    return math.degrees(math.atan2(dy, dx))

def _prod(s):
    """Return the product of a sequence of numbers."""
    return reduce(lambda x, y: x * y, s, 1)

exp = function(math.exp)
log = function(math.log)
log10 = function(math.log10)
sqrt = function(math.sqrt)

degrees = function(math.degrees)
radians = function(math.radians)

sin = function(math.sin)
cos = function(math.cos)
tan = function(math.tan)
asin = function(math.asin)
acos = function(math.acos)
atan = function(math.atan)
atan2 = function(math.atan2)

sind = function(_sind)
cosd = function(_cosd)
tand = function(_tand)
asind = function(_asind)
acosd = function(_acosd)
atand = function(_atand)
atan2d = function(_atan2d)

sinh = function(math.sinh)
cosh = function(math.cosh)
tanh = function(math.tanh)
asinh = function(math.asinh)
acosh = function(math.acosh)
atanh = function(math.atanh)

sum = function(builtins.sum)
prod = function(_prod)

min = function(builtins.min)
max = function(builtins.max)

# Define pickler for numpy ufuncs
#import copy_reg
#def udump(f): return f.__name__
#def uload(name): return getattr(np, name)
#copy_reg.pickle(np.ufunc, udump, uload)
