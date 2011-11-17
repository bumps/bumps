
import math
import __builtin__
from .parameter import function
__all__ = [
    'exp', 'log', 'log10', 'sqrt',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sind', 'cosd', 'tand', 'asind', 'acosd', 'atand', 'atan2d',
    'sinh','cosh','tanh',
    'degrees', 'radians',
    'sum', 'prod'
    ]


exp = function(math.exp)
log = function(math.log)
log10 = function(math.log10)
sqrt = function(math.sqrt)

sin = function(math.sin)
cos = function(math.cos)
tan = function(math.tan)
asin = function(math.asin)
acos = function(math.acos)
atan = function(math.atan)
atan2 = function(math.atan2)

degrees = function(math.degrees)
radians = function(math.radians)

def _cosd(v): return math.cos(math.radians(v))
def _sind(v): return math.sin(math.radians(v))
def _tand(v): return math.tan(math.radians(v))
def _acosd(v): return math.degrees(math.acos(v))
def _asind(v): return math.degrees(math.asin(v))
def _atand(v): return math.degrees(math.atan(v))
def _atan2d(dy,dx): return math.degrees(math.atan2(dy,dx))
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

def _prod(s): return reduce(lambda x,y: x*y, s, 1)
sum = function(__builtin__.sum)
prod = function(_prod)

# Define pickler for numpy ufuncs
#import copy_reg
#def udump(f): return f.__name__
#def uload(name): return getattr(numpy, name)
#copy_reg.pickle(numpy.ufunc, udump, uload)
