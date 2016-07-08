from os.path import join as joinpath, realpath, dirname, exists
from ctypes import CDLL, c_int, c_double, c_void_p, c_char_p, byref

_dll_path = joinpath(realpath(dirname(__file__)), '_compiled.so')
dll = CDLL(_dll_path) if exists(_dll_path) else None

