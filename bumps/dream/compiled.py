from os.path import join as joinpath, realpath, dirname, exists
from ctypes import CDLL

_dll_path = joinpath(realpath(dirname(__file__)), '_compiled.so')
dll = CDLL(_dll_path) if exists(_dll_path) else None
