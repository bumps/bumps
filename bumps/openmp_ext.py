"""
Compile openmp extensions with distutils.

:func:`openmp_ext` returns a replacement *build_ext* command that adds
OpenMP command line parameters to the C compiler for your system.  Use
this is setup.py for any modules that need to be compiled for OpenMP.
"""
import sys
from distutils.command.build_ext import build_ext

__all__ = ['openmp_ext']


def openmp_ext(default=True):
    """
    Enable openmp.

    Add the following to setup.py::

        setup(..., cmdclass={'build_ext': openmp_build_ext()}, ...)

    Enable openmp using "--with-openmp" as a setup parameter, or disable
    it using "--without-openmp".  If no option is specfied, the developer
    *default* value will be used.

    On OS X you will need to specify an openmp compiler::

        CC=openmp-cc CXX=openmp-c++ python setup.py --with-openmp

    Note: when using openmp, you should not use multiprocessing parallelism
    otherwise python will hang.  This is a known bug in the current version
    of python and gcc.  If your modeling code is compiled with openmp, you
    can set OMP_NUM_THREADS=1 in the environment to suppress openmp threading
    when you are running --parallel fits in batch.
    """
    with_openmp = default
    if '--with-openmp' in sys.argv:
        with_openmp = True
        sys.argv.remove('--with-openmp')
    elif '--without-openmp' in sys.argv:
        with_openmp = False
        sys.argv.remove('--without-openmp')

    if not with_openmp:
        return build_ext

    compile_opts = {
        'msvc': ['/openmp'],
        'mingw32': ['-fopenmp'],
        'unix': ['-fopenmp'],
    }
    link_opts = {
        'mingw32': ['-fopenmp'],
        'unix': ['-lgomp'],
    }

    class OpenMPExt(build_ext):

        def build_extensions(self):
            c = self.compiler.compiler_type
            if c in compile_opts:
                for e in self.extensions:
                    e.extra_compile_args = compile_opts[c]
            if c in link_opts:
                for e in self.extensions:
                    e.extra_link_args = link_opts[c]
            build_ext.build_extensions(self)

    return OpenMPExt
