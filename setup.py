#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

# Use our own nose-based test harness
if sys.argv[1] == 'test':
    from subprocess import call
    sys.exit(call([sys.executable, 'test.py'] + sys.argv[2:]))

sys.dont_write_bytecode = True

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

sys.path.insert(0, os.path.dirname(__file__))
import bumps
from bumps.gui.resources import resources as gui_resources
from bumps.openmp_ext import openmp_ext

packages = find_packages(exclude=['amqp_map', 'fit_functions', 'jobqueue'])


def bumpsmodule():
    sources = [os.path.join('bumps', 'lib', f)
               for f in ("bumpsmodule.cc", "methods.cc", "convolve.c", "convolve_sampled.c")]
    module = Extension('bumps._reduction', sources=sources)
    return module

# TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')
# print bumps.package_data()

sys.dont_write_bytecode = False
dist = setup(
    name='bumps',
    version=bumps.__version__,
    author='Paul Kienzle',
    author_email='paul.kienzle@nist.gov',
    url='http://www.reflectometry.org/danse/software.html',
    description='Data fitting with bayesian uncertainty analysis',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    packages=packages,
    package_data=gui_resources.package_data(),
    scripts=['bin/bumps'],
    ext_modules=[bumpsmodule()],
    install_requires=['six', 'numdifftools'],
    #install_requires = ['httplib2', 'numdifftools'],
    cmdclass={'build_ext': openmp_ext(default=False)},
)

# End of file
