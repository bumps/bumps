#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

if sys.argv[1] == 'test':
    sys.exit(os.system(" ".join((sys.executable, 'test.py'))))

sys.dont_write_bytecode = True

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

sys.path.insert(0,os.path.dirname(__file__))
import bumps
from bumps.gui.resources import resources as gui_resources

packages = find_packages(exclude=['amqp_map','fit_functions','jobqueue'])

def bumpsmodule():
    sources = [os.path.join('bumps','lib',f)
               for f in ("bumpsmodule.cc","methods.cc","convolve.c")]
    module = Extension('bumps._reduction', sources=sources)
    return module

#TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')
#print bumps.package_data()

dist = setup(
        name = 'bumps',
        version = bumps.__version__,
        author='Paul Kienzle',
        author_email='pkienzle@nist.gov',
        url='http://www.reflectometry.org/danse/model1d.html',
        description='Data fitting and Bayesian uncertainty modeling for inverse problems',
        long_description=open('README.txt').read(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: Public Domain',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        packages = packages,
        package_data = gui_resources.package_data(),
        scripts = ['bin/bumps'],
        ext_modules = [bumpsmodule()],
        #install_requires = ['httplib2'],
        )

# End of file
