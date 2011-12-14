#!/usr/bin/env python
import sys
import os

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

sys.path.insert(0,os.path.dirname(__file__))
import bumps

packages = find_packages(exclude=['amqp_map','models'])

if len(sys.argv) == 1:
    sys.argv.append('install')

def bumpsmodule():
    sources = [os.path.join('bumps','lib',f)
               for f in ("bumpsmodule.cc","methods.cc","convolve.c")]
    module = Extension('bumps.bumpsmodule', sources=sources)
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
        description='Bayesian uncertainty modeling of parametric systems',
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
        package_data = bumps.package_data(),
        scripts = ['bin/bumps_workerd','bin/bumps'],
        ext_modules = [bumpsmodule()],
        install_requires = ['httplib2'],
        )

# End of file
