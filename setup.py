#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

# Use our own nose-based test harness
if sys.argv[1] == 'test':
    from subprocess import call
    sys.exit(call([sys.executable, 'test.py'] + sys.argv[2:]))

#sys.dont_write_bytecode = True

from setuptools import setup, find_packages

sys.path.insert(0, os.path.dirname(__file__))
import bumps
from bumps.gui.resources import resources as gui_resources

packages = find_packages(exclude=['amqp_map', 'fit_functions', 'jobqueue'])


# TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')
# print bumps.package_data()

scripts = ['bin/bumps.bat'] if os.name == 'nt' else ['bin/bumps']
#sys.dont_write_bytecode = False
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
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    packages=packages,
    package_data=gui_resources.package_data(),
    scripts=scripts,
    install_requires=['six'],
    #install_requires = ['httplib2'],
)

# End of file
