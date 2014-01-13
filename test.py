#!/usr/bin/env python

"""
Run tests for bumps.

Usage:

./test.py
    - run all tests

./test.py --with-coverage
    - run all tests with coverage report
"""

import os, sys, subprocess
from glob import glob
import nose

from distutils.util import get_platform
platform = '.%s-%s'%(get_platform(),sys.version[:3])

# Make sure that we have a private version of mplconfig
mplconfig = os.path.join(os.getcwd(), '.mplconfig')
os.environ['MPLCONFIGDIR'] = mplconfig
if not os.path.exists(mplconfig): os.mkdir(mplconfig)
import matplotlib
matplotlib.use('Agg')
#print(matplotlib.__file__)
import pylab; pylab.hold(False)

def addpath(path):
    """
    Add a directory to the python path environment, and to the PYTHONPATH
    environment variable for subprocesses.
    """
    path = os.path.abspath(path)
    if 'PYTHONPATH' in os.environ:
        PYTHONPATH = path + os.pathsep + os.environ['PYTHONPATH']
    else:
        PYTHONPATH = path
    os.environ['PYTHONPATH'] = PYTHONPATH
    sys.path.insert(0, path)

sys.dont_write_bytecode = True

sys.stderr = sys.stdout # Doctest doesn't see sys.stderr
#import numpy; numpy.seterr(all='raise')

# Check that we are running from the root.
root = os.path.abspath(os.getcwd())
assert os.path.exists(os.path.join(root, 'bumps', 'cli.py')), "Not in bumps root"

# Force a rebuild
print("-"*70)
print("Building bumps ...")
print("-"*70)
subprocess.call((sys.executable, "setup.py", "build"), shell=False)
print("-"*70)

# Add the build dir to the system path
build_path = os.path.join('build','lib'+platform)
addpath(build_path)

# Make sample data and models available
os.environ['BUMPS_DATA'] = os.path.join(root,'doc','_examples')

# Set the nosetest args
nose_args = ['-v', '--all-modules',
             '-m(^_?test_|_test$|^test$)',
             '--with-doctest', '--doctest-extension=.rst',
             '--doctest-options=+ELLIPSIS,+NORMALIZE_WHITESPACE',
             '--cover-package=bumps',
             '-e.*amqp_map.*',
             ]

# exclude gui subdirectory if wx is not available
try: import wx
except ImportError: nose_args.append('-egui')

nose_args += sys.argv[1:]  # allow coverage arguments

# Add targets
nose_args += [build_path]
nose_args += glob('doc/g*/*.rst')
nose_args += glob('doc/_examples/*/*.rst')

print("nosetests "+" ".join(nose_args))
if not nose.run(argv=nose_args): sys.exit(1)

## Run the command line version of bumps which should display help text.
#for p in ['bin/bumps']:
#    ret = subprocess.call((sys.executable, p), shell=False)
#    if ret != 0: sys.exit()
