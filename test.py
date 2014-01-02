#!/usr/bin/env python

"""
Run nose tests for Bumps.

Usage:

./test.py
    - run all tests

./test.py --with-coverage
    - run all tests with coverage report
"""

import os, sys, subprocess
from glob import glob
import nose
import matplotlib
matplotlib.use('Agg')
print(matplotlib.__file__)

sys.dont_write_bytecode = True

sys.stderr = sys.stdout # Doctest doesn't see sys.stderr
#import numpy; numpy.seterr(all='raise')

# Check that we are running from the root.
path = os.path.abspath(os.getcwd())
assert os.path.exists(os.path.join(path, 'bumps', 'cli.py'))

# Make sure that we have a private version of mplconfig
mplconfig = os.path.join(os.getcwd(), '.mplconfig')
os.environ['MPLCONFIGDIR'] = mplconfig
os.putenv('MPLCONFIGDIR', mplconfig)
if not os.path.exists(mplconfig): os.mkdir(mplconfig)
import pylab; pylab.hold(False)

# Force a rebuild
print("-"*70)
print("Building bumps ...")
print("-"*70)
subprocess.call((sys.executable, "setup.py", "build"), shell=False)
print("-"*70)

# Add the build dir to the system path
from distutils.util import get_platform
platform = '.%s-%s'%(get_platform(),sys.version[:3])
build_path = os.path.abspath('build/lib'+platform)
sys.path.insert(0, build_path)

# Run the source tests with the system path augmented such that imports can
# be performed 'from bumps...".  By manipulating the system path in this way,
# we can test without having to install.
nose_args = ['-v', '--all-modules', '--cover-package=bumps',
             '-m(^_?test_|_test$|^test$)', '-I.*amqp_map.*']
if sys.version_info[0] >= 3:
    nose_args.append('-I.*gui.*')
nose_args += sys.argv[1:]  # allow coverage arguments
nose_args.append(build_path)
'''
nose_args += ['tests/bumps', 'bumps',
              #'doc/sphinx/guide'
             ]
'''
print("nosetests "+" ".join(nose_args))
if not nose.run(argv=nose_args): sys.exit(1)

# Run isolated tests in their own environment.  In this case we will have
# to set the PYTHONPATH environment variable before running since it is
# happening in a separate process.
if 'PYTHONPATH' in os.environ:
    PYTHONPATH = build_path + ":" + os.environ['PYTHONPATH']
else:
    PYTHONPATH = build_path
os.putenv('PYTHONPATH', PYTHONPATH)

## Run the command line version of Refl1D which should display help text.
#for p in ['bumps_cli.py']:
#    ret = os.system(" ".join( (sys.executable, os.path.join('bin', '%s'%p)) ))
#    if ret != 0: sys.exit()
