#!/usr/bin/env python

"""
Run nose tests for Refl1D.

Usage:

./test.py
    - run all tests

./test.py --with-coverage
    - run all tests with coverage report
"""

import os, sys, subprocess
import nose
import matplotlib
matplotlib.use('Agg')
print matplotlib.__file__

sys.stderr = sys.stdout # Doctest doesn't see sys.stderr
#import numpy; numpy.seterr(all='raise')

# Check that we are running from the root.
path = os.path.abspath(os.getcwd())
assert os.path.exists(os.path.join(path, 'refl1d', 'model.py'))

# Make sure that we have a private version of mplconfig
mplconfig = os.path.join(os.getcwd(), '.mplconfig')
os.environ['MPLCONFIGDIR'] = mplconfig
os.putenv('MPLCONFIGDIR', mplconfig)
if not os.path.exists(mplconfig): os.mkdir(mplconfig)
import pylab; pylab.hold(False)

# Build reflmodule.pyd if it has not already been built in the source tree.
if not os.path.exists(os.path.join(path, 'refl1d', 'reflmodule.pyd')):
    print "-"*70
    print "Building reflmodule.pyd ..."
    print "-"*70
    if os.name == 'nt': flag = False
    else:               flag = True
    subprocess.call("python setup.py build_ext --inplace", shell=flag)
    print "-"*70

# Run the source tests with the system path augmented such that imports can
# be performed 'from refl1d..." and 'from dream...'.  By manipulating the
# system path in this way, we can test without having to build and install.
# We are adding doc/sphinx to the path because the periodic table extension
# doctests need to be able to find the example extensions.
sys.path.insert(0, path)
sys.path.insert(1, os.path.join(path, 'dream'))
nose_args = [__file__, '-v', '--with-doctest', '--doctest-extension=.rst',
             '--cover-package=refl1d']
nose_args += sys.argv[1:]  # allow coverage arguments
nose_args += [os.path.join('tests', 'refl1d'), 'refl1d',
              #'doc/sphinx/guide'
             ]
'''
nose_args += ['tests/refl1d', 'refl1d',
              #'doc/sphinx/guide'
             ]
'''
if not nose.run(argv=nose_args): sys.exit(1)

# Run isolated tests in their own environment.  In this case we will have
# to set the PYTHONPATH environment variable before running since it is
# happening in a separate process.
if 'PYTHONPATH' in os.environ:
    PYTHONPATH = path + ":" + os.environ['PYTHONPATH']
else:
    PYTHONPATH = path
os.putenv('PYTHONPATH', PYTHONPATH)

## Run the command line version of Refl1D which should display help text.
#for p in ['refl1d_cli.py']:
#    ret = os.system(" ".join( (sys.executable, os.path.join('bin', '%s'%p)) ))
#    if ret != 0: sys.exit()
