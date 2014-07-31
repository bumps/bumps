#!/usr/bin/env python
"""
Build and run bumps.

Usage:

./run.py [bumps cli args]
"""

import os
import sys


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

from contextlib import contextmanager


@contextmanager
def cd(path):
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)


def prepare():
    # Make sure that we have a private version of mplconfig
    mplconfig = os.path.join(os.getcwd(), '.mplconfig')
    os.environ['MPLCONFIGDIR'] = mplconfig
    if not os.path.exists(mplconfig):
        os.mkdir(mplconfig)
    #import matplotlib
    # matplotlib.use('Agg')
    # print matplotlib.__file__
    #import pylab; pylab.hold(False)

    from distutils.util import get_platform
    platform = '.%s-%s' % (get_platform(), sys.version[:3])

    # To avoid cluttering the source tree with .pyc or __pycache__ files, you
    # can suppress the bytecode generation when running in place. Unfortunately
    # this is a pretty big performance hit on Windows, so we are going to
    # suppress this behaviour and rely on .gitignore instead
    #sys.dont_write_bytecode = True

    #import numpy as np; np.seterr(all='raise')
    root = os.path.abspath(os.path.dirname(__file__))

    # Force a rebuild
    import subprocess
    with cd(root), open(os.devnull, 'w') as devnull:
        subprocess.call(
            (sys.executable, "setup.py", "build"), shell=False, stdout=devnull)

    # Add the build dir to the system path
    build_path = os.path.join(root, 'build', 'lib' + platform)
    addpath(build_path)

    # Make sample data and models available
    os.environ['BUMPS_DATA'] = os.path.join(root, 'bumps', 'gui', 'resources')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    prepare()
    import bumps.cli
    bumps.cli.main()
