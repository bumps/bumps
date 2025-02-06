# This program is in the public domain
# Author: Paul Kienzle
"""
Bumps: curve fitter with uncertainty estimation

This package provides tools for modeling parametric systems in a Bayesian
context, with routines for finding the maximum likelihood and the
posterior probability density function.

A graphical interface allows direct manipulation of the model parameters.

See https://bumps.readthedocs.io for online manuals.
"""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"

__schema_version__ = "1"


def data_files():
    """
    Return the data files associated with the package for setup_py2exe.py.

    The format is a list of (directory, [files...]) pairs which can be
    used directly in the py2exe setup script as::

        setup(...,
              data_files=data_files(),
              ...)
    """
    from .gui.utilities import data_files

    return data_files()


def package_data():
    """
    Return the data files associated with the package for setup.py.

    The format is a dictionary of {'fully.qualified.module', [files...]}
    used directly in the setup script as::

        setup(...,
              package_data=package_data(),
              ...)
    """
    from .gui.utilities import package_data

    return package_data()
