# This program is in the public domain
# Author: Paul Kienzle
"""
Refl1D: Specular Reflectometry Modeling and Fitting Software

This package provides tools for modeling a variety of systems from
simple slabs to complex systems with a mixture models involving
smoothly varying layers.

X-ray and neutron and polarized neutron data can be loaded and the model
parameters adjusted to achieve the best fit.

A graphical interface allows direct manipulation of the model profiles.

See http://www.reflectometry.org/danse/reflectometry for online manuals.
"""

__version__ = "0.6.19"

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
