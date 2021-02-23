"""
Generate api docs for all modules in a package.

Drop this file in your sphinx doc directory, and change the constants at
the head of this file as appropriate.  Make sure this file is on the python
path and add the following to the end of conf.py::

    import genmods
    genmods.make()

OPTIONS are the options for gen_api_files().

PACKAGE is the dotted import name for the package.

MODULES is the list fo modules to include in table of contents order.

PACKAGE_TEMPLATE is the template for the api index file.

MODULE_TEMPLATE is the template for each api module.
"""

from __future__ import with_statement

PACKAGE_TEMPLATE=""".. Autogenerated by genmods.py -- DO NOT EDIT --

.. _%(package)s-index:

##############################################################################
Reference: %(package)s
##############################################################################

.. only:: html

   :Release: |version|
   :Date: |today|

.. toctree::
   :hidden:

   %(rsts)s

.. currentmodule:: %(package)s

.. autosummary::

   %(mods)s

"""

MODULE_TEMPLATE=""".. Autogenerated by genmods.py -- DO NOT EDIT --

******************************************************************************
%(prefix)s%(module)s - %(title)s
******************************************************************************

.. currentmodule:: %(package)s.%(module)s

.. autosummary::
   :nosignatures:

   %(members)s

.. automodule:: %(package)s.%(module)s
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

"""

# ===================== Documentation generator =====================

from os import makedirs
from os.path import exists, dirname, getmtime, join as joinpath, abspath
import inspect
import sys

def newer(file1, file2):
    return not exists(file1) or (getmtime(file1) < getmtime(file2))

def get_members(package, module):
    name = package+"."+module
    __import__(name)
    M = sys.modules[name]
    try:
        L = M.__all__
    except Exception:
        L = [s for s in sorted(dir(M))
             if inspect.getmodule(getattr(M,s)) == M and not s.startswith('_')]
    return L

def gen_api_docs(package, modules, dest='api', absolute=True, root=None):
    """
    Generate .rst files in *dir* from *modules* in *package*.

    *dest* is the root path to the constructed rst files.

    *absolute* is True if modules are listed as package.module in the table
    of contents.  Default is True.

    *root* is the path to the package source.  This may be different from
    the location of the package in the python path if the documentation is
    extracted from the build directory rather than the source directory.
    The source is used to check if the module definition has changed since
    the rst file was built.
    """
    #print(f"*** processing {package}")

    # Get path to package source
    if root is None:
        __import__(package)
        M = sys.modules[package]
        root = abspath(dirname(M.__file__))

    # Build path to documentation tree
    if not exists(dest):
        makedirs(dest)

    # Note: prefix used by MODULE_TEMPLATE
    prefix = package+"." if absolute else ""

    # Update any modules that are out of date.  Compiled modules
    # will always be updated since we only check for .py files.
    for module, title in modules:  # Note: title used by MODULE_TEMPLATE
        modfile = joinpath(root, module+'.py')
        rstfile = joinpath(dest, module+'.rst')
        if newer(rstfile, modfile):
            # Note: members is used by MODULE_TEMPLATE
            members = "\n    ".join(get_members(package, module))
            print(f"writing {rstfile} with current={package}.{module}")
            with open(rstfile, 'w') as f:
                f.write(MODULE_TEMPLATE%locals())

    # Update the table of contents, but only if the configuration
    # file containing the module list has changed.  For now, that
    # is the current file.
    api_index = joinpath(dest, 'index.rst')
    if newer(api_index, __file__):
        rsts = "\n   ".join(module+'.rst' for module, _ in modules)
        mods = "\n   ".join(prefix+module for module, _ in modules)
        #print("writing %s"%api_index)
        with open(api_index,'w') as f:
            f.write(PACKAGE_TEMPLATE%locals())


# bumps api

BUMPS_OPTIONS = {
    'absolute': False, # True if package.module in table of contents
    'dest': 'api', # Destination directory for the api docs
    'root': None, # Source directory for the package, or None for default
}

BUMPS_PACKAGE = 'bumps'

BUMPS_MODULES = [
    #('__init__', 'Top level namespace'),
    ('bounds', 'Parameter constraints'),
    ('bspline', 'B-Spline interpolation library'),
    #('caller_name', 'Identify the caller of the function'),
    #('_reduction','Low level calculations'),
    ('cheby', 'Freeform - Chebyshev'),
    ('cli', 'Command line interface'),
    ('curve', 'Model a fit function'),
    ('data', 'Data handling utilities'),
    ('errplot','Plot sample profile uncertainty'),
    ('fitproblem', 'Interface between models and fitters'),
    ('fitservice', 'Remote job plugin for fit jobs'),
    ('fitters', 'Wrappers for various optimization algorithms'),
    ('formatnum', 'Format numbers and uncertainties'),
    ('history', 'Optimizer evaluation trace'),
    ('initpop', 'Population initialization strategies'),
    ('lsqerror', 'Least squares eorror analysis'),
    ('mapper', 'Parallel processing implementations'),
    ('monitor', 'Monitor fit progress'),
    ('mono', 'Freeform - Monotonic Spline'),
    #('mpfit', 'Levenberg-Marquardt with bounds'),  # docstrings don't use rst
    ('names', 'External interface'),
    ('options', 'Command line options processor'),
    #('openmp_ext', 'distutils directive for compiling with OpenMP'),
    ('parameter', 'Optimization parameter definition'),
    ('partemp', 'Parallel tempering optimizer'),
    ('pdfwrapper', 'Model a probability density function'),
    ('plotutil', 'Plotting utilities'),
    ('plugin', 'Domain branding'),
    ('pmath', 'Parametric versions of standard functions'),
    ('pymcfit', 'Wrapper for pyMC models'),
    #('pytwalk', 'MCMC error analysis using T-Walk steps'),
    ('quasinewton', 'BFGS quasi-newton optimizer'),
    ('random_lines', 'Random lines and particle swarm optimizers'),
    ('simplex', 'Nelder-Mead simplex optimizer (amoeba)'),
    ('util', 'Miscellaneous functions'),
    #('vfs', 'Virtual file system for loading models from zip files'),
    ('wsolve', 'Weighted linear and polynomial solver with uncertainty'),
    ]

DREAM_OPTIONS = {
    'absolute': False, # True if package.module in table of contents
    'dest': 'dream', # Destination directory for the api docs
    'root': None, # Source directory for the package, or None for default
}

DREAM_PACKAGE = 'bumps.dream'

DREAM_MODULES = [
    #('__init__', 'Top level namespace'),
    ('acr', 'A C Rencher normal outlier test'),
    ('bounds', 'Bounds handling'),
    ('core', 'DREAM core'),
    #('compiled', 'Load shared object for compiled DE stepper'),
    ('corrplot', 'Correlation plots'),
    ('crossover', 'Adaptive crossover support'),
    ('diffev', 'Differential evolution MCMC stepper'),
    ('entropy', 'Entropy calculation'),
    ('exppow', 'Exponential power density parameter calculator'),
    ('formatnum', 'Format values and uncertainties nicely for printing'),
    ('gelman', 'R-statistic convergence test'),
    ('geweke', 'Geweke convergence test'),
    ('initpop', 'Population initialization routines'),
    ('ksmirnov', 'Kolmogorov-Smirnov test for MCMC convergence'),
    ('mahal', 'Mahalanobis distance calculator'),
    #('matlab', 'Environment for running matlab DREAM models in python'),
    ('metropolis', 'MCMC step acceptance test'),
    ('model', 'MCMC model types'),
    ('outliers', 'Chain outlier tests'),
    ('state', 'Sampling history for MCMC'),
    ('stats', 'Statistics helper functions'),
    ('tile', 'Split a rectangle into n panes'),
    ('util', 'Miscellaneous utilities'),
    ('varplot', 'Plot histograms for indiviual parameters'),
    ('views', 'MCMC plotting methods'),
    #('walk', 'Demo of different kinds of random walk'),
    ]

def make():
    gen_api_docs(BUMPS_PACKAGE, BUMPS_MODULES, **BUMPS_OPTIONS)
    gen_api_docs(DREAM_PACKAGE, DREAM_MODULES, **DREAM_OPTIONS)
