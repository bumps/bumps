.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Bumps |version| is provided in a self-contained Python environment:

    - Windows installer: :slink:`%(winexe)s`
    - Apple installer: :slink:`%(macapp)s`
    - Source: :slink:`%(srczip)s`

The Windows installer is a self-extracting executable that unpacks 
to the location of your choosing, and
once unpacked you can double-click on the "bumps_webview.bat" file to
start the webview server and client.

The Apple .pkg installer unpacks to the Applications directory.  You can
start the application by double-clicking on the "bumps_webview.app".

Building from source
====================

Before building bumps, you will need to set up your python environment.
We depend on many external packages.  The program may work with
older versions of the package, and we will try to keep it compatible with
the latest versions.

Our base scientific python environment contains:

    - python >= 3.8
    - matplotlib
    - numpy
    - scipy

To run tests we use:

    - pytest

To build the HTML documentation we use:

    - sphinx
    - docutils
    - jinja2

The PDF documentation requires a working LaTeX installation.

You can install directly from PyPI using pip::

    pip install bumps

If this fails, then follow the instructions to install from the source
archive directly. Platform specific details for setting up your environment
are given below.

Installing Python
-----------------

You will need to install a python environment.  We recommend using
miniforge, which will install a conda system for you, using the 
"conda-forge" channel for packages (free).

* `miniforge <https://github.com/conda-forge/miniforge/releases/latest>`_

You can also install a python interpreter directly from the python website:

* `Python.org <https://www.python.org/downloads/>`_

To run the program use::

    python -m bumps.cli -h


Fast Stepper for DREAM on MPI
=============================

When running DREAM on larger clusters, we found a significant slowdown as the
number of processes increased.  This is due to Amdahl's law, where the run
time speedup is limited by the slowest serial portion of the code.  In our
case, the DE stepper and the bounds check.  Compiling this in C with OpenMP
allows us to scale to hundreds of nodes until the stepper again becomes a
bottleneck.

Automated build
---------------

To use the compiled DE stepper and bounds checks, use::

    python -m bumps.dream.build_compiled

This will compile the DLL in-place in the dream folder.

Manual build
------------

You can also directly build the compiled module:

To use the compiled DE stepper and bounds checks use::

    (cd bumps/dream && cc compiled.c -I ./random123/include/ -O2 -fopenmp -shared -lm -o _compiled.so -fPIC)

Note: clang doesn't support OpenMP, so on OS/X use::

    (cd bumps/dream && cc compiled.c -I ./random123/include/ -O2 -shared -lm -o _compiled.so -fPIC)

This only works when _compiled.so is in the bumps/dream directory.  If running
from a pip installed version, you will need to fetch the bumps repository::

    $ git clone https://github.com/bumps/bumps.git
    $ cd bumps

Compile as above, then find the bumps install path using the following::

    $ python -c "import bumps.dream; print(bumps.dream.__file__)"
    #dream/path/__init__.py

Copy the compiled module to the install (substituting #dream/path above)::

    $ cp bumps/dream/_compiled.so #dream/path

There is no provision for using _compiled.so in a frozen application.

Run with no more than 64 OMP threads.  If the number of processors is more
than 64, then use:

    OMP_NUM_THREADS=64 ./run.py ...

I don't know how OMP_NUM_THREADS behaves if it is larger than the number
of processors.


.. _docbuild:

Building Documentation
======================

Building the package documentation requires a working Sphinx installation and
a working LaTex installation.  Your latex distribution should include the
following packages:

    multirow, titlesec, framed, threeparttable, wrapfig,
    collection-fontsrecommended

You can then build the documentation as follows::

    (cd doc && make clean html pdf)

Windows users please note that this only works with a unix-like environment
such as *gitbash*, *msys* or *cygwin*.  There is a skeleton *make.bat* in
the directory that will work using the *cmd* console, but it doesn't yet
build PDF files.

You can see the result of the doc build by pointing your browser to::

    bumps/doc/_build/html/index.html
    bumps/doc/_build/latex/Bumps.pdf

ReStructured text format does not have a nice syntax for superscripts and
subscripts.  Units such as |g/cm^3| are entered using macros such as
\|g/cm^3| to hide the details.  The complete list of macros is available in

        doc/sphinx/rst_prolog

In addition to macros for units, we also define cdot, angstrom and degrees
unicode characters here.  The corresponding latex symbols are defined in
doc/sphinx/conf.py.


Building an installer (all platforms)
=====================================

To build a packed distribution for Windows, you will need to install
conda-pack in your base conda environment.  If you don't already have
a base interpreter, install that as well (e.g. on Windows) from
conda-forge::

    conda install -c conda-forge conda-pack bash

Then you can build the packed distribution using::

    bash extra/build_conda_packed.sh

This will create a packed distribution in the dist directory.