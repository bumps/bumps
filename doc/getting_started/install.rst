.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Bumps |version| is provided in a self-contained Python environment:

    - Windows installer: :slink:`%(winexe)s`
    - Apple installer: :slink:`%(macapp)s`

The Windows installer is a self-extracting executable that unpacks 
to the location of your choosing.
Once unpacked you can double-click on the "bumps_webview.bat" file to
start the webview server and client.

The Apple .pkg installer unpacks to the Applications directory.  You can
start the application by double-clicking on the "bumps_webview.app".

For linux you will need to install from source.

Install from source
===================

Bumps is available on `PyPI <https://pypi.org/project/bumps/>`_ so you can
install directly into a python environment with pip. Is is also available
as a package on Debian/Ubuntu and as source from
`Github <https://github.com/bumps/bumps>`_.

To install the Debian/Ubuntu package [pre-1.0 as of this writing]::

    sudo apt install python3-bumps

Otherwise, create a python environment to get the latest release.
We recommend using
`miniforge <https://github.com/conda-forge/miniforge/releases/latest>`_.
This installs a conda system with the "conda-forge" channel for packages::

    conda create -n bumps python matplotlib numpy scipy dill h5py scikit-learn
    conda activate bumps
    # optional dependencies when using webview
    conda install aiohttp blinker plotly mpld3 python-socketio
    pip install bumps[webview]

You can instead use python available on your operating system if it is new
enough (Python 3.10 as of this writing). Again, recommended practice
is to use an isolated python environment. Instructions for Debian/Ubuntu are::

    sudo apt install python3 python3-venv
    python3 -m venv bumps
    . bumps/bin/activate
    pip install bumps[webview]

Python is also available directly from
`Python.org <https://www.python.org/downloads/>`_.

To run the program use::

    # command line interface
    bumps -h
    # graphical user interface
    # point your browser to the URL printed when you run the command
    bumps-webview

TODO: instructions for jupyter and slurm

Fast Stepper for DREAM on MPI
=============================

When running DREAM on larger clusters, we found a significant slowdown as the
number of processes increased.  This is due to Amdahl's law, where the run
time speedup is limited by the slowest serial portion of the code.  In our
case, the DE stepper and the bounds check.  Compiling this in C with OpenMP
allows us to scale to hundreds of nodes until the stepper again becomes a
bottleneck.

The following command should build the fast stepper binary module::

    python -m bumps.dream.build_compiled

If you have installed from source, you must first check out the random123 library::

    git clone --branch v1.14.0 https://github.com/DEShawResearch/random123.git bumps/dream/random123
    python -m bumps.dream.build_compiled

If this fails you can try running the compiler directly. First find the path
to the bumps directory::

    $ python -c "import bumps.dream; print(bumps.dream.__file__)"
    #path/to/bumps/dream/__init__.py

Change into that directory and compile the module::

    (cd path/to/bumps/dream && cc compiled.c -I ./random123/include/ -O2 -DMAX_THREADS=64 -fopenmp -shared -lm -o _compiled.so -fPIC)

Note: clang doesn't support OpenMP, so on macOS use::

    (cd path/to/bumps/dream && cc compiled.c -I ./random123/include/ -O2 -DMAX_THREADS=64 -shared -lm -o _compiled.so -fPIC)

Make sure MAX_THREADS is at least the number of processors on your system
otherwise you will need to set :code:`OMP_NUM_THREADS=MAX_THREADS` in your
environment before running bumps.

.. _docbuild:

Building Documentation
======================

To build the HTML documentation we use:

    - sphinx
    - docutils
    - jinja2

The PDF documentation requires a working LaTeX installation.

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