.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Bumps |version| is provided as a Windows installer or as source:

    - Windows installer: :slink:`%(winexe)s`
    - Apple installer: :slink:`%(macapp)s`
    - Source: :slink:`%(srczip)s`

The Windows installer walks through the steps of setting the program up
to run on your machine and provides the sample data to be used in the
tutorial.

Building from source
====================

Before building bumps, you will need to set up your python environment.
We depend on many external packages.  The versions listed below are a
snapshot of a configuration that we are using.  The program may work with
older versions of the package, and we will try to keep it compatible with
the latest versions.

Our base scientific python environment contains:

    - python 2.7 (also tested on 2.6 and 3.5)
    - matplotlib 1.4.3
    - numpy 1.9.2
    - scipy 0.14.0
    - wxPython 3.0.0.0
    - setuptools 20.1.1

To run tests we use:

    - nose 1.3.0

To build the HTML documentation we use:

    - sphinx 1.3.1
    - docutils 0.12
    - jinja2 2.8

The PDF documentation requires a working LaTeX installation.

You can install directly from PyPI using pip::

    pip install bumps

If this fails, then follow the instructions to install from the source
archive directly. Platform specific details for setting up your environment
are given below.

Windows
-------

There are a number of python environments for windows, including:

* `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
* `Canopy <https://www.enthought.com/products/canopy/>`_
* `Python(X,Y) <http://code.google.com/p/pythonxy/>`_
* `WinPython <http://winpython.sourceforge.net/>`_

You can also build your environment from the individually distributed
python packages.

You may want a C compiler to speed up parts of bumps. Microsoft Visual C++
for Python 2.7 is one option.  Once it is installed, you will need to
enable the compiler using vcvarsall 64.

Alternatively, your python environment may supply the MinGW C/C++ compiler,
but fail to set it as the default compiler.  To do so you will need to create
distutils configuration file in the python lib directory (usually
*C:\Python27\Lib\distutils\distutils.cfg*) with the following content::

    [build]
    compiler=mingw32

Next start a Windows command prompt in the directory containing the source.
This will be a command like the following::

    cd "C:\Documents and Settings\<username>\My Documents\bumps-src"

Now type the command to build and install::

    python setup.py install
    python test.py

Now change to your data directory::

    cd "C:\Documents and Settings\<username>\My Documents\data"

To run the program use::

    python -m bumps.cli -h


Linux
-----

Many linux distributions will provide the base required packages.  You
will need to refer to your distribution documentation for details.

On Ubuntu you can use:

    sudo apt-get install python-matplotlib python-scipy python-nose python-sphinx
    sudo apt-get install python-wxgtk3.0

From a terminal, change to the directory containing the bumps source and type::

    python setup.py build
    python test.py
    sudo python setup.py install

This should install the application somewhere on your path.

To run the program use::

    bumps -h

OS/X
----

Building a useful python environment on OS/X is somewhat involved, and
frequently evolving so this document will likely be out of date.
We've had success using the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
64-bit python 2.7 environment from Continuum Analytics, which provides
the required packages, but other distributions should work as well.

You will need to install XCode from the app store, and set the preferences
to install the command line tools so that a C compiler is available (look
in the Downloads tab of the preferences window).  If any of your models
require fortran, you can download
`gfortran binaries <http://r.research.att.com/tools/>`_ from
r.research.att.com/tools (scroll down to the  Apple Xcode gcc-42 add-ons).
This sets up the basic development environment.

From a terminal, change to the directory containing the source and type::

    conda create -n bumps numpy scipy matplotlib nose sphinx wxpython
    source activate bumps
    python setup.py install
    python test.py
    cd ..

    # Optional: allow bumps to run from outside the bumps environment
	mkdir ~/bin # create user terminal app directory if it doesn't already exist
    ln -s `python -c "import sys;print sys.prefix"`/bin/bumps ~/bin


To run the program, start a new Terminal shell and type::

    bumps -h


Fast Stepper for DREAM on MPI
=============================

When running DREAM on larger clusters, we found a significant slowdown as the
number of processes increased.  This is due to Amdahl's law, where the run
time speedup is limited by the slowest serial portion of the code.  In our
case, the DE stepper and the bounds check.  Compiling this in C with OpenMP
allows us to scale to hundreds of nodes until the stepper again becomes a
bottleneck.

To use the compiled DE stepper and bounds checks use::

    (cd bumps/dream && cc compiled.c -I ../../Random123/include/ -O2 -fopenmp -shared -lm -o _compiled.so -fPIC)

Note: clang doesn't support OpenMP, so on OS/X use::

    (cd bumps/dream && cc compiled.c -I ../../Random123/include/ -O2 -shared -lm -o _compiled.so -fPIC)

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

There is a bug in older sphinx versions (1.0.7 as of this writing) in which
latex tables cannot be created.  You can fix this by changing::

    self.body.append(self.table.colspec)

to::

    self.body.append(self.table.colspec.lower())

in site-packages/sphinx/writers/latex.py.  This may have been fixed in
newer versions.

Windows Installer
=================

To build a windows standalone executable with py2exe you may first need
to create an empty file named
*C:\\Python27\\Lib\\numpy\\distutils\\tests\\__init__.py*.
Without this file, py2exe raises an error when it is searching for
the parts of the numpy package.  This may be fixed on recent versions
of numpy. Next, update the __version__ tag in bumps/__init__.py to mark
it as your own.

Now you can build the standalone executable using::

    python setup_py2exe

This creates a dist subdirectory in the source tree containing
everything needed to run the application including python and
all required packages.

To build the Windows installer, you will need two more downloads:

    - Visual C++ 2008 Redistributable Package (x86) 11/29/2007
    - `Inno Setup <http://www.jrsoftware.org/isdl.php>`_ 5.3.10 QuickStart Pack

The C++ redistributable package is needed for programs compiled with the
Microsoft Visual C++ compiler, including the standard build of the Python
interpreter for Windows.  It is available as vcredist_x86.exe from the
`Microsoft Download Center <http://www.microsoft.com/downloads/>`_.
Be careful to select the version that corresponds to the one used
to build the Python interpreter --- different versions can have the
same name.  For the Python 2.7 standard build, the file is 1.7 Mb
and is dated 11/29/2007.  We have a copy (:slink:`%(vcredist)s`) on
our website for your convenience.  Save it to the *C:\\Python27*
directory so the installer script can find it.

Inno Setup creates the installer executable.  When installing Inno Setup,
be sure to choose the 'Install Inno Setup Preprocessor' option.

With all the pieces in place, you can run through all steps of the
build and install by changing to the top level python directory and
typing::

    python master_builder.py

This creates the redistributable installer bumps-<version>-win32.exe for
Windows one level up in the directory tree.  In addition, source archives
in zip and tar.gz format are produced as well as text files listing the
contents of the installer and the archives.
