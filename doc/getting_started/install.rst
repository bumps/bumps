.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Bumps |version| is provided as a Windows installer or as source:

	- Windows installer: :slink:`%(winexe)s`
	- Apple installer: :slink:`%(macapp)s`
	- Source: :slink:`%(srczip)s`

Th Windows installer walks through the steps of setting the program up
to run on your machine and provides the sample data to be used in the
tutorial.  Installers for other platforms are not yet available, and
must be built from source.

Building from source
====================

Building the application from source requires some preparation.

First you will need to set up your python environment.  We depend on
numerous external packages.  The versions listed below are a snapshot
of a configuration that we are using.  The program may work with older
versions of the package, and we will try to keep it compatible with
the latest versions.

Our base scientific python environment contains:

	- Python 2.7 (not 3.x)
	- Matplotlib 1.0.0
	- Numpy 1.3.0
	- Scipy 0.7.0
	- WxPython 2.8.10.1
	- distribute 0.6.28
	- gcc 3.4.4

To run tests you will need:

	- Nose 0.11

To build the HTML documentation you will need:

	- Sphinx 1.0.4
	- DocUtils 0.5
	- Pyments 1.0
	- Jinja2 2.5.2

The PDF documentation requires a working LaTeX installation.

You can install directly from PyPI using pip::

    pip install bumps

If this fails, then follow the instructions to install from the source
archive directly. Platform specific details for setting up your environment
are given below.

Windows
-------

The `Python(X,Y) <http://code.google.com/p/pythonxy/>`_ package contains
most of the pieces required to build the application.  You can select
"Full Install" for convenience, or you can select "Custom Install" and make
sure the above packages are selected.  In particular, wx is not selected
by default.  Be sure to select py2exe as well, since you may want to
build a self contained release package.

The Python(x,y) package supplies a C/C++ compiler, but the package does
not set it as the default compiler.  To do this you will need to create
*C:\\Python26\\Lib\\distutils\\distutils.cfg* with the following content::

    [build]
    compiler=mingw32

Next change to the directory containing the source.  This will be a command
like the following::

    cd "C:\Documents and Settings\<username>\My Documents\bumps-src"

Now type the command to build and install::

    python setup.py install
    python test.py

Now change to your data directory::

    cd "C:\Documents and Settings\<username>\My Documents\data"

To run the program use::

    python "C:\Python26\Scripts\bumps" -h

Linux
-----

Many linux distributions will provide the base required packages.  You
will need to refer to your distribution documentation for details.

On Ubuntu you can use apt-get to install matplotlib, numpy, scipy, wx,
nose and sphinx.

From a terminal, change to the directory containing the source and type::

    python setup.py install
    python test.py

This should install the application somewhere on your path.

To run the program use::

    bumps_cli -h

OS/X
----

Building a useful python environment on OS/X is somewhat involved, and
frequently evolving so this document will likely be out of date.

We've had success using the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
64-bit python 2.7 environment from Continuum Analytics, which provides most
of the required packages, but other distributions should work as well.
You will need to install XCode from the app store, and set the preferences
to install the command line tools so that a C compiler is available (look
in the Downloads tab of the preferences window).  If any of your models
require fortran, you can download gfortran from r.research.att.com/tools
(scroll down to the  Apple Xcode gcc-42 add-ons). This sets up the basic
development environment.

wxPython
~~~~~~~~

The wxPython package is missing from Anaconda, so we built our own
(:slink:`%(wx4osx)s`).  Download the package and install using::

    conda install wx-2.9.5.0-py27_0.tar.bz2

We built the  wx package from the development release (2.9.5.0) at
`<http://wxpython.org/download.php>`_ using the following commands::


    #### Setup virtual environment with anaconda
    conda create -n wx nose
    source activate wx
    # confirm that there are no untracked files in the environment
    conda package -u

    #### Fetch, build and install wxPython
    curl -LO http://downloads.sourceforge.net/wxpython/wxPython-src-2.9.5.0.tar.bz2
    tar xjf wxPython-src-2.9.5.0.tar.bz2
    cd wxPython-src-2.9.5.0/wxPython
    python build-wxpython.py --build_dir=/tmp/wx --osx_cocoa
    # Note: building with "--prefix=<virtualenvpath> --install" should install
    # all the pieces into the virtual environment:
    #    --prefix=`python -c "import sys;print sys.prefix"` --install
    # This didn't work, so we instead install by hand.
    WXDEST=`python -c "import sys;print sys.prefix"`
    cp -r wx $WXDEST/lib/python2.7/site-packages/wx
    cp wxversion/wxversion.py $WXDEST/lib/python2.7/site-packages
    cp -r /tmp/wx/cocoa/lib/* $WXDEST/lib
    python -m compileall $WXDEST/lib/python2.7/site-packages/wx
    python -m compileall $WXDEST/lib/python2.7/site-packages/wxversion.py
    # optional scripts, #!/usr/bin/env python -> #!/usr/bin/env pythonw
    cd scripts
    for f in `ls | grep -v "[.]py"`; do
        sed -e"s/env python/env pythonw/" $f > $WXDEST/bin/$f
        chmod a+x $WXDEST/bin/$f
    done
    # return to root
    cd ../../..

    #### Construct a conda binary distribution for wx package and clean up
    cd ../../..
    conda package -u  # show what will be in the wx package
    conda package --pkg-name wx --pkg-version 2.9.5.0
    source deactivate
    conda remove -n wx --all
    rm -rf /tmp/wx


The resulting package can be installed into an environment using::

    conda install wx-2.9.5.0-py27_0.tar.bz2

bumps
~~~~~


From a terminal, change to the directory containing the source and type::

    conda create -n bumps numpy scipy matplotlib nose sphinx
    source activate bumps
    conda install wx-2.9.5.0-py27_0.tar.bz2
    python setup.py install
    python test.py
    cd ..

    # Optional: create a bumps binary package and clean up
    conda package --pkg-name wx --pkg-version `python -c "import bumps;print bumps.__version__"`

    # Optional: allow bumps to run from outside the bumps environment
	mkdir ~/bin # create user terminal app directory if it doesn't already exist
    ln -s `python -c "import sys;print sys.prefix"`/bin/bumps ~/bin


To run the program, start a new Terminal shell and type::

    bumps -h


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

Windows users please note that this only works under cygwin/msys for now 
since we are using *make*.  There is a skeleton *make.bat* in the directory
that will work using *cmd* but it doesn't yet build PDF files.

You can see the result by pointing your browser to::

    bumps/doc/_build/html/index.html
    bumps/doc/_build/latex/Bumps.pdf

ReStructured text format does not have a nice syntax for superscripts and
subscripts.  Units such as |g/cm^3| are entered using macros such as
\|g/cm^3| to hide the details.  The complete list of macros is available in

        doc/sphinx/rst_prolog

In addition to macros for units, we also define cdot, angstrom and degrees
unicode characters here.  The corresponding latex symbols are defined in
doc/sphinx/conf.py.

There is a bug in sphinx versions (1.0.7 as of this writing) in which
latex tables cannot be created.  You can fix this by changing::

	self.body.append(self.table.colspec)

to::

    self.body.append(self.table.colspec.lower())

in site-packages/sphinx/writers/latex.py.

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
