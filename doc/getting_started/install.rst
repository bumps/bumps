.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Bumps |version| is provided in a self-contained Python environment:

    - Windows installer: :slink:`%(winexe)s`
    - Apple installer: :slink:`%(macapp)s`
    - Apple (Intel) installer: :slink:`%(imacapp)s`
    - Linux self-contained package: :slink:`%(linuxapp)s`

The Windows installer installs to the `AppData/Local` directory by default,
and adds a shortcut to the Start menu, optionally adding a desktop shortcut.
It also provides an uninstaller through the Control Panel (add/remove programs)

The Apple .dmg installer unpacks to the Applications directory.  You can
start the application by double-clicking on the `bumps_webview.app`.
To uninstall just drag the entire app to the trash from your finder.

For Linux (HPC), you can download the self-contained package and unpack it
to a directory of your choice.  The package contains a python environment
with all the required dependencies, including the webview interface.
To run the application, change into the unpacked directory and run::

    ./bin/python -m bumps

For Debian/Ubuntu Linux, bumps is provided as a package [pre-1.0 as of this writing]::

    sudo apt install python3-bumps

For other linux or for the latest version you will need to install bumps
as a python package.

Python install
==============

Bumps is available on `PyPI <https://pypi.org/project/bumps/>`_ so you can
install directly into a python environment with pip.
To avoid conflicts between python applications it is good practice to create
a separate python environment for each one.

We recommend using
`miniforge <https://github.com/conda-forge/miniforge/releases/latest>`_.
This installs a conda system with the "conda-forge" channel for packages.
Versions are available for Windows, MacOS and Linux.

To create your environment and install bumps use::

    conda create --name bumps python
    conda activate bumps
    pip install bumps[webview]

You could instead download Python directly from
`Python.org <https://www.python.org/downloads/>`_.
Again, recommended practice is to use an isolated python environment::

    python -m venv bumps
    . bumps/bin/activate
    pip install bumps[webview]

Running bumps
=============

Fitting problems in bumps are defined in python files or jupyter notebooks. You
can retrieve the example *curve.py* model
`here <https://github.com/bumps/bumps/blob/master/doc/examples/curvefit/curve.py>`_

To run the webview interface with your problem showing in a browser window use::

    bumps curve.py

To run in batch mode with no interactive interface use::

    bumps -b curve.py --store=session.hdf

This runs a complete fit, appending the results to the session file T1.hdf. To later
view the fit results use::

    bumps --store=session.hdf

There are many command line options for controlling the fit. For a complete list use::

    bumps -h

Jupyter notebooks
=================

The webview interface can be run inside a Jupyter notebook. This allows you to interact with the server
from within the notebook, providing a more integrated experience for users who are already working in a Jupyter environment.

You will need to set up Jupyter. You can install jupyter into your bumps
environment with pip and run the Jupyter server from there. If you are using
jupyterhub, you can install and run ipykernel in your bumps environment to make
it available::

    pip install ipykernel
    python -m ipykernel install --user --name bumps --display-name "bumps"

If running on colab or similar, then you can install bumps from within a
notebook cell using pip:

    %pip install bumps[webview]

To start webview, use the following code cell::

    import asyncio
    from bumps.webview.server import start_bumps_server, api

    # Start the server
    await start_bumps_server()

A link to the server will be printed in the notebook output. You can open this link in a browser to access the server.

In a different cell you can define a problem and load it into the server using the `api` module::

    # Define a problem
    from bumps.fitproblem import FitProblem

    model = MyFitnessClass()
    ...

    problem = FitProblem([model])
    await api.set_problem(problem)


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
