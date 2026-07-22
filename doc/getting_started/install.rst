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

The uv package from astral is a very fast python installer. To set it up following
the instruction on `https://docs.astral.sh/uv/`_. Another options is a conda installer
such as `miniforge <https://github.com/conda-forge/miniforge/releases/latest>`_.

* temporary uv environment

    uv run --with bumps bumps

* permanent uv environment

    uv venv path/to/bumps_env
    source /path/to/bumps_env/bin/activate  # mac, unix
    # /path/to/bumps_env/Scripts/activate.bat  % windows cmd [untested]
    uv pip install bumps

* conda environment

    conda create --name bumps python
    conda activate bumps
    pip install bumps

Running bumps
=============

Fitting problems in bumps are defined in python files or jupyter notebooks. You
can retrieve the example *curve.py* model
`here <https://github.com/bumps/bumps/blob/master/doc/examples/curvefit/curve.py>`_

To run the webview interface with your problem showing in a browser window use::

    bumps curve.py

To run in batch mode with no interactive interface use::

    bumps -b curve.py --session=fit.h5

This runs a complete fit, appending the results to the session file T1.hdf. To later
view the fit results use::

    bumps --session=fit.h5

There are many command line options for controlling the fit. For a complete list use::

    bumps -h

Jupyter notebooks
=================

The webview interface can be run inside a Jupyter notebook. This allows you to interact with the server
from within the notebook, providing a more integrated experience for users who are already working in a Jupyter environment.

* start jupyter in a temporary uv environment

    uv run --with jupyter,bumps jupyter lab

* add jupyter to your permanent environment

    pip install jupyter
    jupyter lab

* add bumps to your jupyter hub server

If you are accessing jupyter through a remote JupyterHub server, you can start a terminal and
create a permanent bumps environment using uv or conda. To register this environment with your
hub server, active the environment and do the following::

    pip install ipykernel
    python -m ipykernel install --user --name bumps --display-name "bumps"

Once you have jupyter running you will need to access bumps withing the notebook.

Start with the following cell::

    ## Uncomment the following to install bumps in your environment
    # %pip install bumps

To start webview, use the following code cell::

    import bumps.names as bp

    # Start the server, with options similar to the command line
    await bp.start_bumps(fit="dream", ...)
    # Show the webview interface
    bp.display_bumps()

A link to the server will be printed in the notebook output. You can open this link in a browser to access the server.

In a different cell you can define a problem and load it into the server using the `api` module::

    # Define a problem
    from bumps.fitproblem import FitProblem

    model = MyFitnessClass()
    ...

    problem = FitProblem([model])
    await api.set_problem(problem)

You may need to work with an unreleased version of bumps, installed from a development branch
on github. The install cell becomes a little more complicated in this case::

    %pip install git+https://github.com/bumps/bumps@BRANCHNAME

    # Check if nodejs is available; if not install it via the python nodeenv package
    !npm -v
    # %pip install nodeenv
    # !nodeenv --prebuilt -p

    # Build the webview client
    !python -m bumps.webview.build_client

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
