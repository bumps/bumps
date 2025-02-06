
Webview Server
==============

The webview server provides a graphical user interface for managing models and fitting data
using the Bumps library. It allows users to interact with the application through a web browser,
providing a more user-friendly experience compared to the command-line interface.

All of the features of the existing WxPython-based graphical user interface are available 
in the webview server, including model editing, data visualization, and fitting controls.

Installation
------------
The webview server is included in the Bumps package, and the additional dependencies needed to run it
can be installed with the following command::

    pip install bumps[webview]

This will install `aiohttp`, `blinker`, `dill`, `matplotlib`, `python-socketio`, `plotly`, `mpld3` and `h5py` as dependencies.

Building the client (only for source install)
---------------------------------------------
The client is a web application that runs in the browser and communicates with the server to perform
model editing, data visualization, and fitting operations. The client is built using modern web technologies
such as TypeScript, Vue.js, and Plotly, and is included (pre-compiled) with the python wheel for 
pip installs and also in the binary installers.

If you are installing from source, you will need to build the client before starting the server.
First install `nodejs` is installed (from conda-forge in your environment, or a system install), then run::

    python -m bumps.webview.build_client

This will install the necessary dependencies and build the client application. 
The client files will be placed in the `bumps/webview/client/dist` directory.

Starting the Server
====================
For binary installations, the server can be started from a shortcut in the bumps application folder, 
e.g. `bumps_webview.app` on macOS or `bumps_webview.bat` on Windows.

To run from the command line, make sure that `bumps` is installed in your python environment and run::

    bumps-webview

This will start the server on a random local port and open a browser window to the server.
For a complete list of options, run::

    bumps-webview --help

A typical invocation might be::

    bumps-webview my_problem_file.py

This will start the server on a random local port and load the problem file `my_problem_file.py`
into the server, and will open a browser window to the server.

To start the server without opening a browser window, use the `--headless` option.

Running the Server inside a Jupyter Notebook
--------------------------------------------
The webview server can also be run inside a Jupyter notebook. This allows you to interact with the server
from within the notebook, providing a more integrated experience for users who are already working in a Jupyter environment.

To run the server inside a Jupyter notebook, use the following code::

    import asyncio
    from bumps.webview.server.webserver import start_app
    from bumps.webview.server import api
    
    # Start the server
    server_task = asyncio.create_task(start_app(jupyter_link=True))

A link to the server will be printed in the notebook output. You can open this link in a browser to access the server.
Then later you can define a problem and load it into the server using the `api` module::

    # Define a problem
    from bumps.fitproblem import FitProblem
    
    model = MyFitnessClass()
    ...

    problem = FitProblem([model])
    await api.set_problem(problem)

