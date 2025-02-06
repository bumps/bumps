"""
Install application into a mod_wsgi stack.

To be sure there is no cross-talk between installed packages, you will
want to create a blank virtualenv to run mod_wsgi and a virtualenv for
each application you are running on the server.

To create the blank environment::

    virtualenv --no-site-packages /usr/local/pythonenv/clean

To configure WSGI, edit /etc/apache2/mods-available/wsgi.conf, setting::

    WSGIPythonHome /usr/local/pythonenv/clean

Then create a bumps user account with its private virtualenv::

    virtualenv --no-site-packages ~/bumps

You can now populate the virtualenv with the required packages::

    cd ~/bumps
    # Use system install for binary packages
    ln -s /usr/share/pyshared/numpy lib/python2.6/site-packages
    ln -s /usr/share/pyshared/matplotlib lib/python2.6/site-packages
    ln -s /usr/share/pyshared/pytz lib/python2.6/site-packages
    ln -s /usr/share/pyshared/scipy lib/python2.6/site-packages
    # Install the less common packages by hand
    bin/pip install flask
    bin/pip install ...
"""

import os

# === Configure virtual environment ===
import site

PYROOT = os.path.abspath(os.path.expanduser("~/bumps"))
sitepackages = "lib/python%d.%d/site-packages" % sys.version_info[0:2]
site.addsitedir(PYROOT + sitepackages)

# === Configure resource locations ===
import jobqueue.server

jobqueue.server.configure(
    jobstore="~/.bumps/server/%s",
    jobkey="~/.bumps/key",
    jobdb="sqlite://" + os.path.abspath(os.path.expanduser("~/.bumps/db")),
    scheduler="dispatch",
)

application = jobqueue.server.app
