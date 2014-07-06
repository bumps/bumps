# This program is in the public domain.
"""
Setup file for constructing OS X applications.

Run using::

    % python setup_py2app.py
"""

import os
import sys
import shutil

sys.dont_write_bytecode = True

# Force build before continuing
os.system('"%s" setup.py build' % sys.executable)

# Remove the current directory from the python path
here = os.path.abspath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != here]

import py2app
from distutils.core import setup
from distutils.util import get_platform

if len(sys.argv) == 1:
    sys.argv.append('py2app')

# Put the build lib on the start of the path.
# For packages with binary extensions, need platform.  If it is a pure
# script library, use an empty platform string.
platform = '.%s-%s' % (get_platform(), sys.version[:3])
#platform = ''
build_lib = os.path.abspath('build/lib' + platform)
sys.path.insert(0, build_lib)

# print "\n".join(sys.path)


# TODO: Combine with setup-py2exe so that consistency is easier.
packages = ['numpy', 'scipy', 'matplotlib', 'pytz']
includes = []
excludes = ['Tkinter', 'PyQt4', '_ssl', '_tkagg', 'numpy.distutils.test',
            'sphinx', 'docutils', 'jinja2', ]
PACKAGE_DATA = {}

import bumps
from bumps.gui.resources import resources as gui_resources

NAME = 'Bumps'
# Until we figure out why packages=... doesn't work reliably,
# use py2app_main with explicit imports of everything we
# might need.
#SCRIPT = 'py2app_main.py'
SCRIPT = 'bin/bumps_gui'
VERSION = bumps.__version__
ICON = 'extra/bumps.icns'
ID = 'Bumps'
COPYRIGHT = 'This program is public domain'
DATA_FILES = gui_resources.data_files()

plist = dict(
    CFBundleIconFile=ICON,
    CFBundleName=NAME,
    CFBundleShortVersionString=' '.join([NAME, VERSION]),
    CFBundleGetInfoString=NAME,
    CFBundleExecutable=NAME,
    CFBundleIdentifier='org.reflectometry.%s' % ID,
    NSHumanReadableCopyright=COPYRIGHT
)


app_data = dict(script=SCRIPT, plist=plist)
py2app_opt = dict(argv_emulation=True,
                  packages=packages,
                  includes=includes,
                  excludes=excludes,
                  iconfile=ICON,
                  optimize=2)
options = dict(py2app=py2app_opt,)


def build_app():
    setup(
        data_files=DATA_FILES,
        package_data=PACKAGE_DATA,
        app=[app_data],
        options=options,
    )
    # Add cli interface to the app directory
    os.system('cp -p extra/appbin/* "dist/%s.app"' % NAME)


def build_dmg():
    """DMG builder; should include docs"""
    PRODUCT = NAME + " " + VERSION
    PRODUCTDASH = NAME + "-" + VERSION
    APP = "dist/%s.app" % PRODUCT
    DMG = "dist/%s.dmg" % PRODUCTDASH
    # Remove previous build if it is still sitting there
    if os.path.exists(APP):
        shutil.rmtree(APP)
    if os.path.exists(DMG):
        os.unlink(DMG)
    os.rename("dist/%s.app" % NAME, APP)
    os.system('cd dist && ../extra/dmgpack.sh "%s" "%s.app" ../doc/_build/html ../doc/examples' %
              (PRODUCTDASH, PRODUCT))
    os.system('chmod a+r "%s"' % DMG)

if __name__ == "__main__":
    build_app()
    build_dmg()
