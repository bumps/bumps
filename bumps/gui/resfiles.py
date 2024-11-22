"""
Package data files
==================

Some python packages, particularly gui packages, but also other packages
with static data need to be able to ship the data with the package.  This
is particularly a problem for py2exe since python does not provide any
facilities for extracting the data files from the bundled exe at runtime.
Instead, the setup.py process needs to ask for the package data files so
it can bundle them separately from the exe.  When the application is running
it will need to find the resource files so that it can load them, regardless
if it is running directly from the source tree, from an installed package
or from an exe or app.

The Resources class handles both the setup and the runtime requirements
for package resources.  You will need to put the resources in a path
within your source tree and initialize a Resources object with the necessary
information.  For example, assuming the resources are in a subdirectory
named "resources", and this package is stored as resfiles.py in the parent,
you would set resources/__init__.py as follows::

    from ..resfiles import Resources
    resources = Resources(package=__name__,
                          patterns=('*.png', '*.jpg', '*.ico', '*.wav'),
                          datadir='bumps-data',
                          check_file='reload.png',
                          env='BUMPS_DATA')

Now a resource file such as 'reload.png' can be accessed from a parent module
using::

    from .resources import resources
    resources.get_path('reload.png')

"""

import sys
import os
import glob


class Resources(object):
    """
    Identify project resource files.

    *package* : string
        Name of the subpackage containing the resources.  From the __init__.py
        file, for the resource directory, this is just __name__.
    *patterns* : list or tuple
        Set of glob patterns used to identify resource files in the resource
        package.
    *datadir* : string
        Name of the installed resource directory.  This is used in setup to
        prepare the resource directory and in the application to locate the
        resources that have been installed.
    *check_file*: string
        Name of a resource file that should be in the resource directory.  This
        is used to check that the resource directory exists in the installed
        application.
    *env* : string (optional)
        Environment variable which contains the complete path to the resource
        directory.  The environment variable overrides other methods of
        accessing the resources except running directly from the source tree.
    """

    def __init__(self, package, patterns, datadir, check_file, env=None):
        self.package = package
        self.patterns = patterns
        self.datadir = datadir
        self.check_file = check_file
        self.env = env
        self._cached_path = None

    def package_data(self):
        """
        Return the data files associated with the package.

        The format is a dictionary of {'fully.qualified.package', [files...]}
        used directly in the setup script as::

            setup(...,
                  package_data=package_data(),
                  ...)
        """
        return {self.package: list(self.patterns)}

    def data_files(self):
        """
        Return the data files associated with the package.

        The format is a list of (directory, [files...]) pairs which can be
        used directly in the py2exe setup script as::

            setup(...,
                  data_files=data_files(),
                  ...)

        Unlike package_data(), which only works from the source tree, data_files
        uses installed data path to locate the resources.
        """
        data_files = [(self.datadir, self._finddata(*self.patterns))]
        return data_files

    def _finddata(self, *patterns):
        path = self.resource_dir()
        files = []
        for p in patterns:
            files += glob.glob(os.path.join(path, p))
        return files

    def resource_dir(self):
        """
        Return the path to the application data.

        This is either in an environment variable, in the source tree next to
        this file, or beside the executable.  The environment key can be set
        using
        """
        # If we already found it, then we are done
        if self._cached_path is not None:
            return self._cached_path

        # Check for data in the package itself (which will be the case when
        # we are running from the source tree).
        path = os.path.abspath(os.path.dirname(sys.modules[self.package].__file__))
        if self._cache_resource_path(path):
            return self._cached_path

        # Check for data path in the environment.  If the environment variable
        # is specified, then the resources have to be there, or the program fails.
        if self.env and self.env in os.environ:
            if not self._cache_resource_path(os.environ[self.env]):
                raise RuntimeError("Environment %s not a directory" % self.env)
            return self._cached_path

        # Check for data next to exe/zip file.
        exepath = os.path.dirname(sys.executable)
        path = os.path.join(exepath, self.datadir)
        if self._cache_resource_path(path):
            return self._cached_path

        # py2app puts the data in Contents/Resources, but the executable
        # is in Contents/MacOS.
        path = os.path.join(exepath, "..", "Resources", self.datadir)
        if self._cache_resource_path(path):
            return self._cached_path

        raise RuntimeError("Could not find the GUI data files")

    def _cache_resource_path(self, path):
        if os.path.exists(os.path.join(path, self.check_file)):
            self._cached_path = path
            return True
        else:
            return False

    def get_path(self, filename):
        return os.path.join(self.resource_dir(), filename)
