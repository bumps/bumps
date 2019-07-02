"""
Miscellaneous utility functions.
"""
from __future__ import division

__all__ = ["erf", "kbhit", "profile",
           "pushdir", "push_seed", "redirect_console"]

import sys
import os
try:  # CRUFT: python 2.x
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
from numpy import ascontiguousarray as _dense
from scipy.special import erf


def parse_errfile(errfile):
    """
    Parse dream statistics from a particular fit.

    Returns overall chisq, list of chisq for individual models and
    a parameter dictionary with attributes for number, name, mean, median,
    p68 for 68% credible interval and p95 for 95% credible interval.

    The parameter dictionary is keyed by parameter name.

    Usually there is only one errfile in a directory, which can be
    retrieved using::

        import glob
        errfile = glob.glob(path+'/*.err')[0]
    """
    from .dream.stats import parse_var
    pars = []
    chisq = []
    overall = None
    with open(errfile) as fid:
        for line in fid:
            if line.startswith("[overall"):
                overall = float(line.split()[1][6:-1])
                continue

            if line.startswith("[chisq"):
                chisq.append(float(line.split()[0][7:-1]))
                continue

            p = parse_var(line)
            if p is not None:
                pars.append(p)

    if overall is None:
        overall = chisq[0]
    pardict = dict((p.name, p) for p in pars)
    return overall, chisq, pardict


def _c_erf(x):
    """
    Error function calculator.
    """
    from ._reduction import _erf
    input = _dense(x, 'd')
    output = np.empty_like(input)
    _erf(input, output)
    return output


def _erf_test():
    assert erf(5) == 2
    assert erf(0.) == 0.
    assert (erf(np.array([0., 0.])) == 0.).all()
    assert abs(erf(3.) - 0.99997790950300136) < 1e-14


def profile(fn, *args, **kw):
    """
    Profile a function called with the given arguments.
    """
    import cProfile
    import pstats

    result = [None]
    def call():
        result[0] = fn(*args, **kw)
    datafile = 'profile.out'
    cProfile.runctx('call()', dict(call=call), {}, datafile)
    stats = pstats.Stats(datafile)
    # order='calls'
    order = 'cumulative'
    # order='pcalls'
    # order='time'
    stats.sort_stats(order)
    stats.print_stats()
    os.unlink(datafile)
    return result[0]


def kbhit():
    """
    Check whether a key has been pressed on the console.
    """
    try:  # Windows
        import msvcrt
        return msvcrt.kbhit()
    except ImportError:  # Unix
        import select
        i, _, _ = select.select([sys.stdin], [], [], 0.0001)
        return sys.stdin in i


class DynamicPackage(object):
    def __init__(self, path):
       self.__path__ = [path]


def relative_import(filename, package="relative_import"):
    """
    Define an empty package allowing relative imports from a script.

    By setting :code:`__package__ = relative_import(__file__)` at the top of
    your script file you can even run your model as a python script.  So long
    as the script behaviour is isolated in :code:`if __name__ == "__main__":`
    code block and :code:`problem = FitProblem(...)` is defined, the same model
    can be used both within and outside of bumps.
    """
    path = os.path.dirname(os.path.abspath(filename))
    if (package in sys.modules
            and not isinstance(sys.modules[package], DynamicPackage)):
        raise ImportError("relative import would override the existing package %s. Use another name"
                          % package)
    sys.modules[package] = DynamicPackage(path)
    return package


class redirect_console(object):
    """
    Console output redirection context

    The output can be redirected to a string, to an already opened file
    (anything with a *write* attribute), or to a filename which will be
    opened for the duration of the with context.  Unless *stderr* is
    specified, then both standard output and standard error are
    redirected to the same file.  The open file handle is returned on
    enter, and (if it was not an already opened file) it is closed on exit.

    If no file is specified, then output is redirected to a StringIO
    object, which has a getvalue() method which can retrieve the string.
    The StringIO object is deleted when the context ends, so be sure to
    retrieve its value within the redirect_console context.

    :Example:

    Show that output is captured in a file:

        >>> from bumps.util import redirect_console
        >>> print("hello")
        hello
        >>> with redirect_console("redirect_out.log"):
        ...     print("captured")
        >>> print("hello")
        hello
        >>> print(open("redirect_out.log").read()[:-1])
        captured
        >>> import os; os.unlink("redirect_out.log")

    Output can also be captured to a string:

        >>> with redirect_console() as fid:
        ...    print("captured to string")
        ...    captured_string = fid.getvalue()
        >>> print(captured_string.strip())
        captured to string

    """
    def __init__(self, stdout=None, stderr=None):
        self.open_files = []
        self.sys_stdout = []
        self.sys_stderr = []

        if stdout is None:
            self.open_files.append(StringIO())
            self.stdout = self.open_files[-1]
        elif hasattr(stdout, 'write'):
            self.stdout = stdout
        else:
            self.open_files.append(open(stdout, 'w'))
            self.stdout = self.open_files[-1]

        if stderr is None:
            self.stderr = self.stdout
        elif hasattr(stderr, 'write'):
            self.stderr = stderr
        else:
            self.open_files.append(open(stderr, 'w'))
            self.stderr = self.open_files[-1]

    def __del__(self):
        for f in self.open_files:
            f.close()

    def __enter__(self):
        self.sys_stdout.append(sys.stdout)
        self.sys_stderr.append(sys.stderr)
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self.open_files[-1]

    def __exit__(self, *args):
        sys.stdout = self.sys_stdout[-1]
        sys.stderr = self.sys_stderr[-1]
        del self.sys_stdout[-1]
        del self.sys_stderr[-1]
        return False

class push_python_path(object):
    """
    Change sys.path for the duration of a with statement.

    :Example:

    Show that the original directory is restored::

        >>> import sys, os
        >>> original_path = list(sys.path)
        >>> with push_python_path('/tmp'):
        ...     assert sys.path[-1] == '/tmp'
        >>> restored_path = list(sys.path)
        >>> assert original_path == restored_path
    """
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.append(self.path)

    def __exit__(self, *args):
        del sys.path[-1]


class pushdir(object):
    """
    Change directories for the duration of a with statement.

    :Example:

    Show that the original directory is restored::

        >>> import sys, os
        >>> original_wd = os.getcwd()
        >>> with pushdir(sys.path[0]):
        ...     pushed_wd = os.getcwd()
        ...     first_site = os.path.abspath(sys.path[0])
        ...     assert pushed_wd == first_site
        >>> restored_wd = os.getcwd()
        >>> assert original_wd == restored_wd
    """
    def __init__(self, path):
        self.path = os.path.abspath(path)

    def __enter__(self):
        self._cwd = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self._cwd)


class push_seed(object):
    """
    Set the seed value for the random number generator.

    When used in a with statement, the random number generator state is
    restored after the with statement is complete.

    :Parameters:

    *seed* : int or array_like, optional
        Seed for RandomState

    :Example:

    Seed can be used directly to set the seed::

        >>> from numpy.random import randint
        >>> push_seed(24)
        <...push_seed object at...>
        >>> print(randint(0,1000000,3))
        [242082    899 211136]

    Seed can also be used in a with statement, which sets the random
    number generator state for the enclosed computations and restores
    it to the previous state on completion::

        >>> with push_seed(24):
        ...    print(randint(0,1000000,3))
        [242082    899 211136]

    Using nested contexts, we can demonstrate that state is indeed
    restored after the block completes::

        >>> with push_seed(24):
        ...    print(randint(0,1000000))
        ...    with push_seed(24):
        ...        print(randint(0,1000000,3))
        ...    print(randint(0,1000000))
        242082
        [242082    899 211136]
        899

    The restore step is protected against exceptions in the block::

        >>> with push_seed(24):
        ...    print(randint(0,1000000))
        ...    try:
        ...        with push_seed(24):
        ...            print(randint(0,1000000,3))
        ...            raise Exception()
        ...    except Exception:
        ...        print("Exception raised")
        ...    print(randint(0,1000000))
        242082
        [242082    899 211136]
        Exception raised
        899
    """
    def __init__(self, seed=None):
        self._state = np.random.get_state()
        np.random.seed(seed)

    def __enter__(self):
        return None

    def __exit__(self, *args):
        np.random.set_state(self._state)
