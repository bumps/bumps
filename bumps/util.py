from __future__ import division

__all__ = ["erf", "profile", "kbhit", "redirect_console", "pushdir", "push_seed"]

import sys
import os

import numpy
from numpy import ascontiguousarray as _dense

def erf(x):
    """
    Error function calculator.
    """
    from bumpsmodule import _erf
    input = _dense(x,'d')
    output = numpy.empty_like(input)
    _erf(input,output)
    return output


def profile(fn, *args, **kw):
    """
    Profile a function called with the given arguments.
    """
    import cProfile, pstats, os
    global call_result
    def call():
        global call_result
        call_result = fn(*args, **kw)
    datafile = 'profile.out'
    cProfile.runctx('call()', dict(call=call), {}, datafile)
    stats = pstats.Stats(datafile)
    #order='calls'
    order='cumulative'
    #order='pcalls'
    #order='time'
    stats.sort_stats(order)
    stats.print_stats()
    os.unlink(datafile)
    return call_result


def kbhit():
    """
    Check whether a key has been pressed on the console.
    """
    try: # Windows
        import msvcrt
        return msvcrt.kbhit()
    except: # Unix
        import sys
        import select
        i,_,_ = select.select([sys.stdin],[],[],0.0001)
        return sys.stdin in i

class redirect_console(object):
    """
    Console output redirection context

    Redirect the console output to a path or file object.

    :Example:

        >>> print "hello"
        hello
        >>> with redirect_console("redirect_out.log"):
        ...     print "hello"
        >>> print "hello"
        hello
        >>> print open("redirect_out.log").read()[:-1]
        hello
        >>> import os; os.unlink("redirect_out.log")
    """
    def __init__(self, stdout=None, stderr=None):
        if stdout is None:
            raise TypeError("stdout must be a path or file object")
        self.open_files = []
        self.sys_stdout = []
        self.sys_stderr = []

        if hasattr(stdout, 'write'):
            self.stdout = stdout
        else:
            self.open_files.append(open(stdout, 'w'))
            self.stdout = self.open_files[-1]

        if stderr is None:
            self.stderr = self.stdout
        elif hasattr(stderr, 'write'):
            self.stderr = stderr
        else:
            self.open_files.append(open(stderr,'w'))
            self.stderr = self.open_files[-1]

    def __del__(self):
        for f in self.open_files:
            f.close()

    def __enter__(self):
        self.sys_stdout.append(sys.stdout)
        self.sys_stderr.append(sys.stderr)
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def __exit__(self, *args):
        sys.stdout = self.sys_stdout[-1]
        sys.stderr = self.sys_stderr[-1]
        del self.sys_stdout[-1]
        del self.sys_stderr[-1]
        return False

class pushdir(object):
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

        >>> import numpy
        >>> push_seed(24) # doctest:+ELLIPSIS
        <...push_seed object at...>
        >>> print numpy.random.randint(0,1000000,3)
        [242082    899 211136]

    Seed can also be used in a with statement, which sets the random
    number generator state for the enclosed computations and restores
    it to the previous state on completion::

        >>> with push_seed(24):
        ...    print numpy.random.randint(0,1000000,3)
        [242082    899 211136]

    Using nested contexts, we can demonstrate that state is indeed
    restored after the block completes::

        >>> with push_seed(24):
        ...    print numpy.random.randint(0,1000000)
        ...    with push_seed(24):
        ...        print numpy.random.randint(0,1000000,3)
        ...    print numpy.random.randint(0,1000000)
        242082
        [242082    899 211136]
        899

    The restore step is protected against exceptions in the block::

        >>> with push_seed(24):
        ...    print numpy.random.randint(0,1000000)
        ...    try:
        ...        with push_seed(24):
        ...            print numpy.random.randint(0,1000000,3)
        ...            raise Exception()
        ...    except:
        ...        print "Exception raised"
        ...    print numpy.random.randint(0,1000000)
        242082
        [242082    899 211136]
        Exception raised
        899
    """
    def __init__(self, seed=None):
        self._state = numpy.random.get_state()
        numpy.random.seed(seed)
    def __enter__(self):
        return None
    def __exit__(self, *args):
        numpy.random.set_state(self._state)
        pass
