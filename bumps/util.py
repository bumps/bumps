from __future__ import division

__all__ = [ "merge_ends", "parse_file", "indfloat",
           "auto_shift", "next_color", "coordinated_colors",
           "dhsv", "profile", "kbhit", "redirect_console",
           "pushdir", "push_seed",
           ]

import numpy
from numpy import inf, nan


def merge_ends(w, p, tol=1e-3):
    """
    given a profile expressed as (dx,y), merge pieces with similar y so
    fewer pieces are required.
    """
    # TODO: accept rho,rhoi pairs as well
    # TODO: make sure we apply an interface to the right as well as the left
    try:
        # Assuming there p[0] != p[-1] within tolerance, we are guaranteed
        # that we will have a first value not equivalent to p[0] on the
        # left, with index > 0 and a last value not equivalent to p[-1] on
        # the right, with index < -1.  We are going to put the first value
        # at left index - 1 and the last value at right index + 1, accumulating
        # the widths of the identical layers.
        lidx = numpy.where(abs(p-p[0]) > tol)[0][0]-1
        ridx = len(p)-numpy.where(abs(p[::-1]-p[-1]) > tol)[0][0]
        w[lidx],p[lidx] = numpy.sum(w[:lidx+1]),p[0]
        w[ridx],p[ridx] = numpy.sum(w[ridx:]),p[-1]
        return w[lidx:ridx+1],p[lidx:ridx+1]
    except:
        # All one big layer
        w[0] = numpy.sum(w)
        return w[0:1],p[0:1]

def parse_file(file):
    """
    Parse a file into a header and data.

    Header lines look like # key value
    Keys can be made multiline by repeating the key
    Data lines look like float float float
    Comment lines look like # float float float
    Data may contain inf or nan values.

    Special hack for TOF data: if the first column contains bin edges, then
    the last row will only have the bin edge.  To make the array square,
    we extend the last row with NaN.
    """
    if hasattr(file, 'readline'):
        fh = file
    elif not string_like(file):
        raise ValueError('file must be a name or a file handle')
    elif file.endswith('.gz'):
        import gzip
        fh = gzip.open(file)
    else:
        fh = open(file)
    header = {}
    data = []
    for line in fh:
        columns,key,value = _parse_line(line)
        if columns:
            data.append([indfloat(v) for v in columns])
        if key:
            if key in header:
                header[key] = "\n".join((header[key],value))
            else:
                header[key] = value
    if fh is not file: fh.close()
    #print data
    #print "\n".join(k+":"+v for k,v in header.items())
    if len(data[-1]) == 1:
        # For TOF data, the first column is the bin edge, which has one
        # more row than the remaining columns; fill those columns with
        # NaN so we get a square array.
        data[-1] = data[-1]+[numpy.nan]*(len(data[0])-1)
    return header, numpy.array(data).T

def string_like(s):
    try: s+''
    except: return False
    return True

def _parse_line(line):
    # Check if line contains comment character
    idx = line.find('#')
    if idx < 0: return line.split(),None,''

    # If comment is after data, ignore the comment
    if idx > 0: return line[:idx].split(),None,''

    # Check if we have '# key value'
    line = line[1:].strip()
    idx = line.find(' ') # should also check for : and =
    if idx < 0: return [],None,None

    # Separate key and value
    key = line[:idx]
    value = line[idx+1:].lstrip()

    # If key is a number, it is simply a commented out data point
    if key[0] in '.-+0123456789': return [], None, None

    # Strip matching quotes off the value
    if (value[0] in ("'",'"')) and value[-1] is value[0]:
        value = value[1:-1]

    return [],key,value

def indfloat(s):
    """
    Convert string to float, with support for inf and nan.

    Example::

        >>> import numpy
        >>> print numpy.isinf(indfloat('inf'))
        True
        >>> print numpy.isinf(indfloat('-inf'))
        True
        >>> print numpy.isnan(indfloat('nan'))
        True
    """
    try:
        return float(s)
    except:
        s = s.lower()
        if s == 'inf': return inf
        if s == '-inf': return -inf
        if s == 'nan': return nan
        raise

def auto_shift(offset):
    from matplotlib.transforms import ScaledTranslation
    import pylab
    ax = pylab.gca()
    if ax.lines:
        ax._auto_shift += offset
    else:
        ax._auto_shift = 0
    trans = pylab.gca().transData
    if ax._auto_shift:
        trans += ScaledTranslation(0,ax._auto_shift/72,
                                   pylab.gcf().dpi_scale_trans)
    return trans

def next_color():
    import pylab
    try:
        base = pylab.gca()._get_lines.color_cycle.next()
    except: # Cruft 1.3 and earlier
        base = pylab.gca()._get_lines._get_next_cycle_color()
    return base

def coordinated_colors(base=None):
    if base is None: base = next_color()
    return dict(base=base,
                light = dhsv(base, dv=+0.3, ds=-0.2),
                dark = dhsv(base, dv=-0.25, ds=+0.35),
                )

# Color functions
def dhsv(color, dh=0, ds=0, dv=0, da=0):
    """
    Modify color on hsv scale.

    *dv* change intensity, e.g., +0.1 to brighten, -0.1 to darken.
    *dh* change hue
    *ds* change saturation
    *da* change transparency

    Color can be any valid matplotlib color.  The hsv scale is [0,1] in
    each dimension.  Saturation, value and alpha scales are clipped to [0,1]
    after changing.  The hue scale wraps between red to violet.

    :Example:

    Make sea green 10% darker:

        >>> darker = dhsv('seagreen', dv=-0.1)
        >>> print [int(v*255) for v in darker]
        [37, 113, 71, 255]
    """
    from matplotlib.colors import colorConverter
    from colorsys import rgb_to_hsv, hsv_to_rgb
    from numpy import clip, array, fmod
    r,g,b,a = colorConverter.to_rgba(color)
    #print "from color",r,g,b,a
    h,s,v = rgb_to_hsv(r,g,b)
    s,v,a = [clip(val,0.,1.) for val in s+ds,v+dv,a+da]
    h = fmod(h+dh,1.)
    r,g,b = hsv_to_rgb(h,s,v)
    #print "to color",r,g,b,a
    return array((r,g,b,a))



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

import sys
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

import os
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
