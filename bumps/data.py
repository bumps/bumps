"""
Data handling utilities.
"""
from __future__ import division

__all__ = ["convolve", "parse_file", "indfloat"]

import numpy
from numpy import inf, nan
from numpy import ascontiguousarray

def _dense(x): return numpy.ascontiguousarray(x,'d')

def convolve(xi,yi,x,dx):
    """
    Apply x-dependent gaussian resolution to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    The theory function is a piece-wise linear spline which does not need to
    be uniformly sampled.  The theory calculation points *xi* should be dense
    enough to capture the "wiggle" in the theory function, and should extend
    beyond the ends of the data measurement points *x*. Convolution at the
    tails is truncated and normalized to area of overlap between the resolution
    function in case the theory does not extend far enough.
    """
    from ._reduction import _convolve
    x = _dense(x)
    y = numpy.empty_like(x)
    _convolve(_dense(xi), _dense(yi), x, _dense(dx), y)
    return y

def convolve_sampled(xi,yi,xp,yp,x,dx):
    """
    Apply x-dependent arbitrary resolution function to the theory.

    Returns convolution y[k] of width dx[k] at points x[k].

    Like :func:`convolve`, the theory (*xi*,*yi*) is represented as a
    piece-wise linear spline which should extend beyond the data
    measurement points *x*.  Instead of a gaussian resolution function,
    resolution (*xp*,*yp*) is also represented as a piece-wise linear
    spline.
    """
    from ._reduction import _convolve_sampled
    x = _dense(x)
    y = numpy.empty_like(x)
    _convolve_sampled(_dense(xi), _dense(yi), _dense(xp), _dense(yp),
                      x, _dense(dx), y)
    return y

def test_convolve_sampled():
    x = [1,2,3,4,5,6,7,8,9,10]
    y = [1,3,1,2,1,3,1,2,1,3]
    xp = [-1,0,1,2,3]
    yp = [1,4,3,2,1]
    _check_convolution("aligned",x,y,xp,yp,dx=1)
    _check_convolution("unaligned",x,y,_dense(xp)-0.2000003,yp,dx=1)
    _check_convolution("wide",x,y,xp,yp,dx=2)
    _check_convolution("super wide",x,y,xp,yp,dx=10)

def _check_convolution(name,x,y,xp,yp,dx):
    ystar = convolve_sampled(x,y,xp,yp,x,dx=numpy.ones_like(x)*dx)

    xp = numpy.array(xp)*dx
    step = 0.0001
    xpfine = numpy.arange(xp[0],xp[-1]+step/10,step)
    ypfine = numpy.interp(xpfine,xp,yp)
    # make sure xfine is wide enough by adding a couple of extra steps
    # at the end
    xfine = numpy.arange(x[0]+xpfine[0],x[-1]+xpfine[-1]+2*step,step)
    yfine = numpy.interp(xfine,x,y,left=0,right=0)
    pidx = numpy.searchsorted(xfine, numpy.array(x)+xp[0])
    left,right = numpy.searchsorted(xfine, [x[0],x[-1]])

    conv = []
    for pi in pidx:
        norm_start = max(0, left-pi)
        norm_end = min(len(xpfine), right-pi)
        norm = step*numpy.sum(ypfine[norm_start:norm_end])
        conv.append(step*numpy.sum(ypfine*yfine[pi:pi+len(xpfine)])/norm)

    #print("checking convolution %s"%(name,))
    #print(" ".join("%7.4f"%yi for yi in ystar))
    #print(" ".join("%7.4f"%yi for yi in conv))
    assert all(abs(yi-fi) < 0.0005 for (yi,fi) in zip(ystar,conv))

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
        >>> print(numpy.isinf(indfloat('inf')))
        True
        >>> print(numpy.isinf(indfloat('-inf')))
        True
        >>> print(numpy.isnan(indfloat('nan')))
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


if __name__ == "__main__":
    test_convolve_sampled()
