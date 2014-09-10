"""
Data handling utilities.
"""
from __future__ import division

__all__ = ["indfloat", "parse_file"]

import numpy as np
from numpy import inf, nan


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
    we replace the bin edges with bin centers.
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
        columns, key, value = _parse_line(line)
        if columns:
            data.append([indfloat(v) for v in columns])
        if key:
            if key in header:
                header[key] = "\n".join((header[key], value))
            else:
                header[key] = value
    if fh is not file:
        fh.close()
    # print data
    # print "\n".join(k+":"+v for k,v in header.items())
    if len(data[-1]) == 1:
        # For TOF data, the first column is the bin edge, which has one
        # more row than the remaining columns; fill those columns with
        # bin centers instead
        last_edge = data[-1][0]
        data = np.array(data[:-1]).T
        edges = np.hstack((data[0],last_edge))
        data[0] = 0.5*(edges[:-1] + edges[1:])
    else:
        data = np.array(data).T

    return header, data


def string_like(s):
    """
    Return True if s operates like a string.
    """
    try:
        s + ''
    except:
        return False
    return True


def _parse_line(line):
    # Check if line contains comment character
    idx = line.find('#')
    if idx < 0:
        return line.split(), None, ''

    # If comment is after data, ignore the comment
    if idx > 0:
        return line[:idx].split(), None, ''

    # Check if we have '# key value'
    line = line[1:].strip()
    idx = line.find(' ')  # should also check for : and =
    if idx < 0:
        return [], None, None

    # Separate key and value
    key = line[:idx]
    value = line[idx + 1:].lstrip()

    # If key is a number, it is simply a commented out data point
    if key[0] in '.-+0123456789':
        return [], None, None

    # Strip matching quotes off the value
    if (value[0] in ("'", '"')) and value[-1] is value[0]:
        value = value[1:-1]

    return [], key, value


def indfloat(s):
    """
    Convert string to float, with support for inf and nan.

    Example::

        >>> from numpy import isinf, isnan
        >>> print(isinf(indfloat('inf')))
        True
        >>> print(isinf(indfloat('-inf')))
        True
        >>> print(isnan(indfloat('nan')))
        True
    """
    try:
        return float(s)
    except:
        s = s.lower()
        if s == 'inf':
            return inf
        if s == '-inf':
            return -inf
        if s == 'nan':
            return nan
        raise

