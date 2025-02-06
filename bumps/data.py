"""
Data handling utilities.
"""

import os
import gzip
from contextlib import contextmanager

import numpy as np
from numpy import inf, nan

__all__ = ["indfloat", "parse_file"]


def parse_multi(file, keysep=None, sep=None, comment="#"):
    """
    Parse a multi-part file.

    Return a list of (header, data) pairs, where header is a key: value
    dictionary and data is a numpy array.

    The header section is list of key value pairs, with the *comment* character
    at the start of each line.  Key and value will be separated by *keysep*,
    or by spaces if *keysep = None*.  The data section is a sequence of
    floating point numbers separated by *sep*, or by spaces if *sep* is None.
    inf and nan are parsed as inf and nan.  Comments at the end of the data
    line will be ignored.  Data points can be commented out by including
    a comment character at the start of the data line, assuming the next
    character is a digit, plus, or decimal separator.

    Quotes around keys are removed, but not around values.  Use
    :func:`strip_quotes` to remove them if they are present.  This is different
    from the :func:`parse_file` interface, which strips quotes around values.
    The new interface allows *json.loads()* calls on values if values are
    stored as *key: json.dumps(value)*.

    Special hack for binned data: if the first column contains bin edges, then
    the last row will only have the bin edge.  To make the array square,
    we replace the bin edges with bin centers.  The original bins can be
    found in the header using the 'bins' key (unless that key already exists
    in the header, in which case the key will be ignored).
    """
    parts = []
    with maybe_open(file) as fh:
        while True:
            header, data, bins = _read_part(fh, comment=comment, multi_part=True, col_sep=sep, key_sep=keysep)
            if header is None:
                break
            if bins is not None:
                header.setdefault("bins", bins)
            parts.append((header, data))
    return parts


def parse_file(file, keysep=None, sep=None, comment="#"):
    """
    Parse a file into a header and data.

    Return a (header, data) pair, where header is a key: value
    dictionary and data is a numpy array.

    The header section is list of key value pairs, with the *comment* character
    at the start of each line.  Key and value will be separated by *keysep*,
    or by spaces if *keysep = None*.  The data section is a sequence of
    floating point numbers separated by *sep*, or by spaces if *sep* is None.
    inf and nan are parsed as inf and nan.  Comments at the end of the data
    line will be ignored.  Data points can be commented out by including
    a comment character at the start of the data line, assuming the next
    character is a digit, plus, or decimal separator.

    Quotes around keys are removed.  For compatibility with the old interface,
    quotes around values are removed as well.

    Special hack for binned data: if the first column contains bin edges, then
    the last row will only have the bin edge.  To make the array square,
    we replace the bin edges with bin centers.  The original bins can be
    found in the header using the 'bins' key (unless that key already exists
    in the header, in which case the key will be ignored).
    """
    with maybe_open(file) as fh:
        header, data, bins = _read_part(fh, comment=comment, multi_part=False, col_sep=sep, key_sep=keysep)
    if header is None:
        raise IOError("data file is empty")
    # compatibility: strip quotes from values in key-value pairs
    header = dict((k, strip_quotes(v)) for k, v in header.items())
    if bins is not None:
        header.setdefault("bins", bins)
    return header, data


def _read_part(fh, key_sep=None, col_sep=None, comment="#", multi_part=False):
    header = {}
    data = []
    iseof = True
    for line in fh:
        # Blank lines indicate a section break.
        if not line.strip():
            # Skip blank lines if we are parsing the data as a single part file
            # or if the "section" has no data due to blank lines in the header.
            if not multi_part or not data:
                continue
            # If we are at the beginning of a section, then iseof is True and
            # continuing to the next loop iteration will skip them. If we have
            # already consumed some non-blank lines, then iseof will be false,
            # and we need to break this section of the data.  If we have blank
            # lines at the end of the file, we will never set iseof to False
            # and they will be ignored.
            if iseof:
                continue
            break

        # Line is not blank, so process it.
        columns, key, value = _parse_line(line, comment=comment, col_sep=col_sep, key_sep=key_sep)
        if columns:
            data.append([indfloat(v) for v in columns])
        if key is not None:
            if key in header:
                header[key] = "\n".join((header[key], value))
            else:
                header[key] = value

        # We have processed some data, so
        iseof = False

    if iseof:
        return None, None, None

    # print data
    # print "\n".join(k+":"+v for k,v in header.items())
    if len(data) and len(data[-1]) == 1:
        # For TOF data, the first column is the bin edge, which has one
        # more row than the remaining columns; fill those columns with
        # bin centers instead
        last_edge = data[-1][0]
        data = np.array(data[:-1]).T
        edges = np.hstack((data[0], last_edge))
        data[0] = 0.5 * (edges[:-1] + edges[1:])
        bins = edges
    else:
        data = np.array(data).T
        bins = None

    return header, data, bins


@contextmanager
def maybe_open(file_or_path):
    """
    A context manager for file opening, given as a file path or an open handle.

    If *file_or_path* is a string ending in ".gz" then open with gzip.
    """
    if hasattr(file_or_path, "readline"):
        # If it is a file handle, yield it and return without closing.
        fh = file_or_path
        yield fh
    else:
        # Otherwise it should be a path. Make sure it is at least a string.
        if not string_like(file_or_path):
            raise ValueError("file must be a name or a file handle")
        # Open file; if name ends in .gz then assume it is compressed.
        path = os.path.expanduser(file_or_path)
        fh = gzip.open(path) if path.endswith(".gz") else open(path)
        try:
            yield fh
        finally:
            fh.close()


def string_like(s):
    """
    Return True if s operates like a string.
    """
    try:
        s + ""
    except Exception:
        return False
    return True


def _parse_line(line, key_sep=None, col_sep=None, comment="#"):
    # Find location of the comment character on the line
    idx = line.find(comment)

    # If the line does not contain a comment character or if the comment
    # character is not in the first column, then this is a data line which
    # should be returned as a sequence of text columns separated by spaces.
    # The caller can turn the columns into numbers or leave them as strings.
    # Data on the line after the comment character is ignored.
    # TODO: allow quoted strings or backslash escaped spaces for text columns
    if idx != 0:
        if idx > 0:
            return line[:idx].split(col_sep), None, ""
        else:
            return line.split(col_sep), None, ""

    # Split line on key separator
    parts = [p.strip() for p in line[1:].split(key_sep, 1)]
    key, value = parts if len(parts) > 1 else (parts[0], "")
    key = strip_quotes(key)

    # If key is a number assume it is simply a commented out data point
    if len(key) and (key[0] in ".-+0123456789" or key == "inf" or key == "nan"):
        return [], None, None

    return [], key, value


def strip_quotes(s):
    return s[1:-1] if len(s) and s[0] in "'\"" and s[0] == s[-1] else s


INF_VALUES = set(("inf", "1/0", "1.#inf", "infinity"))
NAN_VALUES = set(("nan", "0/0", "1.#qnan", "na", "n/a"))


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
    except Exception:
        s = s.lower()
        if s in INF_VALUES:
            return inf
        elif s and s[0] == "-" and s[1:] in INF_VALUES:
            return -inf
        elif s in NAN_VALUES:
            return nan
        raise
