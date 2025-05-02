"""
Print a summary of an hdf file to the console.
"""

import sys

import numpy as np
import h5py as h5

# Typing support
from typing import Iterator, Union
from pathlib import Path


def _str(s):
    if hasattr(s, "dtype") and s.dtype.kind == "V":
        return s.tobytes().rstrip(b"\x00").decode("ascii")
    return s.decode("utf-8") if isinstance(s, bytes) else s


def summarystr(group, indent=0, show_attrs=True, recursive=True):
    # type: (h5.Group, int, bool, bool) -> str
    """
    Return the structure of the HDF 5 tree as a string.

    *group* is the starting group.

    *indent* is the indent for each line.

    *show_attrs* is False if attributes should be hidden

    *recursive* is False to show only the current level of the tree.
    """
    return "\n".join(_tree_format(group, indent, show_attrs, recursive))


def summary(group, indent=0, show_attrs=True, recursive=True):
    # type: (h5.Node, int, bool, bool) -> None
    """
    Print the structure of an HDF5 tree.

    *group* is the starting group.

    *indent* is the indent for each line.

    *show_attrs* is False if attributes should be hidden

    *recursive* is False to show only the current level of the tree.
    """
    for s in _tree_format(group, indent, show_attrs, recursive):
        print(s)


def _tree_format(node, indent, show_attrs, recursive):
    # type: (h5.Node, int, bool, bool) -> Iterator[str]
    """
    Return an iterator for the lines in a formatted HDF5 tree.

    Individual lines are not terminated by newline.
    """
    # Find fields and subgroups within the group; do this ahead of time
    # so that we can show all fields before any subgroups.
    items = [(name, hasattr(child, "dtype")) for (name, child) in node.items()]
    groupnames = [name for (name, isdata) in items if not isdata]
    datasets = [name for (name, isdata) in items if isdata]

    # Yield group as "nodename(nxclass)"
    yield "".join((" " * indent, _group_str(node)))

    # Yield group attributes as "  @attr: value"
    indent += 2
    if show_attrs:
        for s in _yield_attrs(node, indent):
            yield s

    # Yield fields as "  field[NxM]: value"
    for fieldname in sorted(datasets):
        field = node[fieldname]

        # Short circuit links
        path = "/".join((node.name, fieldname))
        if "target" in field.attrs and _str(field.attrs["target"]) != path:
            yield "".join((" " * indent, fieldname, " -> ", _str(field.attrs["target"])))
            continue

        # Format field dimensions
        # print fieldname,field,field.shape,field.dtype,field.attrs['format']
        ndim = len(field.shape)
        if ndim > 1 or (ndim == 1 and field.shape[0] > 1):
            shape = "[" + "x".join(str(dim) for dim in field.shape) + "]"
        else:
            shape = ""
        # shape = '['+'x'.join( str(dim) for dim in field.shape)+']'+str(field.dtype)

        # Format string or numeric value
        # if 'S' in field.attrs['format']:
        if field.dtype.kind in ("S", "O", "V"):
            if ndim == 0:
                # raise ValueError("zero dimensions on string?")
                value = _limited_str(_str(field[()]))
            elif ndim == 1:
                if field.shape[0] == 1:
                    value = _limited_str(_str(field[0]))
                else:
                    values = field[:] if field.shape[0] <= 5 else field[:5]
                    values = [_limited_str(_str(v), width=10) for v in values]
                    if field.shape[0] > 5:
                        values.append("...")
                    value = ", ".join(values)
                value = "[" + value + "]"
            else:
                value = "[[...]]"
        elif field.dtype.kind in ("V"):
            value = f"unknown type {field.dtype.kind}"
        else:
            size = np.prod(field.shape)
            if ndim == 0:
                value = "%g" % field[()]
            elif ndim == 1:
                if size == 0:
                    value = ""
                elif size == 1:
                    value = "%g" % field[0]
                elif size <= 6:
                    value = " ".join("%g" % v for v in field[:])
                else:
                    value = " ".join("%g" % v for v in field[:6]) + " ..."
                value = "[" + value + "]"
            else:
                if size == 0:
                    value = ""
                elif field.shape[-1] <= 6:
                    x = field[:].flatten()[: field.shape[-1]]
                    value = " " + " ".join("%g" % v for v in x)
                else:
                    # print('testing', fieldname,field[:])
                    y = field[:].flatten()[:6]
                    value = " " + " ".join("%g" % v for v in y) + " ..."
                value = "[[" + value + "], ...]"

        dtype = " " + str(field.dtype)

        units = " " + field.attrs.get("units", "") if not show_attrs else ""
        # Maybe using Angstroms in units
        try:
            units = units.decode("UTF-8")
        except AttributeError:
            pass
        except UnicodeDecodeError:
            units = units.decode("ISO-8859-1")

        try:
            value = value.decode("UTF-8")
        except AttributeError:
            pass
        except UnicodeDecodeError:
            value = value.decode("ISO-8859-1")

        # Yield field: value
        yield "".join((" " * indent, fieldname, shape, dtype, units, ": ", value))

        # Yield attributes
        if show_attrs:
            for s in _yield_attrs(field, indent + 2):
                yield s

    # Yield groups.
    # If recursive, show group details, otherwise just show name.
    if recursive:
        for groupname in sorted(groupnames):
            group = node[groupname]

            # Short circuit links
            path = "/".join((node.name, groupname))
            if "target" in group.attrs and _str(group.attrs["target"]) != path:
                yield "".join((" " * indent, groupname, " -> ", _str(group.attrs["target"])))
                continue
            for s in _tree_format(group, indent, show_attrs, recursive):
                yield s
    else:
        for groupname in sorted(groupnames):
            group = node[groupname]

            # Short circuit links
            path = "/".join((node.name, groupname))
            if "target" in group.attrs and _str(group.attrs["target"]) != path:
                yield "".join((" " * indent, groupname, " -> ", _str(group.attrs["target"])))
                continue
            yield "".join((" " * indent, _group_str(node[g])))


def _yield_attrs(node, indent):
    # type: (Union[h5.Group, h5.Dataset], int) -> Iterator[str]
    """
    Iterate over the attribute values of the node, excluding NX_class.
    """
    # print "dumping",node.name,"attrs",node.attrs.keys()
    for k in sorted(node.attrs.keys()):
        if k not in ("NX_class", "target", "binary", "byteorder", "dtype", "format", "shape"):
            v = _str(node.attrs[k])
            vstr = _limited_str(v) if isinstance(v, str) else str(v)
            yield "".join((" " * indent, "@", k, ": ", vstr))


def _group_str(node):
    # type: (h5.Group) -> str
    """
    Return the name and nexus class of a node.
    """
    if node.name == "/":
        return "root"
    nxclass = "(" + _str(node.attrs["NX_class"]) + ")" if "NX_class" in node.attrs else ""
    return node.name.split("/")[-1] + nxclass


def _limited_str(s, width=60):
    # type: (str, int) -> str
    """
    Returns the string trimmed to a maximum of one line of width+3 characters,
    with ... substituted for any trimmed characters.  Leading and trailing
    blanks are removed.
    """
    if "\n" in s:
        ret = s.split("\n")[0]
        if len(ret) > width:
            ret = ret[:width] + "..."
        ret += "\\n..."
    else:
        ret = s if len(s) < width else s[:width] + "..."
    # If it is a string that looks like a float then wrap it in quotes.
    try:
        float(ret)
        ret = f'"{ret}"'
    except ValueError:
        ...
    return ret


def summarize_file(filename: Union[Path, str], show_attrs: bool = True, indent: int = 0):
    print(f"=== {filename} ===")
    with h5.File(filename, "r") as fd:
        summary(fd, indent=indent, show_attrs=show_attrs, recursive=True)


def main():
    """
    Command line interface to file dump.
    """
    # TODO: use option parser
    if len(sys.argv) <= 1:
        print("Usage: {sys.argv[0]} file...")
        sys.exit(1)
    for filename in sys.argv[1:]:
        summarize_file(filename)


if __name__ == "__main__":
    main()
