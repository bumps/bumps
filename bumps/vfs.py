"""
Redirect calls to open, etc. to a virtual file system.

Use this to mount a zip file as a file system, and then all subsequent calls
to chdir, open, etc. will reference files in the zip file instead of the disk.

This will only work for packages which do all their I/O in python, and not
those which use direct calls to the C library, so for example, it will not
work with h5py. XML parsing from a zip file is also unlikely to work since
expat uses the C library directly for parsing, but not tested.

Usage::

    # Do this before importing any other modules!  It sets up hooks for
    # redirecting filesystem access even if the module imports the symbol
    # directly as "from os import getcwd".
    import vfs
    vfs.vfs_init()
    ...
    with vfs.ZipFS('data.zip'):
        data = np.loadtxt('file1.dat')

Filesystems available:

* :class:`RealFS` - uses the builtin python functions to access the O/S.

* :class:`ZipFS` - opens a zip file as the filesystem root.

Calls redirected::

    builtins.open (and __builtin__.open in python 2)
    os.chdir
    os.getcwd
    os.listdir
    os.path.exists
    os.path.isfile
    os.path.isdir
    os.path.abspath
    os.path.realpath

You can also use the file systems directly without using the :func:`vfs_init`
hook or the with statement.  Just call `fs.chdir`, etc. on the file system
object.

*file* in python 2.x is a type as well as constructor, so a simple redirect
to a replacement constructor will not work. Don't try to support it since
it is gone in python 3.

Works with numpy.loadtxt on python 2 and python 3.
Does not work with pandas.
"""
from __future__ import print_function

import os
import os.path
import builtins

# for functions that work for read-only filesystems, use *fn
# for functions implemented as python, use -fn
#
# os functions
#   *chdir, *getcwd, *listdir, rmdir, mkdir, chroot
#   *open, *stat, *access, rename, link, unlink, remove chmod, chown, chflags
#   *getcwdb, *getcwdu  # bytes getcwd (py3), unicode getcwd (py2)
#   -makedirs   # uses exists, split, mkdir
#   -removedirs # uses split, rmdir
#   -walk       # uses islink, join, isdir, listdir

# symbolic link functions
#   *readlink, *lstat, symlink, lchflags, lchmod, lchown
#   *os.path.lexists

# os.path functions
#   *exists, *isfile, *isdir
#   *get[acm]time, *getsize
#   samefile    # uses stat, samestat
#   samestat    # uses filestat.dev and filestat.inode; compares device/inode
#   -isabs      # uses str; returns s.startswith('/')
#   -normpath   # uses str; pure path manipulation
#   -abspath    # uses isabs, normpath, getcwd/getcwdu
#   -realpath   # uses isabs, split, join, islink, readlink
#   -renames    # uses exists, split, rename, makedirs, removedirs
#   -walk       # uses listdir, lstat;  deprecated in favour of os.walk

# os/os.path constants
#   curdir, pardir, sep, pathsep, defpath, extsep, altsep, linesep

# file descriptor operations in os
#   fchdir, fchmod, fchown, fdopen, close, fstat, fstatvfs, fpathconf,
#   lseek, read, dup, dup2, errno, error, closerange, isatty, openpty,
#   mknod

_open = builtins.open
_chdir = os.chdir
_getcwd = os.getcwd
_exists = os.path.exists
_isfile = os.path.isfile
_isdir = os.path.isdir
_listdir = os.listdir

# TODO: maybe use builtin versions?
_abspath = os.path.abspath
_realpath = os.path.realpath

class RealFS(object):
    def __enter__(self):
        pushfs(self)

    def __exit__(self, *args, **kw):
        popfs()

    def open(self, name, mode="r", buffering=True):
        return _open(name, mode=mode, buffering=buffering)

    def getcwd(self):
        return _getcwd()

    def chdir(self, path):
        return _chdir(path)

    def listdir(self, path=None):
        return _listdir(path) if path is not None else _listdir()

    def abspath(self, path):
        return _abspath(path)

    def realpath(self, path):
        return _realpath(path)

    def isfile(self, path):
        return _isfile(path)

    def isdir(self, path):
        return _isdir(path)

    def exists(self, path):
        return _exists(path)

class ZipFS(object):
    """
    Opens a zip file as the root file system.
    """
    def __init__(self, path):
        import zipfile
        # TODO: can we open a zip within a zip?
        # Apparently yes, but only if we read the file into a byte stream and
        # then work from that file.  See the following stackoverflow answer:
        #    https://stackoverflow.com/questions/12025469/how-to-read-from-a-zip-file-within-zip-file-in-python
        self._path = _realpath(path)
        self._wd = "/"
        self._zip = zipfile.ZipFile(path)

    def __enter__(self):
        pushfs(self)
        return self._zip

    def __exit__(self, *args, **kw):
        popfs()

    def open(self, name, mode="r", buffering=True):
        if mode == 'rb':
            mode = 'r'
        with RealFS():
            return self._zip.open(self.abspath(name)[1:], mode)

    def chdir(self, name):
        if self.isdir(name):
            self._wd = self.abspath(name) + "/"

    def _iter_dir(self, path=None):
        path = self._wd if path is None else self.abspath(path)
        path = path[1:]
        n = len(path)
        seen = {}
        for f in self._zip.filelist:
            if not f.filename.startswith(path):
                # it is not part of the tree
                continue
            parts = f.filename.split('/')
            if len(parts) == 1:
                # it is a leaf so report it
                yield parts[0]
            elif parts[0] not in seen:
                # it is a directory, so only report it if it has not already
                # been reported
                seen.add(parts[0])
                yield parts[0]

    def listdir(self, path=None):
        return [f for f in self._iter_dir(path)]

    def abspath(self, name):
        if name[0] != '/':
            name = '/'.join((self._wd[:-1], name))
        return os.path.normpath(name)

    def realpath(self, name):
        return os.path.join(self._path, self.abspath(name))

    def isfile(self, name):
        return any(self.abspath(name)[1:] == f.filename for f in self._zip.filelist)

    def isdir(self, name):
        name = self.abspath(name)[1:] + "/"
        for f in self._zip.filelist:
            if f.filename.startswith(name):
                return True
        return False

    def exists(self, name):
        return self.isfile(name) or self.isdir(name)

# These will be initialized in vfs_init
FS = None  # type: RealFS
FS_STACK = None  # type: List[RealFS]
def pushfs(fs):
    global FS
    FS_STACK.append(FS)
    FS = fs

def popfs():
    global FS
    FS = FS_STACK.pop()

def fs_open(*args, **kw):
    return FS.open(*args, **kw)

def fs_chdir(*args, **kw):
    return FS.chdir(*args, **kw)

def fs_getcwd(*args, **kw):
    return FS.getcwd(*args, **kw)

def fs_listdir(*args, **kw):
    return FS.listdir(*args, **kw)

def fs_exists(*args, **kw):
    return FS.exists(*args)

def fs_isfile(*args, **kw):
    return FS.isfile(*args, **kw)

def fs_isdir(*args, **kw):
    return FS.isdir(*args, **kw)

def fs_abspath(*args, **kw):
    return FS.abspath(*args, **kw)

def fs_realpath(*args, **kw):
    return FS.realpath(*args, **kw)

def vfs_init():
    """
    Call this very early in your program so that various filesystem functions
    will be redirected even if they are expressed as "from module import fn"
    """
    global FS, FS_STACK
    FS = RealFS()
    FS_STACK = []

    builtins.open = fs_open
    os.chdir = fs_chdir
    os.getcwd = fs_getcwd
    os.listdir = fs_listdir
    os.path.abspath = fs_abspath
    os.path.realpath = fs_realpath
    os.path.exists = fs_exists
    os.path.isfile= fs_isfile
    os.path.isdir = fs_isdir

    try: # CRUFT: python 2.x
        import __builtin__
        __builtin__.open = fs_open
    except ImportError:
        pass
