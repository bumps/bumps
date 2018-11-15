"""
Redirect calls to open, etc. to a virtual file system.

Use this to mount a zip file as a file system, and then all subsequent calls
to chdir, open, etc. will reference files in the zip file instead of the disk.

This will only work for packages which do all their I/O in python, and not
those which use direct calls to the C library. Works with numpy.loadtxt.
Not tested with pandas.read_csv.  Will not work with h5py.

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

    __builtin__.open
    os.chdir
    os.getcwd
    os.path.exists
    os.path.isfile
    os.path.isdir
    os.path.abspath
    os.path.realpath

You can also use the file systems directly without using the :func:`vfs_init`
hook or the with statement.  Just call `fs.chdir`, etc. on the file system
object.
"""
from __future__ import print_function

import __builtin__
import os
import os.path

_open = __builtin__.open
_chdir = os.chdir
_getcwd = os.getcwd
_exists = os.path.exists
_isfile = os.path.isfile
_isdir = os.path.isdir

# TODO: maybe use builtin versions?
_abspath = os.path.abspath
_realpath = os.path.realpath

class RealFS:
    # os functions
    #   listdir, rmdir, mkdir, chroot
    #   rename, remove, unlink, stat, chmod, chown, chflags, access, link
    #   getcwdb, getcwdu  # bytes getcwd (py3), unicode getcwd (py2)
    #   -makedirs # uses exists, split, mkdir
    #   -removedirs # uses split, rmdir
    #   -walk # uses islink, join, isdir, listdir

    # symbolic link functions
    #   symlink, readlink, lchflags, lchmod, lchown, lstat
    #   os.path.lexists  # like exists, but True even when link is broken

    # os.path functions
    #   get[acm]time, getsize
    #   samefile  # uses stat, samestat; stat both and compare device/inode
    #   samestat  # compare device and inode fields of stat
    #   -isabs  # s startswith('/')
    #   -normpath # pure path manipulation
    #   -abspath # uses isabs, normpath, getcwd/getcwdu
    #   -realpath # uses isabs, split, join, islink, readlink
    #   -renames # uses exists, split, rename, makedirs, removedirs
    #   -walk  # uses listdir, lstat  (deprecated)

    # os/os.path constants
    #   curdir, pardir, sep, pathsep, defpath, extsep, altsep, linesep

    # file descriptor operations in os
    #   fchdir, fchmod, fchown, fdopen, close, fstat, fstatvfs, fpathconf,
    #   lseek, read, dup, dup2, errno, error, closerange, isatty, openpty,
    #   mknod

    def __enter__(self):
        pushfs(self)

    def __exit__(self, *args, **kw):
        popfs()

    def open(self, name, mode="r", buffering=True):
        return _open(name, mode=mode, buffering=buffering)

    def chdir(self, path):
        return _chdir(path)

    def getcwd(self):
        return _getcwd()

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
    Opens a zip file as the root file system.  On enter into the
    """
    def __init__(self, path):
        import zipfile
        self._path = os.path.realpath(path)
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

    def getcwd(self):
        return self._wd

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

FS = RealFS()
FS_STACK = []
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
    __builtin__.open = fs_open
    __builtin__.file = fs_open  # 2.7 only
    os.chdir = fs_chdir
    os.getcwd = fs_getcwd
    os.path.abspath = fs_abspath
    os.path.realpath = fs_realpath
    os.path.exists = fs_exists
    os.path.isfile= fs_isfile
    os.path.isdir = fs_isdir