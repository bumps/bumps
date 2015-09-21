"""
Generate tutorial docs from the pylit examples directory.

Drop this file in your sphinx doc directory, and change the constants at
the head of this file as appropriate.  Add pylit.py as well. Make sure this
file is on the python path and add the following to the end of conf.py::

    import gentut
    gentut.make()

You may want to change SOURCE_PATH and TARGET_PATH.  Be sure to exclude
the source path using exclude_trees directive in conf.py.
"""

SOURCE_PATH = "examples"
TARGET_PATH = "tutorial"

# =======================================================================
from os.path import join as joinpath, isdir, basename, getmtime, exists
from os import makedirs
from glob import glob
from shutil import copyfile
import pylit

# CRUFT: python 2.x needs to convert unicode to bytes when writing to file
try:
    # python 2.x
    unicode
    def write(fid, s):
        fid.write(s)
except NameError:
    # python 3.x
    def write(fid, s):
        fid.write(s.encode('utf-8') if isinstance(s, str) else s)

def make():
    if not exists(TARGET_PATH):
        makedirs(TARGET_PATH)

    # table of contents
    index_source = joinpath(SOURCE_PATH, "index.rst")
    index_target = joinpath(TARGET_PATH, "index.rst")
    if newer(index_target, index_source):
        copyfile(index_source, index_target)

    # examples
    examples = (f for f in glob(joinpath(SOURCE_PATH,'*')) if isdir(f))
    for f in examples:
        #print "weaving directory",f
        weave(f, joinpath(TARGET_PATH, basename(f)))

def newer(file1, file2):
    return not exists(file1) or (getmtime(file1) < getmtime(file2))

def weave(source, target):
    if not exists(target):
        makedirs(target)
    for f in glob(joinpath(source,'*')):
        if f.endswith('__pycache__') or f.endswith('.pyc'):
            # skip python runtime droppings
            continue
        #print "processing",f
        if f.endswith(".py") and ispylit(f):
            rstfile = joinpath(target, basename(f).replace('.py','.rst'))
            pyclean = joinpath(target, basename(f))
            if newer(rstfile, f):
                # Convert executable literate file to rst file with embedded code
                pylit.main(["--codeindent=4", f, rstfile])
                attach_download(rstfile, basename(f))
                # Convert rst file with embedded code to clean code file
                pylit.main([rstfile, pyclean, "-s", "-t"])
        else:
            dest = joinpath(target, basename(f))
            if newer(dest, f):
                if f.endswith(".py"):
                    print("Warning: file %r is not a pylit file"%(f,))
                #print "copying",f,dest
                copyfile(f, dest)

def attach_download(rstfile, target):
    with open(rstfile, "ab") as fid:
        write(fid, "\n\n.. only:: html\n\n   Download: :download:`%s <%s>`.\n"%(target, target))

def ispylit(f):
    """
    Assume it is a pylit file if it starts with a comment and not a docstring.
    """
    with open(f) as fid:
        line = fid.readline()
        # skip shebang if present
        if line.startswith("#!"):
            line = fid.readline()
        # skip blank lines
        while line != "" and line.strip() == "":
            line = fid.readline()

    return line.startswith("#")


if __name__ == "__main__":
    make()
