#!/usr/bin/env python
"""
Run each fitter on the 3 dimensional Rosenbrock function to make sure they
all converge.
"""
from __future__ import print_function

import sys
import os
import tempfile
import shutil
import glob
from os.path import join as joinpath, realpath, dirname
import traceback
import subprocess

sys.dont_write_bytecode = True

try:
    bytes
    def decode(b):
        return b.decode('utf-8')
except Exception:
    def decode(b):
        return b

# Ask bumps for a list of available fitters
ROOT = realpath(dirname(__file__))
sys.path.insert(0, ROOT)
from bumps.fitters import FIT_AVAILABLE_IDS

RUNPY = joinpath(ROOT, 'run.py')
EXAMPLEDIR = joinpath(ROOT, 'doc', 'examples')

def clear_directory(path, recursive=False):
    """
    Remove all regular files in a directory.

    If *recursive* is True, removes subdirectories as well.

    This does not remove the directory itself.  Use *shutil.rmtree* if
    you want to delete the entire tree.
    """
    for f in os.listdir(path):
        target = joinpath(path, f)
        if not os.path.isdir(target):
            os.unlink(target)
        elif recursive:
            clear_directory(target, recursive)
            os.rmdir(target)

def run_fit(fit_args, model_args, store, seed=1):
    command_parts = ([sys.executable, RUNPY] + fit_args + model_args
                     + ['--store='+store, '--seed=%d'%seed, '--batch'])
    try:
        output = subprocess.check_output(command_parts, stderr=subprocess.STDOUT)
        output = decode(output.strip())
        if output: print(output)
    except subprocess.CalledProcessError as exc:
        output = decode(exc.output.strip())
        if output: print(output)
        if "KeyboardInterrupt" in output:
            raise KeyboardInterrupt()
        else:
            raise RuntimeError("fit failed:\n" + " ".join(command_parts))

def check_fit(fitter, store, targets):
    errfiles = glob.glob(joinpath(store, "*.err"))
    if not errfiles:
        raise ValueError("error in %s: no err file created"%fitter)
    elif len(errfiles) > 1:
        raise ValueError("error in %s: too many err files created"%fitter)
    model_index = 0
    with open(errfiles[0]) as fid:
        for line in fid:
            if line.startswith("[chisq="):
                value = float(line[7:].split("(")[0])
                assert abs(value-targets[model_index]) < 1e-2, \
                    "error in %s: expected %.3f but got %.3f" \
                    % (fitter, targets[model_index], value)
                model_index += 1
    assert model_index == len(targets), \
        "error in %s: not enough models found"%fitter


def run_fits(model_args, store, fitters=FIT_AVAILABLE_IDS, seed=1):
    failed = []
    for f in fitters:
        print("====== fitter: %s"%f)
        try:
            run_fit(["--fit="+f], model_args, store, seed=seed)
            check_fit(f, store, [0.0])
        except Exception as exc:
            #import traceback; traceback.print_exc()
            print(str(exc))
            failed.append(f)
        clear_directory(store)
    return failed

def main():
    fitters = sys.argv[1:] if len(sys.argv) > 1 else FIT_AVAILABLE_IDS
    store = tempfile.mkdtemp(prefix="bumps-test-")
    model = joinpath(EXAMPLEDIR, "test_functions", "model.py")
    #model_args = [model, '"fk(rosenbrock, 3)"']
    model_args = [model, 'gauss', '3']
    seed = 1
    failed = run_fits(model_args, store, fitters=fitters, seed=seed)
    shutil.rmtree(store)
    if failed:
        print("======")
        print("Fits failed for: %s"%(", ".join(failed),))
        sys.exit(1)

if __name__ == "__main__":
    main()
