#!/usr/bin/env python
"""
Run each fitter on the 3 dimensional Rosenbrock function to make sure they
all converge.
"""

import sys
import os
from os.path import join as joinpath
import tempfile
import shutil
import glob
import subprocess
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

# Ask bumps for a list of available fitters
ROOT = Path(__file__).absolute().parent
EXAMPLEDIR = ROOT / "doc" / "examples"

# Add the build dir to the system path
packages = [str(ROOT)]
if "PYTHONPATH" in os.environ:
    packages.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(packages)

# Need bumps on the path to pull in the available fitters
sys.path.insert(0, str(ROOT))
from bumps.fitters import FIT_AVAILABLE_IDS


command = [sys.executable, "-m", "bumps"]


def decode(b):
    return b.decode("utf-8")


def run_fit(fit_args, model_args, store, seed=1):
    command_parts = [*command, *fit_args, *model_args, f"--store={store}", f"--seed={seed}", "--batch"]
    try:
        output = subprocess.check_output(command_parts, stderr=subprocess.STDOUT)
        output = decode(output.strip())
        if output:
            print(output)
    except subprocess.CalledProcessError as exc:
        output = decode(exc.output.strip())
        if output:
            print(output)
        if "KeyboardInterrupt" in output:
            raise KeyboardInterrupt()
        else:
            raise RuntimeError("fit failed:\n" + " ".join(command_parts))


def check_fit(fitter, store, targets):
    errfiles = glob.glob(joinpath(store, "*.err"))
    if not errfiles:
        raise ValueError("error in %s: no err file created" % fitter)
    elif len(errfiles) > 1:
        raise ValueError("error in %s: too many err files created" % fitter)
    model_index = 0
    with open(errfiles[0]) as fid:
        for line in fid:
            if line.startswith("[overall chisq="):
                if line[15:10].lower() == "inf":
                    value = np.inf
                else:
                    value = float(line[15:].split("(")[0])
                assert abs(value - targets[model_index]) < 1e-2, "error in %s: expected %.3f but got %.3f" % (
                    fitter,
                    targets[model_index],
                    value,
                )
                model_index += 1
    assert model_index == len(targets), "error in %s: not enough models found" % fitter


def run_fits(model_args, path, fitters=FIT_AVAILABLE_IDS, seed=1, target=0):
    failed = []
    for f in fitters:
        print(f"====== fitter: {f}")
        try:
            store = Path(path) / f"{f}.hdf"
            run_fit([f"--fit={f}"], model_args, str(store), seed=seed)
            # check_fit(f, store, [target])
        except Exception as exc:
            # import traceback; traceback.print_exc()
            print(str(exc))
            failed.append(f)
    return failed


def main():
    fitters = sys.argv[1:] if len(sys.argv) > 1 else FIT_AVAILABLE_IDS
    # TODO: use a test function that defines residuals
    test_functions = joinpath(EXAMPLEDIR, "test_functions", "model.py")
    # model_args = [test_functions, '"fk(rosenbrock, 3)"']
    model_args, target = [test_functions, "gauss", "3"], 0
    model_args, target = [joinpath(EXAMPLEDIR, "curvefit", "curve.py")], 1.760
    seed = 1
    with tempfile.TemporaryDirectory() as path:
        failed = run_fits(model_args, path, fitters=fitters, seed=seed, target=target)
    if failed:
        print("======")
        print("Fits failed for: %s" % (", ".join(failed),))
        sys.exit(1)


if __name__ == "__main__":
    main()
