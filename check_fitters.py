#!/usr/bin/env python
"""
Run each fitter on the 3 dimensional Rosenbrock function to make sure they
all converge.
"""

import sys
import os
from os.path import join as joinpath
import tempfile
import subprocess
from pathlib import Path
import h5py

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


def check_fit(fitter, store, target):
    """
    Verify overall chisq value matches target within 1% for all fitters.
    """
    with h5py.File(store) as fd:
        group = fd["problem_history"]
        last_item = list(group.keys())[-1]
        chisq_str = group[last_item].attrs["chisq"]
        value = float(chisq_str.split("(")[0])
        assert abs(value - target) / target < 1e-2, f"error in {fitter}: expected {target} but got {value}"


def run_fits(model_args, path, fitters=FIT_AVAILABLE_IDS, seed=1, target=0):
    failed = []
    for f in fitters:
        print(f"====== fitter: {f}")
        try:
            store = Path(path) / f"{f}.hdf"
            run_fit([f"--fit={f}"], model_args, str(store), seed=seed)
            check_fit(f, store, target)
        except Exception as exc:
            # import traceback; traceback.print_exc()
            print(str(exc))
            failed.append(f)
    return failed


def main():
    # Note: bumps.fitters.test_fitters already runs curvefit on the "active" fitters
    fitters = sys.argv[1:] if len(sys.argv) > 1 else FIT_AVAILABLE_IDS
    # TODO: use a test function that defines residuals
    test_functions = EXAMPLEDIR / "test_functions" / "model.py"
    # model_args = [test_functions, '"fk(rosenbrock, 3)"']
    model_args, target = [test_functions, "gauss", "3"], 0
    model_args, target = [EXAMPLEDIR / "curvefit" / "curve.py"], 1.760
    seed = 1
    with tempfile.TemporaryDirectory() as path:
        failed = run_fits(model_args, path, fitters=fitters, seed=seed, target=target)
    if failed:
        print("======")
        print("Fits failed for: %s" % (", ".join(failed),))
        sys.exit(1)


if __name__ == "__main__":
    main()
