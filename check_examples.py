#!/usr/bin/env python
import sys
import os
from pathlib import Path
import subprocess

sys.dont_write_bytecode = True

ROOT = Path(__file__).absolute().parent
EXAMPLEDIR = ROOT / "doc" / "examples"

# Add the build dir to the system path
packages = [str(ROOT)]
if "PYTHONPATH" in os.environ:
    packages.extend(os.environ["PYTHONPATH"].split(os.pathsep))
os.environ["PYTHONPATH"] = os.pathsep.join(packages)
# print("Environment:", os.environ["PYTHONPATH"])


command = [sys.executable, "-m", "bumps"]


class Commands(object):
    @staticmethod
    def edit(f):
        args = "--seed=1 --edit".spit()
        return subprocess.run([*command, f, *args]).returncode

    @staticmethod
    def chisq(f):
        # TODO: fix relative imports from model file
        # Add model directory to the path
        os.environ["PYTHONPATH"] = os.pathsep.join([*packages, str(f.parent)])
        args = "--seed=1 --chisq".split()
        return subprocess.run([*command, f, *args]).returncode

    @staticmethod
    def time(f):
        raise NotImplementedError("model timer no longer available")
        # Note: use --parallel=0 to check parallel execution time
        args = "--seed=1 --chisq --parallel=1 --time-model --steps=20".split()
        return subprocess.run([*command, f, *args]).returncode


examples = [
    EXAMPLEDIR / "peaks" / "model.py",
    EXAMPLEDIR / "curvefit" / "curve.py",
    EXAMPLEDIR / "constraints" / "inequality.py",
    EXAMPLEDIR / "test_functions" / "anticor.py",
]


def main():
    opt = sys.argv[1][2:] if len(sys.argv) == 2 else "chisq"
    command = getattr(Commands, opt, None)
    if command is None:
        print("usage: check_examples.py [--edit|--chisq|--time]")
        return

    for f in examples:
        print(f"\n{f}")
        if command(f) != 0:
            # break
            sys.exit(1)  # example failed
            pass


if __name__ == "__main__":
    main()
