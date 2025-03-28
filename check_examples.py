#!/usr/bin/env python
import sys
import os
from pathlib import Path

sys.dont_write_bytecode = True

BUMPS = f'"{sys.executable}" -m bumps'
ROOT = Path(__file__).absolute().parent
EXAMPLEDIR = ROOT / "doc" / "examples"

# Add the build dir to the system path
packages = [str(ROOT)]
if "PYTHONPATH" in os.environ:
    packages.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(packages)


class Commands(object):
    @staticmethod
    def edit(f):
        return os.system(f'{BUMPS} "{f}" --seed=1 --edit')

    @staticmethod
    def chisq(f):
        # print("Running the following:", f'{BUMPS} "{f}" --seed=1 --chisq')
        return os.system(f'{BUMPS} "{f}" --seed=1 --chisq')

    @staticmethod
    def time(f):
        raise NotImplementedError("model timer no longer available")
        # Note: use --parallel=0 to check parallel execution time
        return os.system(f'{BUMPS} "{f}" --seed=1 --chisq --parallel=1 --time-model --steps=20')


examples = [
    EXAMPLEDIR / "peaks" / "model.py",
    EXAMPLEDIR / "curvefit" / "curve.py",
    EXAMPLEDIR / "constraints" / "inequality.py",
    EXAMPLEDIR / "test_functions" / "anticor.py",
]


def main():
    opt = sys.argv[1][2:] if len(sys.argv) == 2 else "help"
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
