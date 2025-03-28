#!/usr/bin/env python
import sys
import os

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.dirname(__file__))
CLI = "%s %s/bin/bumps %%s %%s" % (sys.executable, ROOT)
BUMPS = f'"{sys.executable}" -m bumps'
EXAMPLEDIR = os.path.join(ROOT, "doc", "examples")

# Add the build dir to the system path
packages = [ROOT]
if "PYTHONPATH" in os.environ:
    packages.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(packages)


class Commands(object):
    @staticmethod
    def edit(f):
        return os.system(f'{BUMPS} "{f}" --seed=1 --edit')

    @staticmethod
    def chisq(f):
        print("Running the following:", f'{BUMPS} "{f}" --seed=1 --chisq')
        return os.system(f'{BUMPS} "{f}" --seed=1 --chisq')

    @staticmethod
    def time(f):
        raise NotImplementedError("model timer no longer available")
        # Note: use --parallel=0 to check parallel execution time
        return os.system(f'{BUMPS} "{f}" --seed=1 --chisq --parallel=1 --time-model --steps=20')


examples = [
    "peaks/model.py",
    "curvefit/curve.py",
    "constraints/inequality.py",
    "test_functions/anticor.py",
]


def main():
    opt = sys.argv[1][2:] if len(sys.argv) == 2 else "help"
    command = getattr(Commands, opt, None)
    if command is None:
        print("usage: check_examples.py [--edit|--chisq|--time]")
        return

    for f in examples:
        print("\n" + f)
        if command(os.path.join(EXAMPLEDIR, f)) != 0:
            # break
            sys.exit(1)  # example failed
            pass


if __name__ == "__main__":
    main()
