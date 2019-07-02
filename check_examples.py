#!/usr/bin/env python
import sys
import os

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.dirname(__file__))
CLI = "%s %s/bin/bumps %%s %%s" % (sys.executable, ROOT)
EXAMPLEDIR = os.path.join(ROOT, 'doc', 'examples')

# Add the build dir to the system path
packages = [ROOT]
if 'PYTHONPATH' in os.environ:
    packages.append(os.environ['PYTHONPATH'])
os.environ['PYTHONPATH'] = os.pathsep.join(packages)


class Commands(object):

    @staticmethod
    def preview(f):
        return os.system(CLI % (f, '--preview --seed=1'))

    @staticmethod
    def edit(f):
        return os.system(CLI % (f, '--edit --seed=1'))

    @staticmethod
    def chisq(f):
        return os.system(CLI % (f, '--chisq --seed=1'))

examples = [
    "peaks/model.py",
    "curvefit/curve.py",
    "constraints/inequality.py",
    "constraints/gmodel.py",
    "test_functions/anticor.py",
    #"test_functions/model.py",
]


def main():
    if len(sys.argv) == 1 or not hasattr(Commands, sys.argv[1][2:]):
        print("usage: check_examples.py [--preview|--edit|--chisq]")
    else:
        command = getattr(Commands, sys.argv[1][2:])
        for f in examples:
            print("\n" + f)
            if command(os.path.join(EXAMPLEDIR, f)) != 0:
                # break
                pass

if __name__ == "__main__":
    main()
