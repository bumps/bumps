#!/usr/bin/env python
import sys, os

sys.dont_write_bytecode = True

examples = [
    "peaks/model.py",
    ]

ROOT = os.path.abspath(os.path.dirname(__file__))
EXAMPLEDIR = os.path.join(ROOT,'doc','examples')
if os.name == 'nt':
    os.environ['PYTHONPATH'] = ROOT+";"+ROOT+"/dream"
else:
    os.environ['PYTHONPATH'] = ROOT+":"+ROOT+"/dream"
PYTHON = sys.executable
CLI = "%s %s/bin/bumps %%s %%s"%(PYTHON,ROOT)

class Commands(object):
    @staticmethod
    def preview(f):
        os.system(CLI%(f,'--preview --seed=1'))

    @staticmethod
    def edit(f):
        os.system(CLI%(f,'--edit --seed=1'))

    @staticmethod
    def chisq(f):
        return os.system(CLI%(f,'--chisq --seed=1'))

def main():
    if len(sys.argv) == 1 or not hasattr(Commands, sys.argv[1]):
        print("usage: check_examples.py [preview|edit|chisq]")
    else:
        command = getattr(Commands, sys.argv[1])
        for f in examples:
            print("Example %s"%f)
            command(os.path.join(EXAMPLEDIR,f))

if __name__ == "__main__": main()
