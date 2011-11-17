#!/usr/bin/env python
import sys, os

examples = [
    "distribution/dist-example.py",
    "ex1/nifilm-web.py",
    "ex1/nifilm-fit-web.py",
    "ex1/nifilm-data-web.py",
    "ex1/nifilm-tof-web.py",
    "freemag/pmf.py",
    "ill_posed/anticor.py",
    "ill_posed/tethered.py",
    "interface/model.py",
    "mixed/mixed-web.py",
    "mixed/mixed_magnetic.py",
    "peaks/model.py",
    "polymer/tethered-web.py",
    "polymer/freeform.py",
    "spinvalve/n101G.py",
    "staj/De2_VATR.py",
    "superlattice/freeform.py",
    "superlattice/NiTi-web.py",
    "superlattice/PEMU-web.py",
    "thick/nifilm.py",
    "TOF/du53.py",
    "xray/mlayer.staj",
    "xray/model.py",
    "xray/staj.py",
    ]

ROOT = os.path.abspath(os.path.dirname(__file__))
EXAMPLEDIR = os.path.join(ROOT,'doc','examples')
if os.name == 'nt':
    os.environ['PYTHONPATH'] = ROOT+";"+ROOT+"/dream"
else:
    os.environ['PYTHONPATH'] = ROOT+":"+ROOT+"/dream"
PYTHON = sys.executable
CLI = "%s %s/bin/refl1d_cli.py %%s %%s"%(PYTHON,ROOT)

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
        print "usage: check_examples.py [preview|edit|chisq]"
    else:
        command = getattr(Commands, sys.argv[1])
        for f in examples:
            print "Example",f
            command(os.path.join(EXAMPLEDIR,f))

if __name__ == "__main__": main()
