#!/bin/sh

set -x

python setup.py build
python test.py
(cd doc && make html pdf)
# make sure the pdf got built by copying it to the current directory
cp doc/_build/latex/Bumps.pdf .
