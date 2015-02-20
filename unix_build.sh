#!/bin/sh

set -x

python setup.py build
python test.py
(cd doc && make html pdf)
cp doc/_build/latex/bumps.pdf .
