#!/bin/sh

# TODO: won't work on windows without a bash environment
# Convert to python using papermill.execute_notebook(source, dest)
cd doc/notebooks
for nb in *.ipynb; do
    echo "Checking $nb"
    papermill "$nb" "/tmp/$nb" || exit 1
done
