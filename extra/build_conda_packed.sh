#!/bin/bash

# Definitions
ENV_NAME="isolated-base"
PYTHON_VERSION="3.12"
OUTPUT="conda_packed"
SCRIPT_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
SRC_DIR=$(dirname "$SCRIPT_DIR")
pkgdir="$SRC_DIR/$OUTPUT"

eval "$(conda shell.bash hook)"

if ! test -f "$ENV_NAME.tar.gz"; then
  conda install -y conda-pack
  echo "creating isolated environment"
  conda remove -n "$ENV_NAME" -y --all
  conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION" "nodejs" "micromamba"
  conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"
fi

# unpack the new environment, that contains only python + pip
# first, clean out the packed folder:
rm -rf "$pkgdir"
mkdir -p "$pkgdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$pkgdir"

# add any icons
mkdir -p $pkgdir/share/icons
cp $SCRIPT_DIR/*.svg $pkgdir/share/icons
cp $SCRIPT_DIR/*.png $pkgdir/share/icons
cp $SCRIPT_DIR/*.ico $pkgdir/share/icons
cp $SCRIPT_DIR/*.icns $pkgdir/share/icons

cd $SRC_DIR
conda activate $pkgdir
python -m pip install --no-input --no-compile .[webview]

# build the client
cd $pkgdir
python -m bumps.webview.build_client --cleanup

# done
