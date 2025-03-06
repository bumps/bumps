#!/bin/bash

# Definitions
ENV_NAME="isolated-base"
PYTHON_VERSION="3.12"
OUTPUT="conda_packed"
SCRIPT_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
SRC_DIR=$(dirname "$SCRIPT_DIR")
pkgdir="$SRC_DIR/$OUTPUT"

conda install -y conda-pack
if ! test -f "$ENV_NAME.tar.gz"; then
  echo "creating isolated environment"
  conda remove -n "$ENV_NAME" -y --all
  conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION" "nodejs" "micromamba"
  conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"
fi

# unpack the new environment, that contains only python + pip
rm -rf "$pkgdir"
envdir="$pkgdir/${PKGNAME:+$PKGNAME-}env"
mkdir -p "$envdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$envdir"

# add any icons
mkdir -p $pkgdir/share/icons
cp $SCRIPT_DIR/*.svg $OUTPUT/share/icons
cp $SCRIPT_DIR/*.png $OUTPUT/share/icons
cp $SCRIPT_DIR/*.ico $OUTPUT/share/icons

# base path to source is in parent of SCRIPT_DIR
conda activate $envdir
pushd $SRC_DIR
pip install --no-input --no-compile .[webview]

# build the client
cd $OUTPUT
python -m bumps.webview.build_client --cleanup

conda deactivate
popd
# done
