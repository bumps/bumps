#!/bin/bash

# Definitions
OUTPUT="conda_packed.tar"
SCRIPT_DIR="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"
SRC_DIR="$(dirname "$SCRIPT_DIR")"

eval "$(conda shell.bash hook)"

if ! command -v conda-pack &> /dev/null; then
  conda install -y conda-pack
fi

ISOLATED_ENV="$(mktemp -d)/env"
conda create -y -p "$ISOLATED_ENV" "python=${PYTHON_VERSION:-3.12}" "nodejs" "micromamba" "pip"

cd "$SCRIPT_DIR"
conda activate "$ISOLATED_ENV"

python -m pip install --no-input --no-compile "..[webview]"
python -m "${PACKAGE_NAME:-bumps}.webview.build_client" --cleanup --install-dependencies

conda deactivate
conda-pack -p "$ISOLATED_ENV" --format=no-archive --output="$SRC_DIR/$OUTPUT"

rm -rf "$ISOLATED_ENV"
# done
