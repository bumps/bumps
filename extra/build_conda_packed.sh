#!/bin/bash

ENV_NAME="isolated-base"
PYTHON_VERSION="3.10"
PKGNAME="bumps"
SUBNAME="packed"
OUTPUT="artifacts"
SCRIPT_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
SRC_DIR=$(dirname "$SCRIPT_DIR")

mkdir -p $OUTPUT

eval "$(conda shell.bash hook)"
conda activate base || { echo 'failed: conda not installed'; exit 1; }

conda install -y conda-pack
if ! test -f "$ENV_NAME.tar.gz"; then
  echo "creating isolated environment"
  conda remove -n "$ENV_NAME" -y --all
  conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION" "nodejs"
  conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"
fi

# unpack the new environment, that contains only python + pip
tmpdir=$(mktemp -d)
pkgdir="$tmpdir/$PKGNAME"
envdir="$pkgdir/env"
mkdir -p "$envdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$envdir"

# activate the unpacked environment and install pip packages
# add our batch script:
case $OSTYPE in
  darwin*) cp -r ./extra/platform_scripts/bumps_webview.app "$pkgdir" ;
           cp -r ./extra/platform_scripts/bumps_shell.app "$pkgdir" ;;
  msys*) cp ./extra/platform_scripts/bumps_webview.bat "$pkgdir" ;
         cp ./extra/platform_scripts/bumps_shell.bat "$pkgdir" ;;
  linux*) cp -r ./extra/platform_scripts/make_linux_desktop_shortcut.sh "$pkgdir" ;
          cp -r ./extra/platform_scripts/bumps-webview "$pkgdir" ;;
esac

case "$OSTYPE" in
 "msys") bindir=$envdir ;
         platform="Windows";;
 *) bindir=$envdir/bin ;
    platform="$(uname -s)";;
esac

mkdir -p $pkgdir/share/icons
cp $SCRIPT_DIR/*.svg $pkgdir/share/icons
cp $SCRIPT_DIR/*.png $pkgdir/share/icons

# base path to source is in parent of SCRIPT_DIR
$bindir/python -m pip install --no-input --no-compile .[webview]

cd $tmpdir
# build the client
$bindir/python -m bumps.webview.build_client --cleanup

version=$($bindir/python -c "import bumps; print(bumps.__version__)")
mv "$tmpdir/$PKGNAME" "$tmpdir/$PKGNAME-$version"

case $OSTYPE in
  # darwin*) cd $tmpdir && hdiutil create -srcfolder  "$PKGNAME-$version" -volname "Refl1D_Jupyter" "$SRC_DIR/Refl1D_Jupyter.dmg" ;;
  darwin*) pkgbuild --root $tmpdir --identifier org.reflectometry.$PKGNAME-$SUBNAME --version $version --ownership preserve --install-location /Applications "$SRC_DIR/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m).pkg" ;;
  msys*) conda install -y 7zip ;
         curl -L https://www.7-zip.org/a/7z2106-x64.exe --output 7z_exe ;
         7z e 7z_exe -aoa 7z.sfx ;
         7z a -mhe=on -mx=1 -sfx".\7z.sfx" "$SRC_DIR/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m)-self-extracting.exe" "$PKGNAME-$version" ;;
esac

cd $tmpdir && tar -czf "$SRC_DIR/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m).tar.gz" "$PKGNAME-$version"
cd $SRC_DIR
rm -rf $tmpdir
