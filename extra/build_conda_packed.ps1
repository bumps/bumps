$ENV_NAME="isolated-base"
$PYTHON_VERSION="3.10"
$DIRNAME="bumps"

conda activate "base"

conda install -y conda-pack
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
$tmpdir="dist"
$destdir="$tmpdir\$DIRNAME"
$envdir = "$destdir\env"
Remove-Item -r -Force "$destdir"
mkdir "$envdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$envdir"

# activate the unpacked environment and install pip packages
conda deactivate
$WORKING_DIRECTORY="$pwd"
echo "WORKING_DIRECTORY=$WORKING_DIRECTORY"
dir .
dir ..
# add our batch script:
Copy-Item .\extra\bumps_webview.bat "$destdir"

& "$envdir\python.exe" -m pip install --no-input --no-compile numba
& "$envdir\python.exe" -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
& "$envdir\python.exe" -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

$version=$(& "$envdir\python.exe" -c "import bumps; print(bumps.__version__)")
# zip it back up
cd $tmpdir
Rename-Item "$DIRNAME" "$DIRNAME-$version"
tar -czf "bumps-webview-Windows-x86_64.tar.gz" "$DIRNAME-$version"
