#!/bin/bash

# This script creates a desktop shortcut for the Bumps webview server.
# The shortcut will be created in the user's desktop directory.
# the executable path is in 'env/bin/python'

# Get the user's desktop directory
desktop_dir=~/Desktop

# Get the path to the bumps webview server
script_dir=$(realpath $(dirname $0))

# Create the desktop shortcut
echo "[Desktop Entry]
Name=Bumps-Webview
Comment=Start the bumps webview server
Exec='$script_dir/env/bin/python' -m bumps.webview.server
Icon=$script_dir/env/share/icons/bumps-icon.svg
Terminal=true
Type=Application
Categories=Development;
" > $desktop_dir/BumpsWebviewServer.desktop

# Make the desktop shortcut executable
chmod +x $desktop_dir/BumpsWebviewServer.desktop