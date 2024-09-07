from typing import Literal

def build_client(install_dependencies=True, mode: Literal["development", "production"]="development", sourcemap=False):
    """Build the webview client."""
    from pathlib import Path
    import os
    import shutil
    # check if npm is installed
    if not shutil.which("npm"):
        raise RuntimeError("npm is not installed. Please install npm.")
    client_folder = Path(__file__).parent / "client"
    # check if the node_modules directory exists
    node_modules = client_folder / "node_modules"
    os.chdir(client_folder)
    if install_dependencies or not node_modules.exists():
        print("Installing node modules...")
        os.system("npm install")
    # build the client
    print("Building the webview client...")
    cmd = f"npm run build -m {mode}"
    if sourcemap:
        cmd += " -- --sourcemap"
    os.system(cmd)
    print("Done.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the webview client.")
    parser.add_argument("--install-dependencies", action="store_true", help="Install dependencies.")
    parser.add_argument("--mode", choices=["development", "production"], default="development", help="Build mode.")
    parser.add_argument("--sourcemap", action="store_true", help="Generate sourcemaps.")
    args = parser.parse_args()
    build_client(install_dependencies=args.install_dependencies, mode=args.mode, sourcemap=args.sourcemap)