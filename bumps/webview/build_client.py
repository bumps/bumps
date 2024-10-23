import os
from pathlib import Path
import shutil


def build_client(
    install_dependencies=False,
    sourcemap=False,
    cleanup=False,
):
    """Build the bumps webview client."""

    # check if npm is installed
    if not shutil.which("npm"):
        raise RuntimeError("npm is not installed. Please install npm.")
    client_folder = (Path(__file__).parent / "client").resolve()
    # check if the node_modules directory exists
    node_modules = client_folder / "node_modules"
    os.chdir(client_folder)
    if install_dependencies or not node_modules.exists():
        print("Installing node modules...")
        os.system("npm install")

    # build the client
    print("Building the webview client...")
    cmd = f"npm run build"
    if sourcemap:
        cmd += " -- --sourcemap"
    os.system(cmd)

    if cleanup:
        print("Cleaning up...")
        shutil.rmtree(node_modules)
        print("node_modules folders removed.")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the webview client.")
    parser.add_argument("--install-dependencies", action="store_true", help="Install dependencies.")
    parser.add_argument("--sourcemap", action="store_true", help="Generate sourcemaps.")
    parser.add_argument("--cleanup", action="store_true", help="Remove the node_modules directory.")
    args = parser.parse_args()
    build_client(
        install_dependencies=args.install_dependencies,
        sourcemap=args.sourcemap,
        cleanup=args.cleanup,
    )
