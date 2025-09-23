import os
from pathlib import Path
import shutil


def build_client(
    no_deps=False,
    sourcemap=False,
    cleanup=False,
):
    """Build the bumps webview client."""
    if shutil.which("bun"):
        tool = "bun"
    elif shutil.which("npm"):
        tool = "npm"
    else:
        raise RuntimeError("npm/bun is not installed. Please install either npm or bun.")

    client_folder = (Path(__file__).parent / "client").resolve()
    node_modules = client_folder / "node_modules"
    os.chdir(client_folder)

    if not no_deps or not node_modules.exists():
        print("Installing node modules...")
        os.system(f"{tool} install")

    # build the client
    print("Building the webview client...")
    cmd = f"{tool} run build"
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
    parser.add_argument("--no-deps", action="store_false", help="Don't install npm dependencies.")
    parser.add_argument("--sourcemap", action="store_true", help="Generate sourcemaps.")
    parser.add_argument("--cleanup", action="store_true", help="Remove the node_modules directory.")
    args = parser.parse_args()
    build_client(
        no_deps=args.no_deps,
        sourcemap=args.sourcemap,
        cleanup=args.cleanup,
    )
