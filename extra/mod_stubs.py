#!/usr/bin/env python3
"""
Generate a .pyi file for a module.

Usage:

    python extra/mod_stubs.py bumps.webview.client

This creates bumps/webview/client.pyi with stubs for each interface in bumps.webview.client.

These stubs are used by your editor for type checking and autocompletion. They are required
when the module in question has dynamically generated symbols. For example, the BumpsClient
class provides client-side proxies for the RPC methods defined in bumps.api. This are too
difficult to synchronize manually, but adding a pre-commit hook to .pre-commit-config.yaml
makes it automatic::

    repos:
      - repo: local
        hooks:
        - id: generate-client-stubs
            name: Generate bumps.webview.client stubs
            # Run the generator and then stage the .pyi file.
            entry: >
            bash -c '
                python extra/mod_stubs.py bumps.webview.client &&
                git add bumps/webview/client.pyi
            '
            language: system # use the host’s Python / Git
            # Trigger when either bumps/api.py or bumps/webview/client.py changes
            files: ^bumps/(api\.py|webview/client\.py)$
            stages: [commit] # default – runs on every commit

"""

import inspect
import importlib
import os
import sys
import typing


# Helper that works on Python 3.8‑3.10 (private) and 3.11+ (public)
if hasattr(typing, "type_to_str"):  # Python ≥3.11

    def _annotation_to_str(ann):
        """Return a stub‑compatible string for a type annotation."""
        # ``type_to_str`` already returns a clean identifier (e.g. "list[int]")
        if ann is inspect.Parameter.empty or ann is inspect.Signature.empty:
            return "Any"
        return typing.type_to_str(ann)
else:  # CRUFT: Python 3.8‑3.10
    from typing import _type_repr  # noqa: W0611  (private import is intentional)

    def _annotation_to_str(ann):
        """Return a stub‑compatible string for a type annotation."""
        if ann is inspect.Parameter.empty or ann is inspect.Signature.empty:
            return "Any"
        try:
            # _type_repr turns objects like <class 'str'>, List[int],
            # Optional[Path] etc. into the source‑code representation.
            return _type_repr(ann)
        except Exception:
            # Fallback – use the object's __name__ if it has one, else Any.
            return getattr(ann, "__name__", "Any")


def generate_pyi(module):
    output = []

    # Get all members of the module
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue  # Skip privates

        if inspect.isclass(obj):
            # Handle Classes
            output.append(f"class {name}:")
            output.append(f'    """{inspect.getdoc(obj) or ""}"""')

            for m_name, m_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                if m_name.startswith("_") and m_name != "__init__":
                    continue

                # Get runtime annotations
                prefix = "async " if inspect.iscoroutinefunction(m_obj) else ""
                sig = inspect.signature(m_obj)
                params = []
                for p_name, p in sig.parameters.items():
                    # Use the runtime annotation or Any
                    ann = p.annotation if p.annotation is not inspect.Parameter.empty else "Any"
                    params.append(f"{p_name}: {ann}")

                ret = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else "Any"

                output.append(f"    {prefix}def {m_name}({', '.join(params)}) -> {ret}:")
                output.append(f"        \"\"\"{inspect.getdoc(m_obj) or ''}\"\"\"")

        elif inspect.isfunction(obj):
            # Handle Module-level functions
            prefix = "async " if inspect.iscoroutinefunction(obj) else ""
            sig = inspect.signature(obj)
            params = [
                f"{p_name}: {p.annotation if p.annotation is not inspect.Parameter.empty else 'Any'}"
                for p_name, p in sig.parameters.items()
            ]
            ret = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else "Any"

            output.append(f"{prefix}def {name}({', '.join(params)}) -> {ret}:")
            output.append(f"    \"\"\"{inspect.getdoc(obj) or ''}\"\"\"")

    return "\n".join(output)


def generate_pyi(module):
    """
    Generate a .pyi stub for *module* where every annotation is rendered
    as a proper type‑hint (e.g. ``str`` instead of ``<class 'str'>``).
    """
    lines = []

    # ---------------------------------------------------------
    # Walk through all members of the module (functions & classes)
    # ---------------------------------------------------------
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue  # skip private names

        # -------------------------------------------------
        # 1️⃣  Classes
        # -------------------------------------------------
        if inspect.isclass(obj):
            lines.append(f"class {name}:")
            doc = inspect.getdoc(obj) or ""
            lines.append(f'    """{doc}"""')

            # iterate over methods defined on the class
            for m_name, m_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                # keep __init__ (constructor), otherwise skip private methods
                if m_name.startswith("_") and m_name != "__init__":
                    continue

                prefix = "async " if inspect.iscoroutinefunction(m_obj) else ""

                sig = inspect.signature(m_obj)
                params = []
                for i, (p_name, p) in enumerate(sig.parameters.items()):
                    # Drop the implicit ``self`` for instance methods
                    if i == 0 and p_name == "self":
                        continue

                    ann = _annotation_to_str(p.annotation)
                    # Preserve defaults when they exist (simple literals only)
                    if p.default is not inspect.Parameter.empty:
                        default_repr = f'"{p.default}"' if isinstance(p.default, str) else repr(p.default)
                        params.append(f"{p_name}: {ann} = {default_repr}")
                    else:
                        params.append(f"{p_name}: {ann}")

                ret_ann = _annotation_to_str(sig.return_annotation)

                lines.append(f"    {prefix}def {m_name}({', '.join(params)}) -> {ret_ann}:")
                method_doc = inspect.getdoc(m_obj) or ""
                if method_doc:
                    lines.append(f'        """{method_doc}"""')

        # -------------------------------------------------
        # 2️⃣  Module-level functions
        # -------------------------------------------------
        elif inspect.isfunction(obj):
            prefix = "async " if inspect.iscoroutinefunction(obj) else ""
            sig = inspect.signature(obj)

            params = []
            for p_name, p in sig.parameters.items():
                ann = _annotation_to_str(p.annotation)
                if p.default is not inspect.Parameter.empty:
                    default_repr = f'"{p.default}"' if isinstance(p.default, str) else repr(p.default)
                    params.append(f"{p_name}: {ann} = {default_repr}")
                else:
                    params.append(f"{p_name}: {ann}")

            ret_ann = _annotation_to_str(sig.return_annotation)

            lines.append(f"{prefix}def {name}({', '.join(params)}) -> {ret_ann}:")
            func_doc = inspect.getdoc(obj) or ""
            if func_doc:
                lines.append(f'    """{func_doc}"""')

    return "\n".join(lines)


def process_one(module_path: str):
    try:
        # 1. Import the module dynamically
        print(f"Importing {module_path}...")
        module = importlib.import_module(module_path)

        # 2. Determine the physical file path of the .py file
        py_file_path = getattr(module, "__file__", None)
        if not py_file_path:
            print(f"Error: Could not find physical file for {module_path}. Is it a built-in module?")
            return

        # Convert /path/to/module.py -> /path/to/module.pyi
        pyi_file_path = os.path.splitext(py_file_path)[0] + ".pyi"

        # 3. Generate the content
        print("Analyzing runtime types...")
        content = generate_pyi(module)

        # 4. Write to disk
        with open(pyi_file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully created stub: {pyi_file_path}")

    except ImportError as e:
        print(f"Import Error: {e}. Make sure the module is in your PYTHONPATH.")


def main():
    if len(sys.argv) == 1:
        print("Usage: python mod_stubs.py bumps.webview.client ...")
        return

    for module_name in sys.argv[1:]:
        process_one(module_name)


if __name__ == "__main__":
    main()
