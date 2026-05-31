#!/usr/bin/env python3
# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Generate install.py for JupyterLite pyodide kernel."""

import argparse
import re
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path


def pkg_name(dep: str) -> str:
    """Extract package name from dependency string."""
    return re.split(r"[>=<!;\[ ]", dep)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate install.py for pyodide")
    parser.add_argument("--pyproject", type=Path, required=True, help="Path to pyproject.toml")
    parser.add_argument("--output", type=Path, required=True, help="Output path for install.py")
    parser.add_argument("--package-name", default="litelm", help="Package name (default: litelm)")
    args = parser.parse_args()

    # Read pyproject.toml
    with open(args.pyproject, "rb") as f:
        proj = tomllib.load(f)

    # Extract dependencies
    exclude = {"aiohttp"}  # replaced by pyfetch on emscripten
    deps = proj.get("project", {}).get("dependencies", [])
    pkgs = [pkg_name(d) for d in deps if pkg_name(d) not in exclude]

    # Get version
    try:
        v = version(args.package_name)
    except Exception as e:
        print(f"Warning: Could not get version for {args.package_name}: {e}", file=sys.stderr)
        v = "0.0.0"

    # Build package list starting with the wheel
    all_pkgs = [f'"emfs:{args.package_name}-{v}-py3-none-any.whl"'] + [f'"{p}"' for p in pkgs]
    pkg_list = ", ".join(all_pkgs)

    # Generate install.py content
    content = (
        "import piplite\n"
        "async def _():\n"
        f"  packages = [{pkg_list}]\n"
        "  await piplite.install(packages, keep_going=True)\n"
    )

    # Write to output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)

    print(f"Generated {args.output}")
    print(content)


if __name__ == "__main__":
    main()
