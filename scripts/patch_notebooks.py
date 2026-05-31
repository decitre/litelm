#!/usr/bin/env python3
# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Patch notebooks to add install line to first code cell."""

import argparse
import json
from pathlib import Path


def patch_notebook(nb_path: Path, install_line: str) -> bool:
    """
    Patch a notebook by adding install line to the first code cell.

    Args:
        nb_path: Path to the notebook file
        install_line: Line to prepend to first code cell

    Returns:
        True if patched, False if no code cells found
    """
    with open(nb_path, "r") as f:
        notebook = json.load(f)

    # Find the first code cell
    first_code_idx = None
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            first_code_idx = idx
            break

    if first_code_idx is None:
        return False

    # Prepend install line to the first code cell
    source = notebook["cells"][first_code_idx].get("source", [])
    if isinstance(source, str):
        source = [source]
    notebook["cells"][first_code_idx]["source"] = [install_line] + source

    # Write back
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=1)
        f.write("\n")  # Add trailing newline

    return True


def main():
    parser = argparse.ArgumentParser(description="Patch notebooks for pyodide")
    parser.add_argument("--notebooks-dir", type=Path, required=True, help="Directory containing notebooks")
    parser.add_argument(
        "--install-line",
        default="from install import _; await _()\n",
        help="Line to add to first code cell",
    )
    args = parser.parse_args()

    if not args.notebooks_dir.is_dir():
        print(f"Error: {args.notebooks_dir} is not a directory")
        return 1

    patched_count = 0
    skipped_count = 0

    for nb_path in sorted(args.notebooks_dir.glob("*.ipynb")):
        if patch_notebook(nb_path, args.install_line):
            print(f"Patched: {nb_path.name}")
            patched_count += 1
        else:
            print(f"Skipped (no code cells): {nb_path.name}")
            skipped_count += 1

    print(f"\nPatched {patched_count} notebook(s), skipped {skipped_count}")
    return 0


if __name__ == "__main__":
    exit(main())
