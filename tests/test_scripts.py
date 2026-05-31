# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
from pathlib import Path


class TestGenerateInstall:
    """Test generate_install.py script."""

    def test_generate_install_no_deps(self):
        """Test generating install.py with no dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal pyproject.toml
            pyproject = tmpdir / "pyproject.toml"
            pyproject.write_text(
                """
[project]
name = "test-package"
dependencies = []
"""
            )

            output = tmpdir / "install.py"

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_install.py",
                    "--pyproject",
                    str(pyproject),
                    "--output",
                    str(output),
                    "--package-name",
                    "test-package",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output.exists()

            content = output.read_text()
            assert "import piplite" in content
            assert "async def _():" in content
            assert "test-package" in content
            assert "await piplite.install" in content

    def test_generate_install_with_deps(self):
        """Test generating install.py with dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pyproject.toml with dependencies
            pyproject = tmpdir / "pyproject.toml"
            pyproject.write_text(
                """
[project]
name = "test-package"
dependencies = [
    "requests>=2.0",
    "numpy",
    "aiohttp",
]
"""
            )

            output = tmpdir / "install.py"

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_install.py",
                    "--pyproject",
                    str(pyproject),
                    "--output",
                    str(output),
                    "--package-name",
                    "test-package",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            content = output.read_text()

            # Check that dependencies are included
            assert "requests" in content
            assert "numpy" in content
            # aiohttp should be excluded
            assert "aiohttp" not in content


class TestPatchNotebooks:
    """Test patch_notebooks.py script."""

    def test_patch_single_notebook(self):
        """Test patching a single notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test notebook
            notebook = tmpdir / "test.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title"]},
                    {"cell_type": "code", "source": ["print('hello')"]},
                    {"cell_type": "code", "source": ["print('world')"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content, indent=1))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Patched: test.ipynb" in result.stdout

            # Check patched notebook
            patched = json.loads(notebook.read_text())
            first_code_cell = patched["cells"][1]
            assert first_code_cell["cell_type"] == "code"
            assert first_code_cell["source"][0] == "from install import _; await _()\n"
            assert "print('hello')" in first_code_cell["source"][1]

            # Second code cell should be unchanged
            second_code_cell = patched["cells"][2]
            assert second_code_cell["source"] == ["print('world')"]

    def test_patch_custom_install_line(self):
        """Test patching with custom install line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            notebook = tmpdir / "test.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "code", "source": ["x = 1"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content))

            # Run script with custom install line
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                    "--install-line",
                    "# Custom setup\n",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            patched = json.loads(notebook.read_text())
            assert patched["cells"][0]["source"][0] == "# Custom setup\n"

    def test_skip_notebook_without_code_cells(self):
        """Test that notebooks without code cells are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            notebook = tmpdir / "markdown-only.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title"]},
                    {"cell_type": "markdown", "source": ["Some text"]},
                ]
            }
            notebook.write_text(json.dumps(nb_content, indent=1))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Skipped (no code cells)" in result.stdout

            # Notebook should be unchanged
            unchanged = json.loads(notebook.read_text())
            assert unchanged == nb_content

    def test_patch_multiple_notebooks(self):
        """Test patching multiple notebooks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple notebooks
            for i in range(3):
                notebook = tmpdir / f"test{i}.ipynb"
                nb_content = {
                    "cells": [
                        {"cell_type": "code", "source": [f"print({i})"]},
                    ]
                }
                notebook.write_text(json.dumps(nb_content))

            # Run script
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/patch_notebooks.py",
                    "--notebooks-dir",
                    str(tmpdir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Patched 3 notebook(s)" in result.stdout

            # Verify all were patched
            for i in range(3):
                notebook = tmpdir / f"test{i}.ipynb"
                patched = json.loads(notebook.read_text())
                assert patched["cells"][0]["source"][0] == "from install import _; await _()\n"
