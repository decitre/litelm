#!/bin/bash

# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$SCRIPT_DIR"/..
NOTEBOOKS_DIR="$SCRIPT_DIR"/../notebooks
TAR_CMD=$(command -v gtar || echo "tar")
PYTHON_CMD=$(command -v python3 || command -v python || echo "python3")
RED='\033[0;31m'
YELLOW='\033[33m'
NC='\033[0m'

printf "${YELLOW}Build workspace${NC}\n"
rm -rf workspace
mkdir -p workspace/content-xeus workspace/content-pyodide

if command -v uv &> /dev/null; then
  pip_cmd="uv pip"
else
  pip_cmd="pip"
  $PYTHON_CMD -m pip -q install --upgrade pip
fi

process_for_xeus() {
    printf "${YELLOW}Create venv-xeus${NC}\n"
    pushd workspace
    mkdir -p content-xeus
    $PYTHON_CMD -m venv venv-xeus
    source venv-xeus/bin/activate

    printf "${YELLOW}Install jupyter/xeus packages:${NC}\n"
    $pip_cmd install jupyter_server jupyterlite-core jupyterlite-xeus libarchive-c

    printf "${YELLOW}Copy static content for jupyterlite-xeus site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb content-xeus/
    ls -l content-xeus

    cat << eof > environment.yml
name: xeus-kernels
channels:
  - https://prefix.dev/emscripten-forge-dev
  - https://prefix.dev/conda-forge
dependencies:
  - xeus-python

eof

    jupyter lite build \
        --contents content-xeus \
        --output-dir public \
        --XeusAddon.environment_file=$(pwd)/environment.yml \
        --XeusAddon.mount_jupyterlite_content=True \
        --XeusAddon.mounts="$ROOT_DIR/src/lmlite:/lib/python3.13/site-packages/lmlite"

    deactivate
    popd
}

process_for_pyodide() {
    printf "${YELLOW}Create venv-pyodide${NC}\n"
    pushd workspace
    mkdir -p content-pyodide
    $PYTHON_CMD -m venv venv-pyodide
    source venv-pyodide/bin/activate

    printf "${YELLOW}Install jupyter/pyodide packages:${NC}\n"
    # https://pyodide.org/en/stable/usage/packages-in-pyodide.html
    $pip_cmd install build -e '..' \
      jupyter_server jupyterlab_server jupyterlite-core jupyterlite-pyodide-kernel libarchive-c

    printf "${YELLOW}Build lmlite wheel${NC}\n"
    $PYTHON_CMD -m build "$ROOT_DIR" --wheel --outdir content-pyodide --skip-dependency-check

    printf "${YELLOW}Copy static content for jupyterlite-pyodide site${NC}\n"
    cp "$NOTEBOOKS_DIR"/*.ipynb content-pyodide/

    printf "${YELLOW}Generate install.py${NC}\n"
    $PYTHON_CMD "$SCRIPT_DIR/generate_install.py" \
        --pyproject "$ROOT_DIR/pyproject.toml" \
        --output content-pyodide/install.py \
        --package-name lmlite

    printf "${YELLOW}Patch notebooks${NC}\n"
    $PYTHON_CMD "$SCRIPT_DIR/patch_notebooks.py" \
        --notebooks-dir content-pyodide

    ls -l content-pyodide

    jupyter lite build \
        --contents content-pyodide \
        --output-dir public/pyodide \
        --lite-dir public/pyodide

    deactivate
    popd
}

process_for_xeus
process_for_pyodide

printf "${YELLOW}Done${NC}\n"
cat << eof
To access the jupyterlite sites, run the following command:
(cd workspace/public; python -m http.server 8000 2>/dev/null)
and navigate to
- http://localhost:8000/ for the jupyterlite-xeus site
- http://localhost:8000/pyodide/ for the jupyterlite-pyodide site
eof
