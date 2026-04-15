#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
echo "Running ${SCRIPT_PATH}"

# get to the root directory (should be ./openfhe-numpy)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd ${ROOT_DIR}
echo "------ in ${PWD}"

# cleanup
rm -fr .venv-docs/ build/ docs/_build/ docs/html/ docs/doctrees/

# preconditions: you need the same basic tools RTD uses
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  build-essential cmake git

# create and activate a clean venv
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip

# install doc dependencies
# installs only Sphinx + runtime deps (not openfhe-numpy yet)
python3 -m pip install -r docs/requirements.txt

# simulate RTD environment variables and build docs for the current branch
export READTHEDOCS=1
export READTHEDOCS_GIT_IDENTIFIER="$(git branch --show-current)"
export READTHEDOCS_GIT_COMMIT_HASH="$(git rev-parse HEAD)"

# build the wheel for docs
WHEEL_PATH="$(bash docs/scripts/build-wheel-for-docs.sh)"
echo "--------------- Wheel built at: $WHEEL_PATH"

# Install the built wheel
python3 -m pip install "$WHEEL_PATH"
# Sanity check
python3 - <<'EOF'
import os, sys
# Prevent importing from the repo checkout ("" or CWD)
cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd)]

import openfhe
print("openfhe import OK")
import openfhe_numpy
print("openfhe_numpy import OK")
print("version:", getattr(openfhe_numpy, "__version__", "unknown"))
EOF

# Build the HTML docs
python3 -m sphinx -b html ${ROOT_DIR}/docs/ ${ROOT_DIR}/docs/_build/html
# # Or equivalently
# cd docs
# make html

# View the docs
firefox ${ROOT_DIR}/docs/_build/html/index.html
