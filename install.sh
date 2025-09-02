#!/usr/bin/env bash
# The script will build OpenFHE-Numpy from source

set -e

INSTALL_PATH=""
if [ "$1" = "--install-path" ]; then
    INSTALL_PATH="$2"
elif [ "$1" = "--help" ]; then
    echo "Usage: ./install.sh [--install-path PATH]"
    echo "Example:"
    echo "    Install to custom path: ./install.sh --install-path /home/user/custom/path"
    echo "    Default installation site-packages/openfhe_numpy: ./install.sh"
    exit 0
fi

# Find Python3 (venv -> conda -> system)
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
    PYTHON_PATH="$VIRTUAL_ENV/bin/python3"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python3" ]]; then
    PYTHON_PATH="$CONDA_PREFIX/bin/python3"
else
    PYTHON_PATH="$(command -v python3)"
fi

[[ -n "${PYTHON_PATH:-}" ]] || { echo "Error: Python3 not found"; exit 1; }
echo "Using Python: $PYTHON_PATH" && $PYTHON_PATH --version

# Get site-packages directory
SITE_PACKAGES="$($PYTHON_PATH -c 'import sysconfig; print(sysconfig.get_path("platlib") or sysconfig.get_path("purelib"))')"
[[ -n "$SITE_PACKAGES" ]] || { echo "Error: Cannot find site-packages"; exit 1; }
echo "Python site-packages: $SITE_PACKAGES"

# Set install prefix
PREFIX=${INSTALL_PATH:-$SITE_PACKAGES/openfhe_numpy}
echo "Installing to: $PREFIX"

# Build and install
rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DPYTHON_EXECUTABLE_PATH="$PYTHON_PATH"
cmake --build . -j$(nproc)
sudo cmake --install .

echo "Installation complete!"
