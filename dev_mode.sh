#!/bin/bash

set -e  # Exit on any error

echo "=== Setting up OpenFHE-NumPy for development ==="

# Install in editable mode
pip install -e .[dev]

# Build C++ extension
echo "Building C++ extension..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

echo "Running unit tests..."


# Create symlink to the extension
echo "Linking C++ extension to package directory..."
extension_path=$(find build -name "openfhe_numpy*.so")
if [ -n "$extension_path" ]; then
    ln -sf $(pwd)/$extension_path $(pwd)/openfhe_numpy/
    echo "Extension linked successfully: $extension_path"
else
    echo "Error: Couldn't find compiled extension"
    exit 1
fi

echo "=== Development setup complete! ==="

echo "You can now run tests with: python3 -m unittest discover -s tests"