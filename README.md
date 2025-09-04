# OpenFHE-NumPy

[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python Versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenFHE Version](https://img.shields.io/badge/OpenFHE-1.4.0%2B-green)](https://github.com/openfheorg/openfhe-development)

OpenFHE-NumPy is a NumPy-like API for homomorphic encryption operations, built on top of OpenFHE. This library enables data scientists and machine learning practitioners to perform computations on encrypted data using familiar NumPy syntax.

## Table of Contents
- [OpenFHE-NumPy](#openfhe-numpy)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installing from Source](#installing-from-source)
    - [Installing using pip (for Ubuntu)](#installing-using-pip-for-ubuntu)
  - [Running Tests](#running-tests)
  - [Code Examples](#code-examples)
  - [Available Operations](#available-operations)
  - [Current Limitations](#current-limitations)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [Contributing](#contributing)
  - [License](#license)

## Project Structure

OpenFHE-NumPy is organized as a hybrid C++/Python project with the following structure:

```
openfhe-numpy/
├── config/              # Configuration files
├── docs/                # Documentation
├── examples/            # Python examples
├── openfhe_numpy/       # Main Python package
│   ├── cpp/             # C++ implementation
│   ├── operations/      # Matrix operations
│   ├── tensor/          # Tensor implementations
│   └── utils/           # Utility functions
├── tests/               # Test suite
└── CMakeLists.txt       # Build configuration
```

## Installation

### Prerequisites

- **C++ compiler**: Supporting C++17 standard
- **CMake**: Version 3.16 or newer
- **Python**: Version 3.10 or newer
- **NumPy**: Any version
- **OpenFHE**: Any version
- **OpenFHE Python**: Any version

### Installing from Source
Before building, make sure you have the following dependencies installed:
- [OpenFHE 1.4.0+](https://github.com/openfheorg/openfhe-development) by following the instructions in [OpenFHE Documentation](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html)
- [OpenFHE Python Bindings](https://github.com/openfheorg/openfhe-python) by following the instructions in [OpenFHE Python Documentation](https://openfheorg.github.io/openfhe-python/html/index.html)

We recommend following OpenFHE C++ and OpenFHE Python installation instructions first (which covers Linux, Windows and MacOS) and then getting back to this repo. If the some package cannot be found when running a Python example (occurs only for some environments), check the `PYTHONPATH` (OpenFHE Python) environment variable and the `LD_LIBRARY_PATH` (OpenFHE libraries). This ensures that the packages can be correctly located and imported.

```bash
# Clone the repository
git clone https://github.com/openfheorg/openfhe-numpy.git
cd openfhe-numpy

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the package
make

# Install
sudo make install
```

### Installing using pip (for Ubuntu)


On Ubuntu, openfhe_numpy can be installed using pip.  All available releases are listed at [Python Package Index OpenFHE-Numpy Release History](https://pypi.org/project/openfhe_numpy/#history). Find the release for your version of Ubuntu and run

```
pip install openfhe_numpy==<openfhe_package_version>
```

Once installed, any python example at https://github.com/openfheorg/openfhe-numpy/tree/main/examples/python can be executed.

Note that Ubuntu LTS 22.04 and 24.04 are currently supported. `pip uninstall` can be used to uninstall the openfhe package.


## Running Tests
Run tests with [unittest](https://docs.python.org/3/library/unittest.html). See the [testing readme](tests/README.md) for detailed instuctions.

## Code Examples

```python
import numpy as np
import openfhe_numpy as onp
from openfhe import *

# Initialize CKKS context
params = CCParamsCKKSRNS()
params.SetMultiplicativeDepth(7)
params.SetScalingModSize(59)
params.SetFirstModSize(60)
params.SetScalingTechnique(FIXEDAUTO)
params.SetSecretKeyDist(UNIFORM_TERNARY)

cc = GenCryptoContext(params)
cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.LEVELEDSHE)
cc.Enable(PKESchemeFeature.ADVANCEDSHE)

# Generate keys
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalSumKeyGen(keys.secretKey)

# Create matrix and encrypt it
A = np.array([[1, 2], [3, 4]])

ring_dim = cc.GetRingDimension()
total_slots = ring_dim // 2

# Encrypt with OpenFHE-NumPy
tensor_A = onp.array(
        cc=cc,
        data=A,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )


# Generate keys
onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, tensor_A.ncols)

# Perform encrypted operations
tensor_product = tensor_A @ tensor_A  # Matrix multiplication
tensor_sum = onp.add(tensor_A, tensor_A)  # Element-wise addition

# Decrypt results
decrypted_product = tensor_product.decrypt(keys.secretKey, unpack_type="original")
decrypted_sum = tensor_sum.decrypt(keys.secretKey, unpack_type="original")

print("Result of A @ A:")
print(decrypted_product)

print("Result of A + A:")
print(decrypted_sum)
```

## Available Operations

OpenFHE-NumPy currently supports the following operations:

| Operation   | Description                 | Example                         |
| ----------- | --------------------------- | ------------------------------- |
| `add`       | Element-wise addition       | `onp.add(a, b)` or `a + b`      |
| `subtract`  | Element-wise subtraction    | `onp.subtract(a, b)` or `a - b` |
| `multiply`  | Element-wise multiplication | `onp.multiply(a, b)` or `a * b` |
| `matmul`    | Matrix multiplication       | `onp.matmul(a, b)` or `a @ b`   |
| `transpose` | Matrix transposition        | `onp.transpose(a)`              |
| `cumulative_sum`    | Cumulative sum along axis   | `onp.cumulative_sum(a, axis)`           |
| `power`     | Element-wise power          | `onp.power(a, exp)`             |
| `dot`       | Dot product                 | `onp.dot(a, b)`                 |
| `sum`       | Sum along axis              | `onp.sum(a, axis)`              |

## Current Limitations
In the current version, the OpenFHE-NumPy package supports operations on single-ciphertext vectors/matrices, where each encrypted array variable (which has type CTArray or PTArray) contains only a single encoding vector.

For example, we can consider a matrix:
```
1 2 3
4 5 6
7 8 9
```
As an encoding vector of the form: ```1 2 3 0 4 5 6 0 7 8 9 0```

The size of the encoded vector must be smaller than the number of available plaintext slots. Certain operations, such as matrix–vector multiplication, may require the ciphertext vector to be duplicated. In such cases, users should ensure that a sufficient number of slots are available for the function to execute correctly.
We plan to release a future version with support for block ciphertexts, which will remove this limitation in the future.

## Documentation

For detailed documentation on the API, please visit our [documentation site](https://openfheorg.github.io/openfhe-numpy/html/index.html).

## Examples

We provide several examples showcasing the library's functionality:

- [1D Convolution](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_convolution.py)
- [Matrix Accumulation](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_matrix_accumulation.py)
- [Matrix Addition](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_matrix_addition.py)
- [Matrix Summation](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_matrix_sum.py)
- [Matrix Transpostion](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_matrix_transpose.py)
- [Matrix-Vector Multiplication](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_matrix_vector_product.py)
- [Square Matrix Multiplication](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_square_matrix_product.py)
- [Vector Operations](https://github.com/openfheorg/openfhe-numpy/blob/main/examples/python/simple_vector_operations.py)


## Contributing
[OpenFHE Development - Contributing Guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/contributing/contributing_workflow.html)

## License

OpenFHE-NumPy is licensed under the BSD 2-Clause License. See the LICENSE file for details.

---

OpenFHE-NumPy is an independent project and is not officially affiliated with NumPy.
