# OpenFHE-NumPy

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenFHE Version](https://img.shields.io/badge/OpenFHE-1.2.3%2B-green)](https://github.com/openfheorg/openfhe-development)

A NumPy-like API for homomorphic encryption operations, built on top of OpenFHE. This library enables data scientists and machine learning practitioners to perform computations on encrypted data using familiar NumPy syntax.

The project is currently in development, with a planned release shortly.
## Table of Contents
- [OpenFHE-NumPy](#openfhe-numpy)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Building from Source](#building-from-source)
  - [Example Usage](#example-usage)
  - [Available Operations](#available-operations)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **NumPy-compatible API**: Use familiar NumPy-style syntax for homomorphic operations
- **Encrypted tensor manipulation**: Create and manipulate encrypted multi-dimensional arrays
- **Matrix operations**: Perform matrix addition, multiplication, transposition on encrypted data
- **Optimized implementation**: Built on top of OpenFHE for optimal performance
- **Type flexibility**: Support for both encrypted (CT) and plaintext (PT) data types
- **Interoperability**: Seamless integration with Python machine learning workflows

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
- **Python**: Version 3.8 or newer
- **NumPy**: Any version
- **OpenFHE**: Any version
- **OpenFHE Python**: Any version

### Building from Source
Please refer to the following repositories for installation instructions:
- [OpenFHE Development](https://github.com/openfheorg/openfhe-development)
- [OpenFHE Python Bindings](https://github.com/openfheorg/openfhe-python)
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
make install
```

<!-- ## Running Tests

```bash
# Run all tests
python -m tests

# Run a specific test
python -m tests.test_matrix_addition
``` -->

## Example Usage

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

## Documentation

For detailed documentation on the API, please visit our [documentation site](https://openfheorg.github.io/openfhe-numpy/html/index.html).

## Examples

We provide several examples showcasing the library's functionality:

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
