# OpenFHE-NumPy

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenFHE Version](https://img.shields.io/badge/OpenFHE-1.3.0%2B-green)](https://github.com/openfheorg/openfhe-development)

A NumPy-like API for homomorphic encryption operations, built on top of OpenFHE. This library enables data scientists and machine learning practitioners to perform computations on encrypted data using familiar NumPy syntax. 

The project is currently in development, with a planned release shortly.


## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing OpenFHE-Python](#installing-openfhe-python)
  - [Installing OpenFHE-NumPy](#installing-openfhe_numpy)
- [Example Usage](#example-usage)
- [Available Operations](#available-operations)
- [Documentation](#documentation)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Features

- **NumPy-like API**: Use familiar NumPy-style syntax for homomorphic operations
- **Encrypted tensor manipulation**: Create and manipulate encrypted multi-dimensional arrays
- **Matrix operations**: Perform matrix addition, multiplication, transposition on encrypted data
- **Optimized implementation**: Built on top of OpenFHE for optimal performance
- **Type flexibility**: Support for both encrypted (CT) and plaintext (PT) data types
- **Interoperability**: Seamless integration with Python machine learning workflows

## Installation

### Prerequisites

- **OpenFHE**: Version 1.2.3 or newer
- **C++ compiler**: Supporting C++20 standard
- **CMake**: Version 3.16 or newer
- **Python**: Version 3.8 or newer
- **NumPy**: Recent version

### Installing OpenFHE-Python

OpenFHE-NumPy requires the OpenFHE-Python bindings:

```bash
# Clone OpenFHE-Python
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python

# Install with pip
pip install .
```

### Installing OpenFHE-NumPy

#### From PyPI (coming soon)

```bash
pip install openfhe_numpy
```

#### From Source

```bash
# Clone the repository
git clone https://github.com/openfheorg/openfhe_numpy.git
cd openfhe_numpy

# Create build directory
mkdir build && cd build

# Configure with CMake
# Set OpenFHE_DIR to your OpenFHE installation path if needed
cmake ..  -DCMAKE_INSTALL_PREFIX=/path/to/openfhe/install

# Build the package
make 

# Install
make install

```

## Example Usage

```python
import numpy as np
import openfhe_numpy as onp
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
)

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
tensor_A = onp.array(cc, A, total_slots, public_key=keys.publicKey)

# Generate keys
onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, tensor_A.ncols)

# Perform encrypted operations
tensor_product = tensor_A @ tensor_A  # Matrix multiplication
tensor_sum = onp.add(tensor_A, tensor_A)  # Element-wise addition

# Decrypt results
decrypted_product = tensor_product.decrypt(keys.secretKey, format_type = True)
decrypted_sum = tensor_sum.decrypt(keys.secretKey, format_type = True)

print("Result of A @ A:")
print(decrypted_product)

print("Result of A + A:")
print(decrypted_sum)
```

## Available Operations

OpenFHE-NumPy currently supports the following operations:

| Operation | Description | Example |
|-----------|-------------|---------|
| `add` | Element-wise addition | `onp.add(a, b)` or `a + b` |
| `subtract` | Element-wise subtraction | `onp.subtract(a, b)` or `a - b` |
| `multiply` | Element-wise multiplication | `onp.multiply(a, b)` or `a * b` |
| `matmul` | Matrix multiplication | `onp.matmul(a, b)` or `a @ b` |
| `transpose` | Matrix transposition | `onp.transpose(a)` |
| `cumsum` | Cumulative sum along axis | `onp.cumsum(a, axis)` |
| `power` | Element-wise power | `onp.power(a, exp)` |
| `dot` | Dot product | `onp.dot(a, b)` |

## Documentation

For detailed documentation on the API, please visit our [documentation site](https://openfheorg.github.io/openfhe_numpy).

## Examples

We provide several examples showcasing the library's functionality:

- [Matrix Addition](https://github.com/openfheorg/openfhe_numpy/blob/main/examples/demo_matrix_addition.py)
- [Matrix Transpose](https://github.com/openfheorg/openfhe_numpy/blob/main/examples/demo_matrix_transpose.py)
- [Matrix-Vector Multiplication](https://github.com/openfheorg/openfhe_numpy/blob/main/examples/demo_matvec_product.py)
- [Square Matrix Multiplication](https://github.com/openfheorg/openfhe_numpy/blob/main/examples/demo_square_matrix_product.py)
- [Cumulative Matrix Operations](https://github.com/openfheorg/openfhe_numpy/blob/main/examples/demo_matrix_accumulation.py)

## Performance

OpenFHE-NumPy is designed for both usability and performance. For optimal performance:
- Use appropriate multiplicative depth for your operations
- Choose ring dimension based on your security requirements and dataset size
- Consider the tradeoff between precision and performance when selecting scaling parameters

## Contributing

Contributions to OpenFHE-NumPy are welcome! Please see our contributing guidelines for details.

## License

OpenFHE-NumPy is licensed under the BSD 3-Clause License. See the LICENSE file for details.

## License

---

OpenFHE-NumPy is an independent project and is not officially affiliated with NumPy.
