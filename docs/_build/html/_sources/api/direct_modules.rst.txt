API Reference (Direct Modules)
===============================

This section provides direct access to individual modules without going through
the main package import, which avoids issues with the compiled C++ extension.

Matrix API Operations
--------------------

.. automodule:: openfhe_numpy.operations.matrix_api
   :members:
   :undoc-members:
   :show-inheritance:

Matrix Arithmetic Backend
------------------------

.. automodule:: openfhe_numpy.operations.matrix_arithmetic
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------

Error Classes
~~~~~~~~~~~~~

.. automodule:: openfhe_numpy.utils.errors
   :members:
   :undoc-members:
   :show-inheritance:

Type Checking
~~~~~~~~~~~~~

.. automodule:: openfhe_numpy.utils.typecheck
   :members:
   :undoc-members:
   :show-inheritance:

Individual Module Code
---------------------

For direct inspection, here are some key modules:

Matrix API Source
~~~~~~~~~~~~~~~~

The matrix_api.py module provides the main NumPy-compatible interface. 
Key functions include:

- ``add(a, b)`` - Element-wise addition
- ``multiply(a, b)`` - Element-wise multiplication  
- ``matmul(a, b)`` - Matrix multiplication
- ``sum(a, axis=None, keepdims=False)`` - Sum reduction
- ``mean(a, axis=None, keepdims=False)`` - Mean calculation
- ``cumulative_sum(a, axis=None)`` - Cumulative sum
- ``transpose(a, axes=None)`` - Transpose operation
- ``roll(a, shift, axis=None)`` - Roll/shift elements

For detailed function signatures and docstrings, see the source files in
``openfhe_numpy/operations/`` directory.
