# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================

"""
matrix_api.py

Handles documenting and the public interface of the matrix operation.

This module provides NumPy-compatible matrix operations that can be performed on
encrypted data using homomorphic encryption. Functions follow NumPy naming
conventions and similar signatures where possible.

All functions use the tensor_function_api decorator to handle different tensor types
and dispatch to the appropriate backend implementation.
"""

"""Public NumPy-style matrix operations for OpenFHE-NumPy tensors."""

from typing import Optional
from numpy.typing import ArrayLike
from .dispatch import tensor_function_api


# ===========================
# Element-wise Operations
# ===========================


@tensor_function_api("add", binary=True)
def add(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Add two arrays element-wise.

    See Also
    --------
    numpy.add

    Examples
    --------
    >>> add([1, 2], [3, 4])
    array([4, 6])
    """
    pass


@tensor_function_api("subtract", binary=True)
def subtract(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Subtract two arrays element-wise.

    See Also
    --------
    numpy.subtract

    Examples
    --------
    >>> subtract([5, 7], [2, 3])
    array([3, 4])
    """
    pass


@tensor_function_api("multiply", binary=True)
def multiply(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Multiply two arrays element-wise.

    Parameters
    ----------
    a : ArrayLike
        First operand.
    b : ArrayLike
        Second operand (array or scalar).

    Returns
    -------
    out : ArrayLike
        Element-wise product.

    See Also
    --------
    numpy.multiply : Corresponding NumPy function.

    Examples
    --------
    >>> multiply([1, 2], [3, 4])
    array([3, 8])
    """
    pass


@tensor_function_api("power", binary=True)
def power(a: ArrayLike, exponent: int) -> ArrayLike:
    """

    Element-wise exponentiation on arrays.
    NumPy v2.5 using power for or element-wise exponentiation on arrays

    Note
    ----
    Only positive integer exponents are supported due to homomorphic-encryption
    constraints. ``exponent == 0`` (all-ones) is not supported and raises.

    Parameters
    ----------
    a : ArrayLike
        Base tensor.
    exponent : int
        Positive integer exponent (``>= 1``).

    Returns
    -------
    out : ArrayLike
        Element-wise ``a`` raised to ``exponent``.

    See Also
    --------
    numpy.power : Corresponding element-wise power function.

    Examples
    --------
    >>> power([1, 2, 3], 2)
    array([1, 4, 9])
    """
    pass


# ===========================
# Matrix Operations
# ===========================


@tensor_function_api("dot", binary=True)
def dot(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Dot product / matrix multiplication.

    - 1-D inputs: inner product
    - 2-D inputs: matrix product

    Parameters
    ----------
    a, b : ArrayLike
        Operands.

    Returns
    -------
    ArrayLike
        Result of the dot product.

    See Also
    --------
    numpy.dot

    Examples
    --------
    >>> dot([1, 2], [3, 4])
    11
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> dot(A, B)
    array([[19, 22],
           [43, 50]])
    """
    pass


@tensor_function_api("matmul", binary=True)
def matmul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Multiply two arrays as matrices.

    Parameters
    ----------
    a : ArrayLike
        First operand.
    b : ArrayLike
        Second operand.

    Returns
    -------
    out : ArrayLike
        Matrix product of 'a' and 'b'.

    See Also
    --------
    numpy.matmul : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> matmul(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    array([[19, 22],
           [43, 50]])
    """
    pass


@tensor_function_api("transpose", binary=False)
def transpose(a: ArrayLike) -> ArrayLike:
    """Transpose a two-dimensional array.

    A one-dimensional array is returned unchanged.

    Parameters
    ----------
    a : ArrayLike
        Input tensor.

    Returns
    -------
    out : ArrayLike
        Transposed tensor.

    See Also
    --------
    numpy.transpose : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> transpose(np.array([[1, 2], [3, 4]]))
    array([[1, 3],
           [2, 4]])
    """
    pass


# ===========================
# Reduction Operations
# ===========================


@tensor_function_api("cumsum", binary=False)
def cumsum(a: ArrayLike, axis: Optional[int] = None) -> ArrayLike:
    """Compute cumulative sums using NumPy axis behavior.

    For vectors, ``None``, ``0``, and ``-1`` scan the vector. For matrices,
    ``None`` scans the C-order flattened values and returns a vector, while
    ``0`` and ``1`` scan down rows and across columns respectively.

    Parameters
    ----------
    a : ArrayLike
        Input tensor.
    axis : int, optional
        Axis along which to compute the sum. Default is ``None``.

    Returns
    -------
    out : ArrayLike
        Cumulative sum along an axis.

    See Also
    --------
    numpy.cumsum : Corresponding NumPy function.

    Notes
    -----
    For a block matrix with ``axis=None``, the current implementation assembles
    logical values in C order one element at a time. It can be
    expensive for large matrices.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> cumsum(a)
    array([ 1,  3,  6, 10])
    >>> cumsum(a, axis=0)
    array([[1, 2],
           [4, 6]])
    >>> cumsum(a, axis=1)
    array([[1, 3],
           [3, 7]])
    """
    pass


@tensor_function_api("cumulative_reduce", binary=False)
def cumulative_reduce(a: ArrayLike, axis: int = 0, keepdims: bool = False) -> ArrayLike:
    """
    Compute the cumulative reduction of tensor elements along a specified axis.\
        - For 1D inputs, axis must be None.
        - For 2D inputs, axis must be 0 or 1.
        - The include_initial argument is not supported.

    Parameters
    ----------
    a : ArrayLike
        Input tensor.
    axis : int, optional
        Axis along which to compute the reduction. Default is 0.

    Returns
    -------
    out : ArrayLike
        Cumulative reduction of 'a'.

    See Also
    --------
    numpy.cumsum : Similar operation for sum.

    Examples
    --------
    >>> import numpy as np
    >>> cumulative_reduce(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
    array([[1, 2, 3],
           [-3, -3, -3]])
    """
    pass


@tensor_function_api("sum", binary=False)
def sum(a: ArrayLike, /, *, axis: Optional[int] = None, keepdims: bool = False) -> ArrayLike:
    """
    Sum of elements over an axis or all.

    Parameters
    ----------
    a : ArrayLike
        Input tensor.
    axis : int, optional
        Axis along which to compute the sum. Default is None.
        0: sum over rows
        1: sum over cols
    keepdims : bool, optional
        If True, retains reduced dimensions. Default is False.

    Returns
    -------
    out : ArrayLike
        Sum of the array elements.

    See Also
    --------
    numpy.sum : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> sum(a)
    10
    >>> sum(a, axis=0)
    array([4, 6])
    >>> sum(a, axis=1)
    array([3, 7])
    """
    pass


@tensor_function_api("mean", binary=False)
def mean(
    a: ArrayLike,
    /,
    *,
    axis: Optional[int] = None,
    dtype=None,
    out=None,
    keepdims: bool = False,
) -> ArrayLike:
    """
    Compute the arithmetic mean along an axis or all elements.

    Returns the average of the array elements. The average is taken over
    the flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : ArrayLike
        Input tensor.
    axis : int, optional
        Axis along which to compute the mean. Default is None.
    keepdims : bool, optional
        If True, retains reduced dimensions. Default is False.

    Returns
    -------
    out : ArrayLike
        Arithmetic mean of the array elements.

    See Also
    --------
    numpy.mean : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> mean(a)
    2.5
    >>> mean(a, axis=0)
    array([2., 3.])
    >>> mean(a, axis=1)
    array([1.5, 3.5])
    """
    pass


@tensor_function_api("roll", binary=False)
def roll(a: ArrayLike, shift: int, axis: Optional[int] = None) -> ArrayLike:
    """
    Roll packed vector elements.

    Elements that roll beyond the last position are re-introduced at the first.

    Current limitation
    ------------------
    This function currently supports only packed vectors with ``axis=None``.
    Matrix roll, tuple-axis roll, and block tensor roll are not implemented yet.

    Parameters
    ----------
    a : ArrayLike
        Input packed vector.
    shift : int
        Number of positions by which elements are shifted.
    axis : None, optional
        Must be None in the current implementation.

    Returns
    -------
    res : ArrayLike
        Output vector with the same shape as ``a``.

    See Also
    --------
    numpy.roll : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    >>> roll(x, 0)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Unsupported examples
    --------------------
    >>> x2 = np.reshape(x, (2, 5))
    >>> roll(x2, 1, axis=0)
    Traceback (most recent call last):
        ...
    ONPNotSupportedError: roll currently supports only packed vectors with axis=None.
    """
    pass
