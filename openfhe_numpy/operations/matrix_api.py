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

from typing import Optional
from numpy.typing import ArrayLike
from .dispatch import tensor_function_api


# ===========================
# Element-wise Operations
# ===========================


def add(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
     Element-wise addition of two arrays (or array and scalar).

    See Also
    --------
    numpy.add
    """
    return _add_dispatch(a, b)


@tensor_function_api("add", binary=True)
def _add_dispatch(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Dispatch for 'add' operation."""
    pass


def subtract(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
     Element-wise subtraction of two arrays (or array and scalar).

    See Also
    --------
    numpy.subtract
    """
    return _subtract_dispatch(a, b)


@tensor_function_api("subtract", binary=True)
def _subtract_dispatch(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Dispatch for 'subtract' operation."""
    pass


@tensor_function_api("multiply", binary=True)
def multiply(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """
    Element-wise multiplication of two arrays (or array and scalar).

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


@tensor_function_api("pow", binary=True)
def pow(a: ArrayLike, exponent: int) -> ArrayLike:
    """

    For each element of the tensor, it raises that element to the given power.

    Note
    ----
    This only supports integer exponents due to homomorphic-encryption constraints.

    Parameters
    ----------
    a : ArrayLike
        Base tensor.
    exponent : int
        Non-negative integer exponent.

    Returns
    -------
    out : ArrayLike
        Element-wise 'a' raised to 'exponent'.

    See Also
    --------
    numpy.pow : Corresponding element-wise power function.
    numpy.linalg.matrix_power : Repeated matrix multiplication for square matrices.

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

    returns
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
    """
    Matrix multiply two tensors.

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
    """
    Transpose array axes (for 2-D: swap rows and columns).
    For 1-D inputs (vectors), the array is returned unchanged.
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


@tensor_function_api("cumulative_sum", binary=False)
def cumulative_sum(x: ArrayLike, /, *, axis: Optional[int] = None) -> ArrayLike:
    """
        Compute the cumulative sum of tensor elements along a specified axis.\
        - For 1D inputs, axis must be None.
        - For 2D inputs, axis must be 0 or 1.
        - The include_initial argument is not supported.

        Parameters
        ----------
        a : ArrayLike
            Input tensor.
        axis : int, optional
            Axis along which to compute the sum. Default is 0.

        Returns
        -------
        out : ArrayLike
            Cumulative sum along an axis.

        See Also
        --------
        numpy.cumulative_sum : Corresponding NumPy function.

        Examples
        --------
        >>> import numpy as onp
        >>> cumulative_sum(np.array([[1, 2], [3, 4]]), axis=1)
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
    numpy.cumulative_sum : Similar operation for sum.

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
        Axis along which to compute the sum. Default is 0.
        0: sum over rows
        1: sum over cols
    keepdims : bool, optional
        If True, retains reduced dimensions. Default is False.

    Returns
    -------
    out : ArrayLike
        Sum of 'a' elements.

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
        Axis along which to compute the mean. Default is 0.
    keepdims : bool, optional
        If True, retains reduced dimensions. Default is False.

    Returns
    -------
    out : ArrayLike
        Mean of 'a' elements.

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
def roll(a: ArrayLike, shift, axis: Optional[int] = None) -> ArrayLike:
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at the first.

    Parameters
    ----------
    a : ArrayLike
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted. If a tuple, then
        axis must be a tuple of the same size, and each of the given axes is
        shifted by the corresponding number. If an int while axis is a tuple
        of ints, then the same value is used for all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted. By default, the array
        is flattened before shifting, after which the original shape is restored.

    Returns
    -------
    res : ArrayLike
        Output array, with the same shape as a.

    See Also
    --------
    numpy.roll : Corresponding NumPy function.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> roll(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> x2 = np.reshape(x, (2, 5))
    >>> roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> roll(x2, (1, 1), axis=(1, 0))  # Multiple axes
    array([[9, 5, 6, 7, 8],
           [4, 0, 1, 2, 3]])
    """
    pass


# ===========================
# Planned Future Functionality
# ===========================

# Array Creation Functions:

# def zeros(shape, crypto_context, key):
#     """Create an encrypted array of zeros."""
#     pass


# def ones(shape, crypto_context, key):
#     """Create an encrypted array of ones."""
#     pass


# def eye(n, crypto_context, key):
#     """Create an encrypted identity matrix."""
#     pass


# Broadcasting Support:

# def _get_broadcast_shape(self, other):
#     """Calculate broadcast shape between tensors."""
#     # Implementation...


# def _broadcast_tensor(self, target_shape):
#     """Broadcast this tensor to target shape."""
#     # Implementation...


# Array Manipulation Methods:

# def reshape(self, new_shape):
#     """Reshape tensor to new dimensions."""
#     # Implementation...


# def concat(tensors, axis=0):
#     """Concatenate tensors along specified axis."""
#     # Implementation...
