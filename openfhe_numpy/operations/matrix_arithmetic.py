# ==============================================================================
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
# ================================================================================
"""
matrix_arithmetic.py

This module implements the core arithmetic operations for encrypted tensors
using the OpenFHE library. Operations include addition, subtraction, multiplication,
matrix multiplication, and other mathematical operations.
"""

# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
from numpy.typing import ArrayLike
from .dispatch import register_tensor_function


from openfhe_numpy.tensor.ctarray import CTArray
from openfhe_numpy.utils.errors import (
    ONPError,
    ONPIncompatibleShapeError,
    ONPNotSupportedError,
    ONPValueError,
    ONPDimensionError,
)
from openfhe_numpy.openfhe_numpy import (
    ArrayEncodingType,
    EvalMatMulSquare,
    EvalReduceCumRows,
    EvalReduceCumCols,
)
from openfhe_numpy.operations.arithmetic_utils import (
    _eval_binary,
    _eval_scalar_binary,
    _normalize_axis,
)


##############################################################################
# BASIC ARITHMETIC OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Addition Operations
# ------------------------------------------------------------------------------
@register_tensor_function("add", [("CTArray", "CTArray"), ("CTArray", "PTArray")])
def add_ct(a, b):
    """Add two tensors."""
    return _eval_binary(a, b, "add")


@register_tensor_function("add", ("CTArray", "scalar"))
def add_cta_scalar(a, scalar):
    """Add a scalar to an encrypted tensor's logical slots."""
    return _eval_scalar_binary(a, scalar, "add")


# ------------------------------------------------------------------------------
# Subtraction Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "subtract",
    [("CTArray", "CTArray"), ("CTArray", "PTArray"), ("PTArray", "CTArray")],
)
def subtract_ct(a, b):
    """Subtract two tensors."""
    return _eval_binary(a, b, "subtract")


@register_tensor_function("subtract", [("CTArray", "scalar")])
def subtract_ct_scalar(a, scalar):
    return _eval_scalar_binary(a, scalar, "subtract")


@register_tensor_function("subtract", [("scalar", "CTArray")])
def subtract_scalar_ct(scalar, a):
    return _eval_scalar_binary(a, scalar, "subtract", reverse=True)


# ------------------------------------------------------------------------------
# Multiplication Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "multiply",
    [("CTArray", "CTArray"), ("CTArray", "PTArray")],
)
def multiply_ct(a, b):
    """Multiply two tensors element-wise."""
    return _eval_binary(a, b, "multiply")


@register_tensor_function("multiply", ("CTArray", "scalar"))
def multiply_ct_scalar(a, scalar):
    """Multiply a tensor by a scalar."""
    return _eval_scalar_binary(a, scalar, "multiply")


##############################################################################
# MATRIX OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Matrix Multiplication Operations
# ------------------------------------------------------------------------------
def _eval_matvec_ct(lhs, rhs):
    """Internal function to evaluate matrix-vector multiplication."""
    cc = rhs.data.GetCryptoContext() if rhs.dtype == "CTArray" else lhs.data.GetCryptoContext()
    if lhs.ndim == 2 and rhs.ndim == 1:
        if lhs.original_shape[1] != rhs.original_shape[0]:
            raise ONPIncompatibleShapeError(
                lhs.original_shape,
                rhs.original_shape,
                f"Matrix dimension [{lhs.original_shape}] mismatch with vector dimension [{rhs.shape}]",
            )

        if lhs.order == ArrayEncodingType.ROW_MAJOR and rhs.order == ArrayEncodingType.COL_MAJOR:
            ct_mult = cc.EvalMult(lhs.data, rhs.data)
            ct_prod = cc.EvalSumCols(ct_mult, lhs.ncols, lhs.extra["colkey"])
            return CTArray(
                ct_prod,
                (lhs.original_shape[0],),
                lhs.batch_size,
                (lhs.shape[0], lhs.shape[1]),
                ArrayEncodingType.ROW_MAJOR,
            )

        elif lhs.order == ArrayEncodingType.COL_MAJOR and rhs.order == ArrayEncodingType.ROW_MAJOR:
            ct_mult = cc.EvalMult(lhs.data, rhs.data)
            ct_prod = cc.EvalSumRows(ct_mult, lhs.nrows, lhs.extra["rowkey"], 0)
            return CTArray(
                ct_prod,
                (lhs.original_shape[0],),
                lhs.batch_size,
                (lhs.shape[1], lhs.shape[0]),
                ArrayEncodingType.COL_MAJOR,
            )

        else:
            raise ONPError(
                f"Encoding styles of matrix ({lhs.order}) and vector ({rhs.order}) must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
            )
    elif lhs.ndim == 1 and rhs.ndim == 1:
        return _dot(lhs, rhs)
    else:
        raise ONPIncompatibleShapeError(lhs.original_shape, rhs.original_shape, "Matrix Product")


def _matmul_ct(lhs, rhs):
    """Internal function to evaluate matrix multiplication."""
    # matrix @ matrix
    if lhs.ndim == 2 and lhs.original_shape == rhs.original_shape:
        return CTArray(
            EvalMatMulSquare(lhs.data, rhs.data, lhs.ncols),
            lhs.original_shape,
            lhs.batch_size,
            lhs.shape,
            lhs.order,
        )

    # matrix @ vector
    elif rhs.ndim == 1:
        return _eval_matvec_ct(lhs, rhs)
    else:
        raise ValueError(
            f"Dimension mismatch for multiplication ({lhs.original_shape} @ {rhs.original_shape})"
        )


@register_tensor_function(
    "matmul", [("CTArray", "CTArray"), ("CTArray", "PTArray"), ("PTArray", "CTArray")]
)
def matmul_ct(a, b):
    """Perform matrix multiplication between two tensors."""
    return _matmul_ct(a, b)


# ------------------------------------------------------------------------------
# Dot Product Operations
# ------------------------------------------------------------------------------
def _dot(lhs, rhs):
    """Internal function to evaluate dot product."""
    crypto_context = lhs.data.GetCryptoContext()

    # inner product: <vector, vector>
    if lhs.ndim == 1 and rhs.ndim == 1:
        ciphertext = crypto_context.EvalInnerProduct(lhs.data, rhs.data, lhs.batch_size)
        return CTArray(ciphertext, (), lhs.batch_size, (), ArrayEncodingType.ROW_MAJOR)
    else:
        return lhs.__matmul__(rhs)


@register_tensor_function("dot", [("CTArray", "CTArray")])
def dot_ct(a, b):
    """Compute dot product between two tensors."""
    return _dot(a, b)


# ------------------------------------------------------------------------------
# Transpose Operations
# ------------------------------------------------------------------------------


@register_tensor_function("transpose", [("CTArray",)])
def transpose_ct(a):
    """Transpose array axes (2-D: swap rows/cols). For 1-D, the array is unchanged."""
    return a._transpose()


##############################################################################
# ADVANCED OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Power Operations
# ------------------------------------------------------------------------------
def _pow(x, exp: int):
    """Raise a tensor to an integer power element-wise (NumPy ``power`` semantics).

    Each element ``x_i`` is raised to ``exp`` via homomorphic element-wise
    multiplication (``x * x * ... * x``).
    """
    if not isinstance(exp, int):
        raise ONPError(f"Exponent must be integer, got {type(exp).__name__}")

    if exp < 0:
        raise ONPError("Negative exponent not supported in homomorphic encryption")

    if exp == 0:
        raise ONPNotSupportedError("Element-wise power with exponent 0 (all-ones) is not supported")

    if exp == 1:
        return x.clone()

    base = x.clone()
    result = None

    while exp:
        if exp & 1:
            result = base if result is None else base * result
        exp >>= 1
        if exp:
            base = base * base
    return result


@register_tensor_function("power", [("CTArray", "int")])
def pow_ct(a, exp):
    """Raise a tensor to an integer power element-wise."""
    return _pow(a, exp)


# ------------------------------------------------------------------------------
# Cumulative Sum Operations
# ------------------------------------------------------------------------------


@register_tensor_function(
    "cumsum",
    [
        ("CTArray",),
        ("CTArray", "NoneType"),
        ("CTArray", "int"),
        ("CTArray", "scalar"),
    ],
)
def cumsum_ct(obj, axis=None):
    """Compute cumulative sum of a tensor along specified axis."""
    return obj.cumsum(axis=axis)


# ------------------------------------------------------------------------------
# Cumulative Reduce Operations
# ------------------------------------------------------------------------------
def _reduce_ct(a, axis=0, keepdims=False):
    """
    Compute the cumulative reduce of tensor elements along a given axis.

    Parameters
    ----------
    a : CTArray
        Input encrypted x.
    axis : int, optional
        Axis along which the cumulative reduction is computed. Default is 0.
    keepdims : bool, optional
        Whether to keep the dimensions of the original x. Default is False.

    Returns
    -------
    CTArray
        A new tensor with cumulative reduction along the specified axis.
    """
    if axis not in (0, 1):
        raise ONPError("Axis must be 0 or 1 for cumulative sum operation")

    if axis == 0:
        ciphertext = EvalReduceCumRows(a.data, a.ncols, a.original_shape[1])
    else:
        ciphertext = EvalReduceCumCols(a.data, a.ncols)
    return a.clone(ciphertext)


@register_tensor_function("cumulative_reduce", [("CTArray", "int", "bool")])
def cumulative_reduce_ct(a, axis=0, keepdims=False):
    """Compute cumulative reduction of a tensor along specified axis."""
    axis = _normalize_axis("cumulative_reduce", axis, a.ndim)
    return _reduce_ct(a, axis, keepdims)


# ------------------------------------------------------------------------------
# Sum Operations
# ------------------------------------------------------------------------------


# NOTE: Sum Operations
# Here is a running example illustrating the behavior of onp.sum when summing over axes 0 and 1
# Original matrix: [11 // 21 // 31 // 26]
# Expected result:
#                   - axis = 0: 8 9
#                   - axis = 1: 2 // 3 // 4 // 8
# Packed matrix behavior
# A. Row-Major: 11 21 31 26
# 1. Sum over rows: axis = 0.
#    using EvalSumRows(rows = 4, cols = 2)
#     11 21 31 22
#     21 31 22 11
#     32 52 53 33
#     53 33 32 52
#     89 89 89 89
# 2. Sum over columns: axis = 1.
#    using EvalSumCols(rows = 4, cols = 2)
#     11 21 31 26
#     12 13 12 61
#     23 34 43 87
#     22 33 44 88
# B. Column-Major: 1232 1116
# 1. Sum over rows: axis = 0.
#    using EvalSumCol(rows = 2, cols = 4)
#     1232 1116
#     8888 9999
# 2. Sum over columns: axis = 1.
#    using EvalSumRows(rows = 2, cols = 4)
#     12 32 11 16
#     11 16 12 32
#     23 48 23 48


def _ct_sum_matrix(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = True):
    """
    This function computes a sum of a padded matrix. It is similar to np.sum
    """

    cc = x.data.GetCryptoContext()
    rows, cols = x.original_shape
    nrows, ncols = x.shape
    order = x.order
    fhe_data = x.data

    if axis is None:
        # Sum all elements in a packed-encoded matrix ciphertext: fhe_data
        ct_sum = cc.EvalSum(fhe_data, nrows * ncols - 1)
        if keepdims:
            shape, padded_shape = (1, 1), x.shape
        else:
            shape, padded_shape = (), ()

    elif axis == 0:
        # Sum across each row of a packed_encoded matrix ciphertext: fhe_data
        if order == ArrayEncodingType.ROW_MAJOR:
            ct_sum = cc.EvalSumRows(fhe_data, ncols, x.extra["rowkey"], 0)
            padded_shape = x.shape
            order = ArrayEncodingType.COL_MAJOR
        elif order == ArrayEncodingType.COL_MAJOR:
            ct_sum = cc.EvalSumCols(fhe_data, nrows, x.extra["colkey"])
            padded_shape = (ncols, nrows)
            order = ArrayEncodingType.ROW_MAJOR

        else:
            raise ONPNotSupportedError(f"Not support the current encoding [{order}] ")

        if keepdims:
            shape = (cols, 1)
        else:
            shape = (cols,)

    elif axis == 1:
        # Sum across each column of a packed_encoded matrix ciphertext: fhe_data
        if order == ArrayEncodingType.ROW_MAJOR:
            ct_sum = cc.EvalSumCols(fhe_data, ncols, x.extra["colkey"])
            padded_shape = x.shape
            order = ArrayEncodingType.ROW_MAJOR
        elif order == ArrayEncodingType.COL_MAJOR:
            ct_sum = cc.EvalSumRows(fhe_data, nrows, x.extra["rowkey"], 0)
            padded_shape = (ncols, nrows)
            order = ArrayEncodingType.COL_MAJOR
        else:
            raise ONPNotSupportedError(f"Not support the current encoding [{order}]")

        if keepdims:
            shape = (rows, 1)
        else:
            shape = (rows,)

    else:
        raise ONPValueError(f"Invalid axis [{axis}]")

    return CTArray(ct_sum, shape, x.batch_size, padded_shape, order)


def _ct_sum_vector(
    x: ArrayLike,
    axis: Optional[int] = None,
):
    crypto_context = x.data.GetCryptoContext()
    if axis not in (None, 0):
        raise ONPDimensionError(f"The dimension is invalid axis = {axis}")
    ct_sum = crypto_context.EvalSum(x.data, x.shape[0])
    return CTArray(ct_sum, (), x.batch_size, x.shape, x.order)


@register_tensor_function("sum", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def sum_ct(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    if x.ndim not in (1, 2):
        raise ONPDimensionError(f"sum requires a vector or matrix; got {x.ndim}D.")

    axis = _normalize_axis("sum", axis, x.ndim)
    if x.ndim == 2:
        return _ct_sum_matrix(x, axis, keepdims)
    return _ct_sum_vector(x, axis)


# ------------------------------------------------------------------------------
# Mean Operations
# ------------------------------------------------------------------------------


@register_tensor_function("mean", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def mean_ct(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    if x.ndim not in (1, 2):
        raise ONPDimensionError(f"mean requires a vector or matrix; got {x.ndim}D.")

    axis = _normalize_axis("mean", axis, x.ndim)
    cc = x.data.GetCryptoContext()
    sum_x = sum_ct(x, axis, keepdims)

    if x.ndim == 1:
        count = x.original_shape[0]
    else:
        nrows, ncols = x.original_shape
        if axis is None:
            count = nrows * ncols
        elif axis == 0:
            count = nrows
        else:
            count = ncols

    ct_mean = cc.EvalMult(sum_x.data, 1.0 / count)

    return CTArray(
        ct_mean,
        sum_x.original_shape,
        sum_x.batch_size,
        sum_x.shape,
        sum_x.order,
    )


# ------------------------------------------------------------------------------
# Rotation Operations
# ------------------------------------------------------------------------------


@register_tensor_function(
    "roll",
    [
        ("CTArray", "int"),
        ("CTArray", "int", "int"),
        ("CTArray", "scalar", "scalar"),
    ],
)
def roll(x: ArrayLike, shift: int, axis: Optional[int] = None) -> ArrayLike:
    if x.ndim != 1:
        raise ONPNotSupportedError("roll currently supports only packed vectors.")

    axis = _normalize_axis("roll", axis, x.ndim)
    if axis is not None:
        raise ONPNotSupportedError(
            "roll currently supports only packed vectors with axis=None."
        )
    return _ct_vector_rotation(x, -shift)


def _ct_vector_rotation(ctv: CTArray, shift: int):
    cc = ctv.data.GetCryptoContext()
    ct_rotated = cc.EvalRotate(ctv.data, shift)
    ctv_cloned = ctv.clone()
    ctv_cloned.data = ct_rotated
    return ctv_cloned
