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
matrix_arithmetic.py

This module implements the core arithmetic operations for encrypted tensors
using the OpenFHE library. Operations include addition, subtraction, multiplication,
matrix multiplication, and other mathematical operations.
"""

# Standard library imports
from typing import Optional

# Third-party imports
from numpy.typing import ArrayLike
from .dispatch import register_tensor_function


from openfhe_numpy.tensor.ctarray import CTArray
from openfhe_numpy.utils.errors import (
    ONP_ERROR,
    ONPIncompatibleShape,
    ONPNotSupportedError,
    ONPValueError,
    ONPDimensionError,
)
from openfhe_numpy.utils.typecheck import is_numeric_scalar
from openfhe_numpy import (
    ArrayEncodingType,
    EvalMatMulSquare,
    EvalReduceCumRows,
    EvalReduceCumCols,
)


##############################################################################
# BASIC ARITHMETIC OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Addition Operations
# ------------------------------------------------------------------------------
def _eval_add(lhs, rhs):
    """Internal function to evaluate addition between encrypted tensors."""
    crypto_context = lhs.data.GetCryptoContext()

    if isinstance(rhs, (int, float)):
        rhs = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)
        result = crypto_context.EvalAdd(lhs.data, rhs)
    else:
        result = crypto_context.EvalAdd(lhs.data, rhs.data)
    return CTArray(result, lhs.original_shape, lhs.batch_size, lhs.shape, lhs.order)


@register_tensor_function("add", [("CTArray", "CTArray"), ("CTArray", "PTArray")])
def add_ct(a, b):
    """Add two tensors."""
    if a.shape == ():
        return _eval_add(b, a)
    elif b.shape == ():
        return _eval_add(a, b)
    elif a.shape == b.shape:
        return _eval_add(a, b)
    raise ONPIncompatibleShape(a.shape, b.shape)


@register_tensor_function("add", ("CTArray", "scalar"))
def add_ct_scalar(a, scalar):
    """Add a scalar to a x."""
    return _eval_add(a, scalar)


@register_tensor_function(
    "add", [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")]
)
def add_block_ct(a, b):
    """Add two block tensors."""
    raise NotImplementedError("BlockPTArray and BlockCTArray addition not implemented yet.")


@register_tensor_function("add", [("BlockCTArray", "scalar")])
def add_block_ct_scalar(a, scalar):
    """Add a scalar to a block x."""
    raise NotImplementedError("BlockPTArray and scalar addition not implemented yet.")


# ------------------------------------------------------------------------------
# Subtraction Operations
# ------------------------------------------------------------------------------
def _eval_sub(lhs, rhs):
    """Internal function to evaluate subtraction between encrypted tensors."""
    crypto_context = (
        rhs.data.GetCryptoContext() if rhs.dtype == "CTArray" else lhs.data.GetCryptoContext()
    )

    if isinstance(rhs, (int, float)):
        rhs = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)
    else:
        rhs = rhs.data

    result = crypto_context.EvalSub(lhs.data, rhs)
    return lhs.clone(result)


@register_tensor_function(
    "subtract",
    [("CTArray", "CTArray"), ("CTArray", "PTArray"), ("PTArray", "CTArray")],
)
def subtract_ct(a, b):
    """Subtract two tensors."""
    if a.shape != b.shape:
        raise ONPIncompatibleShape(a.shape, b.shape)
    return _eval_sub(a, b)


@register_tensor_function("subtract", [("CTArray", "scalar"), ("scalar", "CTArray")])
def subtract_ct_scalar(a, b):
    """Subtract a scalar from a tensor or vice versa."""
    return _eval_sub(a, b.data)


# ------------------------------------------------------------------------------
# Multiplication Operations
# ------------------------------------------------------------------------------
def _eval_multiply(lhs, rhs):
    """Internal function to evaluate element-wise multiplication."""
    crypto_context = lhs.data.GetCryptoContext()
    if is_numeric_scalar(rhs):
        rhs_data = crypto_context.MakeCKKSPackedPlaintext([rhs] * lhs.batch_size)
    else:
        rhs_data = rhs.data

    result = crypto_context.EvalMult(lhs.data, rhs_data)
    return lhs.clone(result)


@register_tensor_function(
    "multiply",
    [("CTArray", "CTArray"), ("CTArray", "int"), ("CTArray", "PTArray")],
)
def multiply_ct(a, b):
    """Multiply two tensors element-wise."""
    if is_numeric_scalar(a) or is_numeric_scalar(b):
        return _eval_multiply(a, b)
    elif a.shape != b.shape:
        raise ONPIncompatibleShape(a.shape, b.shape)
    else:
        return _eval_multiply(a, b)


@register_tensor_function("multiply", ("CTArray", "scalar"))
def multiply_ct_scalar(a, scalar):
    """Multiply a tensor by a scalar."""
    return _eval_multiply(a, scalar)


@register_tensor_function(
    "multiply",
    [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")],
)
def multiply_block_ct(a, b):
    """Multiply two block tensors element-wise."""
    raise NotImplementedError("BlockPTArray multiplication not implemented yet.")


@register_tensor_function("multiply", [("BlockCTArray", "scalar")])
def multiply_block_ct_scalar(a, scalar):
    """Multiply a block tensor by a scalar."""
    raise NotImplementedError("BlockPTArray and scalar multiplication not implemented yet.")


##############################################################################
# MATRIX OPERATIONS
##############################################################################


# ------------------------------------------------------------------------------
# Matrix Multiplication Operations
# ------------------------------------------------------------------------------
def _eval_matvec_ct(lhs, rhs):
    """Internal function to evaluate matrix-vector multiplication."""
    if lhs.ndim == 2 and rhs.ndim == 1:
        if lhs.original_shape[1] != rhs.original_shape[0]:
            ONPIncompatibleShape(
                f"Matrix dimension [{lhs.original_shape}] mismatch with vector dimension [{rhs.shape}]"
            )
        if lhs.order == ArrayEncodingType.ROW_MAJOR and rhs.order == ArrayEncodingType.COL_MAJOR:
            cc = lhs.data.GetCryptoContext()
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
            cc = lhs.data.GetCryptoContext()
            ct_mult = cc.EvalMult(lhs.data, rhs.data)
            ct_prod = cc.EvalSumRows(ct_mult, lhs.nrows, lhs.extra["rowkey"], lhs.batch_size * 4)
            return CTArray(
                ct_prod,
                (lhs.original_shape[0],),
                lhs.batch_size,
                (lhs.shape[1], lhs.shape[0]),
                ArrayEncodingType.COL_MAJOR,
            )
        else:
            ONP_ERROR(
                f"Encoding styles of matrix ({lhs.order}) and vector ({rhs.order}) must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
            )
    elif lhs.ndim == 1 and rhs.ndim == 1:
        return _dot(lhs, rhs)
    else:
        ONPIncompatibleShape(lhs.original_shape, rhs.original_shape, "Matrix Product")


def _matmul_ct(lhs, rhs):
    """Internal function to evaluate matrix multiplication."""
    if lhs.is_encrypted and rhs.is_encrypted:
        if lhs.ndim == 2 and lhs.original_shape == rhs.original_shape:
            return lhs.clone(EvalMatMulSquare(lhs.data, rhs.data, lhs.ncols))
        elif rhs.ndim == 1:
            return _eval_matvec_ct(lhs, rhs)
        else:
            ONPIncompatibleShape(
                f"Matrix dimension mismatch for multiplication: {lhs.shape} and {rhs.shape}"
            )


@register_tensor_function("matmul", [("CTArray", "CTArray")])
def matmul_ct(a, b):
    """Perform matrix multiplication between two tensors."""
    return _matmul_ct(a, b)


# ------------------------------------------------------------------------------
# Dot Product Operations
# ------------------------------------------------------------------------------
def _dot(lhs, rhs):
    """Internal function to evaluate dot product."""
    if lhs.ndim == 1 and rhs.ndim == 1:
        crypto_context = lhs.data.GetCryptoContext()
        ciphertext = crypto_context.EvalInnerProduct(lhs.data, rhs.data, lhs.batch_size)
        return CTArray(ciphertext, (), lhs.batch_size, (), ArrayEncodingType.ROW_MAJOR)
    else:
        return lhs._matmul(rhs)


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
    """Exponentiate a matrix to power k using homomorphic multiplication."""
    if not isinstance(exp, int):
        ONP_ERROR(f"Exponent must be integer, got {type(exp).__name__}")

    if exp < 0:
        ONP_ERROR("Negative exponent not supported in homomorphic encryption")

    if exp == 0:
        # return algebra.eye(tensor))
        pass

    if exp == 1:
        return x.clone()

    # Binary exponentiation implementation
    base = x.clone()
    result = None

    while exp:
        if exp & 1:
            result = base if result is None else base @ result
        base = base @ base
        exp >>= 1
    return result


@register_tensor_function("pow", [("CTArray", "int")])
def pow_ct(a, exp):
    """Raise a tensor to an integer power."""
    return _pow(a, exp)


@register_tensor_function("pow", [("BlockCTArray", "int")])
def pow_block_ct(a, exp):
    """Raise a block tensor to an integer power."""
    raise NotImplementedError("BlockPTArray power not implemented yet.")


# ------------------------------------------------------------------------------
# Cumulative Sum Operations
# ------------------------------------------------------------------------------


@register_tensor_function(
    "cumulative_sum",
    [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")],
)
def cumulative_sum_ct(obj, axis=0, keepdims=True):
    """Compute cumulative sum of a tensor along specified axis."""
    # return _cumulative_sum_ct(a, axis, keepdims)
    return obj.cumulative_sum(axis)


@register_tensor_function(
    "cumulative_sum", [("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")]
)
def cumulative_sum_block_ct(obj, axis=0, keepdims=True):
    """Compute cumulative sum of a block tensor along specified axis."""
    raise NotImplementedError("BlockPTArray cumulative not implemented yet.")


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
        ONP_ERROR("Axis must be 0 or 1 for cumulative sum operation")

    if axis == 0:
        ciphertext = EvalReduceCumRows(a.data, a.ncols, a.original_shape[1])
    else:
        ciphertext = EvalReduceCumCols(a.data, a.ncols)
    return a.clone(ciphertext)


@register_tensor_function("cumulative_reduce", [("CTArray", "int", "bool")])
def cumulative_reduce_ct(a, axis=0, keepdims=False):
    """Compute cumulative reduction of a tensor along specified axis."""
    return _reduce_ct(a, axis, keepdims)


@register_tensor_function("cumulative_reduce", [("BlockCTArray", "int")])
def cumulative_reduce_block_ct(a, axis=0, keepdims=False):
    """Compute cumulative reduction of a block tensor along specified axis."""
    raise NotImplementedError("BlockPTArray power not implemented yet.")


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
            ct_sum = cc.EvalSumRows(fhe_data, ncols, x.extra["rowkey"], x.batch_size * 4)
            padded_shape = x.shape
            order = ArrayEncodingType.COL_MAJOR
        elif order == ArrayEncodingType.COL_MAJOR:
            ct_sum = cc.EvalSumCols(fhe_data, nrows, x.extra["colkey"])
            padded_shape = (ncols, nrows)
            order = ArrayEncodingType.ROW_MAJOR

        else:
            ONPNotSupportedError(f"Not support the current encoding [{order}] ")

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
            ct_sum = cc.EvalSumRows(fhe_data, nrows, x.extra["rowkey"], x.batch_size * 4)
            padded_shape = (ncols, nrows)
            order = ArrayEncodingType.COL_MAJOR
        else:
            ONPNotSupportedError(f"Not support the current encoding [{order}]")

        if keepdims:
            shape = (rows, 1)
        else:
            shape = (rows,)

    else:
        ONPValueError(f"Invalid axis [{axis}]")

    return CTArray(ct_sum, shape, x.batch_size, padded_shape, order)


def _ct_sum_vector(
    x: ArrayLike,
    axis: Optional[int] = None,
):
    crypto_context = x.data.GetCryptoContext()
    if axis is not None:
        ONPDimensionError(f"The dimension is invalid axis = {axis}")
    ct_sum = crypto_context.EvalSum(x.data, x.shape[0])
    return CTArray(ct_sum, (), x.batch_size, x.shape, x.order)


@register_tensor_function("sum", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def sum_ct(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    if x.ndim == 2:
        return _ct_sum_matrix(x, axis, keepdims)
    elif x.ndim == 1:
        return _ct_sum_vector(x, axis)
    else:
        ONPDimensionError(f"The dimension is invalid = {x.ndims}")


# ------------------------------------------------------------------------------
# Mean Operations
# ------------------------------------------------------------------------------


@register_tensor_function("mean", [("CTArray",), ("CTArray", "int"), ("CTArray", "int", "bool")])
def mean_ct(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False):
    cc = x.data.GetCryptoContext()
    nrows, ncols = x.original_shape
    sum_x = sum_ct(x, axis, keepdims)
    if axis is None:
        ct_mean = cc.EvalMult(sum_x.data, 1.0 / (nrows * ncols))
    elif axis == 0:  # sum over rows
        ct_mean = cc.EvalMult(sum_x.data, 1.0 / nrows)
    elif axis == 1:  # sum over cols
        ct_mean = cc.EvalMult(sum_x.data, 1.0 / ncols)
    else:
        ONPDimensionError(f"The dimension is invalid axis = {axis}")

    return CTArray(ct_mean, sum_x.original_shape, sum_x.batch_size, sum_x.shape, sum_x.order)


# ------------------------------------------------------------------------------
# Rotation Operations
# ------------------------------------------------------------------------------


@register_tensor_function("roll", [("CTArray", "int"), ("CTArray", "int", "int")])
def roll(x: ArrayLike, shift: int, axis: Optional[int] = None) -> ArrayLike:
    if axis is None:
        return _ct_vector_rotation(x, -shift)
    else:
        ONP_ERROR(f"This function only supports packed vector")


def _ct_vector_rotation(ctv: CTArray, shift: int):
    cc = ctv.data.GetCryptoContext()
    ct_rotated = cc.EvalRotate(ctv.data, shift)
    ctv_cloned = ctv.clone()
    ctv_cloned.data = ct_rotated
    return ctv_cloned
