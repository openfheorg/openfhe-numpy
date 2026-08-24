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
Crypto context operations and key generation utilities for homomorphic operations.

This module provides functions for generating rotation, accumulation, and other specialized
keys needed for various homomorphic operations in OpenFHE-NumPy.
"""

from typing import Any
import openfhe
import openfhe_numpy as backend  # Import from cpp source
from ..utils.packing import _is_row_major, _is_col_major
from ..utils.errors import ONPValueError


def accumulation_depth(nrows: int, ncols: int, accumulate_by_rows: bool):
    """
    Compute the CKKS multiplicative depth needed to sum over a matrix.

    Parameters
    ----------
    nrows : int
        Number of rows in the matrix
    ncols : int
        Number of columns in the matrix
    accumulate_by_rows : bool
        Whether to sum over rows or columns

    Returns
    -------
    int
        Required multiplicative depth
    """
    return backend.MulDepthAccumulation(nrows, ncols, accumulate_by_rows)


def sum_row_keys(secret_key: openfhe.PrivateKey, ncols: int = 0, slots: int = 0):
    """
    Generate keys for summing rows in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of cols for the matrix, by default 0
    slots: int
        The total plaintext slots

    Returns
    -------
    object
        Generated sum keys
    """
    context = secret_key.GetCryptoContext()
    return context.EvalSumRowsKeyGen(secret_key, None, ncols, 0)


def sum_col_keys(secret_key: openfhe.PrivateKey, ncols: int = 0):
    """
    Generate keys for summing columns in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of columns in the matrix, by default 0
    """
    context = secret_key.GetCryptoContext()
    return context.EvalSumColsKeyGen(secret_key)


def gen_sum_key(secret_key: openfhe.PrivateKey):
    """
    Generate keys for summing all slots

    Parameters
    ----------
    secret_key : openfhe.PrivateKey

    """
    context = secret_key.GetCryptoContext()
    context.EvalSumKeyGen(secret_key)


def gen_accumulate_rows_key(secret_key: openfhe.PrivateKey, ncols: int):
    """
    Generate keys for cumulative sum of rows in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    backend.EvalSumCumRowsKeyGen(secret_key, ncols)


def gen_accumulate_cols_key(secret_key: openfhe.PrivateKey, ncols: int):
    """
    Generate keys for cumulative sum of columns in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    backend.EvalSumCumColsKeyGen(secret_key, ncols)


def gen_rotation_keys(secret_key: openfhe.PrivateKey, shifts: list):
    """
    Generate rotation keys for the specified indices.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    shifts : list
        List of rotation indices to generate keys for (negated OpenFHE implementation).
    """

    standard_indices = [-x for x in shifts]
    context = secret_key.GetCryptoContext()
    context.EvalRotateKeyGen(secret_key, standard_indices)


def gen_lintrans_keys(
    secret_key: openfhe.PrivateKey,
    block_size: int,
    linear_transform_type,
    repetitions: int = 0,
):
    """
    Generate keys for linear transformations.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    block_size : int
        linear_transform_type size for the matrix
    linear_transform_type : LinTransType
        Type of linear transformation
    repetitions : int, optional
        Number of repetitions, by default 0
    """
    backend.EvalLinTransKeyGen(secret_key, block_size, linear_transform_type, repetitions)


def gen_square_matmult_key(secret_key: openfhe.PrivateKey, block_size: int):
    """
    Generate keys for square matrix multiplication.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    block_size : int
        Block size for the matrix
    """
    backend.EvalSquareMatMultRotateKeyGen(secret_key, block_size)


def gen_transpose_keys(secret_key: openfhe.PrivateKey, ctm_matrix):
    """
    Generate keys for matrix transposition.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ctm_matrix : CTArray
        The ciphertext matrix to transpose
    """
    if ctm_matrix.ndim < 2:
        return

    rows, cols = ctm_matrix.shape
    if rows == 1 or cols == 1:
        return
    backend.EvalTransposeKeyGen(secret_key, rows, cols)


def gen_transform_keys(secret_key: openfhe.PrivateKey, tensor):
    """Generate the keys needed to transform a matrix in either direction."""
    if tensor.ndim < 2:
        return

    from ..tensor.block_ctarray import BlockCTArray

    if isinstance(tensor, BlockCTArray):
        tensor = tensor.data[0]
    if tensor.order not in (
        backend.ArrayEncodingType.ROW_MAJOR,
        backend.ArrayEncodingType.COL_MAJOR,
    ):
        raise ValueError("Order transform supports only ROW_MAJOR and COL_MAJOR.")

    gen_transpose_keys(secret_key, tensor)


##############################################################################
# BLOCK ARITHMETIC OPERATIONS
##############################################################################


# [CTArray] attach key for mat@vec product
def attach_matvec_keys(matrix, secret_key, emit: bool = False) -> tuple[str, Any] | None:
    """Attach the summation key required for matrix-vector multiplication.

    Required keys:
    - ROW_MAJOR matrix @ COL_MAJOR vector uses extra["colkey"].
    - COL_MAJOR matrix @ ROW_MAJOR vector uses extra["rowkey"].

    Limitation:
    - The key is generated from one matrix layout. All blocks in a block matrix
      are expected to share the same packing order and logical column count.
    """
    if matrix.ndim != 2:
        raise ONPValueError("attach_matvec_keys expects a 2-D matrix.")

    if _is_row_major(matrix.order):
        key_name = "colkey"
        key = sum_col_keys(secret_key)

    elif _is_col_major(matrix.order):
        key_name = "rowkey"
        key = sum_row_keys(secret_key, matrix.ncols, matrix.batch_size)

    else:
        raise ONPValueError(f"Unsupported packing order: {matrix.order}")

    matrix.extra[key_name] = key

    if emit:
        return key_name, key

    return None


# [BlockCTArray] attach key for mat@vec product
def attach_block_matvec_keys(
    block_matrix, secret_key, emit: bool = False
) -> tuple[str, Any] | None:
    """Attach summation keys to encrypted matrix blocks for block matvec.

    Required keys:
    - ROW_MAJOR matrix @ COL_MAJOR vector uses ``extra["colkey"]``.
    - COL_MAJOR matrix @ ROW_MAJOR vector uses ``extra["rowkey"]``.

    The key-generation parameters must match the CTArray evaluator:
    - EvalSumCols(..., lhs.ncols, ...)
    - EvalSumRows(..., lhs.nrows, ..., lhs.batch_size)
    """

    if block_matrix.ndim != 2 or len(block_matrix.data) <= 0:
        raise ONPValueError("attach_matvec_keys expects a 2-D block matrix.")

    reference = block_matrix.data[0]
    key_name, key = attach_matvec_keys(reference, secret_key, True)

    for block in block_matrix.data:
        block.extra[key_name] = key

    if emit:
        return key_name, key

    return None


def attach_block_sum_keys(block_matrix, secret_key) -> None:
    """Attach both axis-reduction keys to every encrypted matrix block.

    All blocks share one physical shape, so the two generated key maps can be
    reused. For example, after ``attach_block_sum_keys(x, sk)``, both
    ``x.sum(axis=0)`` and ``x.sum(axis=1)`` are available regardless of packing
    order.
    """
    if block_matrix.ndim != 2 or not block_matrix.data:
        raise ONPValueError("attach_block_sum_keys expects a non-empty 2-D block matrix.")

    reference = block_matrix.data[0]
    if _is_row_major(reference.order):
        row_width = reference.ncols
    elif _is_col_major(reference.order):
        row_width = reference.nrows
    else:
        raise ONPValueError(f"Unsupported packing order: {reference.order}")

    row_key = sum_row_keys(secret_key, row_width, reference.batch_size)
    col_key = sum_col_keys(secret_key)
    for block in block_matrix.data:
        block.extra["rowkey"] = row_key
        block.extra["colkey"] = col_key


def gen_block_transpose_keys(secret_key: openfhe.PrivateKey, block_matrix) -> None:
    """Generate transpose keys from one representative matrix block."""
    if block_matrix.ndim == 1:
        return
    if block_matrix.ndim != 2 or not block_matrix.data:
        raise ONPValueError("gen_block_transpose_keys expects a non-empty 2-D block matrix.")

    gen_transpose_keys(secret_key, block_matrix.data[0])
