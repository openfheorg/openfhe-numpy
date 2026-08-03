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
from ..utils.matlib import next_power_of_two
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
    if ctm_matrix.ndim == 1:
        ncols = 1
    else:
        ncols = ctm_matrix.ncols

    backend.EvalLinTransKeyGen(secret_key, ncols, backend.LinTransType.TRANSPOSE)


def generate_slicing_key(secret_key, original_shape):
    """
    Pre-generate all rotation keys needed for any possible slicing
    of a CTArray with the given original_shape.
    """

    indices = set()

    if len(original_shape) == 1:
        n = original_shape[0]
        for i in range(n):
            indices.add(i)
        for i in range(1, n):
            indices.add(-i)

    elif len(original_shape) == 2:
        nrow, ncol = original_shape
        nrow_pow_2 = next_power_of_two(nrow)
        ncol_pow_2 = next_power_of_two(ncol)

        for r in range(nrow):
            for c in range(ncol):
                indices.add(r * ncol_pow_2 + c)
                indices.add(c * nrow_pow_2 + r)

        for i in range(1, max(nrow, ncol)):
            indices.add(-i)

        # Rotation indices to collapse any sub-matrix result back to slot 0.
        #
        # The naive enumeration scans every (res_nrow, res_ncol) result shape and
        # every (r, c) offset within it -- Theta(nrow^2 * ncol^2). Each added value
        # depends on only one of the two result extents (through its power-of-two
        # padding), so the identical index set is produced in O(nrow * ncol) by
        # grouping result extents that share a padding and taking the widest offset
        # range for that group.
        def _widest_extent_by_padding(size):
            """Map next_power_of_two(k) -> largest k in 1..size sharing that padding."""
            widest = {}
            for k in range(1, size + 1):
                widest[next_power_of_two(k)] = k  # k ascends, so this stays the max
            return widest

        # Column-padded offsets: value depends on res_ncol's padding; rows span 0..nrow.
        for pad_c, max_c in _widest_extent_by_padding(ncol).items():
            for r in range(nrow):
                for c in range(max_c):
                    indices.add(-(pad_c * r + c))

        # Row-padded offsets: value depends on res_nrow's padding; cols span 0..ncol.
        for pad_r, max_r in _widest_extent_by_padding(nrow).items():
            for c in range(ncol):
                for r in range(max_r):
                    indices.add(-(pad_r * c + r))

    # NOTE: index 0 is intentionally kept. Element extraction rotates the target
    # element to slot 0 via EvalRotate(ct, 0) for position-0 elements, which needs
    # the identity-rotation key (OpenFHE automorphism index 1).
    if indices:
        cc = secret_key.GetCryptoContext()
        cc.EvalRotateKeyGen(secret_key, sorted(indices))


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
