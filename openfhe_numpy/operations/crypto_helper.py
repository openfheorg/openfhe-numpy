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

import openfhe
import openfhe_numpy as backend  # Import from cpp source


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
    return context.EvalSumRowsKeyGen(secret_key, None, ncols, slots * 4)


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
        List of rotation indices to generate keys for
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
