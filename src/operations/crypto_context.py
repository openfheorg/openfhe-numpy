"""
Crypto context operations and key generation utilities for homomorphic operations.

This module provides functions for generating rotation, accumulation, and other specialized
keys needed for various homomorphic operations in OpenFHE-NumPy.
"""

import openfhe_numpy as onp


def accumulation_depth(nrows, ncols, accumulate_by_rows):
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
    return onp.MulDepthAccumulation(nrows, ncols, accumulate_by_rows)


def sum_row_keys(context, secret_key, ncols=0):
    """
    Generate keys for summing rows in a matrix.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of cols for the matrix, by default 0

    Returns
    -------
    object
        Generated sum keys
    """
    return context.EvalSumRowsKeyGen(secret_key, None, ncols)


def sum_col_keys(context, secret_key, ncols=0):
    """
    Generate keys for summing columns in a matrix.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int, optional
        Number of columns in the matrix, by default 0
    """
    # import numpy as np

    # base = np.arange(ncols) * ncols
    # indices = np.empty(2 * ncols, dtype=base.dtype)
    # indices[0::2] = base
    # indices[1::2] = -base

    # indices = [x for i in range(ncols) for x in (i * ncols, -i * ncols)]
    # context.EvalRotateKeyGen(secret_key, indices)
    return context.EvalSumColsKeyGen(secret_key)


def gen_accumulate_rows_key(secret_key, ncols):
    """
    Generate keys for cumulative sum of rows in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    onp.EvalSumCumRowsKeyGen(secret_key, ncols)


def gen_accumulate_cols_key(secret_key, ncols):
    """
    Generate keys for cumulative sum of columns in a matrix.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ncols : int
        Number of columns in the matrix
    """
    onp.EvalSumCumColsKeyGen(secret_key, ncols)


def gen_rotation_keys(context, secret_key, rotation_indices):
    """
    Generate rotation keys for the specified indices.

    Parameters
    ----------
    context : CryptoContext
        The OpenFHE crypto context
    secret_key : PrivateKey
        The private key to use for key generation
    rotation_indices : list
        List of rotation indices to generate keys for
    """
    context.EvalRotateKeyGen(secret_key, rotation_indices)


def gen_lintrans_keys(secret_key, block_size, linear_transform_type, repetitions=0):
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
    onp.EvalLinTransKeyGen(secret_key, block_size, linear_transform_type, repetitions)


def gen_square_matmult_key(secret_key, block_size):
    """
    Generate keys for square matrix multiplication.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    block_size : int
        Block size for the matrix
    """
    onp.EvalSquareMatMultRotateKeyGen(secret_key, block_size)


def gen_transpose_keys(secret_key, ctm_matrix):
    """
    Generate keys for matrix transposition.

    Parameters
    ----------
    secret_key : PrivateKey
        The private key to use for key generation
    ctm_matrix : CTArray
        The ciphertext matrix to transpose
    """
    onp.EvalLinTransKeyGen(
        secret_key, ctm_matrix.ncols, onp.LinTransType.TRANSPOSE
    )
