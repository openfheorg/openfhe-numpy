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
Array constructor functions for OpenFHE-NumPy.

This module provides functions to create FHE array from various input types,
including support for block-based tensor operations.
"""

# Third‐party imports
from typing import Literal, Optional, Union
import numpy as np
from openfhe import CryptoContext, PublicKey

# Package-level imports
from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from openfhe_numpy.utils.errors import ONP_ERROR
from openfhe_numpy.utils.matlib import is_power_of_two
from openfhe_numpy.utils.packing import (
    _pack_matrix_col_wise,
    _pack_matrix_row_wise,
    _pack_vector_col_wise,
    _pack_vector_row_wise,
)
from openfhe_numpy.utils.typecheck import (
    Number,
    is_numeric_arraylike,
    is_numeric_scalar,
)


# Tensor imports
from .ctarray import CTArray
from .ptarray import PTArray
from .tensor import FHETensor, PackedArrayInformation


def _get_block_dimensions(data: np.ndarray, slots: int) -> tuple[int, int]:
    """
    TODO: Compute the block‐matrix dimensions (rows, cols)
    given raw `data` and number of slots.
    """
    pass


def block_array(
    cc: CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    fhe_type: Literal["C", "P"] = "C",
    mode: str = "tile",
    package: Optional[dict] = None,
    public_key: PublicKey = None,
    **kwargs,
) -> FHETensor:
    """
    Construct a block‐plaintext or block‐ciphertext array from raw input.

    Parameters
    ----------
    cc         : CryptoContext
    data       : np.ndarray | Number | list
    batch_size : Optional[int]
    order      : ArrayEncodingType
    type      : "C" for ciphertext, "P" for plaintext
    mode       : padding mode ("tile" or "zero")
    package    : Optional prepacked dict from `_pack_array`
    public_key : PublicKey (required for encryption)

    Returns
    -------
    FHETensor
    """
    pass


def _pack_array(
    data: np.ndarray | Number | list,
    batch_size: int,
    order: int = ArrayEncodingType.ROW_MAJOR,
    mode: str = "tile",
    **kwargs,
) -> PackedArrayInformation:
    """
    Flatten a scalar, vector, or matrix into a 1D array, padding
    or tiling elements to fill all slots.

    Parameters
    ----------
    data       : np.ndarray | Number | list
    batch_size : int
        Number of available plaintext slots (must be a power of two).
    order      : ArrayEncodingType
    mode       : str
        "tile" to duplicate values, "zero" to pad with zeros.
    **kwargs   : extra args for matrix/vector packing

    Returns
    -------
    metadata (PackedArrayInformation) with keys:
      - data           : packed 1D numpy array
      - original_shape : tuple
      - ndim           : int
      - batch_size     : int
      - shape          : tuple (rows, cols)
      - order          : int
    """
    if batch_size < 0:
        ONP_ERROR("The batch size cannot be negative.")
    if not is_power_of_two(batch_size):
        ONP_ERROR(f"Batch size [{batch_size}] must be a power of two.")

    data = np.array(data)

    if is_numeric_scalar(data):
        if mode == "zero":
            packed = np.zeros(batch_size, dtype=data.dtype)
            packed[0] = data
        elif mode == "tile":
            packed = np.full(batch_size, data)
        else:
            ONP_ERROR(f"Invalid padding mode: '{mode}'. Use 'zero' or 'tile'.")
        shape = (batch_size, 1)

    elif is_numeric_arraylike(data):
        if data.ndim == 2:
            packed, shape = _ravel_matrix(data, batch_size, order, True, mode, **kwargs)
        elif data.ndim == 1:
            packed, shape = _ravel_vector(data, batch_size, order, True, mode, **kwargs)
        else:
            ONP_ERROR(f"Unsupported data dimension [{data.ndim}].")

    else:
        ONP_ERROR("Input is not numeric.")

    return PackedArrayInformation(
        data=packed,
        original_shape=data.shape,
        ndim=data.ndim,
        batch_size=batch_size,
        shape=shape,
        order=order,
    )


def array(
    cc: CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    fhe_type: Literal["C", "P"] = "P",
    mode: str = "tile",
    package: Optional[PackedArrayInformation] = None,
    public_key: PublicKey = None,
    **kwargs,
) -> FHETensor:
    """
    Construct a ciphertext or plaintext FHETensor from raw input.

    Parameters
    ----------
    cc         : CryptoContext
    data       : matrix | vector | scalar
    batch_size : Optional[int]
    order      : ArrayEncodingType
    fhe_type   : "C" (ciphertext) or "P" (plaintext)
    package    : dict from `_pack_array` (optional)
    public_key : required if type == "C"

    Returns
    -------
    FHETensor
    """
    if cc is None:
        ONP_ERROR("CryptoContext does not exist")

    if batch_size is None:
        batch_size = cc.GetBatchSize()
    if not isinstance(batch_size, int) or batch_size < 0:
        ONP_ERROR(f"batch_size must be a non-negative int or None, got {batch_size}.")

    if not package:
        package = _pack_array(data, batch_size, order, mode, **kwargs)

    try:
        plaintext = cc.MakeCKKSPackedPlaintext(package.data)
    except Exception as e:
        ONP_ERROR("Error: " + str(e))

    if fhe_type == "P":
        return PTArray(
            plaintext,  # data
            package.original_shape,  # original_shape
            package.batch_size,  # batch_size
            package.shape,  # new_shape
            package.order,  # order
        )
    elif fhe_type == "C":
        if public_key is None:
            ONP_ERROR("Public key must be provided for ciphertext encoding.")
        try:
            ciphertext = cc.Encrypt(public_key, plaintext)
            return CTArray(
                ciphertext,  # data
                package.original_shape,  # original_shape
                package.batch_size,  # batch_size
                package.shape,  # new_shape
                package.order,  # order
            )
        except Exception as e:
            ONP_ERROR(f"Failed to encrypt: {e}")
    else:
        ONP_ERROR(f"type must be 'C' or 'P', got '{fhe_type}'.")


def _ravel_matrix(
    data: np.ndarray,
    batch_size: int,
    order: int = ArrayEncodingType.ROW_MAJOR,
    pad_to_pow2: bool = True,
    mode: str = "tile",
    **kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Encode a 2D matrix into a packed array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D matrix to encode
    batch_size : int
        Number of available plaintext slots
    order : int, optional
        Encoding order (default: ROW_MAJOR)
    pad_to_pow2 : bool, optional
        Whether to pad to power of 2 (default: True)
    mode : str, optional
        Padding mode "tile" or "zero" (default: "tile")
    **kwargs
        Additional keyword arguments

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        Packed array and shape tuple (rows, cols)
    """

    if order == ArrayEncodingType.ROW_MAJOR:
        return _pack_matrix_row_wise(data, batch_size, pad_to_pow2, mode)
    elif order == ArrayEncodingType.COL_MAJOR:
        return _pack_matrix_col_wise(data, batch_size, pad_to_pow2, mode)
    else:
        raise ValueError("Unsupported encoding order")


def _ravel_vector(
    data: list | np.ndarray,
    batch_size: int,
    order: int = ArrayEncodingType.ROW_MAJOR,
    pad_to_pow2: bool = True,
    mode: str = "tile",
    **kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Encode a 1D vector into a packed array.
    Parameters
    ----------
    data : list | np.ndarray
        Input 1D vector to encode
    batch_size : int
        Number of available plaintext slots
    order : int, optional
        Encoding order (default: ROW_MAJOR)
    pad_to_pow2 : bool, optional
        Whether to pad to power of 2 (default: True)
    mode : str, optional
        Padding mode "tile" or "zero" (default: "tile")
    **kwargs
        Additional keyword arguments including target_cols, pad_value, expand

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        Packed array and shape tuple (rows, cols)
    """
    target_cols = kwargs.get("target_cols")
    if target_cols is not None and not (isinstance(target_cols, int) and target_cols > 0):
        ONP_ERROR(f"target_cols must be positive int or None, got {target_cols!r}.")

    pad_value = kwargs.get("pad_value", "tile")
    expand = kwargs.get("expand", "tile")

    if order == ArrayEncodingType.ROW_MAJOR:
        return _pack_vector_row_wise(
            data, batch_size, target_cols, expand, mode, pad_to_pow2, pad_value
        )
    elif order == ArrayEncodingType.COL_MAJOR:
        return _pack_vector_col_wise(
            data, batch_size, target_cols, expand, mode, pad_to_pow2, pad_value
        )
    else:
        ONP_ERROR("Unsupported encoding order")
