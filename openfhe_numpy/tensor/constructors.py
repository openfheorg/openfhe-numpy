"""
Array constructor functions for OpenFHE-NumPy.

This module provides functions to create FHE array from various input types,
including support for block-based tensor operations.
"""

# Third‐party imports
from typing import Literal, Optional

import numpy as np
import openfhe

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
from .tensor import FHETensor


def _get_block_dimensions(data, slots) -> tuple[int, int]:
    """
    TODO: Compute the block‐matrix dimensions (rows, cols)
    given raw `data` and number of slots.
    """
    pass


def block_array(
    cc: openfhe.CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    type: Literal["C", "P"] = "C",
    mode: str = "tile",
    package: Optional[dict] = None,
    public_key: openfhe.PublicKey = None,
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
) -> dict:
    """
    Flatten a scalar, vector, or matrix into a 1D array, padding
    or tileing elements to fill all slots.

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
    dict with keys:
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
            packed, shape = _ravel_matrix(
                data, batch_size, order, True, mode, **kwargs
            )
        elif data.ndim == 1:
            packed, shape = _ravel_vector(
                data, batch_size, order, True, mode, **kwargs
            )
        else:
            ONP_ERROR(f"Unsupported data dimension [{data.ndim}].")
        packed = packed
    else:
        ONP_ERROR("Input is not numeric.")

    return {
        "data": packed,
        "original_shape": data.shape,
        "ndim": data.ndim,
        "batch_size": batch_size,
        "shape": shape,
        "order": order,
    }


def array(
    cc: openfhe.CryptoContext,
    data: np.ndarray | Number | list,
    batch_size: Optional[int] = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    type: Literal["C", "P"] = "C",
    mode: str = "tile",
    package: dict = {},
    public_key: openfhe.PublicKey = None,
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
    type      : "C" or "P"
    package    : dict from `_pack_array` (optional)
    public_key : required if type == "C"

    Returns
    -------
    FHETensor
    """
    if cc is None:
        ONP_ERROR("CryptoContext cannot be None.")

    if batch_size is not None and not isinstance(batch_size, int):
        ONP_ERROR(
            f"batch_size must be int or None, got {type(batch_size).__name__}."
        )

    if type not in ["C", "P"]:
        ONP_ERROR(f"type must be 'C' or 'P', got '{type}'.")

    if not package:
        package = _pack_array(data, batch_size, order, mode, **kwargs)

    if mode not in ["tile", "zero"]:
        ONP_ERROR(f"mode must be 'tile' or 'zero', got '{mode}'.")

    packed_data = package["data"]
    batch_size = package["batch_size"]
    shape = package["shape"]
    order = package["order"]

    try:
        plaintext = cc.MakeCKKSPackedPlaintext(packed_data)
    except Exception as e:
        ONP_ERROR(f"Failed to create plaintext: {e}")

    if type == "P":
        result = PTArray(
            plaintext, package["original_shape"], batch_size, shape, order
        )
    else:
        if public_key is None:
            ONP_ERROR("Public key must be provided for ciphertext encoding.")
        try:
            ciphertext = cc.Encrypt(public_key, plaintext)
            result = CTArray(
                ciphertext, package["original_shape"], batch_size, shape, order
            )
        except Exception as e:
            ONP_ERROR(f"Failed to encrypt: {e}")

    result.set_shape(shape)
    result.set_batch_size(batch_size)
    return result


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
    tile: str = "tile",
    **kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Encode a 1D vector into a packed array.
    """
    target_cols = kwargs.get("target_cols")
    if target_cols is not None and not (
        isinstance(target_cols, int) and target_cols > 0
    ):
        ONP_ERROR(
            f"target_cols must be positive int or None, got {target_cols!r}."
        )

    pad_value = kwargs.get("pad_value", "tile")
    expand = kwargs.get("expand", "tile")

    if order == ArrayEncodingType.ROW_MAJOR:
        return _pack_vector_row_wise(
            data, batch_size, target_cols, expand, tile, pad_to_pow2, pad_value
        )
    elif order == ArrayEncodingType.COL_MAJOR:
        return _pack_vector_col_wise(
            data, batch_size, target_cols, expand, tile, pad_to_pow2, pad_value
        )
    else:
        ONP_ERROR("Unsupported encoding order")
