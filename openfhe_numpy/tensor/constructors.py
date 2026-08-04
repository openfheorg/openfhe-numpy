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
Constructor functions for OpenFHE-NumPy.

This module provides functions to create FHE array from various input types,
including support for block-based tensor operations.
"""

from __future__ import annotations

# Third‐party imports
from math import ceil, prod
from typing import Any, Literal
import numpy as np
from openfhe import CryptoContext, PublicKey

# Package-level imports
from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from openfhe_numpy.utils.errors import ONPError
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
from .tensor import FHETensor, PackedArrayInformation
from .ctarray import CTArray
from .ptarray import PTArray
from .block_tensor import BlockFHETensor
from .block_ctarray import BlockCTArray
from .block_ptarray import BlockPTArray


def _shape(data: Any) -> tuple[int, ...]:
    """Return NumPy-style shape."""
    if isinstance(data, Number):
        return ()
    shape = getattr(data, "shape", None)
    if shape is None:
        shape = np.shape(data)
    return tuple(shape)


def _compute_block_dimensions(
    shape: tuple[int, ...],
    batch_size: int,
    block_shape: tuple[int, ...] | None = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    target_cols: int | None = None,
    compact: bool = False,
) -> tuple[int, ...]:
    """Choose block dimensions if block_shape is None.
    Default rules:
        Vector  (n,)    -> (batch_size,), or (side,) where
                           side = 2^floor(log2(batch_size)/2) if compact=True
        Column  (m, 1)  -> (batch_size, 1)
        Row     (1, n)  -> (1, batch_size)
        General (m, n)  -> (side, side) where side = 2^floor(log2(batch_size)/2)
    ``compact=True`` requests the smaller, square-compatible vector block
    size required to pair a block vector with a block matrix for block
    matrix-vector multiplication (see ``block_array``'s ``compact``
    parameter). Plain block vectors used for vector-vector arithmetic
    (add/sub/multiply/dot/matmul) should use the default, non-compact sizing.
    TODO:
    [OPTIONAL] For rectangular matrices where one dimension fits in a single
    block, use rectangular block_shape to reduce wasted slots.
    """
    if len(shape) not in (1, 2):
        raise ValueError(f"Only 1-D or 2-D shapes are supported; got {shape}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive; got {shape}.")
    if block_shape is not None:
        block_shape = tuple(block_shape)
        if len(block_shape) != len(shape):
            raise ValueError(
                f"block_shape rank must match shape rank; "
                f"got block_shape={block_shape}, shape={shape}."
            )
        if any(dim <= 0 for dim in block_shape):
            raise ValueError(f"block_shape must be positive; got {block_shape}.")
        if prod(block_shape) > batch_size:
            raise ValueError(
                f"block_shape={block_shape} uses {prod(block_shape)} slots, "
                f"but batch_size={batch_size}."
            )
        if len(block_shape) == 2:
            br, bc = block_shape
            if not is_power_of_two(br) or not is_power_of_two(bc):
                raise ValueError(
                    f"Matrix block dimensions must be powers of two; got block_shape={block_shape}."
                )
        return block_shape
    if target_cols is not None:
        if len(shape) != 1:
            raise ValueError("target_cols is only valid for block vectors.")
        if not isinstance(target_cols, int) or target_cols <= 0:
            raise ValueError(f"target_cols must be a positive integer, got {target_cols!r}.")
        if target_cols > batch_size:
            raise ValueError(f"target_cols={target_cols} exceeds batch_size={batch_size}.")
        return (target_cols,)
    if len(shape) == 1:
        if not compact:
            return (batch_size,)
        side = 1 << ((batch_size.bit_length() - 1) // 2)
        return (side,)
    m, n = shape
    if n == 1:
        return (batch_size, 1)
    if m == 1:
        return (1, batch_size)
    side = 1 << ((batch_size.bit_length() - 1) // 2)
    return (side, side)


def _compute_grid_shape(
    original_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the number of blocks along each dimension."""
    if len(original_shape) != len(block_shape):
        raise ValueError(
            f"original_shape and block_shape must have the same rank; "
            f"got original_shape={original_shape}, block_shape={block_shape}."
        )
    return tuple(ceil(s / b) for s, b in zip(original_shape, block_shape))


def _pack_block(
    data: np.ndarray,
    original_shape: tuple[int, ...],
    batch_size: int,
    order: int,
    mode: str,
    **kwargs,
) -> PackedArrayInformation:
    """Pack a padded block while preserving its logical shape."""
    package = _pack_array(data, batch_size, order, mode, **kwargs)
    package.original_shape = original_shape
    return package


def block_array(
    cc,
    data: np.ndarray | list,
    block_shape: tuple | None = None,
    batch_size: int | None = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    mode: str = "tile",
    fhe_type: Literal["C", "P"] = "C",
    public_key=None,
    target_cols: int | None = None,
    compact: bool = False,
) -> BlockFHETensor:
    """Construct a block-encoded plaintext or ciphertext tensor.

    ``block_shape`` determines the partition. A vector is divided into blocks of
    ``b`` elements; a matrix is divided into ``br x bc`` tiles. Thus,

    - vector: ``grid_shape = (ceil(n / b),)``;
    - matrix: ``grid_shape = (ceil(m / br), ceil(n / bc))``.

    Matrix block ``(i, j)`` contains
    ``data[i*br:(i+1)*br, j*bc:(j+1)*bc]``. Boundary blocks are zero-padded to
    ``block_shape`` and stored in row-major grid order.

    If ``block_shape`` is omitted, let ``B = batch_size`` and let ``s`` be the
    largest power of two satisfying ``s**2 <= B``. The defaults are

    - ``(B,)`` for a vector;
    - ``(s,)`` for a compact vector;
    - ``(B, 1)`` for a column matrix;
    - ``(1, B)`` for a row matrix;
    - ``(s, s)`` for a general matrix.

    Each block is packed into one CKKS plaintext or ciphertext. ``order`` controls
    the slot order within a block, while ``mode`` controls whether unused slots are
    tiled or zero-filled.

    For vectors, ``compact=True`` expands each length-``b`` block into the
    ``b x b`` layout required by block matrix-vector multiplication, with
    ``b**2 <= batch_size``. Compatible packing orders are

    - ``ROW_MAJOR`` matrix with ``COL_MAJOR`` vector;
    - ``COL_MAJOR`` matrix with ``ROW_MAJOR`` vector.

    Compact packing replicates vector entries and is unsuitable for ordinary
    vector arithmetic, dot products, or vector-vector matrix multiplication.
    Providing ``target_cols`` also enables compact packing.

    Parameters
    ----------
    cc
        OpenFHE crypto context.
    data
        Nonempty one- or two-dimensional input.
    block_shape
        Shape of each logical block.
    batch_size
        CKKS slots per block. Defaults to ``cc.GetBatchSize()``.
    order
        Encoding order within each block.
    mode
        Unused-slot filling mode: ``"tile"`` or ``"zero"``.
    fhe_type
        ``"C"`` for ciphertext or ``"P"`` for plaintext.
    public_key
        Required when ``fhe_type="C"``.
    target_cols
        Compact vector block length when ``block_shape`` is omitted.
    compact
        Enable matrix-vector-compatible vector packing.

    Returns
    -------
    BlockFHETensor
        The resulting ``BlockCTArray`` or ``BlockPTArray``.
    """

    if cc is None:
        raise ValueError("CryptoContext is required.")
    if fhe_type not in ("C", "P"):
        raise ValueError(f"fhe_type must be 'C' or 'P'; got {fhe_type!r}.")
    if fhe_type == "C" and public_key is None:
        raise ValueError("public_key is required for encryption.")

    if batch_size is None:
        batch_size = cc.GetBatchSize()
    arr = np.asarray(data)
    original_shape = arr.shape
    if arr.ndim == 0:
        raise ValueError("Scalar input not supported. Use array() instead.")
    if arr.ndim > 2:
        raise ValueError(f"Only 1-D and 2-D supported; got shape {arr.shape}.")
    is_compact = compact or target_cols is not None
    block_shape = _compute_block_dimensions(
        original_shape,
        batch_size,
        block_shape,
        order=order,
        target_cols=target_cols,
        compact=is_compact,
    )
    grid_shape = _compute_grid_shape(original_shape, block_shape)
    block_cls = BlockCTArray if fhe_type == "C" else BlockPTArray
    blocks = []
    if arr.ndim == 1:
        chunk = block_shape[0]
        for i in range(grid_shape[0]):
            start = i * chunk
            stop = min(start + chunk, original_shape[0])
            tile = np.zeros(block_shape, dtype=arr.dtype)
            tile[: stop - start] = arr[start:stop]
            # Compact packing repeats vector entries for matrix-vector multiplication.
            vector_target_cols = chunk if is_compact and chunk * chunk <= batch_size else None
            package = _pack_block(
                tile,
                (stop - start,),
                batch_size,
                order,
                mode,
                target_cols=vector_target_cols,
            )
            blocks.append(
                array(
                    cc=cc,
                    data=tile,
                    batch_size=batch_size,
                    order=order,
                    mode=mode,
                    fhe_type=fhe_type,
                    public_key=public_key,
                    package=package,
                )
            )
    else:
        br, bc = block_shape
        for gi in range(grid_shape[0]):
            r0, r1 = gi * br, min((gi + 1) * br, original_shape[0])
            for gj in range(grid_shape[1]):
                c0, c1 = gj * bc, min((gj + 1) * bc, original_shape[1])
                tile = np.zeros(block_shape, dtype=arr.dtype)
                tile[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
                package = _pack_block(
                    tile,
                    (r1 - r0, c1 - c0),
                    batch_size,
                    order,
                    mode,
                )
                blocks.append(
                    array(
                        cc=cc,
                        data=tile,
                        batch_size=batch_size,
                        order=order,
                        mode=mode,
                        fhe_type=fhe_type,
                        public_key=public_key,
                        package=package,
                    )
                )

    return block_cls(
        data=blocks,
        grid_shape=grid_shape,
        block_shape=block_shape,
        original_shape=original_shape,
        batch_size=batch_size,
        order=order,
    )


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
        raise ONPError("The batch size cannot be negative.")
    if not is_power_of_two(batch_size):
        raise ONPError(f"Batch size [{batch_size}] must be a power of two.")

    data = np.asarray(data)

    if is_numeric_scalar(data):
        if mode == "zero":
            packed = np.zeros(batch_size, dtype=data.dtype)
            packed[0] = data
        elif mode == "tile":
            packed = np.full(batch_size, data)
        else:
            raise ONPError(f"Invalid padding mode: '{mode}'. Use 'zero' or 'tile'.")
        shape = (batch_size, 1)

    elif is_numeric_arraylike(data):
        if data.ndim == 2:
            packed, shape = _ravel_matrix(data, batch_size, order, True, mode, **kwargs)
        elif data.ndim == 1:
            packed, shape = _ravel_vector(data, batch_size, order, True, mode, **kwargs)
        else:
            raise ONPError(f"Unsupported data dimension [{data.ndim}].")

    else:
        raise ONPError("Input is not numeric.")

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
    batch_size: int | None = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    fhe_type: Literal["C", "P"] = "P",
    mode: str = "tile",
    package: PackedArrayInformation | None = None,
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
        raise ONPError("CryptoContext does not exist")

    if batch_size is None:
        batch_size = cc.GetBatchSize()
    if not isinstance(batch_size, int) or batch_size < 0:
        raise ONPError(f"batch_size must be a non-negative int or None, got {batch_size}.")

    if package is None:
        package = _pack_array(data, batch_size, order, mode, **kwargs)

    try:
        plaintext = cc.MakeCKKSPackedPlaintext(package.data)
    except Exception as e:
        raise ONPError("Error: " + str(e))

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
            raise ONPError("Public key must be provided for ciphertext encoding.")
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
            raise ONPError(f"Failed to encrypt: {e}")
    else:
        raise ONPError(f"type must be 'C' or 'P', got '{fhe_type}'.")


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
    if order == ArrayEncodingType.COL_MAJOR:
        return _pack_matrix_col_wise(data, batch_size, pad_to_pow2, mode)
    raise ValueError(f"Unsupported encoding order: {order}")


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
        raise ONPError(f"target_cols must be a positive int or None, got {target_cols!r}.")

    pad_value = kwargs.get("pad_value", "tile")
    expand = kwargs.get("expand", "tile")

    if order == ArrayEncodingType.ROW_MAJOR:
        return _pack_vector_row_wise(
            vector=data,
            batch_size=batch_size,
            target_cols=target_cols,
            expand=expand,
            tile=mode,
            pad_to_power_of_2=pad_to_pow2,
            pad_value=pad_value,
        )
    if order == ArrayEncodingType.COL_MAJOR:
        return _pack_vector_col_wise(
            vector=data,
            batch_size=batch_size,
            target_cols=target_cols,
            expand=expand,
            tile=mode,
            pad_to_power_of_2=pad_to_pow2,
            pad_value=pad_value,
        )
    raise ONPError(f"Unsupported encoding order: {order}")
