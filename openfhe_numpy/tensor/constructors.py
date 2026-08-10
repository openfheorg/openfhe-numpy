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
from openfhe_numpy.utils.matlib import is_power_of_two, next_power_of_two
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
from .tensor import FHETensor, PackedArrayInformation, FramePacking
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
    target_cols: int | None = None,
) -> tuple[int, ...]:
    """Compute block dimensions or choose the largest size by default.
    Default rules:
    Vector  (n,)    -> (batch_size,), or (side,) where
                       side = 2^floor(log2(batch_size)/2) if compact=True
    Column  (m, 1)  -> (batch_size, 1)
    Row     (1, n)  -> (1, batch_size)
    General (m, n)  -> (side, side) where side = 2^floor(log2(batch_size)/2)
    """
    if len(shape) not in (1, 2):
        raise ValueError(f"Only 1-D or 2-D shapes are supported; got {shape}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive; got {shape}.")

    physical_cols = 1
    if target_cols is not None:
        if len(shape) != 1:
            raise ValueError("target_cols is only valid for block vectors.")
        if not isinstance(target_cols, int) or target_cols <= 0:
            raise ValueError(f"target_cols must be a positive integer, got {target_cols!r}.")
        physical_cols = next_power_of_two(target_cols)
        if physical_cols > batch_size:
            raise ValueError(f"target_cols={target_cols} exceeds batch_size={batch_size}.")

    if block_shape is not None:
        block_shape = tuple(block_shape)
        if len(block_shape) != len(shape):
            raise ValueError(
                f"block_shape rank must match shape rank; "
                f"got block_shape={block_shape}, shape={shape}."
            )
        if any(dim <= 0 for dim in block_shape):
            raise ValueError(f"block_shape must be positive; got {block_shape}.")

        required_slots = prod(block_shape)
        if len(shape) == 1:
            required_slots = next_power_of_two(block_shape[0]) * physical_cols

        if required_slots > batch_size:
            raise ValueError(
                f"block_shape={block_shape} uses {required_slots} slots, "
                f"but batch_size={batch_size}."
            )
        if len(block_shape) == 2:
            br, bc = block_shape
            if not is_power_of_two(br) or not is_power_of_two(bc):
                raise ValueError(
                    f"Matrix block dimensions must be powers of two; got block_shape={block_shape}."
                )
        return block_shape

    if len(shape) == 1:
        return (batch_size // physical_cols,)

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
    original_shape = tuple(original_shape)
    ndim = len(original_shape)
    active = (original_shape[0], package.geometry.active[1]) if ndim == 1 else original_shape
    package.original_shape = original_shape
    package.ndim = ndim
    package.geometry = FramePacking(
        active=active,
        padding=package.geometry.padding,
        repeats=package.geometry.repeats,
    )
    return package


def block_array(
    cc,
    data: np.ndarray | list,
    block_shape: tuple | None = None,
    batch_size: int | None = None,
    order: int = ArrayEncodingType.ROW_MAJOR,
    mode: str = "zero",
    fhe_type: Literal["C", "P"] = "C",
    public_key=None,
    target_cols: int | None = None,
    expand: Literal["tile", "zero"] = "tile",
    pad_value: Literal["tile", "zero"] = "zero",
) -> BlockFHETensor:
    """Construct a block-encoded plaintext or ciphertext tensor.

    The input is divided into blocks of ``block_shape``. Boundary blocks are
    zero-padded, and each block is packed into one CKKS plaintext or ciphertext.
    When ``block_shape`` is omitted, a suitable shape is selected from
    ``batch_size``.

    For vectors, ``target_cols`` creates a matrix-compatible frame. ``expand``
    controls value replication, while ``pad_value`` controls padded columns.

    Parameters
    ----------
    cc : CryptoContext
        OpenFHE crypto context.
    data : array-like
        Nonempty vector or matrix.
    block_shape : tuple, optional
        Block dimensions. Selected automatically when omitted.
    batch_size : int, optional
        CKKS slots per block. Defaults to ``cc.GetBatchSize()``.
    order : ArrayEncodingType
        Slot order within each block.
    mode : {"tile", "zero"}
        Repeat each block frame or zero-fill unused slots.
    fhe_type : {"C", "P"}
        Ciphertext or plaintext output.
    public_key : PublicKey, optional
        Required when ``fhe_type="C"``.
    target_cols : int, optional
        Active columns in an expanded vector block.
    expand : {"tile", "zero"}
        Repeat vector values across columns or use only the first column.
    pad_value : {"tile", "zero"}
        Repeat values into padded columns or leave them zero.

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
    if expand not in ("tile", "zero"):
        raise ValueError("expand must be 'tile' or 'zero'.")
    if pad_value not in ("tile", "zero"):
        raise ValueError("pad_value must be 'tile' or 'zero'.")
    if arr.ndim != 1 and (expand != "tile" or pad_value != "zero"):
        raise ValueError("expand and pad_value are valid only for block vectors.")
    block_shape = _compute_block_dimensions(
        original_shape,
        batch_size,
        block_shape,
        target_cols=target_cols,
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
            package = _pack_block(
                tile,
                (stop - start,),
                batch_size,
                order,
                mode,
                target_cols=target_cols,
                expand=expand,
                pad_value=pad_value,
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
    if batch_size <= 0:
        raise ONPError("The batch size must be positive.")
    if not is_power_of_two(batch_size):
        raise ONPError(f"Batch size [{batch_size}] must be a power of two.")
    if mode not in ("zero", "tile"):
        raise ONPError(f"Invalid padding mode: {mode!r}.")

    data = np.asarray(data)

    if is_numeric_scalar(data):
        if kwargs:
            raise ONPError("target_cols, expand, and pad_value are valid only for vectors.")

        if mode == "zero":
            packed = np.zeros(batch_size, dtype=data.dtype)
            packed[0] = data
        else:
            packed = np.full(batch_size, data)

        shape = (1, 1)
        active = (1, 1)
        padding = "zero"

    elif is_numeric_arraylike(data) and data.ndim == 1:
        target_cols = kwargs.get("target_cols")
        expand = kwargs.get("expand", "tile")
        pad_value = kwargs.get("pad_value", "tile")

        if expand not in ("tile", "zero"):
            raise ONPError("expand must be 'tile' or 'zero'.")
        if pad_value not in ("tile", "zero"):
            raise ONPError("pad_value must be 'tile' or 'zero'.")

        packed, shape = _ravel_vector(data, batch_size, order, True, mode, **kwargs)

        if target_cols is None:
            active = (len(data), 1)
            padding = "zero"
        elif expand == "tile":
            active = (len(data), target_cols)
            padding = pad_value if target_cols < shape[1] else "zero"
        else:
            active = (len(data), 1)
            padding = "zero"

    elif is_numeric_arraylike(data) and data.ndim == 2:
        if kwargs:
            raise ONPError("target_cols, expand, and pad_value are valid only for vectors.")

        packed, shape = _ravel_matrix(data, batch_size, order, True, mode)
        active = tuple(data.shape)
        padding = "zero"

    elif is_numeric_arraylike(data):
        raise ONPError(f"Unsupported data dimension [{data.ndim}].")

    else:
        raise ONPError("Input is not numeric.")

    frame_size = shape[0] * shape[1]
    repeats = 1 if mode == "zero" else batch_size // frame_size

    return PackedArrayInformation(
        data=packed,
        original_shape=data.shape,
        ndim=data.ndim,
        batch_size=batch_size,
        shape=shape,
        order=order,
        geometry=FramePacking(
            active=active,
            padding=padding,
            repeats=repeats,
        ),
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
    cc : CryptoContext
        OpenFHE crypto context.
    data : array-like
        Scalar, vector, or matrix to pack.
    batch_size : int, optional
        Number of CKKS slots. Defaults to ``cc.GetBatchSize()``.
    order : ArrayEncodingType
        ``ROW_MAJOR`` or ``COL_MAJOR``. Default is ``ROW_MAJOR``.
    fhe_type : {"C", "P"}
        Ciphertext or plaintext output. Default is ``"P"``.
    mode : {"tile", "zero"}, optional
        Repeat the packed frame or zero-fill unused slots. Default is ``"tile"``.
    package : PackedArrayInformation, optional
        Prepacked data used internally by ``block_array``.
    public_key : PublicKey, optional
        Required for ``fhe_type="C"``.
    **kwargs
        Vector packing options: ``target_cols``, ``expand``, and ``pad_value``.

    Returns
    -------
    FHETensor
        ``CTArray`` for ``"C"`` or ``PTArray`` for ``"P"``.

    Examples
    --------
    Use the matrix and logical column vector::

        matrix = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        vector = [1, 2, 3]

    Their dimensions pad from 3 to 4. With one frame and zero tail::

        >>> array(cc, matrix, batch_size=32, order=ROW_MAJOR, mode="zero")
        # 1 2 3 0 | 4 5 6 0 | 7 8 9 0 | 0 0 0 0 | then 16 zeros

        >>> array(cc, matrix, batch_size=32, order=COL_MAJOR, mode="zero")
        # 1 4 7 0 | 2 5 8 0 | 3 6 9 0 | 0 0 0 0 | then 16 zeros

        >>> array(cc, vector, batch_size=32, mode="zero")
        # 1 2 3 0 | then 28 zeros

    ``target_cols=3`` creates a ``4 x 4`` physical vector frame::

        >>> array(cc, vector, batch_size=32, target_cols=3, mode="zero")
        # 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 0 0 0 0 | then 16 zeros

        >>> array(cc, vector, batch_size=32, target_cols=3,
        ...       pad_value="zero", mode="zero")
        # 1 1 1 0 | 2 2 2 0 | 3 3 3 0 | 0 0 0 0 | then 16 zeros

        >>> array(cc, vector, batch_size=32, target_cols=3,
        ...       expand="zero", mode="zero")
        # 1 0 0 0 | 2 0 0 0 | 3 0 0 0 | 0 0 0 0 | then 16 zeros

    ``mode="tile"`` repeats the padded frame to fill all slots.
    """
    if cc is None:
        raise ONPError("CryptoContext does not exist")

    if batch_size is None:
        batch_size = cc.GetBatchSize()

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
            geometry=package.geometry,
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
                geometry=package.geometry,
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
