# ================================================================================
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
"""Broadcast scalar-like, row, and column packed tensors."""

from math import prod
from operator import index as operator_index

import numpy as np

from ..openfhe_numpy import ArrayEncodingType
from ..tensor.constructors import _pack_block, array
from ..tensor.ctarray import CTArray
from ..tensor.tensor import FramePacking
from ..utils._helper_slots_ops import (
    _create_masking,
    _replicate_pattern,
    _replication_steps,
)
from ..utils.errors import (
    ONPDimensionError,
    ONPIncompatibleShapeError,
    ONPNotSupportedError,
    ONPValueError,
    _require,
)
from ..utils.matlib import next_power_of_two


# ------------------------------------------------------------------------------
# Shape and layout validation
# ------------------------------------------------------------------------------


def _normalize_shape(shape, name):
    """Normalize a logical or physical shape.

    Parameters
    ----------
    shape
        Iterable of integer-compatible dimensions. The empty scalar shape is
        allowed.
    name
        User-facing name included in validation errors.

    Returns
    -------
    tuple[int, ...]
        Shape containing positive Python integers, or ``()`` for a scalar.

    Raises
    ------
    ONPValueError
        If ``shape`` is not iterable, contains a non-integral dimension, or
        contains a nonpositive dimension.
    """
    try:
        normalized = tuple(operator_index(size) for size in shape)
    except TypeError as exc:
        raise ONPValueError(f"{name} must be a shape, got {shape!r}") from exc

    if any(size <= 0 for size in normalized):
        raise ONPValueError(f"{name} must contain positive dimensions, got {normalized}")
    return normalized


def _require_broadcast_target(source_shape, logical_shape):
    """Require NumPy broadcasting to produce the requested logical shape.

    Parameters
    ----------
    source_shape
        Normalized logical shape of the source tensor.
    logical_shape
        Normalized logical shape requested by the caller.

    Returns
    -------
    tuple[int, ...]
        NumPy broadcast result, which is guaranteed to equal ``logical_shape``.

    Raises
    ------
    ONPIncompatibleShapeError
        If the shapes cannot broadcast or broadcasting produces a different
        result shape.
    """
    try:
        result_shape = tuple(np.broadcast_shapes(source_shape, logical_shape))
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            source_shape,
            logical_shape,
            f"cannot broadcast {source_shape} to {logical_shape}",
        ) from exc
    if result_shape != logical_shape:
        raise ONPIncompatibleShapeError(
            source_shape,
            logical_shape,
            f"broadcasting produces {result_shape}, not {logical_shape}",
        )
    return result_shape


def _slot_source_kind(source_shape):
    """Classify a packed source for matrix broadcasting.

    Parameters
    ----------
    source_shape
        Logical source shape already validated against the matrix target.

    Returns
    -------
    str
        ``scalar``, ``row``, or ``column``.

    Raises
    ------
    ONPNotSupportedError
        If the source is a general matrix or otherwise has no broadcast kernel.
    """
    if prod(source_shape) == 1:
        return "scalar"
    if len(source_shape) == 1:
        return "row"
    if len(source_shape) == 2 and source_shape[0] == 1:
        return "row"
    if len(source_shape) == 2 and source_shape[1] == 1:
        return "column"
    raise ONPNotSupportedError(f"unsupported broadcast source shape {source_shape}")


def _validate_matrix_broadcast_geometry(
    source_shape,
    logical_shape,
    physical_shape,
    order,
    *,
    allow_order_union,
):
    """Validate one logical matrix broadcast and its physical slot frame.

    Parameters
    ----------
    source_shape
        Logical source shape.
    logical_shape
        Requested two-dimensional logical result shape.
    physical_shape
        Two-dimensional padded slot frame for the result.
    order
        Required packing order, or ``None`` when key planning may union both
        supported orders.
    allow_order_union
        Whether ``order=None`` is valid.

    Returns
    -------
    tuple
        Normalized ``(source_shape, logical_shape, physical_shape)``.

    Raises
    ------
    ONPDimensionError
        If source or frame ranks are unsupported.
    ONPIncompatibleShapeError
        If the logical result exceeds the physical frame or the source cannot
        broadcast to the requested logical shape.
    ONPValueError
        If the packing order is missing or invalid.
    """
    source_shape = _normalize_shape(source_shape, "source shape")
    logical_shape = _normalize_shape(logical_shape, "logical shape")
    physical_shape = _normalize_shape(physical_shape, "physical shape")

    _require(
        len(source_shape) <= 2,
        source_shape,
        logical_shape,
        f"unsupported source rank {len(source_shape)}",
        error_cls=ONPDimensionError,
    )
    _require(
        len(logical_shape) == 2 and len(physical_shape) == 2,
        logical_shape,
        physical_shape,
        "matrix broadcasting requires two-dimensional frames",
        error_cls=ONPDimensionError,
    )

    valid_orders = (
        ArrayEncodingType.ROW_MAJOR,
        ArrayEncodingType.COL_MAJOR,
    )
    if order is None:
        _require(
            allow_order_union,
            order,
            None,
            "matrix packing order is required",
            error_cls=ONPValueError,
        )
    else:
        _require(
            order in valid_orders,
            order,
            None,
            f"invalid matrix packing order {order!r}",
            error_cls=ONPValueError,
        )

    _require(
        all(logical <= physical for logical, physical in zip(logical_shape, physical_shape)),
        logical_shape,
        physical_shape,
        f"logical_shape={logical_shape} exceeds physical_shape={physical_shape}",
    )

    _require_broadcast_target(source_shape, logical_shape)

    return source_shape, logical_shape, physical_shape


# ------------------------------------------------------------------------------
# Rotation planning and key generation
# ------------------------------------------------------------------------------


def _broadcast_rotation_indices(
    source_shape,
    logical_shape,
    physical_shape,
    order=None,
):
    """Plan rotations for one logical/physical broadcast layout.

    Parameters
    ----------
    source_shape
        Logical source shape.
    logical_shape
        Requested logical result shape.
    physical_shape
        Padded physical slot frame.
    order
        Packing order to plan, or ``None`` to union both supported matrix
        orders.

    Returns
    -------
    set[int]
        Nonzero signed rotation indices used by the runtime kernels.

    Raises
    ------
    ONPDimensionError
        If source or frame ranks are unsupported.
    ONPIncompatibleShapeError
        If the source cannot broadcast to the requested logical frame.
    ONPNotSupportedError
        If no vector or matrix broadcast kernel supports the source shape.
    ONPValueError
        If the physical frame or packing order is invalid.
    """
    source_shape = _normalize_shape(source_shape, "source shape")
    logical_shape = _normalize_shape(logical_shape, "logical shape")
    physical_shape = _normalize_shape(physical_shape, "physical shape")
    indices = set()

    def add_replication(copies, stride):
        indices.update(rotation for rotation, _ in _replication_steps(copies, stride))

    if len(logical_shape) == 1:
        _require(
            len(physical_shape) == 2
            and physical_shape[1] == 1
            and logical_shape[0] <= physical_shape[0],
            logical_shape,
            physical_shape,
            f"invalid vector frame: logical={logical_shape}, physical={physical_shape}",
            error_cls=ONPValueError,
        )

        _require_broadcast_target(source_shape, logical_shape)
        _require(
            _slot_source_kind(source_shape) == "scalar",
            source_shape,
            logical_shape,
            "only scalar-like data can expand to a vector",
            error_cls=ONPNotSupportedError,
        )

        if order is not None:
            _require(
                order in (ArrayEncodingType.ROW_MAJOR, ArrayEncodingType.COL_MAJOR),
                order,
                None,
                f"invalid vector packing order {order!r}",
                error_cls=ONPValueError,
            )

        add_replication(logical_shape[0], 1)
        return indices

    source_shape, logical_shape, physical_shape = _validate_matrix_broadcast_geometry(
        source_shape=source_shape,
        logical_shape=logical_shape,
        physical_shape=physical_shape,
        order=order,
        allow_order_union=True,
    )
    source_kind = _slot_source_kind(source_shape)

    logical_rows, logical_cols = logical_shape
    physical_rows, physical_cols = physical_shape
    orders = (
        (ArrayEncodingType.ROW_MAJOR, ArrayEncodingType.COL_MAJOR) if order is None else (order,)
    )

    for current_order in orders:
        if source_kind == "scalar":
            if current_order == ArrayEncodingType.ROW_MAJOR:
                add_replication(logical_cols, 1)
                add_replication(logical_rows, physical_cols)
            else:
                add_replication(logical_rows, 1)
                add_replication(logical_cols, physical_rows)

        elif source_kind == "row":
            source_cols = source_shape[-1]
            if current_order == ArrayEncodingType.ROW_MAJOR:
                add_replication(logical_rows, physical_cols)
            else:
                indices.update(-i * (physical_rows - 1) for i in range(1, source_cols))
                add_replication(logical_rows, 1)

        elif source_kind == "column":
            source_rows = source_shape[0]
            if current_order == ArrayEncodingType.COL_MAJOR:
                add_replication(logical_cols, physical_rows)
            else:
                indices.update(-i * (physical_cols - 1) for i in range(1, source_rows))
                add_replication(logical_cols, 1)

    indices.discard(0)
    return indices


def generate_broadcast_key(secret_key, original_shape, target_shape):
    """Generate rotations needed to broadcast one packed tensor."""
    logical_shape = _normalize_shape(target_shape, "target_shape")

    # Direct broadcasting preserves logical identity without repacking. The source
    # shape is normalized and validated inside _broadcast_rotation_indices, so it is
    # passed through raw rather than normalized a second time here.
    if tuple(original_shape) == logical_shape:
        return

    if len(logical_shape) == 1:
        physical_shape = (next_power_of_two(logical_shape[0]), 1)
    elif len(logical_shape) == 2:
        physical_shape = tuple(next_power_of_two(size) for size in logical_shape)
    else:
        raise ONPNotSupportedError(f"target shape {logical_shape} is not supported")

    context = secret_key.GetCryptoContext()
    _require(
        physical_shape[0] * physical_shape[1] <= context.GetBatchSize(),
        physical_shape,
        None,
        f"physical_shape={physical_shape} exceeds the crypto-context batch size",
        error_cls=ONPValueError,
    )

    indices = _broadcast_rotation_indices(
        source_shape=original_shape,
        logical_shape=logical_shape,
        physical_shape=physical_shape,
        order=None,
    )
    if indices:
        context.EvalRotateKeyGen(secret_key, sorted(indices))


# ------------------------------------------------------------------------------
# Public runtime dispatch
# ------------------------------------------------------------------------------


def broadcast_to(x, target_shape, order=None, cc=None):
    """Broadcast ``x`` to a logical NumPy target shape."""
    source_shape = _normalize_shape(x.original_shape, "source shape")
    logical_shape = _normalize_shape(target_shape, "target_shape")

    _require_broadcast_target(source_shape, logical_shape)

    # Preserve the existing logical no-op, including order=None and tiled tails.
    if source_shape == logical_shape:
        return x

    if len(logical_shape) == 1:
        return _broadcast_to_vector(x, logical_shape, order=order, cc=cc)

    if len(logical_shape) == 2:
        physical_shape = tuple(next_power_of_two(size) for size in logical_shape)
        return _broadcast_to_physical_slots(
            x,
            logical_shape=logical_shape,
            physical_shape=physical_shape,
            order=order,
            cc=cc,
        )

    raise ONPNotSupportedError(f"target shape {logical_shape} is not supported")


# ------------------------------------------------------------------------------
# Vector broadcasting
# ------------------------------------------------------------------------------


def _broadcast_to_vector(x, logical_shape, order=None, cc=None):
    """Broadcast scalar-like packed data into a logical vector.

    Parameters
    ----------
    x
        Scalar-like ``CTArray`` or ``PTArray`` source.
    logical_shape
        Normalized one-dimensional target shape.
    order
        Requested packing order, or ``None`` to preserve the source order.
    cc
        Crypto context required when constructing a plaintext result.

    Returns
    -------
    CTArray or PTArray
        Vector with physical shape ``(next_power_of_two(length), 1)`` and zero
        padding outside its logical region.

    Raises
    ------
    ONPNotSupportedError
        If the source is not scalar-like or has an unsupported tensor type.
    ONPValueError
        If the physical vector exceeds the batch, the order is invalid, or a
        plaintext source is missing its crypto context.
    """
    logical_length = logical_shape[0]
    padded_length = next_power_of_two(logical_length)
    physical_shape = (padded_length, 1)

    _require(
        padded_length <= x.batch_size,
        padded_length,
        x.batch_size,
        f"physical vector length {padded_length} exceeds batch_size={x.batch_size}",
        error_cls=ONPValueError,
    )
    _require(
        _slot_source_kind(tuple(x.original_shape)) == "scalar",
        x.original_shape,
        None,
        f"only scalar-like data can expand to a vector; got {x.original_shape}",
        error_cls=ONPNotSupportedError,
    )

    resolved_order = x.order if order is None else order
    _require(
        resolved_order in (ArrayEncodingType.ROW_MAJOR, ArrayEncodingType.COL_MAJOR),
        resolved_order,
        None,
        f"invalid vector packing order {resolved_order!r}",
        error_cls=ONPValueError,
    )

    if x.dtype == "CTArray":
        seed = _isolate_ciphertext(x.data, [0], x.batch_size)
        result_data = _replicate_pattern(
            seed,
            copies=logical_length,
            stride=1,
        )
        return CTArray(
            data=result_data,
            original_shape=logical_shape,
            batch_size=x.batch_size,
            new_shape=physical_shape,
            order=resolved_order,
            geometry=FramePacking(
                active=(logical_length, 1),
                padding="zero",
                repeats=1,
            ),
        )

    if x.dtype == "PTArray":
        _require(
            cc is not None,
            None,
            None,
            "broadcasting plaintext data requires a crypto context",
            error_cls=ONPValueError,
        )
        logical_values = np.broadcast_to(
            x.decode(unpack_type="original"),
            logical_shape,
        )
        return array(
            cc=cc,
            data=np.asarray(logical_values),
            batch_size=x.batch_size,
            order=resolved_order,
            fhe_type="P",
            mode="zero",
        )

    raise ONPNotSupportedError(f"broadcast does not support {type(x)}")


# ------------------------------------------------------------------------------
# Matrix broadcasting
# ------------------------------------------------------------------------------


def _broadcast_to_physical_slots(
    x,
    logical_shape,
    physical_shape,
    order=None,
    cc=None,
):
    """Broadcast packed data into a specified matrix slot frame.

    Parameters
    ----------
    x
        Packed ``CTArray`` or ``PTArray`` source.
    logical_shape
        Requested logical matrix shape.
    physical_shape
        Padded physical matrix frame.
    order
        Required row-major or column-major packing order.
    cc
        Crypto context required when constructing a plaintext result.

    Returns
    -------
    CTArray or PTArray
        Broadcast result carrying the requested logical and physical metadata.

    Raises
    ------
    ONPDimensionError
        If source or frame ranks are unsupported.
    ONPIncompatibleShapeError
        If source, logical, and physical shapes are incompatible.
    ONPNotSupportedError
        If the source shape or tensor type has no supported kernel.
    ONPValueError
        If the order, batch capacity, or plaintext context is invalid.
    """
    source_shape, logical_shape, physical_shape = _validate_matrix_broadcast_geometry(
        source_shape=x.original_shape,
        logical_shape=logical_shape,
        physical_shape=physical_shape,
        order=order,
        allow_order_union=False,
    )

    _require(
        physical_shape[0] * physical_shape[1] <= x.batch_size,
        physical_shape,
        x.batch_size,
        f"physical_shape={physical_shape} exceeds batch_size={x.batch_size}",
        error_cls=ONPValueError,
    )

    # Packed identity requires logical, physical, and order equality.
    if source_shape == logical_shape and tuple(x.shape) == physical_shape and x.order == order:
        return x

    source_kind = _slot_source_kind(source_shape)
    if x.dtype == "CTArray":
        return _ct_broadcast_to(
            x,
            logical_shape,
            physical_shape,
            order,
            source_kind,
        )
    if x.dtype == "PTArray":
        return _pt_broadcast_to(
            x,
            logical_shape,
            physical_shape,
            order,
            cc,
        )
    raise ONPNotSupportedError(f"broadcast does not support {type(x)}")


# ------------------------------------------------------------------------------
# Ciphertext broadcast helpers
# ------------------------------------------------------------------------------


def _isolate_ciphertext(data, indices, batch_size):
    """Mask a ciphertext to a selected set of source slots.

    Parameters
    ----------
    data
        Ciphertext containing the packed source values.
    indices
        Slot indices to retain.
    batch_size
        Total number of packed slots.

    Returns
    -------
    Ciphertext
        Ciphertext with every slot outside ``indices`` set to zero.
    """
    context = data.GetCryptoContext()
    mask = _create_masking(indices, batch_size)
    return context.EvalMult(data, context.MakeCKKSPackedPlaintext(mask))


def _ct_broadcast_to(x, logical_shape, physical_shape, order, source_kind):
    context = x.data.GetCryptoContext()
    logical_rows, logical_cols = logical_shape
    physical_rows, physical_cols = physical_shape

    if source_kind == "scalar":
        seed = _isolate_ciphertext(x.data, [0], x.batch_size)
        if order == ArrayEncodingType.ROW_MAJOR:
            first_row = _replicate_pattern(seed, logical_cols, 1)
            result_data = _replicate_pattern(
                first_row,
                logical_rows,
                physical_cols,
            )
        else:
            first_column = _replicate_pattern(seed, logical_rows, 1)
            result_data = _replicate_pattern(
                first_column,
                logical_cols,
                physical_rows,
            )

    elif source_kind == "row":
        source_cols = tuple(x.original_shape)[-1]
        if order == ArrayEncodingType.ROW_MAJOR:
            seed_row = _isolate_ciphertext(
                x.data,
                range(source_cols),
                x.batch_size,
            )
            result_data = _replicate_pattern(
                seed_row,
                logical_rows,
                physical_cols,
            )
        else:
            first_row = _isolate_ciphertext(x.data, [0], x.batch_size)
            for i in range(1, source_cols):
                element = _isolate_ciphertext(x.data, [i], x.batch_size)
                rotation = -i * (physical_rows - 1)
                if rotation:
                    element = context.EvalRotate(element, rotation)
                first_row = context.EvalAdd(first_row, element)
            result_data = _replicate_pattern(first_row, logical_rows, 1)

    elif source_kind == "column":
        source_rows = tuple(x.original_shape)[0]
        if order == ArrayEncodingType.COL_MAJOR:
            seed_column = _isolate_ciphertext(
                x.data,
                range(source_rows),
                x.batch_size,
            )
            result_data = _replicate_pattern(
                seed_column,
                logical_cols,
                physical_rows,
            )
        else:
            first_column = _isolate_ciphertext(x.data, [0], x.batch_size)
            for i in range(1, source_rows):
                element = _isolate_ciphertext(x.data, [i], x.batch_size)
                rotation = -i * (physical_cols - 1)
                if rotation:
                    element = context.EvalRotate(element, rotation)
                first_column = context.EvalAdd(first_column, element)
            result_data = _replicate_pattern(first_column, logical_cols, 1)

    else:  # Defensive: source kinds are produced only by _slot_source_kind().
        raise ONPValueError(f"unsupported broadcast source kind {source_kind!r}")

    return CTArray(
        data=result_data,
        original_shape=logical_shape,
        batch_size=x.batch_size,
        new_shape=physical_shape,
        order=order,
        geometry=FramePacking(
            active=logical_shape,
            padding="zero",
            repeats=1,
        ),
    )


# ------------------------------------------------------------------------------
# Plaintext matrix broadcasting
# ------------------------------------------------------------------------------


def _pt_broadcast_to(
    pta_x,
    logical_shape,
    physical_shape,
    order=None,
    cc=None,
):
    _require(
        cc is not None,
        None,
        None,
        "broadcasting plaintext data requires a crypto context",
        error_cls=ONPValueError,
    )

    source_values = pta_x.decode(unpack_type="original")
    logical_values = np.broadcast_to(source_values, logical_shape)

    frame = np.zeros(physical_shape, dtype=logical_values.dtype)
    frame[tuple(slice(0, size) for size in logical_shape)] = logical_values

    package = _pack_block(
        data=frame,
        original_shape=logical_shape,
        batch_size=pta_x.batch_size,
        order=order,
        mode="zero",
    )
    return array(
        cc=cc,
        data=frame,
        batch_size=pta_x.batch_size,
        order=order,
        fhe_type="P",
        package=package,
    )
