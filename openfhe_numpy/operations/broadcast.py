# ==============================================================================
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
"""NumPy-style broadcasting for packed plaintext and ciphertext tensors"""

from math import prod
from operator import index as operator_index

import numpy as np

from ..openfhe_numpy import ArrayEncodingType
from ..tensor.constructors import array
from ..utils._helper_slots_ops import _create_masking, _duplicate_block
from ..utils.errors import (
    ONPDimensionError,
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPValueError,
)
from ..utils.matlib import next_power_of_two


_ORDERS = (ArrayEncodingType.ROW_MAJOR, ArrayEncodingType.COL_MAJOR)


def broadcast_shapes(x_shape, y_shape):
    """Return the NumPy broadcast shape of two logical shapes."""
    return np.broadcast_shapes(x_shape, y_shape)


def _broadcast_spec(source_shape, target_shape):
    """Return validated ``(source_shape, target_shape, kind)``."""

    def normalize(shape, name):
        try:
            shape = (operator_index(shape),)
        except TypeError:
            try:
                shape = tuple(operator_index(dimension) for dimension in shape)
            except TypeError as exc:
                raise ONPDimensionError(f"Invalid {name}: {shape!r}.") from exc

        if len(shape) > 2:
            raise ONPDimensionError(f"broadcast_to supports ranks zero through two; got {shape}.")
        if any(dimension <= 0 for dimension in shape):
            raise ONPDimensionError(f"Empty shapes are not supported; got {shape}.")
        return shape

    source_shape = normalize(source_shape, "source_shape")
    target_shape = normalize(target_shape, "target_shape")
    try:
        result_shape = tuple(np.broadcast_shapes(source_shape, target_shape))
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            source_shape,
            target_shape,
            f"Cannot broadcast {source_shape} to {target_shape}.",
        ) from exc

    if result_shape != target_shape:
        raise ONPIncompatibleShapeError(
            source_shape,
            target_shape,
            f"Broadcasting produces {result_shape}, not {target_shape}.",
        )

    if source_shape == target_shape:
        kind = "identity"
    elif not source_shape or all(dimension == 1 for dimension in source_shape):
        kind = "scalar"
    elif len(target_shape) == 2 and (len(source_shape) == 1 or source_shape[0] == 1):
        kind = "row"
    elif len(target_shape) == 2 and source_shape[1] == 1:
        kind = "column"
    else:
        raise ONPNotImplementedError(
            f"Broadcasting from {source_shape} to {target_shape} is not implemented."
        )
    return source_shape, target_shape, kind


def _padded_shape(shape):
    return tuple(next_power_of_two(dimension) for dimension in shape)


def _resolve_order(tensor, requested=None):
    if tensor.order not in _ORDERS:
        raise ONPValueError(f"Invalid packing order: {tensor.order!r}.")
    if requested is not None and requested not in _ORDERS:
        raise ONPValueError(f"Invalid packing order: {requested!r}.")
    if requested is not None and requested != tensor.order:
        raise ONPNotImplementedError(
            "broadcast_to preserves packing order; convert the order separately."
        )
    return tensor.order


def _vector_layout(kind, source_shape, target_shape, order):
    """Return count, repetitions, alignment, block stride, and scatter stride."""
    nrows, ncols = target_shape
    padded_rows, padded_cols = _padded_shape(target_shape)
    if kind == "row":
        return (
            source_shape[-1],
            nrows,
            order == ArrayEncodingType.ROW_MAJOR,
            padded_cols,
            padded_rows,
        )
    return (
        source_shape[0],
        ncols,
        order == ArrayEncodingType.COL_MAJOR,
        padded_rows,
        padded_cols,
    )


def _rotation_indices(source_shape, target_shape, kind, order):
    """Return exactly the rotations used by the ciphertext kernel."""

    def duplicate(count, block_size):
        rotations = set()
        offset = block_size
        while offset < block_size * count:
            rotations.add(-offset)
            offset *= 2
        return rotations

    if kind == "identity":
        return set()
    if len(target_shape) == 1:
        return duplicate(target_shape[0], 1)
    if kind == "scalar":
        nrows, ncols = target_shape
        padded_rows, padded_cols = _padded_shape(target_shape)
        if order == ArrayEncodingType.ROW_MAJOR:
            return duplicate(nrows, padded_cols) | duplicate(ncols, 1)
        return duplicate(ncols, padded_rows) | duplicate(nrows, 1)

    count, repeats, aligned, block_stride, scatter_stride = _vector_layout(
        kind, source_shape, target_shape, order
    )
    if aligned:
        return duplicate(repeats, block_stride)
    scatter = {-index * (scatter_stride - 1) for index in range(1, count) if scatter_stride > 1}
    return scatter | duplicate(repeats, 1)


def generate_broadcast_key(secret_key, source_shape, target_shape, order=None):
    """Generate rotation keys for one broadcast.

    ``order=None`` generates keys for both packing orders. Pass an order for the
    smaller exact key set.
    """
    source_shape, target_shape, kind = _broadcast_spec(source_shape, target_shape)
    if order is not None and order not in _ORDERS:
        raise ONPValueError(f"Invalid packing order: {order!r}.")

    rotations = set()
    for packing_order in _ORDERS if order is None else (order,):
        rotations.update(_rotation_indices(source_shape, target_shape, kind, packing_order))
    if rotations:
        secret_key.GetCryptoContext().EvalRotateKeyGen(secret_key, sorted(rotations))


def generate_block_broadcast_key(secret_key, block_matrix):
    """Generate row/column broadcast keys shared by all matrix blocks."""
    if block_matrix.ndim != 2:
        raise ONPDimensionError("Expected a two-dimensional block matrix.")
    if block_matrix.order not in _ORDERS:
        raise ONPValueError(f"Invalid packing order: {block_matrix.order!r}.")

    block_rows, block_cols = block_matrix.block_shape
    rotations = set()
    for source_shape in ((block_cols,), (block_rows, 1)):
        source_shape, target_shape, kind = _broadcast_spec(source_shape, block_matrix.block_shape)
        rotations.update(_rotation_indices(source_shape, target_shape, kind, block_matrix.order))
    if rotations:
        secret_key.GetCryptoContext().EvalRotateKeyGen(secret_key, sorted(rotations))


def broadcast_to(tensor, target_shape, order=None, cc=None):
    """Broadcast ``tensor`` to ``target_shape`` while preserving its order.

    Ciphertexts require keys from :func:`generate_broadcast_key`. Plaintexts
    require ``cc`` to encode the result. ``order`` is retained for compatibility
    and must equal ``tensor.order``.
    """
    from ..tensor.ctarray import CTArray
    from ..tensor.ptarray import PTArray

    if isinstance(tensor, CTArray):
        return _broadcast_ct(tensor, target_shape, order)
    if isinstance(tensor, PTArray):
        return _broadcast_pt(tensor, target_shape, order, cc)
    raise ONPValueError(f"Broadcasting does not support {type(tensor).__name__}.")


def _broadcast_pt(tensor, target_shape, requested_order, cc):
    source_shape, target_shape, kind = _broadcast_spec(tensor.original_shape, target_shape)
    order = _resolve_order(tensor, requested_order)
    if kind == "identity":
        return tensor
    if cc is None:
        raise ONPValueError("A crypto context is required for plaintext broadcasting.")

    padded_shape = _padded_shape(target_shape)
    if prod(padded_shape) > tensor.batch_size:
        raise ONPDimensionError(
            f"Padded target {padded_shape} exceeds batch_size={tensor.batch_size}."
        )
    source = np.asarray(tensor.decode())
    if source.shape != source_shape:
        raise ONPValueError(f"Decoded shape {source.shape} does not match metadata {source_shape}.")
    return array(
        cc=cc,
        data=np.broadcast_to(source, target_shape),
        batch_size=tensor.batch_size,
        order=order,
        fhe_type="P",
        mode="zero",
    )


def _mask(cc, ciphertext, indices, batch_size):
    values = _create_masking(indices, batch_size)
    return cc.EvalMult(ciphertext, cc.MakeCKKSPackedPlaintext(values))


def _scatter(cc, ciphertext, count, stride, batch_size):
    """Move source slot ``i`` to target slot ``i * stride``."""
    result = None
    for index in range(count):
        term = _mask(cc, ciphertext, [index], batch_size)
        shift = index * (stride - 1)
        if shift:
            term = cc.EvalRotate(term, -shift)
        result = term if result is None else cc.EvalAdd(result, term)
    return result


def _broadcast_ct(tensor, target_shape, requested_order):
    from ..tensor.ctarray import CTArray

    source_shape, target_shape, kind = _broadcast_spec(tensor.original_shape, target_shape)
    order = _resolve_order(tensor, requested_order)
    if kind == "identity":
        return tensor

    padded_shape = _padded_shape(target_shape)
    if prod(padded_shape) > tensor.batch_size:
        raise ONPDimensionError(
            f"Padded target {padded_shape} exceeds batch_size={tensor.batch_size}."
        )
    if kind in ("row", "column"):
        expected_size = prod(_padded_shape(source_shape))
        if prod(tuple(tensor.shape)) != expected_size:
            raise ONPNotImplementedError(
                "Compact vectors cannot be broadcast; re-encode without target_cols."
            )

    cc = tensor.data.GetCryptoContext()
    if kind == "scalar":
        result = _mask(cc, tensor.data, [0], tensor.batch_size)
        if len(target_shape) == 1:
            result = _duplicate_block(result, target_shape[0], 1)
        else:
            nrows, ncols = target_shape
            padded_rows, padded_cols = padded_shape
            if order == ArrayEncodingType.ROW_MAJOR:
                result = _duplicate_block(result, nrows, padded_cols)
                result = _duplicate_block(result, ncols, 1)
            else:
                result = _duplicate_block(result, ncols, padded_rows)
                result = _duplicate_block(result, nrows, 1)
    else:
        count, repeats, aligned, block_stride, scatter_stride = _vector_layout(
            kind, source_shape, target_shape, order
        )
        if aligned:
            result = _mask(cc, tensor.data, range(count), tensor.batch_size)
            result = _duplicate_block(result, repeats, block_stride)
        else:
            result = _scatter(cc, tensor.data, count, scatter_stride, tensor.batch_size)
            result = _duplicate_block(result, repeats, 1)

    if target_shape != padded_shape:
        nrows, ncols = target_shape if len(target_shape) == 2 else (1, target_shape[0])
        padded_rows, padded_cols = padded_shape if len(target_shape) == 2 else (1, padded_shape[0])
        if len(target_shape) == 1:
            valid_slots = range(ncols)
        elif order == ArrayEncodingType.ROW_MAJOR:
            valid_slots = [row * padded_cols + col for row in range(nrows) for col in range(ncols)]
        else:
            valid_slots = [col * padded_rows + row for col in range(ncols) for row in range(nrows)]
        result = _mask(cc, result, valid_slots, tensor.batch_size)

    physical_shape = (padded_shape[0], 1) if len(target_shape) == 1 else padded_shape
    return CTArray(
        data=result,
        original_shape=target_shape,
        batch_size=tensor.batch_size,
        new_shape=physical_shape,
        order=order,
    )
