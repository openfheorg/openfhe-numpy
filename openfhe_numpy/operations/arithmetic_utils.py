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
Shared arithmetic helpers used by both plain (CTArray/PTArray) and block
tensor operations.
"""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

import numpy as np

from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from openfhe_numpy.operations.broadcast import broadcast_to
from openfhe_numpy.tensor.tensor import BaseTensor
from openfhe_numpy.utils.errors import (
    ONPDimensionError,
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPNotSupportedError,
    ONPValueError,
    _require,
)
from openfhe_numpy.utils.packing import _is_col_major, _is_row_major
from openfhe_numpy.utils.typecheck import is_numeric_scalar


# ------------------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------------------


def _result_cls(a: BaseTensor, b: BaseTensor | None = None) -> type[BaseTensor]:
    """Return the encrypted tensor class if either operand is encrypted."""
    if a.is_encrypted:
        return type(a)

    if b is not None and b.is_encrypted:
        return type(b)

    raise ONPValueError("Expected at least one encrypted tensor operand.")


# ------------------------------------------------------------------------------
# Arithmetic: Validation
# ------------------------------------------------------------------------------


def _require_matvec_order(matrix_order: Any, vector_order: Any) -> None:
    """Raise ``ONPValueError`` unless the matrix-vector packing pair is supported.

    Supported pairs are ROW_MAJOR matrix @ COL_MAJOR vector, or COL_MAJOR matrix
    @ ROW_MAJOR vector.
    """
    supported = (_is_row_major(matrix_order) and _is_col_major(vector_order)) or (
        _is_col_major(matrix_order) and _is_row_major(vector_order)
    )
    _require(
        supported,
        (matrix_order,),
        (vector_order,),
        "Block matvec requires ROW_MAJOR matrix @ COL_MAJOR vector, "
        "or COL_MAJOR matrix @ ROW_MAJOR vector.",
        error_cls=ONPValueError,
    )


def _get_matvec_key_name(matrix_order: Any) -> str:
    """Return the key name required by matrix-vector multiplication."""
    if _is_row_major(matrix_order):
        return "colkey"

    if _is_col_major(matrix_order):
        return "rowkey"

    raise ONPValueError(f"Unsupported packing order: {matrix_order}")


# ------------------------------------------------------------------------------
# Arithmetic: Numpy axis convention
# ------------------------------------------------------------------------------


_AXIS_SUPPORT = {
    "cumsum": frozenset({"none", "integer"}),
    "sum": frozenset({"none", "integer"}),
    "mean": frozenset({"none", "integer"}),
    "roll": frozenset({"none"}),
    "cumulative_reduce": frozenset({"integer"}),
}


def _normalize_axis(
    operation: str,
    axis,
    ndim: int,
) -> int | None:
    """Validate and normalize an axis for one registered operation.

    Axis tuples are intentionally rejected until multi-axis runtimes exist.
    """
    try:
        supported = _AXIS_SUPPORT[operation]
    except KeyError as exc:
        raise ValueError(f"unknown axis operation {operation!r}.") from exc

    if axis is None:
        if "none" in supported:
            return None
        raise TypeError(f"{operation} requires an integer axis.")

    if isinstance(axis, tuple):
        if "tuple" not in supported:
            raise ONPNotSupportedError(
                f"{operation} does not support tuple axes."
            )
        raise ONPNotImplementedError(
            f"tuple-axis normalization is not implemented for {operation}."
        )

    if "integer" not in supported:
        raise ONPNotSupportedError(
            f"{operation} currently supports only axis=None."
        )

    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer, not boolean.")
    try:
        axis = operator_index(axis)
    except TypeError as exc:
        raise TypeError(f"axis must be an integer, got {type(axis).__name__}.") from exc

    if axis < -ndim or axis >= ndim:
        raise ONPDimensionError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions.")

    return axis + ndim if axis < 0 else axis


# ------------------------------------------------------------------------------
# Element-wise operation dispatch
# ------------------------------------------------------------------------------

_EVAL_METHODS: dict[str, str] = {
    "add": "EvalAdd",
    "subtract": "EvalSub",
    "multiply": "EvalMult",
}


def _resolve_eval_op(crypto_context, op_name: str):
    """Resolve an element-wise operation to its OpenFHE evaluator.

    Parameters
    ----------
    crypto_context
        Crypto context that owns the evaluator methods.
    op_name
        Supported operation name: ``add``, ``subtract``, or ``multiply``.

    Returns
    -------
    Callable
        Bound OpenFHE evaluator method.

    Raises
    ------
    ONPNotImplementedError
        If ``op_name`` is not supported.
    """
    try:
        method_name = _EVAL_METHODS[op_name]
    except KeyError:
        supported = ", ".join(sorted(_EVAL_METHODS))
        raise ONPNotImplementedError(
            f"Unsupported element-wise operation {op_name!r}. Supported operations: {supported}."
        ) from None

    return getattr(crypto_context, method_name)


# ------------------------------------------------------------------------------
# Packed binary operand context and alignment
# ------------------------------------------------------------------------------


def _binary_crypto_context(a, b):
    """Return the shared crypto context for two packed operands.

    Parameters
    ----------
    a, b
        Packed operands. At least one must be encrypted.

    Returns
    -------
    CryptoContext
        Crypto context shared by every encrypted operand.

    Raises
    ------
    ONPValueError
        If neither operand is encrypted, or encrypted operands use different
        crypto contexts or key tags.
    """
    ciphertexts = [operand.data for operand in (a, b) if operand.is_encrypted]

    if not ciphertexts:
        raise ONPValueError("Arithmetic requires at least one encrypted operand.")

    context = ciphertexts[0].GetCryptoContext()
    key_tag = ciphertexts[0].GetKeyTag() if hasattr(ciphertexts[0], "GetKeyTag") else None
    for ciphertext in ciphertexts[1:]:
        current_key_tag = ciphertext.GetKeyTag() if hasattr(ciphertext, "GetKeyTag") else None
        if ciphertext.GetCryptoContext() != context or current_key_tag != key_tag:
            raise ONPValueError("Operands must use the same crypto context and key tag.")

    return context


def _align_binary_operands(a, b):
    """Validate and broadcast two packed operands to one logical layout.

    Parameters
    ----------
    a, b
        Packed operands for an element-wise operation.

    Returns
    -------
    tuple
        ``(crypto_context, aligned_a, aligned_b)`` with both operands carrying
        the NumPy broadcast result as their logical shape and matching physical
        layouts.

    Raises
    ------
    ONPIncompatibleShapeError
        If the logical shapes are not broadcast-compatible.
    ONPValueError
        If batch sizes, packing orders, crypto contexts, key tags, or aligned
        physical frames differ.
    """
    source_a = tuple(a.original_shape)
    source_b = tuple(b.original_shape)
    try:
        output_shape = tuple(np.broadcast_shapes(source_a, source_b))
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            source_a,
            source_b,
            f"cannot broadcast {source_a} and {source_b}",
        ) from exc

    _require(
        a.batch_size == b.batch_size,
        (a.batch_size,),
        (b.batch_size,),
        "Element-wise arithmetic requires equal batch_size; "
        f"got {a.batch_size} and {b.batch_size}.",
        error_cls=ONPValueError,
    )
    _require(
        a.order == b.order,
        (a.order,),
        (b.order,),
        "Element-wise arithmetic requires matching packing order; "
        f"got {a.order!r} and {b.order!r}.",
        error_cls=ONPValueError,
    )

    crypto_context = _binary_crypto_context(a, b)
    if source_a != output_shape:
        a = broadcast_to(a, output_shape, order=a.order, cc=crypto_context)
    if source_b != output_shape:
        b = broadcast_to(b, output_shape, order=b.order, cc=crypto_context)

    _require(
        tuple(a.shape) == tuple(b.shape),
        a.shape,
        b.shape,
        f"Logically aligned operands use different physical frames; got {a.shape} and {b.shape}.",
        error_cls=ONPValueError,
    )

    return crypto_context, a, b


# ------------------------------------------------------------------------------
# Scalar logical-slot handling
# ------------------------------------------------------------------------------


def _logical_slot_indices(tensor):
    """Return the physical slot indices representing logical tensor values.

    Parameters
    ----------
    tensor
        Packed tensor whose logical region is described by ``original_shape``.

    Returns
    -------
    list[int]
        Active slot indices in the tensor's packing order. Physical padding and
        the batch tail are excluded; replicated vector lanes and repeated
        frames are included.

    Raises
    ------
    ONPDimensionError
        If the logical or physical rank is unsupported.
    ONPValueError
        If the logical shape exceeds the physical frame or the packing order is
        unsupported.
    """
    logical_shape = tuple(tensor.original_shape)
    physical_shape = tuple(tensor.shape)

    if len(logical_shape) == 0:
        frame_indices = [0]
    elif len(logical_shape) == 1:
        logical_length = logical_shape[0]
        if len(physical_shape) == 1 or physical_shape[1] == 1:
            frame_indices = list(range(logical_length))
        else:
            physical_rows, physical_cols = physical_shape
            if logical_length > physical_rows:
                raise ONPValueError(
                    f"logical shape {logical_shape} exceeds physical frame {tensor.shape}."
                )

            participating_cols = physical_cols
            if tensor.geometry is not None and tensor.geometry.padding == "zero":
                participating_cols = tensor.geometry.active[1]

            if tensor.order == ArrayEncodingType.ROW_MAJOR:
                frame_indices = [
                    row * physical_cols + col
                    for row in range(logical_length)
                    for col in range(participating_cols)
                ]
            elif tensor.order == ArrayEncodingType.COL_MAJOR:
                frame_indices = [
                    col * physical_rows + row
                    for col in range(participating_cols)
                    for row in range(logical_length)
                ]
            else:
                raise ONPValueError(f"Unsupported packing order {tensor.order!r}.")
    else:
        if len(logical_shape) != 2 or len(physical_shape) != 2:
            raise ONPDimensionError(
                f"Scalar arithmetic supports logical rank zero, one, or two; got {logical_shape}."
            )

        logical_rows, logical_cols = logical_shape
        physical_rows, physical_cols = physical_shape
        if logical_rows > physical_rows or logical_cols > physical_cols:
            raise ONPValueError(
                f"logical shape {logical_shape} exceeds physical frame {tensor.shape}."
            )

        if tensor.order == ArrayEncodingType.ROW_MAJOR:
            frame_indices = [
                row * physical_cols + col
                for row in range(logical_rows)
                for col in range(logical_cols)
            ]
        elif tensor.order == ArrayEncodingType.COL_MAJOR:
            frame_indices = [
                col * physical_rows + row
                for col in range(logical_cols)
                for row in range(logical_rows)
            ]
        else:
            raise ONPValueError(f"Unsupported packing order {tensor.order!r}.")

    frame_size = int(np.prod(physical_shape))
    repeats = 1 if tensor.geometry is None else tensor.geometry.repeats
    return [
        repeat_idx * frame_size + frame_idx
        for repeat_idx in range(repeats)
        for frame_idx in frame_indices
    ]


def _logical_plaintext(tensor, value, indices):
    """Encode a scalar in logical slots and zero every inactive slot.

    Parameters
    ----------
    tensor
        Packed tensor providing the crypto context and batch size.
    value
        Python or NumPy numeric scalar to encode.
    indices
        Active slot indices returned by :func:`_logical_slot_indices`.

    Returns
    -------
    Plaintext
        CKKS packed plaintext with ``value`` in active slots.

    Raises
    ------
    ONPValueError
        If ``value`` is not a numeric scalar or an index exceeds the batch.
    """
    scalar = np.asarray(value)
    if scalar.shape != () or not is_numeric_scalar(scalar):
        raise ONPValueError(f"Expected a numeric scalar, got {value!r}.")
    scalar = scalar.item()

    slots = [0.0] * tensor.batch_size
    for index in indices:
        if index < 0 or index >= tensor.batch_size:
            raise ONPValueError(
                f"logical slot index {index} exceeds batch_size={tensor.batch_size}."
            )
        slots[index] = scalar

    return tensor.data.GetCryptoContext().MakeCKKSPackedPlaintext(slots)


def _mask_ciphertext_to_logical_slots(tensor, indices):
    """Zero a ciphertext's physical padding and batch tail.

    Parameters
    ----------
    tensor
        Encrypted packed tensor to mask.
    indices
        Slot indices that belong to the tensor's logical region.

    Returns
    -------
    Ciphertext
        Original ciphertext when every batch slot is logical; otherwise a
        masked ciphertext containing only logical slots.
    """
    if len(indices) == tensor.batch_size and all(
        index == position for position, index in enumerate(indices)
    ):
        return tensor.data

    crypto_context = tensor.data.GetCryptoContext()
    mask = [0.0] * tensor.batch_size
    for index in indices:
        mask[index] = 1.0
    return crypto_context.EvalMult(
        tensor.data,
        crypto_context.MakeCKKSPackedPlaintext(mask),
    )


# ------------------------------------------------------------------------------
# Element-wise evaluation
# ------------------------------------------------------------------------------


def _eval_scalar_binary(tensor, scalar, operation, *, reverse=False):
    """Evaluate scalar arithmetic while keeping inactive result slots zero.

    Parameters
    ----------
    tensor
        Encrypted packed tensor operand.
    scalar
        Python or NumPy numeric scalar operand.
    operation
        Supported operation name: ``add``, ``subtract``, or ``multiply``.
    reverse
        Evaluate ``scalar operation tensor`` when true; otherwise evaluate
        ``tensor operation scalar``.

    Returns
    -------
    CTArray
        Metadata-preserving encrypted result.
    """
    crypto_context = tensor.data.GetCryptoContext()
    indices = _logical_slot_indices(tensor)
    scalar_plaintext = _logical_plaintext(tensor, scalar, indices)
    eval_op = _resolve_eval_op(crypto_context, operation)

    tensor_data = (
        tensor.data
        if operation == "multiply"
        else _mask_ciphertext_to_logical_slots(tensor, indices)
    )
    lhs_data, rhs_data = (
        (scalar_plaintext, tensor_data) if reverse else (tensor_data, scalar_plaintext)
    )
    return tensor.clone(eval_op(lhs_data, rhs_data))


def _eval_binary(a, b, op_name):
    """Align and evaluate one packed element-wise operation.

    Parameters
    ----------
    a, b
        Packed ciphertext/plaintext operands. At least one must be encrypted.
    op_name
        Supported operation name: ``add``, ``subtract``, or ``multiply``.

    Returns
    -------
    CTArray
        Encrypted result with the aligned logical/physical layout and metadata
        from both operands.
    """
    crypto_context, a, b = _align_binary_operands(a, b)
    result_data = _resolve_eval_op(crypto_context, op_name)(a.data, b.data)

    encrypted, other = (a, b) if a.is_encrypted else (b, a)
    result = encrypted.clone(result_data)
    result.extra.update(other.extra)
    return result
