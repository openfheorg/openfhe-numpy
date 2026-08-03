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

from __future__ import annotations

import operator
from typing import Any, Callable, Optional

import numpy as np

from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from openfhe_numpy.operations.arithmetic_utils import (
    _get_matvec_key_name,
    _normalize_axis,
    _normalize_sum_axis,
    _require,
    _require_matvec_order,
    _result_cls,
)
from openfhe_numpy.operations.block_broadcast import _block_broadcast_to
from openfhe_numpy.operations.matrix_arithmetic import _reduce_ct, sum_ct
from openfhe_numpy.tensor.block_ctarray import BlockCTArray
from openfhe_numpy.tensor.block_tensor import BlockFHETensor
from openfhe_numpy.utils._helper_slots_ops import _create_masking
from openfhe_numpy.utils.errors import (
    ONPDimensionError,
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPValueError,
)
from openfhe_numpy.utils.matlib import _sum_terms
from openfhe_numpy.utils.packing import _is_col_major, _is_row_major
from openfhe_numpy.utils.typecheck import Number


# ------------------------------------------------------------------------------
# Type aliases
# ------------------------------------------------------------------------------

BlockOp = Callable[[Any, Any], Any]


# ------------------------------------------------------------------------------
# Supported element-wise operators
# ------------------------------------------------------------------------------

_OPS: dict[str, BlockOp] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
}


def _resolve_op(op_name: str) -> BlockOp:
    """Return the blockwise Python operator for ``op_name``."""
    try:
        return _OPS[op_name]
    except KeyError:
        supported = ", ".join(sorted(_OPS))
        raise ONPNotImplementedError(
            f"Unsupported block operation {op_name!r}. Supported operations: {supported}."
        ) from None


# ------------------------------------------------------------------------------
# Generic block helpers
# ------------------------------------------------------------------------------


def _build_block_result(
    reference: BlockFHETensor,
    blocks: list[Any],
    *,
    result_cls: type[BlockFHETensor] | None = None,
    grid_shape: tuple[int, ...] | None = None,
    block_shape: tuple[int, ...] | None = None,
    original_shape: tuple[int, ...] | None = None,
    order: Any | None = None,
) -> BlockFHETensor:
    """Construct a block tensor result while preserving reference metadata by default."""
    cls = result_cls or type(reference)

    return cls(
        data=blocks,
        grid_shape=reference.grid_shape if grid_shape is None else grid_shape,
        block_shape=reference.block_shape if block_shape is None else block_shape,
        original_shape=reference.original_shape if original_shape is None else original_shape,
        batch_size=reference.batch_size,
        order=reference.order if order is None else order,
    )


def _inherit_compatible_keys(result: Any, *sources: Any) -> Any:
    """Copy evaluation-key metadata from encrypted, layout-compatible sources."""
    if not getattr(result, "is_encrypted", False):
        return result

    layout_attrs = ("original_shape", "shape", "batch_size", "order")
    for source in sources:
        if not getattr(source, "is_encrypted", False):
            continue
        if not all(getattr(source, attr) == getattr(result, attr) for attr in layout_attrs):
            continue
        for name, key in source.extra.items():
            result.extra.setdefault(name, key)

    return result


# ------------------------------------------------------------------------------
# Element-wise operations
# ------------------------------------------------------------------------------


def _eval_block_binary(a: BlockFHETensor, b: BlockFHETensor, op_name: str) -> BlockFHETensor:
    """Evaluate a blockwise binary operation on two block tensors.

    When both operands share the same block layout the operation is applied
    block by block. Otherwise the smaller operand is broadcast to the layout of
    the full block matrix (see :func:`_eval_block_broadcast_binary`).
    """
    if not a.same_layout(b):
        return _eval_block_broadcast_binary(a, b, op_name)

    op = _resolve_op(op_name)
    blocks = [
        _inherit_compatible_keys(op(left, right), left, right)
        for left, right in zip(a.data, b.data)
    ]
    return _build_block_result(a, blocks, result_cls=_result_cls(a, b))


def _eval_block_broadcast_binary(
    a: BlockFHETensor, b: BlockFHETensor, op_name: str
) -> BlockFHETensor:
    """Evaluate ``a op b`` when the operands have different block layouts.

    One operand must already have the broadcast output shape (the full matrix);
    the other is broadcast up to it. Operand order is preserved so that
    non-commutative operations (e.g. ``subtract``) keep NumPy semantics.
    """
    try:
        out_shape = tuple(np.broadcast_shapes(a.original_shape, b.original_shape))
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            f"Block {op_name} operands are not broadcast-compatible.",
        ) from exc

    a_is_full = a.original_shape == out_shape
    b_is_full = b.original_shape == out_shape

    if a_is_full and not b_is_full:
        target, source, source_on_left = a, b, False
    elif b_is_full and not a_is_full:
        target, source, source_on_left = b, a, True
    else:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            f"Block {op_name} requires broadcast-compatible block layouts. The "
            "operands share a logical shape but differ in block tiling, or both "
            "would need expansion (unsupported).",
        )

    expanded = _block_broadcast_to(source, target)

    op = _resolve_op(op_name)
    if source_on_left:
        blocks = [
            _inherit_compatible_keys(op(left, right), right)
            for left, right in zip(expanded.data, target.data)
        ]
    else:
        blocks = [
            _inherit_compatible_keys(op(left, right), left)
            for left, right in zip(target.data, expanded.data)
        ]

    result = _build_block_result(target, blocks, result_cls=_result_cls(a, b))
    if op_name in {"add", "subtract"}:
        return _clear_block_padding(result)
    return result


def _eval_block_scalar(a: BlockFHETensor, scalar: Number, op_name: str) -> BlockFHETensor:
    """Evaluate ``block_tensor op scalar`` blockwise."""
    op = _resolve_op(op_name)
    blocks = [_inherit_compatible_keys(op(block, scalar), block) for block in a.data]

    result = _build_block_result(a, blocks)
    if op_name in {"add", "subtract"}:
        return _clear_block_padding(result)
    return result


def _eval_scalar_block(scalar: Number, a: BlockFHETensor, op_name: str) -> BlockFHETensor:
    """Evaluate ``scalar op block_tensor`` blockwise."""
    op = _resolve_op(op_name)
    blocks = [_inherit_compatible_keys(op(scalar, block), block) for block in a.data]

    result = _build_block_result(a, blocks)
    if op_name == "subtract":
        return _clear_block_padding(result)
    return result


# ------------------------------------------------------------------------------
# Outer-edge masking
# ------------------------------------------------------------------------------
#
# Element-wise scalar operations and broadcasts can turn constructor padding into
# non-zero values. Decryption crops those away, but a later block reduction would
# otherwise sum them in, so edge blocks are re-zeroed after those operations.


def _flatten_mask(mask: np.ndarray, order: Any) -> np.ndarray:
    """Flatten a logical validity mask using the ciphertext packing order."""
    if _is_row_major(order):
        return mask.ravel(order="C")
    if _is_col_major(order):
        return mask.ravel(order="F")
    raise ONPValueError(f"Unsupported packing order: {order}")


def _build_block_padding_mask(block, valid_shape: tuple[int, ...]) -> list[float]:
    """Build a batch-sized mask, retaining valid cells in every packed copy."""
    physical_shape = tuple(block.shape)

    if len(valid_shape) == 1:
        valid_length = valid_shape[0]
        if len(physical_shape) == 1:
            logical_mask = np.zeros(physical_shape, dtype=np.float64)
            logical_mask[:valid_length] = 1.0
        elif len(physical_shape) == 2:
            # Compact vectors store logical vector entries along matrix rows in
            # both packing orders; the columns are duplicated copies.
            logical_mask = np.zeros(physical_shape, dtype=np.float64)
            logical_mask[:valid_length, :] = 1.0
        else:
            raise ONPValueError(
                f"Unsupported packed vector shape for edge masking: {physical_shape}."
            )
    else:
        valid_rows, valid_cols = valid_shape
        if len(physical_shape) != 2:
            raise ONPValueError(
                f"Unsupported packed matrix shape for edge masking: {physical_shape}."
            )
        logical_mask = np.zeros(physical_shape, dtype=np.float64)
        logical_mask[:valid_rows, :valid_cols] = 1.0

    one_copy = _flatten_mask(logical_mask, block.order)
    packed = np.zeros(block.batch_size, dtype=np.float64)
    copy_size = len(one_copy)

    for start in range(0, block.batch_size, copy_size):
        stop = min(start + copy_size, block.batch_size)
        packed[start:stop] = one_copy[: stop - start]

    return packed.tolist()


def _clear_block_padding(tensor: BlockFHETensor) -> BlockFHETensor:
    """Set padding cells in partial boundary blocks to zero."""
    if not tensor.is_encrypted:
        return tensor

    mask_cache: dict[tuple[Any, ...], Any] = {}
    blocks = []

    for index in tensor.iter_block_indices():
        block = tensor.get_block(*index)

        if tensor.ndim == 1:
            block_length = tensor.block_shape[0]
            valid_length = max(
                0,
                min(block_length, tensor.original_shape[0] - index[0] * block_length),
            )
            valid_shape = (valid_length,)
            is_full = valid_length == block_length
        else:
            block_rows, block_cols = tensor.block_shape
            valid_rows = max(
                0,
                min(block_rows, tensor.original_shape[0] - index[0] * block_rows),
            )
            valid_cols = max(
                0,
                min(block_cols, tensor.original_shape[1] - index[1] * block_cols),
            )
            valid_shape = (valid_rows, valid_cols)
            is_full = valid_shape == tensor.block_shape

        if is_full:
            blocks.append(block)
            continue

        cache_key = (
            tuple(block.shape),
            block.batch_size,
            block.order,
            valid_shape,
        )
        pt_mask = mask_cache.get(cache_key)
        if pt_mask is None:
            mask = _build_block_padding_mask(block, valid_shape)
            pt_mask = block.crypto_context.MakeCKKSPackedPlaintext(mask)
            mask_cache[cache_key] = pt_mask

        masked = block.clone(block.crypto_context.EvalMult(block.data, pt_mask))
        masked.extra.update(block.extra)
        blocks.append(masked)

    return tensor.clone(data=blocks)


# ------------------------------------------------------------------------------
# Reduction operations
# ------------------------------------------------------------------------------


def _cumsum_block_ct(
    obj: BlockCTArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
):
    """Compute cumulative sums for an encrypted block tensor."""
    if keepdims:
        raise ONPNotImplementedError("Block cumsum does not support keepdims=True yet.")

    if obj.ndim not in (1, 2):
        raise ONPDimensionError(f"cumsum requires a 1-D or 2-D tensor, got {obj.ndim}-D.")

    axis = _normalize_sum_axis(axis, obj.ndim)

    if obj.ndim == 1:
        axis = 0 if axis is None else axis

        if any(len(block.shape) > 1 and block.shape[1] != 1 for block in obj.data):
            raise ONPNotImplementedError(
                "Block cumsum does not support compact block-vector packing."
            )

        blocks = []
        carry = None

        total_mask = None
        if obj.num_blocks > 1:
            cc = obj.data[0].crypto_context
            block_width = obj.block_shape[0]
            mask = _create_masking(range(block_width), obj.batch_size)
            total_mask = cc.MakeCKKSPackedPlaintext(mask)

        for index, block in enumerate(obj.data):
            cumulative = block.cumsum(axis=axis)

            if carry is not None:
                cumulative = cumulative + carry

            blocks.append(cumulative)

            if index + 1 < obj.num_blocks:
                one_copy = cc.EvalMult(block.data, total_mask)
                total_data = cc.EvalSum(one_copy, block.batch_size)
                block_total = block.clone(total_data)

                carry = block_total if carry is None else carry + block_total

        return obj.clone(data=blocks)

    if axis is None:
        raise ONPNotImplementedError("Block cumsum(axis=None) is not implemented for 2-D tensors.")

    if obj.grid_shape[axis] != 1:
        direction = "rows" if axis == 0 else "columns"
        raise ONPNotImplementedError(
            f"Block cumsum(axis={axis}) across multiple block {direction} is not implemented."
        )

    return obj.clone(data=[block.cumsum(axis=axis) for block in obj.data])


def _cumulative_reduce_block_ct(
    a: BlockCTArray,
    axis: int = 0,
    keepdims: bool = False,
):
    """Compute cumulative reductions inside each encrypted block."""
    if keepdims:
        raise ONPNotImplementedError("Block cumulative_reduce does not support keepdims=True yet.")

    if a.ndim != 2:
        raise ONPDimensionError(f"cumulative_reduce requires a 2D block tensor, got {a.ndim}D.")

    axis = _normalize_axis(axis, a.ndim)

    if axis == 0 and a.grid_shape[0] != 1:
        raise ONPNotImplementedError(
            "Block cumulative_reduce(axis=0) across multiple block rows is not implemented yet."
        )

    if axis == 1 and a.grid_shape[1] != 1:
        raise ONPNotImplementedError(
            "Block cumulative_reduce(axis=1) across multiple block columns is not implemented yet."
        )

    return a.clone(data=[_reduce_ct(block, axis, keepdims) for block in a.data])


def _sum_block_ct(
    x: BlockCTArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
):
    """Sum encrypted block tensor elements."""
    if keepdims:
        raise ONPNotImplementedError("Block sum does not support keepdims=True yet.")

    axis = _normalize_sum_axis(axis, x.ndim)

    if x.ndim == 1:
        return _sum_terms(sum_ct(block, None, False) for block in x.data)

    if x.ndim != 2:
        raise ONPDimensionError(f"sum requires 1D or 2D tensor, got {x.ndim}D.")

    if axis is None:
        return _sum_terms(sum_ct(block, None, False) for block in x.data)

    grid_rows, grid_cols = x.grid_shape

    if axis == 0:
        blocks = [
            _sum_terms(
                sum_ct(
                    x.get_block(i, j),
                    axis=0,
                    keepdims=False,
                )
                for i in range(grid_rows)
            )
            for j in range(grid_cols)
        ]

        output_block_shape = (x.block_shape[1],)
        output_original_shape = (x.original_shape[1],)

    else:
        blocks = [
            _sum_terms(
                sum_ct(
                    x.get_block(i, j),
                    axis=1,
                    keepdims=False,
                )
                for j in range(grid_cols)
            )
            for i in range(grid_rows)
        ]

        output_block_shape = (x.block_shape[0],)
        output_original_shape = (x.original_shape[0],)

    return type(x)(
        data=blocks,
        grid_shape=(len(blocks),),
        block_shape=output_block_shape,
        original_shape=output_original_shape,
        batch_size=x.batch_size,
        order=blocks[0].order,
    )


def _mean_block_ct(
    x: BlockCTArray,
    axis: Optional[int] = None,
    keepdims: bool = False,
):
    """Compute the arithmetic mean for an encrypted block tensor."""
    if keepdims:
        raise ONPNotImplementedError("Block mean does not support keepdims=True yet.")

    axis = _normalize_sum_axis(axis, x.ndim)
    mean_x = _sum_block_ct(x, axis, keepdims=False)

    if x.ndim == 1:
        n = x.original_shape[0]

    elif x.ndim == 2:
        nrows, ncols = x.original_shape
        if axis is None:
            n = nrows * ncols
        elif axis == 0:
            n = nrows
        elif axis == 1:
            n = ncols
        else:
            raise ONPDimensionError(f"Invalid axis {axis} for 2D block tensor.")

    else:
        raise ONPDimensionError(f"mean requires 1D or 2D tensor, got {x.ndim}D.")

    return mean_x * (1.0 / n)


# ------------------------------------------------------------------------------
# Linear-algebra operations
# ------------------------------------------------------------------------------


def _transpose_block_ct(a: BlockFHETensor) -> BlockFHETensor:
    """Transpose an encrypted block tensor."""
    if a.ndim == 1:
        return a

    if a.ndim != 2:
        raise ONPDimensionError(f"transpose requires 1D or 2D tensor, got {a.ndim}D.")

    grid_rows, grid_cols = a.grid_shape
    blocks = [a.get_block(i, j).transpose() for j in range(grid_cols) for i in range(grid_rows)]

    return _build_block_result(
        a,
        blocks,
        grid_shape=(grid_cols, grid_rows),
        block_shape=(a.block_shape[1], a.block_shape[0]),
        original_shape=(a.original_shape[1], a.original_shape[0]),
    )


def _eval_block_dot(a: BlockFHETensor, b: BlockFHETensor) -> Any:
    """Compute a block-vector inner product."""
    _require(
        a.ndim == 1 and b.ndim == 1,
        a.original_shape,
        b.original_shape,
        "Block dot requires two 1-D block vectors.",
    )

    if not a.same_layout(b):
        raise ONPValueError("Incompatible layout")

    if not (a.is_encrypted or b.is_encrypted):
        raise ONPNotImplementedError("Operation requires at least one encrypted operand.")

    return _sum_terms(left @ right for left, right in zip(a.data, b.data))


def _assert_block_matvec_compatible(a: BlockFHETensor, b: BlockFHETensor) -> None:
    """Validate block matrix-vector multiplication inputs."""
    _require(
        a.ndim == 2 and b.ndim == 1,
        a.original_shape,
        b.original_shape,
        "Block matvec requires a 2-D block matrix and a 1-D block vector.",
    )

    if not (a.is_encrypted or b.is_encrypted):
        raise ONPNotImplementedError("Operation requires at least one encrypted operand.")

    _require(
        a.original_shape[1] == b.original_shape[0],
        a.original_shape,
        b.original_shape,
        "Block matvec dimension mismatch.",
    )
    _require(
        a.grid_shape[1] == b.grid_shape[0],
        a.grid_shape,
        b.grid_shape,
        "Block matvec requires matching inner block dimension.",
    )
    _require(
        len(a.block_shape) == 2 and len(b.block_shape) == 1,
        a.block_shape,
        b.block_shape,
        "Matrix blocks must be 2-D, vector blocks 1-D.",
    )

    _, block_cols = a.block_shape

    _require(
        b.block_shape == (block_cols,),
        a.block_shape,
        b.block_shape,
        f"Vector block_shape must be ({block_cols},) to match matrix block_shape={a.block_shape}.",
    )
    _require(
        a.batch_size == b.batch_size,
        (a.batch_size,),
        (b.batch_size,),
        "Block matvec requires equal batch_size.",
        error_cls=ONPValueError,
    )
    _require_matvec_order(a.order, b.order)

    key_name = _get_matvec_key_name(a.order)

    for index in a.iter_block_indices():
        block = a.get_block(*index)
        if key_name not in block.extra:
            raise ONPValueError(
                f"Matrix block {index} is missing extra[{key_name!r}]. "
                "Call attach_block_matvec_keys(...) before block matrix-vector multiplication."
            )


def _eval_block_matvec(a: BlockFHETensor, b: BlockFHETensor) -> BlockFHETensor:
    """Compute block matrix-vector multiplication."""
    _assert_block_matvec_compatible(a, b)

    block_rows, _ = a.block_shape
    grid_rows, grid_inner = a.grid_shape

    blocks = [
        _sum_terms(a.get_block(i, k) @ b.get_block(k) for k in range(grid_inner))
        for i in range(grid_rows)
    ]

    return _build_block_result(
        a,
        blocks,
        result_cls=BlockCTArray,
        grid_shape=(grid_rows,),
        block_shape=(block_rows,),
        original_shape=(a.original_shape[0],),
        order=a.order,
    )


def _assert_block_matmul_compatible(a: BlockFHETensor, b: BlockFHETensor) -> None:
    """Validate block matrix-matrix multiplication inputs."""
    _require(
        a.ndim == 2 and b.ndim == 2,
        a.original_shape,
        b.original_shape,
        "Block matmul requires two 2-D block matrices.",
    )

    if not (a.is_encrypted or b.is_encrypted):
        raise ONPNotImplementedError("Operation requires at least one encrypted operand.")

    _require(
        a.original_shape[1] == b.original_shape[0],
        a.original_shape,
        b.original_shape,
        "Block matmul dimension mismatch.",
    )
    _require(
        a.grid_shape[1] == b.grid_shape[0],
        a.grid_shape,
        b.grid_shape,
        "Block matmul requires matching inner block dimension.",
    )
    _require(
        a.block_shape == b.block_shape,
        a.block_shape,
        b.block_shape,
        "Block matmul requires equal block_shape.",
    )
    _require(
        len(a.block_shape) == 2,
        a.block_shape,
        b.block_shape,
        "Block matmul requires 2-D matrix blocks.",
    )

    block_rows, _ = a.block_shape
    _require(
        block_rows == a.block_shape[1],
        a.block_shape,
        b.block_shape,
        "Block matmul requires square blocks.",
    )
    _require(
        a.batch_size == b.batch_size,
        (a.batch_size,),
        (b.batch_size,),
        "Block matmul requires equal batch_size.",
        error_cls=ONPValueError,
    )
    _require(
        a.order == b.order == ArrayEncodingType.ROW_MAJOR,
        (a.order,),
        (b.order,),
        "Block matmul currently requires both operands to use ROW_MAJOR packing.",
        error_cls=ONPValueError,
    )


def _eval_block_matmat(a: BlockFHETensor, b: BlockFHETensor) -> BlockFHETensor:
    """Compute block matrix-matrix multiplication."""
    _assert_block_matmul_compatible(a, b)

    grid_rows = a.grid_shape[0]
    grid_inner = a.grid_shape[1]
    grid_cols = b.grid_shape[1]

    blocks = [
        _sum_terms(a.get_block(i, k) @ b.get_block(k, j) for k in range(grid_inner))
        for i in range(grid_rows)
        for j in range(grid_cols)
    ]

    return _build_block_result(
        a,
        blocks,
        result_cls=_result_cls(a, b),
        grid_shape=(grid_rows, grid_cols),
        block_shape=a.block_shape,
        original_shape=(a.original_shape[0], b.original_shape[1]),
        order=a.order,
    )


def _eval_block_matmul(a: BlockFHETensor, b: BlockFHETensor) -> Any:
    """Dispatch block matmul by operand ranks."""
    if a.ndim == 1 and b.ndim == 1:
        return _eval_block_dot(a, b)

    if a.ndim == 2 and b.ndim == 1:
        return _eval_block_matvec(a, b)

    if a.ndim == 2 and b.ndim == 2:
        return _eval_block_matmat(a, b)

    raise ONPNotImplementedError(f"Block matmul does not support ndim={a.ndim} @ ndim={b.ndim}.")
