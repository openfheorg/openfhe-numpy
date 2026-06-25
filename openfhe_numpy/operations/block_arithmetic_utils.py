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
from typing import Any, Callable

from openfhe_numpy import ArrayEncodingType
from openfhe_numpy.tensor.block_tensor import BlockFHETensor
from openfhe_numpy.tensor.block_ctarray import BlockCTArray
from openfhe_numpy.utils.errors import ONPNotImplementedError, ONPValueError
from openfhe_numpy.utils.typecheck import Number
from openfhe_numpy.utils.matlib import _sum_terms
from openfhe_numpy.operations.arithmetic_utils import (
    _result_cls,
    _require,
    _assert_matvec_order,
    _get_matvec_key_name,
)


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
        grid_shape=grid_shape or reference.grid_shape,
        block_shape=block_shape or reference.block_shape,
        original_shape=original_shape or reference.original_shape,
        batch_size=reference.batch_size,
        order=reference.order if order is None else order,
    )


# ------------------------------------------------------------------------------
# Element-wise operations
# ------------------------------------------------------------------------------


def _eval_block_binary(a: BlockFHETensor, b: BlockFHETensor, op_name: str) -> BlockFHETensor:
    """Evaluate a blockwise binary operation on two block tensors."""

    _require(
        a.same_layout(b),
        a.original_shape,
        b.original_shape,
        f"Block {op_name} requires identical block layout.",
    )

    op = _resolve_op(op_name)
    blocks = [op(left, right) for left, right in zip(a.data, b.data)]

    return _build_block_result(a, blocks, result_cls=BlockCTArray)


def _eval_block_scalar(a: BlockFHETensor, scalar: Number, op_name: str) -> BlockFHETensor:
    """Evaluate ``block_tensor op scalar`` blockwise."""
    op = _resolve_op(op_name)
    blocks = [op(block, scalar) for block in a.data]

    return _build_block_result(a, blocks)


def _eval_scalar_block(scalar: Number, a: BlockFHETensor, op_name: str) -> BlockFHETensor:
    """Evaluate ``scalar op block_tensor`` blockwise."""
    op = _resolve_op(op_name)
    blocks = [op(scalar, block) for block in a.data]

    return _build_block_result(a, blocks)


# ------------------------------------------------------------------------------
# Vector @ vector
# ------------------------------------------------------------------------------


def _eval_block_dot(a: BlockFHETensor, b: BlockFHETensor) -> Any:
    """Compute a block-vector inner product."""

    if not a.same_layout(b):
        raise ONPValueError("Incompatible layout")

    if not (a.is_encrypted or b.is_encrypted):
        raise ONPNotImplementedError("Operation requires at least one encrypted operand.")

    return _sum_terms(left @ right for left, right in zip(a.data, b.data))


# ------------------------------------------------------------------------------
# Matrix @ vector
# ------------------------------------------------------------------------------


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
    )
    _require(
        _assert_matvec_order(a.order, b.order),
        (a.order,),
        (b.order,),
        "Block matvec requires ROW_MAJOR matrix @ COL_MAJOR vector, or COL_MAJOR matrix @ ROW_MAJOR vector.",
    )

    key_name = _get_matvec_key_name(a.order)

    for index in a.iter_block_indices():
        block = a.get_block(*index)

        if block.is_encrypted and key_name not in block.extra:
            raise ONPValueError(
                f"Matrix block {index} is missing extra[{key_name!r}]. "
                "Call attach_matvec_keys(...) before block matrix-vector multiplication."
            )


def _eval_block_matvec(a: BlockFHETensor, b: BlockFHETensor) -> BlockFHETensor:
    """Compute block matrix-vector multiplication.

    For a block matrix A and block vector x, this computes y[i] = sum_k A[i,k] @ x[k].
    """
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


# ------------------------------------------------------------------------------
# Matrix @ matrix
# ------------------------------------------------------------------------------


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

    block_rows, block_cols = a.block_shape

    _require(
        block_rows == block_cols,
        a.block_shape,
        b.block_shape,
        "Block matmul requires square blocks.",
    )
    _require(
        a.batch_size == b.batch_size,
        (a.batch_size,),
        (b.batch_size,),
        "Block matmul requires equal batch_size.",
    )
    _require(
        a.order == b.order == ArrayEncodingType.ROW_MAJOR,
        (a.order,),
        (b.order,),
        "Block matmul currently requires both operands to use ROW_MAJOR packing.",
    )


def _eval_block_matmat(a: BlockFHETensor, b: BlockFHETensor) -> BlockFHETensor:
    """Compute block matrix-matrix multiplication.

    For block matrices A and B, this computes C[i,j] = sum_k A[i,k] @ B[k,j].

    Required logical condition:
    - a.shape = (m, k), b.shape = (k, n).

    Current implementation limitations:
    - Both tensors must be 2-D block matrices.
    - Blocks must have the same square block_shape.
    - Both operands must use ROW_MAJOR packing.
    - At least one operand must be encrypted.

    """
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


# ------------------------------------------------------------------------------
# Matmul dispatch
# ------------------------------------------------------------------------------


def _eval_block_matmul(a: BlockFHETensor, b: BlockFHETensor) -> Any:
    """Dispatch block matmul by operand ranks."""
    if a.ndim == 1 and b.ndim == 1:
        return _eval_block_dot(a, b)

    if a.ndim == 2 and b.ndim == 1:
        return _eval_block_matvec(a, b)

    if a.ndim == 2 and b.ndim == 2:
        return _eval_block_matmat(a, b)

    raise ONPNotImplementedError(f"Block matmul does not support ndim={a.ndim} @ ndim={b.ndim}.")
