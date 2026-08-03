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

"""Broadcast a block row ``(n,)`` / ``(1, n)`` or column ``(m, 1)`` into a matrix.

Each source block is expanded into one matrix block via the per-block
:func:`broadcast_to`, then reused across the opposite grid dimension.
"""

from __future__ import annotations

import numpy as np

from openfhe_numpy.operations.arithmetic_utils import _require
from openfhe_numpy.operations.broadcast import broadcast_to
from openfhe_numpy.tensor.block_tensor import BlockFHETensor
from openfhe_numpy.utils.errors import (
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPValueError,
)


def _block_broadcast_context(*tensors: BlockFHETensor):
    """Return the crypto context of the first encrypted block tensor."""
    for tensor in tensors:
        if tensor.is_encrypted and tensor.data:
            return tensor.data[0].crypto_context

    raise ONPValueError("Block broadcasting requires at least one encrypted operand.")


def _validate_logical_broadcast(source: BlockFHETensor, target: BlockFHETensor) -> None:
    """Require ``source`` to broadcast exactly to ``target``'s logical shape.

    Block padding can make geometrically-aligned tensors look compatible when
    their logical shapes are not, so validate on ``original_shape`` here rather
    than relying only on the arithmetic caller.
    """
    try:
        result_shape = tuple(
            np.broadcast_shapes(source.original_shape, target.original_shape)
        )
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            source.original_shape,
            target.original_shape,
            "Block operands are not broadcast-compatible.",
        ) from exc

    if result_shape != target.original_shape:
        raise ONPIncompatibleShapeError(
            source.original_shape,
            target.original_shape,
            f"Broadcasting would produce {result_shape}, not {target.original_shape}.",
        )


def _block_broadcast_to(
    source: BlockFHETensor,
    target: BlockFHETensor,
) -> BlockFHETensor:
    """Broadcast ``source`` to the block layout of the 2-D ``target``.

    A source block is expanded once per shared-axis lane, then reused across the
    opposite grid dimension. The returned wrapper preserves ``type(source)``:
    plaintext expansion remains a ``BlockPTArray`` and encrypted expansion
    remains a ``BlockCTArray``.
    """
    if target.ndim != 2:
        raise ONPNotImplementedError(
            "Block broadcasting is only supported for a 2-D target block matrix."
        )

    _validate_logical_broadcast(source, target)

    # A logical singleton (1,)/(1, 1) is a valid NumPy broadcast, but the row and
    # column kernels below would replicate its zero-padded block (s, 0, ...)
    # instead of the scalar s. Reject it explicitly; use a scalar operation.
    if all(dim == 1 for dim in source.original_shape):
        raise ONPNotImplementedError(
            "Block broadcasting of a logical singleton (1,)/(1, 1) is not supported; "
            "use a scalar operation instead."
        )

    grid_rows, grid_cols = target.grid_shape
    block_rows, block_cols = target.block_shape
    order = target.order
    cc = _block_broadcast_context(source, target)

    _require(
        source.batch_size == target.batch_size,
        (source.batch_size,),
        (target.batch_size,),
        "Block broadcasting requires equal batch_size.",
        error_cls=ONPValueError,
    )
    _require(
        source.order == target.order,
        (source.order,),
        (target.order,),
        "Block broadcasting requires matching packing order.",
        error_cls=ONPValueError,
    )

    if source.ndim == 1:
        _require(
            source.block_shape[0] == block_cols and source.grid_shape[0] == grid_cols,
            source.block_shape,
            target.block_shape,
            "Row broadcasting requires the vector block/grid columns to match the "
            f"matrix: expected block_shape=({block_cols},), grid_shape=({grid_cols},).",
        )
        lane_blocks = [
            broadcast_to(source.get_block(j), (block_rows, block_cols), order, cc)
            for j in range(grid_cols)
        ]
        blocks = [lane_blocks[j] for _ in range(grid_rows) for j in range(grid_cols)]

    elif source.ndim == 2 and source.original_shape[0] == 1:
        _require(
            source.block_shape == (1, block_cols) and source.grid_shape == (1, grid_cols),
            source.block_shape,
            target.block_shape,
            "Row-matrix broadcasting requires the (1, n) block/grid columns to match the "
            f"matrix: expected block_shape=(1, {block_cols}), grid_shape=(1, {grid_cols}).",
        )
        lane_blocks = [
            broadcast_to(source.get_block(0, j), (block_rows, block_cols), order, cc)
            for j in range(grid_cols)
        ]
        blocks = [lane_blocks[j] for _ in range(grid_rows) for j in range(grid_cols)]

    elif source.ndim == 2 and source.original_shape[1] == 1:
        _require(
            source.block_shape == (block_rows, 1) and source.grid_shape == (grid_rows, 1),
            source.block_shape,
            target.block_shape,
            "Column broadcasting requires the vector block/grid rows to match the "
            f"matrix: expected block_shape=({block_rows}, 1), grid_shape=({grid_rows}, 1).",
        )
        lane_blocks = [
            broadcast_to(source.get_block(i, 0), (block_rows, block_cols), order, cc)
            for i in range(grid_rows)
        ]
        blocks = [lane_blocks[i] for i in range(grid_rows) for _ in range(grid_cols)]

    else:
        raise ONPNotImplementedError(
            f"Block broadcasting from original_shape={source.original_shape} to "
            f"{target.original_shape} is not supported. Supported sources: row vector "
            "(n,), row matrix (1, n), or column vector (m, 1) into a matrix (m, n)."
        )

    return type(source)(
        data=blocks,
        grid_shape=target.grid_shape,
        block_shape=target.block_shape,
        original_shape=target.original_shape,
        batch_size=target.batch_size,
        order=target.order,
    )
