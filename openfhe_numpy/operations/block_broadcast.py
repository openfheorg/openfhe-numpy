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

"""Broadcast aligned block vectors and matrices.

Broadcasting follows NumPy rules for logical shapes while preserving existing
block boundaries. Supported sources have shape ``(n,)``, ``(1, n)``, or
``(m, 1)``. Re-tiling and compact vector packing are not supported, and at least
one operand must be encrypted.

Use :func:`generate_block_broadcast_key` to generate the required rotation keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openfhe_numpy.operations.arithmetic_utils import _binary_crypto_context, _require
from openfhe_numpy.operations.broadcast import (
    _broadcast_rotation_indices,
    _broadcast_to_physical_slots,
)
from openfhe_numpy.tensor.block_ctarray import BlockCTArray
from openfhe_numpy.tensor.block_tensor import BlockFHETensor
from openfhe_numpy.utils.errors import (
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPNotSupportedError,
    ONPTypeError,
    ONPValueError,
)


# ----------------------------------------------------------------------------
# Shape / layout helpers
# ----------------------------------------------------------------------------


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the common NumPy broadcast shape.

    Parameters
    ----------
    *shapes : tuple[int, ...]
        Logical tensor shapes.

    Returns
    -------
    tuple[int, ...]
        Shape obtained by applying NumPy broadcasting rules.

    Raises
    ------
    ONPIncompatibleShapeError
        If the shapes are not broadcast-compatible.
    """
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise ONPIncompatibleShapeError(
            shapes[0],
            shapes[-1],
            "Block operands are not broadcast-compatible.",
        ) from exc


def _is_standard_block_layout(tensor: BlockFHETensor) -> bool:
    """Check whether a tensor uses standard block packing.

    Parameters
    ----------
    tensor : BlockFHETensor
        Block tensor to inspect.

    Returns
    -------
    bool
        ``True`` if the physical block shape matches the logical block shape.

    Notes
    -----
    A standard vector block of logical shape ``(b,)`` has physical shape
    ``(b, 1)``. Compact vector blocks use a square physical layout.
    """
    block_shape = tensor.block_shape
    # Standard 1-D packing has physical shape (b, 1); compact packing is square.
    standard = (block_shape[0], 1) if len(block_shape) == 1 else block_shape
    return tuple(tensor.data[0].shape) == standard


def _source_kind(source: BlockFHETensor) -> str:
    """Classify a supported broadcast source.

    Parameters
    ----------
    source : BlockFHETensor
        Source tensor to classify.

    Returns
    -------
    {"row_vector", "row_matrix", "column"}
        Source category determined from the logical shape.

    Raises
    ------
    ONPNotSupportedError
        If ``source`` is not a vector, row matrix, or column matrix.
    """
    if source.ndim == 1:
        return "row_vector"
    if source.original_shape[0] == 1:
        return "row_matrix"
    if source.original_shape[1] == 1:
        return "column"
    raise ONPNotSupportedError(
        "Block broadcasting supports only (n,), (1, n), or (m, 1) sources; "
        f"got shape {source.original_shape}."
    )


def _block_original_shape(
    original_shape: tuple[int, int],
    block_shape: tuple[int, int],
    row: int,
    column: int,
) -> tuple[int, int]:
    """Compute the unpadded shape of one result block.

    Parameters
    ----------
    original_shape : tuple[int, int]
        Logical shape of the result matrix.
    block_shape : tuple[int, int]
        Shape of each padded result block.
    row : int
        Block-grid row index.
    column : int
        Block-grid column index.

    Returns
    -------
    tuple[int, int]
        Logical shape of the selected block before edge padding.
    """
    rows, cols = original_shape
    block_rows, block_cols = block_shape
    return (
        min(block_rows, rows - row * block_rows),
        min(block_cols, cols - column * block_cols),
    )


# ----------------------------------------------------------------------------
# Per-block expansion
# ----------------------------------------------------------------------------


def _source_block(source: BlockFHETensor, kind: str, row: int, column: int):
    """Select the source block for a result-grid position.

    Parameters
    ----------
    source : BlockFHETensor
        Block row or column being broadcast.
    kind : {"row_vector", "row_matrix", "column"}
        Source category returned by :func:`_source_kind`.
    row : int
        Result block-grid row index.
    column : int
        Result block-grid column index.

    Returns
    -------
    CTArray or PTArray
        Encoded block reused at position ``(row, column)``.
    """
    # A row block is reused down its grid column; a column block is reused
    # across its grid row.
    if kind == "row_vector":
        return source.get_block(column)
    if kind == "row_matrix":
        return source.get_block(0, column)
    return source.get_block(row, 0)


# ----------------------------------------------------------------------------
# Layout-driven broadcast
# ----------------------------------------------------------------------------


def _validate_source_for_layout(
    source: BlockFHETensor,
    original_shape: tuple[int, int],
    block_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    batch_size: int,
    order: Any,
) -> str:
    """Validate a row or column source against a result block layout."""
    _require(
        len(original_shape) == 2,
        original_shape,
        source.original_shape,
        "Block broadcasting produces a matrix result.",
        error_cls=ONPNotSupportedError,
    )
    result_shape = _broadcast_shape(source.original_shape, original_shape)
    _require(
        result_shape == original_shape,
        source.original_shape,
        original_shape,
        f"Broadcasting produces {result_shape}, not target shape {original_shape}.",
    )
    _require(
        source.batch_size == batch_size,
        source.batch_size,
        batch_size,
        f"Block broadcasting requires equal batch_size; got {source.batch_size} and {batch_size}.",
        error_cls=ONPValueError,
    )
    _require(
        source.order == order,
        source.order,
        order,
        f"Block broadcasting requires matching packing order; got {source.order!r} and {order!r}.",
        error_cls=ONPValueError,
    )
    _require(
        not all(size == 1 for size in source.original_shape),
        source.original_shape,
        original_shape,
        "Singleton block tensors are not broadcast operands; use a Python scalar.",
        error_cls=ONPNotImplementedError,
    )
    _require(
        _is_standard_block_layout(source),
        tuple(source.data[0].shape),
        source.block_shape,
        "Block broadcasting requires standard non-compact packing; construct "
        "vector sources with compact=False.",
        error_cls=ONPNotSupportedError,
    )

    kind = _source_kind(source)
    block_rows, block_cols = block_shape
    grid_rows, grid_cols = grid_shape
    expected_block, expected_grid = {
        "row_vector": ((block_cols,), (grid_cols,)),
        "row_matrix": ((1, block_cols), (1, grid_cols)),
        "column": ((block_rows, 1), (grid_rows, 1)),
    }[kind]
    _require(
        source.block_shape == expected_block and source.grid_shape == expected_grid,
        source.block_shape,
        block_shape,
        "Incompatible block layout: result "
        f"block_shape={block_shape} requires source "
        f"block_shape={expected_block} and grid_shape={expected_grid}; "
        f"got block_shape={source.block_shape} and grid_shape={source.grid_shape}.",
    )
    return kind


def _broadcast_into_layout(
    source: BlockFHETensor,
    original_shape: tuple[int, int],
    block_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    batch_size: int,
    order: Any,
    context,
    result_cls: type[BlockFHETensor],
) -> BlockFHETensor:
    """Expand a source into a specified block-matrix layout.

    Parameters
    ----------
    source : BlockFHETensor
        Row or column source to broadcast.
    original_shape : tuple[int, int]
        Logical shape of the result matrix.
    block_shape : tuple[int, int]
        Shape of each result block.
    grid_shape : tuple[int, int]
        Shape of the result block grid.
    batch_size : int
        Number of CKKS slots available per block.
    order : ArrayEncodingType
        Packing order required for every result block.
    context : openfhe.CryptoContext
        Crypto context used during block expansion.
    result_cls : type[BlockFHETensor]
        Concrete block-tensor class to construct.

    Returns
    -------
    BlockFHETensor
        Broadcast source represented in the requested layout.

    Raises
    ------
    ONPIncompatibleShapeError
        If source block boundaries do not match the result layout.
    ONPNotSupportedError
        If the source shape or packing layout is unsupported.
    ONPValueError
        If the batch size or packing order differs from the result layout.

    Notes
    -----
    This function expands blocks but never changes their boundaries. For result
    blocks of shape ``(br, bc)``, source blocks must have shape ``(bc,)``,
    ``(1, bc)``, or ``(br, 1)``.
    """
    kind = _validate_source_for_layout(
        source,
        original_shape,
        block_shape,
        grid_shape,
        batch_size,
        order,
    )
    cache = {}
    blocks = []
    for block_row, block_col in np.ndindex(*grid_shape):
        source_block = _source_block(
            source,
            kind,
            block_row,
            block_col,
        )
        logical_shape = _block_original_shape(
            original_shape,
            block_shape,
            block_row,
            block_col,
        )
        key = (id(source_block), logical_shape)
        if key not in cache:
            cache[key] = _broadcast_to_physical_slots(
                source_block.clone(),
                logical_shape=logical_shape,
                physical_shape=block_shape,
                order=order,
                cc=context,
            )
        blocks.append(cache[key].clone())

    return result_cls(
        data=blocks,
        grid_shape=grid_shape,
        block_shape=block_shape,
        original_shape=original_shape,
        batch_size=batch_size,
        order=order,
    )


def _block_broadcast_to(source: BlockFHETensor, target: BlockFHETensor) -> BlockFHETensor:
    """Broadcast an aligned source to an existing matrix layout.

    Parameters
    ----------
    source : BlockFHETensor
        Row or column source to expand.
    target : BlockFHETensor
        Matrix whose logical shape and block layout define the result.

    Returns
    -------
    BlockFHETensor
        Broadcast copy of ``source`` with the target layout and source type.

    Raises
    ------
    ONPTypeError
        If either argument is not a block tensor.
    ONPIncompatibleShapeError
        If the logical shapes or block layouts are incompatible.
    ONPNotSupportedError
        If ``target`` is not a matrix or ``source`` has an unsupported layout.
    ONPValueError
        If the operands do not share compatible encryption metadata.
    """
    _require(
        isinstance(source, BlockFHETensor) and isinstance(target, BlockFHETensor),
        type(source).__name__,
        type(target).__name__,
        "Block broadcasting requires two block tensors.",
        error_cls=ONPTypeError,
    )
    _require(
        target.ndim == 2,
        source.original_shape,
        target.original_shape,
        "Block broadcasting requires a matrix target.",
        error_cls=ONPNotSupportedError,
    )
    result_shape = _broadcast_shape(source.original_shape, target.original_shape)
    _require(
        result_shape == target.original_shape,
        source.original_shape,
        target.original_shape,
        f"Broadcasting produces {result_shape}, not target shape {target.original_shape}.",
    )
    context = _binary_crypto_context(source.data[0], target.data[0])
    return _broadcast_into_layout(
        source,
        target.original_shape,
        target.block_shape,
        target.grid_shape,
        target.batch_size,
        target.order,
        context,
        type(source),
    )


def _two_sided_layout(
    a: BlockFHETensor,
    b: BlockFHETensor,
    result_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], int, Any]:
    """Validate a column-plus-row broadcast and return its common layout."""
    _require(
        len(result_shape) == 2,
        a.original_shape,
        b.original_shape,
        "Two-sided block broadcasting produces a matrix; got operands "
        f"{a.original_shape} and {b.original_shape}.",
        error_cls=ONPNotSupportedError,
    )
    a_is_column = _source_kind(a) == "column"
    b_is_column = _source_kind(b) == "column"
    _require(
        a_is_column != b_is_column,
        a.original_shape,
        b.original_shape,
        "Two-sided block broadcasting needs one column (m, 1) and one row "
        f"((n,) or (1, n)); got {a.original_shape} and {b.original_shape}.",
        error_cls=ONPNotSupportedError,
    )
    column, row = (a, b) if a_is_column else (b, a)
    _require(
        column.batch_size == row.batch_size,
        column.batch_size,
        row.batch_size,
        "Two-sided block broadcasting requires equal batch_size.",
        error_cls=ONPValueError,
    )
    _require(
        column.order == row.order,
        column.order,
        row.order,
        "Two-sided block broadcasting requires matching packing order.",
        error_cls=ONPValueError,
    )

    block_shape = (column.block_shape[0], row.block_shape[-1])
    grid_shape = (column.grid_shape[0], row.grid_shape[-1])
    return block_shape, grid_shape, column.batch_size, column.order


def _two_sided_broadcast(
    a: BlockFHETensor, b: BlockFHETensor, result_shape: tuple[int, int]
) -> tuple[BlockFHETensor, BlockFHETensor]:
    """Broadcast a column and a row to a common matrix shape.

    Parameters
    ----------
    a, b : BlockFHETensor
        One column tensor and one row tensor.
    result_shape : tuple[int, int]
        Common logical matrix shape.

    Returns
    -------
    tuple[BlockFHETensor, BlockFHETensor]
        Expanded operands in their original order.

    Raises
    ------
    ONPIncompatibleShapeError
        If either source does not match the derived result layout.
    ONPNotSupportedError
        If the operands are not one column and one row.
    ONPValueError
        If their batch sizes, packing orders, or encryption metadata differ.
    """
    a_is_column = _source_kind(a) == "column"
    column, row = (a, b) if a_is_column else (b, a)
    block_shape, grid_shape, batch_size, order = _two_sided_layout(
        column,
        row,
        result_shape,
    )
    context = _binary_crypto_context(column.data[0], row.data[0])

    column_full = _broadcast_into_layout(
        column,
        result_shape,
        block_shape,
        grid_shape,
        batch_size,
        order,
        context,
        type(column),
    )
    row_full = _broadcast_into_layout(
        row,
        result_shape,
        block_shape,
        grid_shape,
        batch_size,
        order,
        context,
        type(row),
    )
    return (column_full, row_full) if a is column else (row_full, column_full)


def _align_block_operands(
    a: BlockFHETensor, b: BlockFHETensor
) -> tuple[BlockFHETensor, BlockFHETensor]:
    """Align two operands to a common block-broadcast layout.

    Parameters
    ----------
    a, b : BlockFHETensor
        Operands to align.

    Returns
    -------
    tuple[BlockFHETensor, BlockFHETensor]
        Original or expanded operands in ``(a, b)`` order.

    Raises
    ------
    ONPIncompatibleShapeError
        If logical shapes or block boundaries are incompatible.
    ONPNotSupportedError
        If an operand shape or packing layout is unsupported.
    ONPValueError
        If encryption metadata, batch sizes, or packing orders are incompatible.

    Notes
    -----
    Operands with identical layouts are returned unchanged. Otherwise, a source
    is expanded into a full matrix layout; encrypted re-tiling is not performed.
    """
    if a.same_layout(b):
        return a, b

    result_shape = _broadcast_shape(a.original_shape, b.original_shape)
    a_is_full = a.original_shape == result_shape
    b_is_full = b.original_shape == result_shape

    if a_is_full and not b_is_full:
        return a, _block_broadcast_to(b, a)
    if b_is_full and not a_is_full:
        return _block_broadcast_to(a, b), b
    if not a_is_full and not b_is_full:
        return _two_sided_broadcast(a, b, result_shape)

    # Both already carry the result shape but with different tilings.
    _require(
        False,
        a.original_shape,
        b.original_shape,
        "Block operands share a logical shape but different block layouts; "
        "broadcasting cannot re-tile. Build them with the same block_shape.",
    )


# ----------------------------------------------------------------------------
# Key generation
# ----------------------------------------------------------------------------


def _verify_secret_key(secret_key: Any, block_tensor: BlockFHETensor) -> None:
    """Validate a secret key against an encrypted block tensor.

    Parameters
    ----------
    secret_key : openfhe.PrivateKey
        Candidate secret key.
    block_tensor : BlockFHETensor
        Tensor whose crypto context and key tag are checked.

    Raises
    ------
    ONPValueError
        If the key does not match an encrypted tensor.

    Notes
    -----
    Plaintext block tensors do not require key validation.
    """
    if not isinstance(block_tensor, BlockCTArray):
        return
    data = block_tensor.data[0].data
    same_key = secret_key.GetCryptoContext() == data.GetCryptoContext()
    if same_key and hasattr(secret_key, "GetKeyTag") and hasattr(data, "GetKeyTag"):
        same_key = secret_key.GetKeyTag() == data.GetKeyTag()
    _require(
        same_key,
        "secret_key",
        "block_tensor",
        "The secret key does not match the encrypted block tensor.",
        error_cls=ONPValueError,
    )


def generate_block_broadcast_key(secret_key: Any, *operands: BlockFHETensor) -> None:
    """Generate rotation keys for block broadcasting.

    Parameters
    ----------
    secret_key : openfhe.PrivateKey
        Secret key associated with the encrypted operands.
    *operands : BlockFHETensor
        Tensors that will participate in the broadcast operation. A single
        operand must be a matrix; this generates keys for both row and column
        broadcasting.

    Returns
    -------
    None

    Raises
    ------
    ONPTypeError
        If the key or an operand has an invalid type, or if a single operand is
        not a matrix.
    ONPIncompatibleShapeError
        If operand block shapes are not broadcast-compatible.
    ONPNotSupportedError
        If an operand has compact packing or an unsupported source shape.
    ONPValueError
        If no operand is given or the key does not match an encrypted operand.

    Notes
    -----
    Multiple operands generate keys only for the required broadcast directions.
    Their block boundaries must already align; this function does not prepare
    keys for re-tiling.
    """
    _require(
        hasattr(secret_key, "GetCryptoContext"),
        type(secret_key).__name__,
        "private key",
        "secret_key must be an OpenFHE private key.",
        error_cls=ONPTypeError,
    )
    _require(
        len(operands) >= 1,
        len(operands),
        1,
        "generate_block_broadcast_key needs at least one block tensor.",
        error_cls=ONPValueError,
    )
    for operand in operands:
        _require(
            isinstance(operand, BlockFHETensor),
            type(operand).__name__,
            "BlockFHETensor",
            "generate_block_broadcast_key expects block tensors.",
            error_cls=ONPTypeError,
        )
        _require(
            _is_standard_block_layout(operand),
            tuple(operand.data[0].shape),
            operand.block_shape,
            "Block broadcast key generation requires standard (non-compact) block layout.",
            error_cls=ONPNotSupportedError,
        )
        _verify_secret_key(secret_key, operand)

    requests = set()

    if len(operands) == 1:
        matrix = operands[0]
        _require(
            matrix.ndim == 2,
            matrix.original_shape,
            "(m, n)",
            "A single-operand call must be a block matrix.",
            error_cls=ONPTypeError,
        )
        result_shape = matrix.original_shape
        physical_shape = matrix.block_shape
        grid_shape = matrix.grid_shape
        order = matrix.order

        for block_row, block_col in np.ndindex(*grid_shape):
            logical_shape = _block_original_shape(
                result_shape,
                physical_shape,
                block_row,
                block_col,
            )
            logical_rows, logical_cols = logical_shape

            # Row vectors and row matrices use the same packed-slot kernel.
            requests.add(((logical_cols,), logical_shape))
            requests.add(((logical_rows, 1), logical_shape))

    else:
        result_shape = _broadcast_shape(*(operand.original_shape for operand in operands))
        if len(result_shape) != 2:
            return

        full_operands = [operand for operand in operands if operand.original_shape == result_shape]
        if full_operands:
            target = full_operands[0]
            for other in full_operands[1:]:
                _require(
                    target.same_layout(other),
                    target.block_shape,
                    other.block_shape,
                    "Full matrix operands use different layouts; broadcasting cannot re-tile them.",
                )

            physical_shape = target.block_shape
            grid_shape = target.grid_shape
            batch_size = target.batch_size
            order = target.order

        else:
            classified = [(operand, _source_kind(operand)) for operand in operands]
            column = next(
                (operand for operand, kind in classified if kind == "column"),
                None,
            )
            row = next(
                (operand for operand, kind in classified if kind != "column"),
                None,
            )
            _require(
                column is not None and row is not None,
                tuple(operand.original_shape for operand in operands),
                result_shape,
                "Two-sided block broadcasting needs at least one column and one row "
                "when no full matrix operand defines the result layout.",
                error_cls=ONPNotSupportedError,
            )
            physical_shape, grid_shape, batch_size, order = _two_sided_layout(
                column,
                row,
                result_shape,
            )

        for operand in operands:
            if operand.original_shape == result_shape:
                # Full operands define this layout and require no expansion.
                continue

            kind = _validate_source_for_layout(
                operand,
                result_shape,
                physical_shape,
                grid_shape,
                batch_size,
                order,
            )
            for block_row, block_col in np.ndindex(*grid_shape):
                logical_shape = _block_original_shape(
                    result_shape,
                    physical_shape,
                    block_row,
                    block_col,
                )
                source_block = _source_block(
                    operand,
                    kind,
                    block_row,
                    block_col,
                )

                # A logical match alone does not prove packed-slot identity.
                if (
                    tuple(source_block.original_shape) == logical_shape
                    and tuple(source_block.shape) == physical_shape
                    and source_block.order == order
                ):
                    continue

                requests.add((tuple(source_block.original_shape), logical_shape))

    indices = set()
    for source_shape, logical_shape in requests:
        indices.update(
            _broadcast_rotation_indices(
                source_shape=source_shape,
                logical_shape=logical_shape,
                physical_shape=physical_shape,
                order=order,
            )
        )

    indices.discard(0)
    if indices:
        secret_key.GetCryptoContext().EvalRotateKeyGen(
            secret_key,
            sorted(indices),
        )
