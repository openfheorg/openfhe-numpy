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

"""Plan and evaluate NumPy-compatible cumulative sums on encrypted tensors.

The public key-generation functions install the rotation keys needed by one
``cumsum`` axis and packed layout. The private helpers share the same rotation
plans with runtime evaluation, validate block geometry, flatten matrix blocks
in logical C order for ``axis=None``, and propagate cumulative totals between
ciphertext blocks.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import isclose, prod

import openfhe
from openfhe import FIXEDMANUAL

from ..tensor.block_ctarray import BlockCTArray
from ..tensor.ctarray import CTArray
from ..tensor.tensor import FramePacking
from ..utils._helper_slots_ops import (
    _create_masking,
    _get_slot_index,
    _replicate_pattern,
    _replication_steps,
)
from ..utils.errors import ONPValueError, _require
from ..utils.matlib import next_power_of_two
from ..utils.packing import _is_col_major, _is_row_major
from .arithmetic_utils import _normalize_axis


# ------------------------------------------------------------------------------
# Shared cumsum planning and geometry helpers
# ------------------------------------------------------------------------------


def _get_cumsum_lane_parameters(tensor: CTArray, axis: int) -> tuple[int, int, int]:
    """Return the active length, lane count, and slot stride for one scan axis."""
    geometry = tensor.geometry
    _require(
        geometry is not None,
        geometry,
        None,
        "cumsum requires packed geometry.",
        error_cls=ONPValueError,
    )

    frame_rows, frame_cols = tensor.shape
    active_rows, active_cols = geometry.active
    _require(
        _is_row_major(tensor.order) or _is_col_major(tensor.order),
        tensor.order,
        None,
        f"unsupported cumsum packing order {tensor.order!r}.",
        error_cls=ONPValueError,
    )
    _require(
        geometry.padding in ("tile", "zero"),
        geometry.padding,
        ("tile", "zero"),
        f"unsupported cumsum padding mode {geometry.padding!r}.",
        error_cls=ONPValueError,
    )

    if geometry.padding == "tile":
        scan_cols = frame_cols
    else:
        scan_cols = active_cols

    if axis == 0:
        lane_size = active_rows
        num_lanes = scan_cols
        slot_stride = frame_cols if _is_row_major(tensor.order) else 1
    else:
        lane_size = scan_cols
        num_lanes = active_rows
        slot_stride = 1 if _is_row_major(tensor.order) else frame_rows

    return lane_size, num_lanes, slot_stride


def _can_flatten_ctarray_without_slot_moves(tensor: CTArray) -> bool:
    """Return whether matrix ``axis=None`` can reuse ciphertext slots directly."""
    geometry = tensor.geometry
    if tensor.ndim != 2 or geometry is None or not _is_row_major(tensor.order):
        return False

    frame_rows, frame_cols = tensor.shape
    active_rows, active_cols = geometry.active
    return (
        geometry.padding == "zero"
        and geometry.active == tuple(tensor.original_shape)
        and active_rows <= frame_rows
        and active_cols == frame_cols
    )


def _plan_cumsum_lane_rotations(lane_size: int, slot_stride: int) -> set[int]:
    """Return rotations for a power-of-two prefix scan over one lane."""
    rotations = set()
    offset = 1
    while offset < lane_size:
        rotations.add(-(offset * slot_stride))
        offset *= 2
    return rotations


def _plan_block_matrix_flatten_moves(
    block_tensor: BlockCTArray,
) -> tuple[int, int, list[dict[tuple[int, int], list[int]]]]:
    """Group the mask-and-rotate moves needed for C-order block flattening."""
    block_rows, block_cols = block_tensor.block_shape
    block_size = block_rows * block_cols
    frame_rows = next_power_of_two(block_size)
    _require(
        frame_rows <= block_tensor.batch_size,
        frame_rows,
        block_tensor.batch_size,
        "flattened cumsum frame exceeds batch_size.",
        error_cls=ONPValueError,
    )

    rows, cols = block_tensor.original_shape
    _, grid_cols = block_tensor.grid_shape
    num_chunks = (rows * cols + block_size - 1) // block_size
    chunks = [{} for _ in range(num_chunks)]

    for block_idx, block in enumerate(block_tensor.data):
        grid_row, grid_col = divmod(block_idx, grid_cols)
        active_rows, active_cols = block.geometry.active
        for cell_row in range(active_rows):
            row = grid_row * block_rows + cell_row
            for cell_col in range(active_cols):
                col = grid_col * block_cols + cell_col
                logical_idx = row * cols + col
                chunk_idx, target_slot = divmod(logical_idx, block_size)
                source_slot = _get_slot_index(
                    cell_row,
                    cell_col,
                    block.shape,
                    block.order,
                )
                rotation = source_slot - target_slot
                key = block_idx, rotation
                chunks[chunk_idx].setdefault(key, []).append(source_slot)

    return block_size, frame_rows, chunks


def _can_view_block_matrix_as_c_order_vector(block_tensor: BlockCTArray) -> bool:
    """Return whether block-list order already matches logical C order."""
    if (
        block_tensor.ndim != 2
        or not _is_row_major(block_tensor.order)
        or block_tensor.grid_shape[1] != 1
    ):
        return False

    return all(_can_flatten_ctarray_without_slot_moves(block) for block in block_tensor.data)


def _view_block_matrix_as_c_order_vector(block_tensor: BlockCTArray) -> BlockCTArray:
    """Return a metadata-only vector view that shares the matrix ciphertexts."""
    blocks = []
    for block in block_tensor.data:
        frame_size = prod(block.shape)
        logical_size = prod(block.original_shape)
        blocks.append(
            CTArray(
                data=block.data,
                original_shape=(logical_size,),
                batch_size=block.batch_size,
                new_shape=(frame_size, 1),
                order=block.order,
                geometry=FramePacking(
                    active=(logical_size, 1),
                    padding="zero",
                    repeats=block.geometry.repeats,
                ),
            )
        )

    frame_size = prod(block_tensor.data[0].shape)
    return BlockCTArray(
        data=blocks,
        grid_shape=(len(blocks),),
        block_shape=(frame_size,),
        original_shape=(prod(block_tensor.original_shape),),
        batch_size=block_tensor.batch_size,
        order=block_tensor.order,
    )


# ------------------------------------------------------------------------------
# CTArray key generation
# ------------------------------------------------------------------------------


def _plan_ctarray_cumsum_rotations(
    tensor: CTArray,
    axis: int | None = None,
) -> set[int]:
    """Return every rotation used by ``CTArray.cumsum`` for one axis."""
    _require(
        tensor.geometry is not None,
        tensor.geometry,
        None,
        "cumsum requires packed geometry; construct the tensor with array().",
        error_cls=ONPValueError,
    )
    _require(
        tensor.ndim in (0, 1, 2),
        tensor.ndim,
        (0, 1, 2),
        f"cumsum keys require scalar, vector, or matrix geometry; got {tensor.ndim}D.",
        error_cls=ONPValueError,
    )

    axis = _normalize_axis(
        "cumsum",
        axis,
        1 if tensor.ndim <= 1 else tensor.ndim,
    )

    if tensor.ndim <= 1 and axis is None:
        axis = 0

    if axis is not None:
        lane_size, _, slot_stride = _get_cumsum_lane_parameters(
            tensor,
            axis,
        )
        return _plan_cumsum_lane_rotations(lane_size, slot_stride)

    if _can_flatten_ctarray_without_slot_moves(tensor):
        return _plan_cumsum_lane_rotations(prod(tensor.original_shape), 1)

    rotations = set()
    active_rows, active_cols = tensor.geometry.active
    for row in range(active_rows):
        for col in range(active_cols):
            slot = _get_slot_index(
                row,
                col,
                tensor.shape,
                tensor.order,
            )
            if slot:
                rotations.add(slot)
    size = active_rows * active_cols
    rotations.update(-offset for offset in range(1, size))
    return rotations


def gen_cumsum_key(
    secret_key: openfhe.PrivateKey,
    tensor: CTArray,
    axis: int | None = None,
) -> None:
    """Install rotation keys required by one ``CTArray.cumsum`` call.

    Parameters
    ----------
    secret_key : openfhe.PrivateKey
        Private key used to install evaluation keys in its crypto context.
    tensor : CTArray
        Encrypted tensor whose packed shape, order, and geometry determine the
        required rotations.
    axis : int or None, optional
        Axis that will be passed to ``tensor.cumsum``. For vectors, ``None`` is
        equivalent to ``0``. For matrices, ``None`` requests logical C-order
        flattening.

    Returns
    -------
    None
        Keys are stored in the crypto context associated with ``secret_key``.

    Raises
    ------
    TypeError
        If ``axis`` is neither ``None`` nor an integer.
    ONPDimensionError
        If ``axis`` is outside the tensor rank.
    ONPValueError
        If the arguments have the wrong type, key and tensor do not share a
        crypto context and key tag, or the packed layout is unsupported.

    Notes
    -----
    Rotation keys depend on both ``axis`` and the tensor's packed geometry. Call
    this after the final layout is known and before evaluating the corresponding
    cumsum. Scalar inputs and length-one scan lanes may require no rotations.
    """
    _require(
        isinstance(secret_key, openfhe.PrivateKey),
        type(secret_key),
        openfhe.PrivateKey,
        "gen_cumsum_key expects a PrivateKey.",
        error_cls=ONPValueError,
    )
    _require(
        isinstance(tensor, CTArray),
        type(tensor),
        CTArray,
        "gen_cumsum_key expects a CTArray.",
        error_cls=ONPValueError,
    )
    same_key_domain = (
        secret_key.GetCryptoContext() == tensor.data.GetCryptoContext()
        and secret_key.GetKeyTag() == tensor.data.GetKeyTag()
    )
    _require(
        same_key_domain,
        secret_key,
        tensor,
        "secret_key and tensor must share a crypto context and key tag.",
        error_cls=ONPValueError,
    )
    rotations = _plan_ctarray_cumsum_rotations(tensor, axis=axis)
    if rotations:
        secret_key.GetCryptoContext().EvalRotateKeyGen(
            secret_key,
            sorted(rotations),
        )


# ------------------------------------------------------------------------------
# BlockCTArray preparation
# ------------------------------------------------------------------------------


def _validate_and_plan_block_cumsum(
    block_tensor: BlockCTArray,
    axis: int | None,
) -> tuple[int | None, tuple[tuple[int, ...], ...]]:
    """Validate block geometry and return the normalized axis and block chains."""
    axis = _normalize_axis("cumsum", axis, block_tensor.ndim)
    if block_tensor.ndim == 1 and axis is None:
        axis = 0

    if block_tensor.ndim == 1:
        block_chains = (tuple(range(block_tensor.num_blocks)),)
    else:
        grid_rows, grid_cols = block_tensor.grid_shape
        if axis is None:
            block_chains = ()
        elif axis == 0:
            block_chains = tuple(
                tuple(grid_row * grid_cols + grid_col for grid_row in range(grid_rows))
                for grid_col in range(grid_cols)
            )
        else:
            block_chains = tuple(
                tuple(grid_row * grid_cols + grid_col for grid_col in range(grid_cols))
                for grid_row in range(grid_rows)
            )

    ref_block = block_tensor.data[0]
    ref_geometry = ref_block.geometry
    _require(
        ref_geometry is not None,
        ref_geometry,
        None,
        "block child 0 has no packed geometry; construct it with "
        "block_array() before calling cumsum.",
        error_cls=ONPValueError,
    )

    for block_idx, block in enumerate(block_tensor.data):
        geometry = block.geometry
        _require(
            geometry is not None,
            geometry,
            None,
            f"block child {block_idx} has no packed geometry; "
            "construct it with block_array() before calling cumsum.",
            error_cls=ONPValueError,
        )

        frame_rows, frame_cols = block.shape
        active_rows, active_cols = geometry.active
        valid = (
            1 <= active_rows <= frame_rows
            and 1 <= active_cols <= frame_cols
            and geometry.repeats >= 1
            and geometry.repeats * frame_rows * frame_cols <= block.batch_size
        )
        _require(
            valid,
            geometry,
            block.shape,
            f"block child {block_idx} has invalid packed geometry {geometry}.",
            error_cls=ONPValueError,
        )

        compatible = (
            block.shape == ref_block.shape
            and block.order == block_tensor.order
            and block.batch_size == block_tensor.batch_size
            and geometry.padding == ref_geometry.padding
            and geometry.repeats == ref_geometry.repeats
            and block.crypto_context == ref_block.crypto_context
            and block.data.GetKeyTag() == ref_block.data.GetKeyTag()
        )
        _require(
            compatible,
            block,
            ref_block,
            f"incompatible cumsum child {block_idx}; all blocks must share "
            "one frame shape, order, slot domain, padding mode, repeat count, "
            "crypto context, and key tag.",
            error_cls=ONPValueError,
        )

    if axis is None:
        _require(
            ref_block.shape == tuple(block_tensor.block_shape),
            ref_block.shape,
            block_tensor.block_shape,
            "matrix cumsum(axis=None) requires child frame shape to match block_shape.",
            error_cls=ONPValueError,
        )
        return None, ()

    lane_capacity = block_tensor.block_shape[axis]

    for block_chain in block_chains:
        _, expected_lanes, _ = _get_cumsum_lane_parameters(
            block_tensor.data[block_chain[0]],
            axis,
        )
        last_idx = block_chain[-1]
        for block_idx in block_chain:
            lane_size, num_lanes, _ = _get_cumsum_lane_parameters(
                block_tensor.data[block_idx],
                axis,
            )
            _require(
                num_lanes == expected_lanes,
                num_lanes,
                expected_lanes,
                f"incompatible non-axis geometry in cumsum block chain {block_chain}.",
                error_cls=ONPValueError,
            )
            _require(
                block_idx == last_idx or lane_size == lane_capacity,
                lane_size,
                lane_capacity,
                f"cumsum block chain {block_chain} shrinks before its final child.",
                error_cls=ONPValueError,
            )
            _require(
                1 <= lane_size <= lane_capacity,
                lane_size,
                lane_capacity,
                f"invalid active cumsum length {lane_size} in child {block_idx}.",
                error_cls=ONPValueError,
            )

    return axis, block_chains


# ------------------------------------------------------------------------------
# BlockCTArray key generation
# ------------------------------------------------------------------------------


def _plan_block_cumsum_rotations(
    block_tensor: BlockCTArray,
    axis: int | None = None,
) -> set[int]:
    """Return every rotation used by block cumsum for one axis and layout."""
    axis, block_chains = _validate_and_plan_block_cumsum(block_tensor, axis)
    rotations = set()
    if axis is None:
        if _can_view_block_matrix_as_c_order_vector(block_tensor):
            flattened = _view_block_matrix_as_c_order_vector(block_tensor)
            return _plan_block_cumsum_rotations(flattened, axis=0)

        block_size, _, chunks = _plan_block_matrix_flatten_moves(block_tensor)
        rotations.update(rotation for chunk in chunks for _, rotation in chunk if rotation)
        total_cells = prod(block_tensor.original_shape)
        for chunk_idx in range(len(chunks)):
            start = chunk_idx * block_size
            num_cells = min(block_size, total_cells - start)
            rotations.update(_plan_cumsum_lane_rotations(num_cells, 1))
            if chunk_idx < len(chunks) - 1 and num_cells > 1:
                rotations.add(num_cells - 1)
            if chunk_idx > 0:
                rotations.update(rotation for rotation, _ in _replication_steps(num_cells, 1))
    else:
        for block_chain in block_chains:
            for block_idx in block_chain:
                block = block_tensor.data[block_idx]
                lane_size, _, slot_stride = _get_cumsum_lane_parameters(
                    block,
                    axis,
                )
                rotations.update(
                    _plan_cumsum_lane_rotations(
                        lane_size,
                        slot_stride,
                    )
                )
                if block_idx != block_chain[-1]:
                    total_rotation = (lane_size - 1) * slot_stride
                    if total_rotation:
                        rotations.add(total_rotation)
                if block_idx != block_chain[0]:
                    rotations.update(
                        rotation
                        for rotation, _ in _replication_steps(
                            lane_size,
                            slot_stride,
                        )
                    )

    rotations.discard(0)
    return rotations


def gen_block_cumsum_keys(
    secret_key: openfhe.PrivateKey,
    block_tensor: BlockCTArray,
    axis: int | None = None,
) -> None:
    """Install rotation keys required by one ``BlockCTArray.cumsum`` call.

    Parameters
    ----------
    secret_key : openfhe.PrivateKey
        Private key used to install evaluation keys in its crypto context.
    block_tensor : BlockCTArray
        Encrypted block tensor whose child geometry and grid layout determine
        the required rotations.
    axis : int or None, optional
        Axis that will be passed to ``block_tensor.cumsum``. Vector ``None`` is
        equivalent to ``0``. Matrix ``None`` flattens in logical C order.

    Returns
    -------
    None
        Keys are stored in the crypto context associated with ``secret_key``.

    Raises
    ------
    TypeError
        If ``axis`` is neither ``None`` nor an integer.
    ONPDimensionError
        If ``axis`` is outside the tensor rank.
    ONPValueError
        If arguments have the wrong type, the tensor is empty or unsupported,
        child geometry is incompatible, or key and ciphertexts do not share a
        crypto context and key tag.

    Notes
    -----
    The installed set covers local prefix scans, movement of each block's
    terminal values, replication of accumulated carry into following blocks,
    and matrix-flattening moves when ``axis=None``. Generate keys after the
    block layout is final and for the same axis used at evaluation time.
    """
    _require(
        isinstance(secret_key, openfhe.PrivateKey),
        type(secret_key),
        openfhe.PrivateKey,
        "gen_block_cumsum_keys expects a PrivateKey.",
        error_cls=ONPValueError,
    )
    _require(
        isinstance(block_tensor, BlockCTArray),
        type(block_tensor),
        BlockCTArray,
        "gen_block_cumsum_keys expects a BlockCTArray.",
        error_cls=ONPValueError,
    )
    _require(
        block_tensor.ndim in (1, 2) and bool(block_tensor.data),
        block_tensor.ndim,
        (1, 2),
        "gen_block_cumsum_keys expects a nonempty block vector or matrix.",
        error_cls=ONPValueError,
    )

    rotations = _plan_block_cumsum_rotations(block_tensor, axis)
    ref_block = block_tensor.data[0]
    same_key_domain = (
        secret_key.GetCryptoContext() == ref_block.data.GetCryptoContext()
        and secret_key.GetKeyTag() == ref_block.data.GetKeyTag()
    )
    _require(
        same_key_domain,
        secret_key,
        ref_block,
        "secret_key and every child must share a context and key tag.",
        error_cls=ONPValueError,
    )

    if rotations:
        secret_key.GetCryptoContext().EvalRotateKeyGen(
            secret_key,
            sorted(rotations),
        )


# ------------------------------------------------------------------------------
# CKKS level and scale alignment
# ------------------------------------------------------------------------------


def _advance_ciphertext_to_level(
    cc: openfhe.CryptoContext,
    ct: openfhe.Ciphertext,
    target_level: int,
) -> openfhe.Ciphertext:
    """Advance a ciphertext to ``target_level`` without changing its value."""
    result = ct
    fixed_manual = cc.GetScalingTechnique() == FIXEDMANUAL
    while result.GetLevel() < target_level:
        result = cc.EvalMult(result, 1.0)
        if fixed_manual:
            cc.ModReduceInPlace(result)
    return result


def _prepare_ciphertexts_for_addition(
    cc: openfhe.CryptoContext,
    lhs: openfhe.Ciphertext,
    rhs: openfhe.Ciphertext,
) -> tuple[openfhe.Ciphertext, openfhe.Ciphertext]:
    """Prepare ciphertext levels and scales for addition under ``FIXEDMANUAL``."""
    if cc.GetScalingTechnique() != FIXEDMANUAL:
        return lhs, rhs

    if (
        lhs.GetLevel() == rhs.GetLevel()
        and lhs.GetNoiseScaleDeg() == 1
        and rhs.GetNoiseScaleDeg() == 1
        and isclose(
            lhs.GetScalingFactor(),
            rhs.GetScalingFactor(),
            rel_tol=1e-10,
            abs_tol=0.0,
        )
    ):
        return lhs, rhs

    def normalize_scale(ct):
        result = ct.Clone()
        while result.GetNoiseScaleDeg() > 1:
            cc.ModReduceInPlace(result)
        return result

    lhs = normalize_scale(lhs)
    rhs = normalize_scale(rhs)
    target_level = max(lhs.GetLevel(), rhs.GetLevel())
    lhs = _advance_ciphertext_to_level(cc, lhs, target_level)
    rhs = _advance_ciphertext_to_level(cc, rhs, target_level)

    _require(
        isclose(
            lhs.GetScalingFactor(),
            rhs.GetScalingFactor(),
            rel_tol=1e-10,
            abs_tol=0.0,
        ),
        lhs.GetScalingFactor(),
        rhs.GetScalingFactor(),
        "ciphertexts have incompatible scales for addition.",
        error_cls=ONPValueError,
    )
    return lhs, rhs


# ------------------------------------------------------------------------------
# Block flattening and cross-block carry evaluation
# ------------------------------------------------------------------------------


def _flatten_block_matrix_to_c_order_vector(
    block_tensor: BlockCTArray,
) -> BlockCTArray:
    """Homomorphically repack a block matrix into a C-order block vector."""
    block_size, frame_rows, chunks = _plan_block_matrix_flatten_moves(block_tensor)
    ref_block = block_tensor.data[0]
    blocks = []
    total_cells = prod(block_tensor.original_shape)
    cc = ref_block.crypto_context
    fixed_manual = cc.GetScalingTechnique() == FIXEDMANUAL

    for chunk_idx, chunk in enumerate(chunks):
        data = None
        for (block_idx, rotation), source_slots in chunk.items():
            source = block_tensor.data[block_idx]
            mask = cc.MakeCKKSPackedPlaintext(
                _create_masking(source_slots, block_tensor.batch_size),
                1,
                source.data.GetLevel(),
                None,
                block_tensor.batch_size,
            )
            term = cc.EvalMult(source.data, mask)
            if fixed_manual:
                cc.ModReduceInPlace(term)
            if rotation:
                term = cc.EvalRotate(term, rotation)

            if data is None:
                data = term
            else:
                data, term = _prepare_ciphertexts_for_addition(
                    cc,
                    data,
                    term,
                )
                data = cc.EvalAdd(data, term)

        if data is None:
            raise RuntimeError("internal cumsum error: flattened block is empty.")

        start = chunk_idx * block_size
        num_cells = min(block_size, total_cells - start)
        blocks.append(
            CTArray(
                data=data,
                original_shape=(num_cells,),
                batch_size=block_tensor.batch_size,
                new_shape=(frame_rows, 1),
                order=block_tensor.order,
                geometry=FramePacking(
                    active=(num_cells, 1),
                    padding="zero",
                    repeats=1,
                ),
            )
        )

    return BlockCTArray(
        data=blocks,
        grid_shape=(len(blocks),),
        block_shape=(block_size,),
        original_shape=(prod(block_tensor.original_shape),),
        batch_size=block_tensor.batch_size,
        order=block_tensor.order,
    )


def _iter_block_chain_cumsum_with_carry(
    block_tensor: BlockCTArray,
    block_chain: tuple[int, ...],
    axis: int,
) -> Iterator[tuple[int, CTArray]]:
    """Yield local cumsums while propagating carry through one block chain.

    Notes
    -----
    The terminal value of each lane is moved to its origin and accumulated.
    Before the next block is yielded, that carry is replicated across its active
    lane and added to its local cumsum.
    """
    ref_block = block_tensor.data[block_chain[0]]
    cc = ref_block.crypto_context
    frame_rows, frame_cols = ref_block.shape
    frame_size = frame_rows * frame_cols
    repeats = ref_block.geometry.repeats
    lane_size, num_lanes, slot_stride = _get_cumsum_lane_parameters(
        ref_block,
        axis,
    )

    last_idx = block_chain[-1]
    needs_carry = len(block_chain) > 1
    last_lane_size = lane_size
    if needs_carry:
        last_lane_size, _, _ = _get_cumsum_lane_parameters(
            block_tensor.data[last_idx],
            axis,
        )

    total_slots = []
    if needs_carry:
        last_pos = lane_size - 1
        for frame_idx in range(repeats):
            frame_offset = frame_idx * frame_size
            for lane_idx in range(num_lanes):
                if axis == 0:
                    cell_row, cell_col = last_pos, lane_idx
                else:
                    cell_row, cell_col = lane_idx, last_pos
                total_slots.append(
                    frame_offset
                    + _get_slot_index(
                        cell_row,
                        cell_col,
                        ref_block.shape,
                        ref_block.order,
                    )
                )

    mask_values = _create_masking(total_slots, ref_block.batch_size) if total_slots else None
    total_rotation = (lane_size - 1) * slot_stride
    masks = {}
    carry_total = None

    for block_idx in block_chain:
        block = block_tensor.data[block_idx]
        local_cumsum = block.cumsum(axis=axis)
        is_last = block_idx == last_idx

        result = local_cumsum
        if carry_total is not None:
            current_lane_size = last_lane_size if is_last else lane_size
            carry = _replicate_pattern(
                carry_total,
                copies=current_lane_size,
                stride=slot_stride,
            )
            local_data, carry = _prepare_ciphertexts_for_addition(cc, local_cumsum.data, carry)
            result = local_cumsum.clone(data=cc.EvalAdd(local_data, carry))

        if not is_last:
            level = local_cumsum.data.GetLevel()
            if level not in masks:
                masks[level] = cc.MakeCKKSPackedPlaintext(
                    mask_values,
                    1,
                    level,
                    None,
                    ref_block.batch_size,
                )
            total = cc.EvalMult(
                local_cumsum.data,
                masks[level],
            )
            if cc.GetScalingTechnique() == FIXEDMANUAL:
                cc.ModReduceInPlace(total)

            if total_rotation:
                total = cc.EvalRotate(total, total_rotation)
            if carry_total is None:
                carry_total = total
            else:
                carry_total, total = _prepare_ciphertexts_for_addition(cc, carry_total, total)
                carry_total = cc.EvalAdd(carry_total, total)

        yield block_idx, result


# ------------------------------------------------------------------------------
# BlockCTArray evaluation entry point
# ------------------------------------------------------------------------------


def _eval_block_cumsum(
    block_tensor: BlockCTArray,
    axis: int | None = None,
) -> BlockCTArray:
    """Evaluate one logical cumsum across an encrypted block tensor.

    Parameters
    ----------
    block_tensor : BlockCTArray
        Encrypted block vector or matrix with compatible child geometry.
    axis : int or None, optional
        NumPy-style axis. Vector ``None`` is equivalent to ``0``. Matrix
        ``None`` first flattens values in logical C order and then scans the
        resulting vector.

    Returns
    -------
    BlockCTArray
        Block tensor containing the complete logical cumsum. Matrix
        ``axis=None`` returns a one-dimensional ``BlockCTArray``; axis-specific
        matrix scans retain the input block layout. All result children are
        advanced to one common ciphertext level.

    Raises
    ------
    TypeError
        If ``axis`` is neither ``None`` nor an integer.
    ONPDimensionError
        If ``axis`` is outside the tensor rank.
    ONPValueError
        If child geometry, packing layout, or ciphertext scales are
        incompatible with the selected cumsum path.

    Notes
    -----
    A row-major matrix with one block column and contiguous child frames uses a
    metadata-only flattening fast path. Other matrix layouts use homomorphic
    mask-and-rotate flattening. Axis-specific scans keep the original blocks and
    propagate exact carry between consecutive children along each block chain.
    """
    axis, block_chains = _validate_and_plan_block_cumsum(block_tensor, axis)

    if axis is None:
        if _can_view_block_matrix_as_c_order_vector(block_tensor):
            block_tensor = _view_block_matrix_as_c_order_vector(block_tensor)
        else:
            block_tensor = _flatten_block_matrix_to_c_order_vector(block_tensor)
        axis, block_chains = _validate_and_plan_block_cumsum(block_tensor, axis=0)

    results = [None] * block_tensor.num_blocks

    for block_chain in block_chains:
        for block_idx, result in _iter_block_chain_cumsum_with_carry(
            block_tensor,
            block_chain,
            axis,
        ):
            results[block_idx] = result

    if any(block is None for block in results):
        raise RuntimeError("internal cumsum error: not every result block was produced.")

    target_level = max(block.data.GetLevel() for block in results)
    for block_idx, block in enumerate(results):
        data = _advance_ciphertext_to_level(
            block.crypto_context,
            block.data,
            target_level,
        )
        if data is not block.data:
            results[block_idx] = block.clone(data=data)

    return block_tensor.clone(data=results)
