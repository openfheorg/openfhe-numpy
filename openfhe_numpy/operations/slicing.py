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

"""NumPy-style basic slicing for direct and block ciphertext arrays."""

from __future__ import annotations

from itertools import product
from math import ceil, isclose, prod
from operator import index as operator_index

import numpy as np

from ..tensor.block_ctarray import BlockCTArray
from ..tensor.ctarray import CTArray
from ..tensor.tensor import FramePacking
from ..utils._helper_slots_ops import (
    _get_elements_at_slots,
    _get_packed_slot_index,
)
from ..utils.errors import ONPValueError
from ..utils.matlib import next_power_of_two
from .arithmetic_utils import _binary_crypto_context


# ==================================================================================
# Public entry points
# ==================================================================================


def ctarray_getitem(tensor, key):
    """Return the NumPy-style basic-indexing result for one ``CTArray``."""
    if not isinstance(tensor, CTArray):
        raise TypeError(f"unsupported ciphertext tensor type {type(tensor).__name__}")
    if tensor.ndim == 0:
        raise TypeError("'int' object is not subscriptable")

    axis_indices, axis_is_integer, result_shape = _normalize_basic_index(
        key,
        tensor.original_shape,
    )
    if _is_full_selection(axis_indices, axis_is_integer, tensor.original_shape):
        return tensor.clone()

    if tensor.ndim == 1 and tensor.shape[1] != 1:
        raise NotImplementedError("non-full slicing of expanded vector frames is not supported")

    physical_shape, repeats = _get_ctarray_index_layout(
        tensor, axis_indices, axis_is_integer, result_shape
    )
    slot_move_groups = _plan_ctarray_index(
        tensor, axis_indices, axis_is_integer, result_shape, physical_shape, repeats
    )
    result_data = _evaluate_slot_move_groups(tensor, {0: tensor}, slot_move_groups)
    return _build_ctarray_index_result(tensor, result_data, result_shape, physical_shape, repeats)


def block_ctarray_getitem(tensor, key):
    """Return the NumPy-style basic-indexing result for one ``BlockCTArray``."""
    if not isinstance(tensor, BlockCTArray):
        raise TypeError(f"unsupported block tensor type {type(tensor).__name__}")

    axis_indices, axis_is_integer, result_shape = _normalize_basic_index(key, tensor.original_shape)
    if _is_full_selection(axis_indices, axis_is_integer, tensor.original_shape):
        return tensor.clone()

    # A scalar reads only one child, so it does not require uniform child layouts.
    if all(axis_is_integer):
        source_coordinate = tuple(indices[0] for indices in axis_indices)
        source_grid_index, source_block_coordinate = tensor._locate_index(source_coordinate)
        return tensor.get_block(source_grid_index)[source_block_coordinate]

    reference_block = _validate_block_indexing_sources(tensor)
    result_block_shape, result_grid_shape, physical_shape, repeats = (
        _get_block_ctarray_index_layout(tensor, reference_block, result_shape, axis_is_integer)
    )

    result_blocks = []
    for result_grid_index in product(*map(range, result_grid_shape)):
        output_shape, source_blocks, slot_move_groups = _plan_block_ctarray_index(
            tensor,
            result_grid_index,
            axis_indices,
            axis_is_integer,
            result_shape,
            result_block_shape,
            physical_shape,
            repeats,
        )
        result_data = _evaluate_slot_move_groups(reference_block, source_blocks, slot_move_groups)
        result_blocks.append(
            _build_ctarray_index_result(
                reference_block, result_data, output_shape, physical_shape, repeats
            )
        )

    return BlockCTArray(
        data=result_blocks,
        grid_shape=result_grid_shape,
        block_shape=result_block_shape,
        original_shape=result_shape,
        batch_size=tensor.batch_size,
        order=tensor.order,
    )


def generate_slicing_key(secret_key, original_shape, *, physical_shape=None):
    """Generate rotation keys covering basic slicing of one packed layout.

    Both row-major and column-major packing are covered because the public API
    historically accepts only the logical ``original_shape``. Pass the actual
    ``physical_shape`` when slicing a matrix whose packed frame is nonstandard.
    """
    indices = _get_slicing_rotation_indices(original_shape, physical_shape)
    if indices:
        secret_key.GetCryptoContext().EvalRotateKeyGen(secret_key, sorted(indices))


# ==================================================================================
# Basic-index normalization
# ==================================================================================


def _normalize_basic_index(
    key,
    shape,
) -> tuple[tuple[range, ...], tuple[bool, ...], tuple[int, ...]]:
    """Normalize the supported subset of NumPy basic indexing."""
    shape = tuple(shape)
    ndim = len(shape)
    if ndim not in (1, 2):
        raise IndexError(f"basic indexing supports only 1-D/2-D tensors; got {shape}")

    items = key if isinstance(key, tuple) else (key,)
    ellipsis_positions = [i for i, item in enumerate(items) if item is Ellipsis]
    if len(ellipsis_positions) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_count = len(ellipsis_positions)
    position = ellipsis_positions[0] if ellipsis_positions else len(items)
    fill = ndim - (len(items) - ellipsis_count)
    if fill < 0:
        raise IndexError("too many indices for array")
    items = items[:position] + (slice(None),) * fill + items[position + ellipsis_count :]

    axis_indices = []
    axis_is_integer = []
    for axis, (item, size) in enumerate(zip(items, shape)):
        if item is None:
            raise NotImplementedError("None/newaxis indexing is not supported")
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("boolean indexing is not supported")

        if isinstance(item, slice):
            axis_indices.append(range(*item.indices(size)))
            axis_is_integer.append(False)
            continue

        try:
            index = operator_index(item)
        except TypeError as exc:
            raise TypeError(f"invalid index type: {type(item).__name__}") from exc

        normalized = index + size if index < 0 else index
        if not 0 <= normalized < size:
            raise IndexError(f"index {index} is out of bounds for axis {axis} with size {size}")
        axis_indices.append(range(normalized, normalized + 1))
        axis_is_integer.append(True)

    axis_indices = tuple(axis_indices)
    axis_is_integer = tuple(axis_is_integer)
    result_shape = tuple(
        len(indices) for indices, is_integer in zip(axis_indices, axis_is_integer) if not is_integer
    )
    if any(axis_size == 0 for axis_size in result_shape):
        raise IndexError("empty slicing results are not supported")
    return axis_indices, axis_is_integer, result_shape


def _is_full_selection(axis_indices, axis_is_integer, shape):
    """Return whether the index selects the entire array without changing rank."""
    return not any(axis_is_integer) and all(
        indices == range(axis_size) for indices, axis_size in zip(axis_indices, shape)
    )


def _get_source_coordinate(axis_indices, axis_is_integer, result_coordinate):
    """Map one result coordinate back to its source-array coordinate."""
    source_coordinate = []
    result_axis = 0
    for indices, is_integer in zip(axis_indices, axis_is_integer):
        if is_integer:
            source_coordinate.append(indices[0])
        else:
            source_coordinate.append(indices[result_coordinate[result_axis]])
            result_axis += 1
    return tuple(source_coordinate)


# ==================================================================================
# Slot-move planning
# ==================================================================================


def _plan_ctarray_index(
    tensor,
    axis_indices,
    axis_is_integer,
    result_shape,
    physical_shape,
    repeats,
):
    """Plan source-to-destination slot moves for direct-array indexing."""
    source_frame_size = prod(tensor.shape)
    slot_move_groups = {}

    for result_coordinate in product(*map(range, result_shape)):
        source_coordinate = _get_source_coordinate(axis_indices, axis_is_integer, result_coordinate)
        source_slot = _get_packed_slot_index(source_coordinate, tensor.shape, tensor.order)
        destination_slot = _get_packed_slot_index(result_coordinate, physical_shape, tensor.order)
        _record_slot_move(
            slot_move_groups, 0, source_slot, destination_slot, source_frame_size, repeats
        )

    return slot_move_groups


def _plan_block_ctarray_index(
    tensor,
    result_grid_index,
    axis_indices,
    axis_is_integer,
    result_shape,
    result_block_shape,
    physical_shape,
    repeats,
):
    """Plan one output block assembled from one or more source blocks."""
    reference_block = tensor.data[0]
    source_frame_size = prod(reference_block.shape)
    output_shape = tuple(
        max(0, min(block_size, axis_size - grid_index * block_size))
        for axis_size, block_size, grid_index in zip(
            result_shape, result_block_shape, result_grid_index
        )
    )

    source_blocks = {}
    slot_move_groups = {}
    for output_coordinate in product(*map(range, output_shape)):
        result_coordinate = tuple(
            grid_index * block_size + offset
            for grid_index, block_size, offset in zip(
                result_grid_index, result_block_shape, output_coordinate
            )
        )
        source_coordinate = _get_source_coordinate(axis_indices, axis_is_integer, result_coordinate)
        source_grid_index, source_block_coordinate = tensor._locate_index(source_coordinate)
        source_block = tensor.get_block(source_grid_index)
        source_blocks[source_grid_index] = source_block

        source_slot = _get_packed_slot_index(
            source_block_coordinate, source_block.shape, tensor.order
        )
        destination_slot = _get_packed_slot_index(output_coordinate, physical_shape, tensor.order)
        _record_slot_move(
            slot_move_groups,
            source_grid_index,
            source_slot,
            destination_slot,
            source_frame_size,
            repeats,
        )

    return output_shape, source_blocks, slot_move_groups


def _record_slot_move(
    slot_move_groups,
    source_key,
    source_slot,
    destination_slot,
    source_frame_size,
    repeats,
):
    """Group slots sharing one source ciphertext and rotation."""
    rotation = source_slot - destination_slot
    source_slots = slot_move_groups.setdefault((source_key, rotation), [])
    source_slots.extend(source_slot + frame * source_frame_size for frame in range(repeats))


# ==================================================================================
# Result layouts and evaluation
# ==================================================================================


def _get_ctarray_index_layout(tensor, axis_indices, axis_is_integer, result_shape):
    """Return the physical frame and repeat count for a direct-array result."""
    preserves_frame = not any(axis_is_integer) and all(
        indices.step == 1 for indices in axis_indices
    )
    if preserves_frame:
        repeats = tensor.geometry.repeats if tensor.geometry is not None else 1
        return tuple(tensor.shape), repeats
    if not result_shape:
        return (1, 1), 1

    physical_shape = tuple(next_power_of_two(axis_size) for axis_size in result_shape)
    if len(physical_shape) == 1:
        physical_shape = (physical_shape[0], 1)
    return physical_shape, 1


def _get_block_ctarray_index_layout(tensor, reference_block, result_shape, axis_is_integer):
    """Return block-grid and child-frame layout for a block-array result."""
    result_block_shape = tuple(
        block_size
        for block_size, is_integer in zip(tensor.block_shape, axis_is_integer)
        if not is_integer
    )
    result_grid_shape = tuple(
        max(1, ceil(axis_size / block_size))
        for axis_size, block_size in zip(result_shape, result_block_shape)
    )

    same_rank = len(result_shape) == tensor.ndim
    physical_shape = tuple(reference_block.shape) if same_rank else (result_block_shape[0], 1)
    repeats = (
        reference_block.geometry.repeats
        if same_rank and reference_block.geometry is not None
        else 1
    )
    return result_block_shape, result_grid_shape, physical_shape, repeats


def _evaluate_slot_move_groups(reference, source_arrays, slot_move_groups):
    """Mask and rotate each planned group, then add the pieces together."""
    if not slot_move_groups:
        raise RuntimeError("slicing produced an empty slot-move plan")

    pieces = []
    for (source_key, rotation), source_slots in slot_move_groups.items():
        source_data = source_arrays[source_key].data
        pieces.append(
            _get_elements_at_slots(source_data, source_slots, reference.batch_size, rotation)
        )

    if len(pieces) == 1:
        return pieces[0]
    return reference.crypto_context.EvalAddMany(pieces)


def _build_ctarray_index_result(reference, result_data, result_shape, physical_shape, repeats):
    """Wrap evaluated slot moves in a ``CTArray`` with result metadata."""
    if not result_shape:
        active_shape = (1, 1)
    elif len(result_shape) == 1:
        active_shape = (result_shape[0], 1)
    else:
        active_shape = result_shape

    result = CTArray(
        data=result_data,
        original_shape=result_shape,
        batch_size=reference.batch_size,
        new_shape=physical_shape,
        order=reference.order,
        geometry=FramePacking(
            active=active_shape,
            padding="zero",
            repeats=repeats,
        ),
    )
    if len(result_shape) == 2 and physical_shape == tuple(reference.shape):
        for name in ("rowkey", "colkey"):
            if name in reference.extra:
                result.extra[name] = reference.extra[name]
    return result


# ==================================================================================
# Block-source validation
# ==================================================================================


def _validate_block_indexing_sources(tensor):
    """Validate the child compatibility required to assemble an indexed result."""
    reference_block = tensor.data[0]
    if not isinstance(reference_block, CTArray):
        raise TypeError("BlockCTArray children must be CTArray objects")
    reference_data = reference_block.data

    for child in tensor.data[1:]:
        if not isinstance(child, CTArray):
            raise TypeError("BlockCTArray children must be CTArray objects")
        if not reference_block.same_layout(child, mode="physical"):
            raise ValueError("BlockCTArray children must share one physical layout")
        _binary_crypto_context(reference_block, child)
        same_state = (
            child.data.GetLevel() == reference_data.GetLevel()
            and child.data.GetNoiseScaleDeg() == reference_data.GetNoiseScaleDeg()
            and isclose(
                child.data.GetScalingFactor(),
                reference_data.GetScalingFactor(),
                rel_tol=1e-10,
                abs_tol=0.0,
            )
        )
        if not same_state:
            raise ONPValueError("BlockCTArray children must share level and scale.")

    if tensor.ndim == 1:
        if len(reference_block.shape) != 2 or reference_block.shape[1] != 1:
            raise NotImplementedError(
                "non-full slicing of expanded block-vector frames is not supported"
            )
    elif not tensor.is_standard_layout():
        raise ValueError("matrix child physical shape must equal block_shape")

    if reference_block.batch_size != tensor.batch_size or reference_block.order != tensor.order:
        raise ValueError("BlockCTArray child layout must match its parent")

    repeats = reference_block.geometry.repeats if reference_block.geometry is not None else 1
    if repeats < 1 or repeats * prod(reference_block.shape) > tensor.batch_size:
        raise ValueError("child repeat count exceeds the ciphertext batch")

    return reference_block


# ==================================================================================
# Rotation-key planning
# ==================================================================================


def _get_slicing_rotation_indices(original_shape, physical_shape=None):
    """Return rotations covering basic slicing of one packed frame."""
    try:
        shape = tuple(operator_index(axis_size) for axis_size in original_shape)
    except TypeError as exc:
        raise TypeError("original_shape must contain integer dimensions") from exc

    if len(shape) not in (1, 2):
        raise ValueError(f"slicing supports only 1-D/2-D shapes; got {shape}")
    if any(axis_size <= 0 for axis_size in shape):
        raise ValueError(f"original_shape must be positive; got {shape}")

    if physical_shape is None:
        if len(shape) == 1:
            physical_shape = (next_power_of_two(shape[0]), 1)
        else:
            physical_shape = tuple(next_power_of_two(size) for size in shape)
    else:
        try:
            physical_shape = tuple(operator_index(axis_size) for axis_size in physical_shape)
        except TypeError as exc:
            raise TypeError("physical_shape must contain integer dimensions") from exc

    if len(physical_shape) != 2 or any(size <= 0 for size in physical_shape):
        raise ValueError(
            f"physical_shape must contain two positive dimensions; got {physical_shape}"
        )

    if len(shape) == 1:
        if shape[0] > prod(physical_shape):
            raise ValueError(
                f"original_shape={shape} cannot exceed physical_shape={physical_shape}"
            )
        max_slot = shape[0] - 1
    else:
        rows, columns = shape
        physical_rows, physical_columns = physical_shape
        if rows > physical_rows or columns > physical_columns:
            raise ValueError(
                f"original_shape={shape} cannot exceed physical_shape={physical_shape}"
            )
        row_major_max = (rows - 1) * physical_columns + columns - 1
        column_major_max = (columns - 1) * physical_rows + rows - 1
        max_slot = max(row_major_max, column_major_max)

    return set(range(-max_slot, max_slot + 1)) - {0}
