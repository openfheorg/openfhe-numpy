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
block_arithmetic.py

This module implements block tensor arithmetic operations for encrypted tensors.
Block operations are registered through the tensor dispatch system and usually
delegate per-block work to CTArray operations.
"""

# Standard library imports
from typing import Optional

# Third-party imports
from ..tensor.block_ctarray import BlockCTArray
from .dispatch import register_tensor_function

from ..utils.errors import (
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
    ONPValueError,
    ONPDimensionError,
)
from .block_arithmetic_utils import (
    _eval_block_binary,
    _eval_block_scalar,
    _eval_scalar_block,
    _eval_block_dot,
    _eval_block_matmul,
)
from .matrix_arithmetic import (
    _pow,
    _reduce_ct,
    sum_ct,
)
from .arithmetic_utils import _normalize_axis, _normalize_sum_axis

# ==============================================================================
# Basic block arithmetic operations
# ==============================================================================


# ------------------------------------------------------------------------------
# Addition Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "add", [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")]
)
def add_block_ct(a, b):
    """Add two block tensors."""
    return _eval_block_binary(a, b, "add")


@register_tensor_function("add", [("BlockCTArray", "scalar")])
def add_block_ct_scalar(a, scalar):
    """Add a scalar to a block ciphertext tensor."""
    return _eval_block_scalar(a, scalar, "add")


# ------------------------------------------------------------------------------
# Subtraction Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "subtract",
    [
        ("BlockCTArray", "BlockCTArray"),
        ("BlockCTArray", "BlockPTArray"),
        ("BlockPTArray", "BlockCTArray"),
    ],
)
def subtract_block_ct(a, b):
    """Subtract two block tensors."""
    return _eval_block_binary(a, b, "subtract")


@register_tensor_function("subtract", [("BlockCTArray", "scalar")])
def subtract_block_ct_scalar(a, scalar):
    """Subtract a scalar from a block ciphertext tensor."""
    return _eval_block_scalar(a, scalar, "subtract")


@register_tensor_function("subtract", [("scalar", "BlockCTArray")])
def subtract_scalar_block_ct(scalar, a):
    """Subtract a block ciphertext tensor from a scalar."""
    return _eval_scalar_block(scalar, a, "subtract")


# ------------------------------------------------------------------------------
# Multiplication Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "multiply", [("BlockCTArray", "BlockCTArray"), ("BlockCTArray", "BlockPTArray")]
)
def multiply_block_ct(a, b):
    """Multiply two block tensors element-wise."""
    return _eval_block_binary(a, b, "multiply")


@register_tensor_function("multiply", [("BlockCTArray", "scalar")])
def multiply_block_ct_scalar(a, scalar):
    """Multiply a block tensor by a scalar."""
    return _eval_block_scalar(a, scalar, "multiply")


# ==============================================================================
# Linear Algebra Operations
# ==============================================================================


# ------------------------------------------------------------------------------
# Matrix Multiplication Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "matmul",
    [
        ("BlockCTArray", "BlockCTArray"),
        ("BlockCTArray", "BlockPTArray"),
        ("BlockPTArray", "BlockCTArray"),
    ],
)
def matmul_block_ct(a, b):
    """Perform block vector dot or block matrix multiplication."""
    return _eval_block_matmul(a, b)


# ------------------------------------------------------------------------------
# Dot Product Operations
# ------------------------------------------------------------------------------
@register_tensor_function("dot", [("BlockCTArray", "BlockCTArray")])
def dot_block_ct(a, b):
    """Compute dot product of two block vectors."""
    return _eval_block_dot(a, b)


# ------------------------------------------------------------------------------
# Transpose Operations
# ------------------------------------------------------------------------------
@register_tensor_function("transpose", [("BlockCTArray",)])
def transpose_block_ct(a):
    """Transpose a block ciphertext tensor."""
    if a.ndim == 1:
        return a

    if a.ndim != 2:
        raise ONPDimensionError(f"transpose requires 1D or 2D tensor, got {a.ndim}D.")

    grid_rows, grid_cols = a.grid_shape
    blocks = []
    for j in range(grid_cols):
        for i in range(grid_rows):
            blocks.append(a.get_block(i, j).transpose())

    return type(a)(
        data=blocks,
        grid_shape=(grid_cols, grid_rows),
        block_shape=(a.block_shape[1], a.block_shape[0]),
        original_shape=(a.original_shape[1], a.original_shape[0]),
        batch_size=a.batch_size,
        order=a.order,
    )


# ==============================================================================
# Advanced block operations
# ==============================================================================


# ------------------------------------------------------------------------------
# Power Operations
# ------------------------------------------------------------------------------
@register_tensor_function("power", [("BlockCTArray", "int")])
def power_block_ct(a, exp):
    """Raise a block ciphertext matrix to a nonnegative integer power."""
    if a.ndim != 2:
        raise ONPNotImplementedError("Block tensor power currently supports only block matrices.")

    if a.original_shape[0] != a.original_shape[1]:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            a.original_shape,
            "Block matrix power requires a square logical shape.",
        )

    if a.grid_shape[0] != a.grid_shape[1]:
        raise ONPIncompatibleShapeError(
            a.grid_shape,
            a.grid_shape,
            "Block matrix power requires a square block grid.",
        )

    return _pow(a, exp)


# ------------------------------------------------------------------------------
# Cumulative Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "cumsum",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def cumsum_block_ct(obj, axis=None, keepdims=False):
    """Compute cumulative sums for encrypted block tensors.

    Currently supports:
    - 1D block tensors across multiple blocks.
    - 2D block tensors only when the cumulative axis stays inside each block.
    """
    if keepdims:
        raise ONPNotImplementedError("Block cumsum does not support keepdims=True yet.")

    if obj.ndim == 1:
        if axis is None:
            axis = 0

        axis = _normalize_axis(axis, obj.ndim)

        blocks = []
        offset = None

        for block in obj.data:
            cumulative = block.cumsum(axis=0)

            if offset is not None:
                cumulative = cumulative + offset

            blocks.append(cumulative)

            block_sum = block.sum()
            offset = block_sum if offset is None else offset + block_sum

        return obj.clone(data=blocks)

    if obj.ndim == 2:
        if axis is None:
            raise ONPNotImplementedError("Block cumsum(axis=None) is not implemented yet.")

        axis = _normalize_axis(axis, obj.ndim)

        if axis == 0 and obj.grid_shape[0] != 1:
            raise ONPNotImplementedError(
                "Block cumsum(axis=0) across multiple block rows is not implemented yet."
            )

        if axis == 1 and obj.grid_shape[1] != 1:
            raise ONPNotImplementedError(
                "Block cumsum(axis=1) across multiple block columns is not implemented yet."
            )

        return obj.clone(data=[block.cumsum(axis=axis) for block in obj.data])

    raise ONPDimensionError(f"cumsum requires 1D or 2D tensor, got {obj.ndim}D.")


@register_tensor_function(
    "cumulative_reduce",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def cumulative_reduce_block_ct(a, axis=0, keepdims=False):
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


# ------------------------------------------------------------------------------
# Sum Operations
# ------------------------------------------------------------------------------
def _sum_ct_scalars(values):
    """Add scalar CTArray values into one scalar CTArray."""
    if not values:
        raise ONPValueError("Cannot sum an empty block tensor.")

    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


@register_tensor_function(
    "sum",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def sum_block_ct(x: BlockCTArray, axis: Optional[int] = None, keepdims: bool = False):
    """Sum encrypted block tensor elements."""
    if keepdims:
        raise ONPNotImplementedError("Block sum does not support keepdims=True yet.")

    axis = _normalize_sum_axis(axis, x.ndim)

    if x.ndim == 1:
        return _sum_ct_scalars([sum_ct(block, None, False) for block in x.data])

    if x.ndim != 2:
        raise ONPDimensionError(f"sum requires 1D or 2D tensor, got {x.ndim}D.")

    if axis is None:
        return _sum_ct_scalars([sum_ct(block, None, False) for block in x.data])

    if axis == 0:
        grid_rows, grid_cols = x.grid_shape
        blocks = []
        for j in range(grid_cols):
            col_sum = sum_ct(x.get_block(0, j), axis=0, keepdims=False)
            for i in range(1, grid_rows):
                col_sum = col_sum + sum_ct(x.get_block(i, j), axis=0, keepdims=False)
            blocks.append(col_sum)

        return type(x)(
            data=blocks,
            grid_shape=(grid_cols,),
            block_shape=(x.block_shape[1],),
            original_shape=(x.original_shape[1],),
            batch_size=x.batch_size,
            order=blocks[0].order,
        )

    if axis == 1:
        grid_rows, grid_cols = x.grid_shape
        blocks = []
        for i in range(grid_rows):
            row_sum = sum_ct(x.get_block(i, 0), axis=1, keepdims=False)
            for j in range(1, grid_cols):
                row_sum = row_sum + sum_ct(x.get_block(i, j), axis=1, keepdims=False)
            blocks.append(row_sum)

        return type(x)(
            data=blocks,
            grid_shape=(grid_rows,),
            block_shape=(x.block_shape[0],),
            original_shape=(x.original_shape[0],),
            batch_size=x.batch_size,
            order=blocks[0].order,
        )

    raise ONPValueError(f"Invalid axis {axis}.")


# ------------------------------------------------------------------------------
# Mean Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "mean",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def mean_block_ct(x: BlockCTArray, axis: Optional[int] = None, keepdims: bool = False):
    """Compute the arithmetic mean for encrypted block tensors.

    - axis=None: mean over all entries.
    - axis=0: mean down rows and return one value per column.
    - axis=1: mean across columns and return one value per row.

    Limitation:
    - keepdims=True is not supported for block tensors yet.
    """
    if keepdims:
        raise ONPNotImplementedError("Block mean does not support keepdims=True yet.")

    axis = _normalize_sum_axis(axis, x.ndim)
    mean_x = sum_block_ct(x, axis, keepdims=False)

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
# Rotation Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "roll",
    [("BlockCTArray", "int"), ("BlockCTArray", "int", "int")],
)
def roll_block_ct(x: BlockCTArray, shift: int, axis: Optional[int] = None) -> BlockCTArray:
    """Block tensor roll is not implemented yet."""
    raise ONPNotImplementedError("Block roll is not implemented yet.")
