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
    ONPNotImplementedError,
)
from .block_arithmetic_utils import (
    _cumsum_block_ct,
    _cumulative_reduce_block_ct,
    _eval_block_binary,
    _eval_block_dot,
    _eval_block_matmul,
    _eval_block_scalar,
    _eval_scalar_block,
    _mean_block_ct,
    _sum_block_ct,
    _transpose_block_ct,
)
from .matrix_arithmetic import _pow

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
    return _transpose_block_ct(a)


# ==============================================================================
# Advanced block operations
# ==============================================================================


# ------------------------------------------------------------------------------
# Power Operations
# ------------------------------------------------------------------------------
@register_tensor_function("power", [("BlockCTArray", "int")])
def power_block_ct(a, exp):
    """Raise a block ciphertext tensor to an integer power element-wise.

    Element-wise power (``a_i ** exp``) applies to any block shape; no square
    logical/grid constraint is required.
    """
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

    Supported:
    - 1-D cumsum across multiple blocks.
    - 2-D cumsum when the cumulative axis stays within each block.

    Not supported:
    - 2-D axis=None.
    - 2-D cumsum across block boundaries.
    - Compact block-vector packing.
    """
    return _cumsum_block_ct(obj, axis, keepdims)


@register_tensor_function(
    "cumulative_reduce",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def cumulative_reduce_block_ct(a, axis=0, keepdims=False):
    """Compute cumulative reductions inside each encrypted block."""
    return _cumulative_reduce_block_ct(a, axis, keepdims)


# ------------------------------------------------------------------------------
# Sum Operations
# ------------------------------------------------------------------------------
@register_tensor_function(
    "sum",
    [("BlockCTArray",), ("BlockCTArray", "int"), ("BlockCTArray", "int", "bool")],
)
def sum_block_ct(x: BlockCTArray, axis: Optional[int] = None, keepdims: bool = False):
    """Sum encrypted block tensor elements.

    Supported:
    - 1-D sum over all elements.
    - 2-D sum over all elements.
    - 2-D sum along axis 0 or axis 1.

    Block tensors should be constructed with mode="zero".
    """
    return _sum_block_ct(x, axis, keepdims)


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
    return _mean_block_ct(x, axis, keepdims)


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
