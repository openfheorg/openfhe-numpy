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

from typing import Any, TypeVar

from openfhe_numpy.tensor.tensor import BaseTensor
from openfhe_numpy.utils.errors import ONPIncompatibleShapeError, ONPValueError, ONPDimensionError
from openfhe_numpy.utils.packing import _is_row_major, _is_col_major


# ------------------------------------------------------------------------------
# Arithmetic: Convention
# ------------------------------------------------------------------------------
def _require(condition: bool, left: Any, right: Any, message: str) -> None:
    """Raise a shape error when ``condition`` is false."""
    if not condition:
        raise ONPIncompatibleShapeError(left, right, message)


def _result_cls(a: BaseTensor, b: BaseTensor | None = None) -> BaseTensor:
    """Return the encrypted tensor class if either operand is encrypted."""
    if a.is_encrypted:
        return type(a)

    if b is not None and b.is_encrypted:
        return type(b)

    raise ONPValueError("Expected at least one encrypted tensor operand.")


# ------------------------------------------------------------------------------
# Arithmetic: Validation
# ------------------------------------------------------------------------------


def _assert_matvec_order(matrix_order: Any, vector_order: Any) -> bool:
    """Return True for supported matrix-vector packing pairs."""
    return (_is_row_major(matrix_order) and _is_col_major(vector_order)) or (
        _is_col_major(matrix_order) and _is_row_major(vector_order)
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


def _normalize_axis(axis, ndim: int) -> int:
    """Normalize an integer axis.

    Negative axes are converted to their positive equivalents.

    Examples
    --------
    ndim=2:
    - axis=0  -> 0
    - axis=1  -> 1
    - axis=-1 -> 1
    - axis=-2 -> 0

    Remark: Tuple axes are not supported
    """
    if not isinstance(axis, int):
        raise ONPDimensionError(f"axis must be an integer, got {type(axis).__name__}.")

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise ONPDimensionError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions.")

    return axis


def _normalize_sum_axis(axis, ndim: int) -> int | None:
    """Normalize a NumPy-style reduction axis.

    Unlike _normalize_axis, this helper allows axis=None.

    - axis=None means reduce over all entries.
    - negative axes are converted to positive axes.

    Remark: Tuple axes are not supported
    """
    if axis is None:
        return None

    if not isinstance(axis, int):
        raise ONPDimensionError(f"axis must be an integer or None, got {type(axis).__name__}.")

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise ONPDimensionError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions.")

    return axis
