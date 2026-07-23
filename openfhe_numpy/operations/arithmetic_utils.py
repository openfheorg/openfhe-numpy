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

from typing import Any

from openfhe_numpy.tensor.tensor import BaseTensor
from openfhe_numpy.utils.errors import ONPIncompatibleShapeError, ONPValueError, ONPDimensionError
from openfhe_numpy.utils.packing import _is_row_major, _is_col_major


# ------------------------------------------------------------------------------
# Arithmetic: Convention
# ------------------------------------------------------------------------------
def _require(
    condition: bool,
    left: Any,
    right: Any,
    message: str,
    *,
    error_cls: type[Exception] = ONPIncompatibleShapeError,
) -> None:
    """Raise ``error_cls`` when ``condition`` is false.

    By default this reports an ``ONPIncompatibleShapeError`` built from ``left``
    and ``right``. Pass ``error_cls`` (e.g. ``ONPValueError``) for non-shape
    failures such as batch-size or packing-order mismatches; those are raised
    with ``message`` only.
    """
    if condition:
        return

    if error_cls is ONPIncompatibleShapeError:
        raise ONPIncompatibleShapeError(left, right, message)

    raise error_cls(message)


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
def _normalize_axis(axis, ndim: int) -> int:
    """Normalize integer axis.

    Tuple axes and NumPy integer types are not supported.
    """
    if type(axis) is not int:
        raise ONPDimensionError(f"axis must be an integer, got {type(axis).__name__}.")

    if axis < -ndim or axis >= ndim:
        raise ONPDimensionError(f"axis {axis} is out of bounds for tensor with {ndim} dimensions.")

    return axis + ndim if axis < 0 else axis


def _normalize_sum_axis(axis, ndim: int) -> int | None:
    """Normalize a single reduction axis or None.

    Tuple axes are not supported.
    """
    if axis is None:
        return None

    return _normalize_axis(axis, ndim)
