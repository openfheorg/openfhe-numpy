# ==============================================================================
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
# ================================================================================
"""
matrix_arithmetic.py

This module implements the core arithmetic operations for encrypted tensors
using the OpenFHE library. Operations include addition, subtraction, multiplication,
matrix multiplication, and other mathematical operations.
"""

# Third-party imports
import math
import numpy as np
from ..openfhe_numpy import ArrayEncodingType
from ..utils.matlib import next_power_of_two
from ..utils._helper_slots_ops import _create_masking, _duplicate_block
from ..tensor.constructors import array


# ------------------------------------------------------------------------------
# Utilities functions for broadcasting
# ------------------------------------------------------------------------------


def broadcast_shapes(x_shape, y_shape):
    return np.broadcast_shapes(x_shape, y_shape)


# Broadcasting rules: expand operands to a common shape (x, y) -> (a, b)
#
# 1. Scalar: () -> (1, 1)
#    1 + 2                     -> 3
#    1 + [1, 2, 3]             -> [2, 3, 4]
#    1 + [[1, 2], [3, 4]]      -> [[6, 7], [8, 9]]
#
# 2. Vector: (n,) -> (1, n)
#    - (n,) + (n,)             -> (n,)
#      Example: [1, 2, 3] + [10, 20, 30]
#               -> [11, 22, 33]
#
#    - (m, n) + (n,)           -> (m, n)
#      Example: [[1, 2, 3],
#                [4, 5, 6]] + [10, 20, 30]
#               -> [[11, 22, 33],
#                   [14, 25, 36]]
#
# 3. Matrix: (m, n) -> (m, n)


# ------------------------------------------------------------------------------
# Rotation index helpers
# ------------------------------------------------------------------------------


def _duplicate_block_indices(duplicate_count, block_size) -> set:
    """
    Mirrors the rotation pattern inside _duplicate_block:
    """
    indices = set()
    rotation = block_size
    while rotation < block_size * duplicate_count:
        indices.add(-rotation)
        rotation *= 2
    return indices


# ------------------------------------------------------------------------------
# Key generation
# ------------------------------------------------------------------------------


def generate_broadcast_key(secret_key, original_shape, target_shape):
    """
    Pre-generate all rotation keys needed by broadcast_to for a given
    (original_shape, target_shape) pair - for both ROW_MAJOR and COL_MAJOR.

    Broadcasting cases covered:
      Scalar  ()      -> Vector (n,)
      Scalar  ()      -> Matrix (m, n)
      Vector  (n,)    -> Matrix (m, n)
      ColVec  (m, 1)  -> Matrix (m, n)
    """
    if target_shape == ():
        return

    cc = secret_key.GetCryptoContext()
    indices = set()

    # --- Scalar -> Vector ---
    if len(target_shape) == 1:
        nrow = target_shape[0]
        if original_shape == ():
            rotation = 1
            while rotation < nrow:
                indices.add(-rotation)
                rotation *= 2

    # --- Any -> Matrix ---
    elif len(target_shape) == 2:
        nrow, ncol = target_shape
        nrow_pow_2 = next_power_of_two(nrow)
        ncol_pow_2 = next_power_of_two(ncol)

        if original_shape == ():
            indices.update(_duplicate_block_indices(nrow, ncol_pow_2))
            indices.update(_duplicate_block_indices(ncol, 1))
            indices.update(_duplicate_block_indices(ncol, nrow_pow_2))
            indices.update(_duplicate_block_indices(nrow, 1))

        elif len(original_shape) == 1:
            indices.update(_duplicate_block_indices(nrow, ncol_pow_2))
            for i in range(1, original_shape[0]):
                indices.add(-i * (nrow_pow_2 - 1))
            indices.update(_duplicate_block_indices(nrow, 1))

        elif len(original_shape) == 2:
            indices.update(_duplicate_block_indices(ncol_pow_2, nrow_pow_2))
            for i in range(1, original_shape[0]):
                indices.add(-i * (ncol_pow_2 - 1))
            indices.update(_duplicate_block_indices(ncol, 1))

    cc.EvalRotateKeyGen(secret_key, sorted(indices))


# ------------------------------------------------------------------------------
# Broadcasting
# ------------------------------------------------------------------------------


def broadcast_to(x, target_shape, order=None, cc=None):
    if x.dtype == "CTArray":
        return _ct_broadcast_to(x, target_shape, order)
    elif x.dtype == "PTArray":
        return _pt_broadcast_to(x, target_shape, order, cc)
    else:
        raise ValueError(f"Broadcast doesn't support {type(x)}")


def _pt_broadcast_to(pta_x, target_shape, order=None, cc=None):
    target_shape = tuple(target_shape)

    if target_shape == pta_x.original_shape:
        return pta_x

    def _make_array(data):
        if cc is not None:
            return array(
                cc=cc,
                data=data,
                batch_size=pta_x.batch_size,
                order=order,
                fhe_type="P",
                mode="zero",
            )
        raise ValueError("Broadcasting operation requires a crypto context")

    packed = pta_x.data.GetRealPackedValue()

    # --- Scalar () -> anything ---
    if pta_x.original_shape == ():
        x = np.broadcast_to(np.array(packed[0]), target_shape)
        return _make_array(x)

    # --- 1D (n,) -> (m, n) ---
    if len(pta_x.original_shape) == 1:
        n = pta_x.original_shape[0]
        x = np.array(packed[:n])  # shape (n,)
        x_broadcasted = np.broadcast_to(x, target_shape)
        return _make_array(x_broadcasted)

    # --- 2D (m,1) -> (m,n)  or  (1,n) -> (m,n) ---
    if len(pta_x.original_shape) == 2:
        m, n = pta_x.original_shape
        x = np.array(packed[: m * n]).reshape(pta_x.original_shape)  # restore 2D shape
        x_broadcasted = np.broadcast_to(x, target_shape)
        return _make_array(x_broadcasted)

    raise ValueError(
        f"Incompatible shapes: {pta_x.original_shape} "
        f"cannot be broadcast to target matrix shape {target_shape}."
    )


def _ct_broadcast_to(x, target_shape, order=None):
    from openfhe_numpy.tensor.ctarray import CTArray

    target_shape = tuple(target_shape)
    if target_shape == x.original_shape:
        return x

    cc = x.data.GetCryptoContext()

    # --- Scalar -> anything ---
    if x.original_shape == ():
        if target_shape == ():
            return x

        # Scalar -> Vector
        elif len(target_shape) == 1:
            mask = _create_masking([0], x.batch_size)
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)
            ct_res = cc.EvalMult(x.data, pt_mask)

            rotation = 1
            while rotation < target_shape[0]:
                ct_rotated = cc.EvalRotate(ct_res, -rotation)
                ct_res = cc.EvalAdd(ct_res, ct_rotated)
                rotation *= 2

            mask = _create_masking(list(range(target_shape[0])), x.batch_size)
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)
            ct_res = cc.EvalMult(ct_res, pt_mask)

            return CTArray(
                data=ct_res,
                original_shape=target_shape,
                batch_size=x.batch_size,
                new_shape=(next_power_of_two(target_shape[0]),),
                order=ArrayEncodingType.ROW_MAJOR,
            )

        # Scalar -> Matrix
        elif len(target_shape) == 2:
            nrow, ncol = target_shape
            mask = _create_masking([0], x.batch_size)
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)
            ct_res = cc.EvalMult(x.data, pt_mask)

            if order == ArrayEncodingType.ROW_MAJOR:
                ncol_pow_2 = next_power_of_two(ncol)
                mask = [0] * x.batch_size
                for i in range(nrow):
                    for j in range(ncol):
                        mask[i * ncol_pow_2 + j] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = _duplicate_block(ct_res, nrow, ncol_pow_2)
                ct_res = _duplicate_block(ct_res, ncol, 1, pt_mask)

            elif order == ArrayEncodingType.COL_MAJOR:
                nrow_pow_2 = next_power_of_two(nrow)
                mask = [0] * x.batch_size
                for i in range(ncol):
                    for j in range(nrow):
                        mask[i * nrow_pow_2 + j] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = _duplicate_block(ct_res, ncol, nrow_pow_2)
                ct_res = _duplicate_block(ct_res, nrow, 1, pt_mask)

            else:
                raise ValueError(f"Invalid order ({order})")

            return CTArray(
                data=ct_res,
                original_shape=target_shape,
                batch_size=x.batch_size,
                new_shape=(next_power_of_two(nrow), next_power_of_two(ncol)),
                order=order,
            )

        raise ValueError(f"Target shape ({target_shape}) is not supported")

    # --- Vector (n,) -> Matrix (m, n) ---
    if len(x.original_shape) == 1:
        if len(target_shape) == 2:
            if target_shape[1] != x.original_shape[0]:
                raise ValueError(
                    f"Incompatible shapes: vector length {x.original_shape[0]} "
                    f"cannot be broadcast to target matrix shape {target_shape}. "
                    "Only supports broadcasting from (n,) to (m, n)."
                )

            nrow, ncol = target_shape
            ncol_pow_2 = next_power_of_two(ncol)
            nrow_pow_2 = next_power_of_two(nrow)

            if order == ArrayEncodingType.ROW_MAJOR:
                mask = _create_masking(list(range(x.original_shape[0])), x.batch_size)
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = cc.EvalMult(x.data, pt_mask)
                ct_res = _duplicate_block(ct_res, nrow, ncol_pow_2)

                return CTArray(
                    data=ct_res,
                    original_shape=target_shape,
                    batch_size=x.batch_size,
                    new_shape=(nrow_pow_2, ncol_pow_2),
                    order=order,
                )

            elif order == ArrayEncodingType.COL_MAJOR:
                mask = [0] * x.batch_size
                mask[0] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = cc.EvalMult(x.data, pt_mask)

                for i in range(1, x.original_shape[0]):
                    mask[i - 1] = 0
                    mask[i] = 1
                    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                    ct_scalar = cc.EvalMult(x.data, pt_mask)
                    ct_scalar = cc.EvalRotate(ct_scalar, -(nrow_pow_2 * i - i))
                    ct_res = cc.EvalAdd(ct_res, ct_scalar)
                mask[x.original_shape[0] - 1] = 0  # reset

                mask = [0] * x.batch_size
                for i in range(ncol):
                    for j in range(nrow):
                        mask[i * nrow_pow_2 + j] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = _duplicate_block(ct_res, nrow, 1, pt_mask)

                return CTArray(
                    data=ct_res,
                    original_shape=target_shape,
                    batch_size=x.batch_size,
                    new_shape=(nrow_pow_2, ncol_pow_2),
                    order=order,
                )

            else:
                raise ValueError(f"Invalid order ({order})")

    # --- ColVec (m, 1) -> Matrix (m, n) ---
    elif len(x.original_shape) == 2:
        if len(target_shape) == 2:
            try:
                new_shape = np.broadcast_shapes(x.original_shape, target_shape)
            except ValueError as e:
                raise ValueError(
                    f"Incompatible shapes: {x.original_shape} cannot be broadcast "
                    f"to target shape {target_shape}. "
                    "Only supports broadcasting from (m,1) to (m,n) or (1,n) to (m,n) "
                ) from e
            nrow, ncol = target_shape
            ncol_pow_2 = next_power_of_two(ncol)
            nrow_pow_2 = next_power_of_two(nrow)

            if order == ArrayEncodingType.COL_MAJOR:
                mask = _create_masking(list(range(nrow)), x.batch_size)
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_x_cleared = cc.EvalMult(x.data, pt_mask)
                ct_x_duplicated = _duplicate_block(ct_x_cleared, ncol_pow_2, nrow_pow_2)

                return CTArray(
                    data=ct_x_duplicated,
                    original_shape=target_shape,
                    batch_size=x.batch_size,
                    new_shape=(nrow_pow_2, ncol_pow_2),
                    order=order,
                )

            elif order == ArrayEncodingType.ROW_MAJOR:
                mask = [0] * x.batch_size
                mask[0] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = cc.EvalMult(x.data, pt_mask)

                for i in range(1, x.original_shape[0]):
                    mask[i - 1] = 0
                    mask[i] = 1
                    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                    ct_scalar = cc.EvalMult(x.data, pt_mask)
                    ct_scalar = cc.EvalRotate(ct_scalar, -(ncol_pow_2 * i - i))
                    ct_res = cc.EvalAdd(ct_res, ct_scalar)
                mask[x.original_shape[0] - 1] = 0  # reset

                mask = [0] * x.batch_size
                for i in range(nrow):
                    for j in range(ncol):
                        mask[i * ncol_pow_2 + j] = 1
                pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                ct_res = _duplicate_block(ct_res, ncol, 1, pt_mask)

                return CTArray(
                    data=ct_res,
                    original_shape=target_shape,
                    batch_size=x.batch_size,
                    new_shape=(nrow_pow_2, ncol_pow_2),
                    order=order,
                )

            else:
                raise ValueError(f"Invalid order ({order})")

    raise ValueError(
        f"Incompatible shapes: {x.original_shape} "
        f"cannot be broadcast to target shape {target_shape}."
    )
