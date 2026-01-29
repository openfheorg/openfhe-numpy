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
import numpy as np
from openfhe_numpy.utils.matlib import next_power_of_two
from openfhe_numpy.tensor.ctarray import CTArray
from openfhe_numpy import (
    ArrayEncodingType,
)


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


def _create_masking(indices, size):
    """
    Create a binary mask with 1s at specified indices

    Args:
        indices: List/array of indices to set to 1
        size: Total size of the mask

    Returns:
        List with 1s at indices, 0s elsewhere
    """
    mask = [0] * size
    for idx in indices:
        mask[idx] = 1
    return mask


def _duplicate_block(x, duplicate_count, block_size, pt_mask=None):
    cc = x.GetCryptoContext()
    rotation = block_size
    ct_res = x

    while rotation < block_size * duplicate_count:
        ct_rotated = cc.EvalRotate(ct_res, -rotation)
        ct_res = cc.EvalAdd(ct_res, ct_rotated)
        rotation *= 2
    if pt_mask != None:
        ct_res = cc.EvalMult(pt_mask, ct_res)
    return ct_res


def broadcast_to(x, target_shape, order=None):
    # Broadcasting needs to generate rotation keys at the beginning.
    target_shape = tuple(target_shape)
    if target_shape == x.original_shape:
        return x

    cc = x.data.GetCryptoContext()
    # Scalar
    if x.original_shape == ():
        # Scalar to Scalar
        if target_shape == ():
            return x
        # Scalar to Vector
        elif len(target_shape) == 1:
            mask = _create_masking([0], x.batch_size)
            pt_mask = cc.MakeCKKSPackedPlaintext(mask)

            ct_res = cc.EvalMult(x.data, pt_mask)
            rotation = 1
            while rotation < target_shape[0]:
                ct_rotated = cc.EvalRotate(ct_res, -rotation)
                ct_res = cc.EvalAdd(ct_res, ct_rotated)
                rotation *= 2
                print(f"rotation = {rotation}")

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
        # Scalar to Matrix
        elif len(target_shape) == 2:
            nrow, ncol = target_shape[0], target_shape[1]
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
                ct_res = _duplicate_block(ct_res, ncol, next_power_of_two(nrow_pow_2))
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

    # 1D row vector
    #   original shape:  (n,)
    #   original packing: [a, b, c, 0, 0 0]. The vector is always packed by ROW_MAJOR
    if len(x.original_shape) == 1:
        # Vector to Matrix
        if len(target_shape) == 2:
            if target_shape[1] != x.original_shape[0]:
                raise ValueError(
                    f"Incompatible shapes: vector length {x.original_shape[0]} "
                    f"cannot be broadcast to target matrix shape {target_shape}. "
                    "Only supports broadcasting from (n,) to (m, n)."
                )

            nrow, ncol = target_shape[0], target_shape[1]
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
                    mask = [0] * x.batch_size
                    mask[i] = 1
                    pt_mask = cc.MakeCKKSPackedPlaintext(mask)
                    ct_scalar = cc.EvalMult(x.data, pt_mask)
                    ct_scalar = cc.EvalRotate(ct_scalar, -(nrow_pow_2 * i - i))
                    ct_res = cc.EvalAdd(ct_res, ct_scalar)

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

    raise ValueError(
        f"Incompatible shapes: vector length {x.original_shape[0]} "
        f"cannot be broadcast to target matrix shape {target_shape}. "
        "Only supports broadcasting from (n,) to (m, n)."
    )
