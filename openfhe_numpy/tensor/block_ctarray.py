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


from __future__ import annotations

from typing import Any, Callable

import numpy as np
from openfhe import Ciphertext

from ..utils.constants import UnpackType
from .block_tensor import BlockFHETensor


class BlockCTArray(BlockFHETensor[Ciphertext]):
    """Block tensor of encrypted blocks (CTArray).

    Storage and metadata logic live in BlockFHETensor. BlockCTArray adds:

        - is_encrypted = True
        - decrypt()
        - higher tensor_priority for dispatch
    """

    tensor_priority = 40
    is_encrypted = True

    def __getitem__(self, key):
        from ..operations.slicing import block_ctarray_getitem

        return block_ctarray_getitem(self, key)

    def decrypt(
        self,
        secret_key,
        unpack_type: UnpackType = UnpackType.ORIGINAL,
        new_shape=None,
    ) -> np.ndarray:
        """Decrypt all blocks and return a NumPy array with original_shape."""
        if isinstance(unpack_type, str):
            unpack_type = UnpackType(unpack_type.lower())

        if unpack_type != UnpackType.ORIGINAL:
            raise NotImplementedError(
                "BlockCTArray.decrypt currently supports only unpack_type='original'."
            )

        if self.ndim == 1:
            if new_shape is not None:
                raise NotImplementedError("new_shape is not supported for 1-D BlockCTArray.")
            chunks = [
                np.asarray(block.decrypt(secret_key, unpack_type=unpack_type)).reshape(
                    block.original_shape
                )
                for block in self.data
            ]
            full = np.concatenate(chunks)
            return full[: self.original_shape[0]]

        rows, cols = self.original_shape
        br, bc = self.block_shape
        grid_rows, grid_cols = self.grid_shape

        full = np.zeros(self.shape, dtype=np.float64)

        for gi in range(grid_rows):
            for gj in range(grid_cols):
                block = self.get_block(gi, gj)
                block_rows, block_cols = block.original_shape
                plain = np.asarray(block.decrypt(secret_key, unpack_type=unpack_type)).reshape(
                    block.original_shape
                )
                r0, c0 = gi * br, gj * bc
                full[r0 : r0 + block_rows, c0 : c0 + block_cols] = plain

        if new_shape is not None:
            return full[:rows, :cols].reshape(new_shape)
        return full[:rows, :cols]

    def __neg__(self) -> BlockCTArray:
        """Return the blockwise homomorphic negation."""
        return self.clone(data=[-block for block in self.data])

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> "BlockCTArray":
        """Apply a ciphertext-level function to every encrypted block.

        The function is applied to each block's underlying ciphertext while block
        tensor metadata is preserved.
        """
        if not callable(func):
            raise TypeError(f"apply expects a callable, got {type(func).__name__}.")

        ct_result = [block.apply(func, *args, **kwargs) for block in self.data]
        return self.clone(data=ct_result)
