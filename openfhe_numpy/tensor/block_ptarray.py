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
import numpy as np

from openfhe import Plaintext
from .block_tensor import BlockFHETensor


class BlockPTArray(BlockFHETensor[Plaintext]):
    """Block tensor of plaintext blocks (PTArray).

    All storage, indexing, and metadata logic lives in BlockFHETensor.
    BlockPTArray adds plaintext decoding and a higher tensor priority than
    PTArray for dispatch.
    """

    tensor_priority = 35
    is_encrypted = False

    def decrypt(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(
            "Decrypt is not defined for plaintext block arrays. Use decode()."
        )

    def decode(self) -> np.ndarray:
        """Decode all blocks and reassemble the original logical array.

        Returns a NumPy array with shape == original_shape.
        """
        if self.ndim == 1:
            chunks = [
                np.asarray(block.decode()).reshape(block.original_shape) for block in self.data
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
                plain = np.asarray(block.decode()).reshape(block.original_shape)
                r0 = gi * br
                c0 = gj * bc
                full[r0 : r0 + block_rows, c0 : c0 + block_cols] = plain

        return full[:rows, :cols]
