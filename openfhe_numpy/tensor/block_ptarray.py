from __future__ import annotations

import numpy as np

from openfhe import Plaintext

from .block_tensor import BlockFHETensor


class BlockPTArray(BlockFHETensor[Plaintext]):
    """Block tensor of plaintext blocks (PTArray).

    All storage, indexing, and metadata
    logic lives in BlockFHETensor. BlockPTArray adds only:

        - decode()
        - higher tensor_priority than PTArray for dispatch
    """

    tensor_priority = 35
    is_encrypted = False

    def decrypt(self, *args, **kwargs):
        raise NotImplementedError(
            "Decrypt is not defined for plaintext block arrays. Use decode()."
        )

    def decode(self) -> np.ndarray:
        """Decode all blocks and reassemble the original logical array.

        Returns a NumPy array with shape == original_shape.
        """
        if self.ndim == 1:
            chunks = [block.decode() for block in self.data]
            full = np.concatenate(chunks)
            return full[: self.original_shape[0]]

        rows, cols = self.original_shape
        br, bc = self.block_shape
        grid_rows, grid_cols = self.grid_shape

        full = np.zeros(self.shape, dtype=np.float64)

        for gi in range(grid_rows):
            for gj in range(grid_cols):
                block = self.get_block(gi, gj)
                plain = block.decode()
                r0 = gi * br
                c0 = gj * bc
                full[r0 : r0 + br, c0 : c0 + bc] = plain

        return full[:rows, :cols]

    def __repr__(self) -> str:
        return (
            f"BlockPTArray("
            f"original_shape={self.original_shape}, "
            f"block_shape={self.block_shape}, "
            f"grid_shape={self.grid_shape}, "
            f"num_blocks={self.num_blocks}, "
            f"batch_size={self.batch_size}, "
            f"slot_utilization={self.slot_utilization:.3f})"
        )
