from __future__ import annotations

import numpy as np

from openfhe import Ciphertext

from ..utils.constants import UnpackType
from .block_tensor import BlockFHETensor


class BlockCTArray(BlockFHETensor[Ciphertext]):
    """Block tensor of encrypted blocks (CTArray).

    This is a thin subclass. All storage, indexing, and metadata
    logic lives in BlockFHETensor. BlockCTArray adds only:

        - decrypt()
        - higher tensor_priority for dispatch
    """

    tensor_priority = 40

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
            chunks = [
                np.asarray(block.decrypt(secret_key, unpack_type=unpack_type)).reshape(
                    self.block_shape
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
                plain = np.asarray(block.decrypt(secret_key, unpack_type=unpack_type)).reshape(
                    self.block_shape
                )

                r0 = gi * br
                c0 = gj * bc
                full[r0 : r0 + br, c0 : c0 + bc] = plain

        if new_shape is not None:
            return full[:rows, :cols].reshape(new_shape)

        return full[:rows, :cols]

    def __repr__(self) -> str:
        return (
            f"BlockCTArray("
            f"original_shape={self.original_shape}, "
            f"shape={self.shape}, "
            f"block_shape={self.block_shape}, "
            f"grid_shape={self.grid_shape}, "
            f"num_blocks={self.num_blocks}, "
            f"batch_size={self.batch_size}, "
            f"slot_utilization={self.slot_utilization:.3f})"
        )
