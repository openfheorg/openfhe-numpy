from openfhe import Ciphertext
from .block_tensor import BlockFHETensor


class BlockCTArray(BlockFHETensor[Ciphertext]):
    tensor_priority = 40  # Higher priority than CTArray

    def __str__(self):
        return f"BlockCTArray(shape={self.original_shape}, cell={len(self._blocks)} {len(self._blocks[0]) if self._blocks else 0}, block_shape={self.block_shape})"

    def __repr__(self):
        return self.__str__()

    def clone(self, blocks=None):
        pass

    def decrypt(self, secret_key):
        pass
