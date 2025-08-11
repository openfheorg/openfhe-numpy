from typing import Generic
from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from .tensor import FHETensor, TPL


class BlockFHETensor(FHETensor[TPL], Generic[TPL]):
    """Base class for block tensor implementations"""

    def __init__(
        self,
        blocks,
        block_shape,
        original_shape,
        batch_size,
        ncols=1,
        order=ArrayEncodingType.ROW_MAJOR,
    ):
        self._blocks = blocks
        self._block_shape = block_shape
        super().__init__(None, original_shape, batch_size, ncols, order)

    @property
    def blocks(self):
        return self._blocks

    @property
    def block_shape(self):
        return self._block_shape
