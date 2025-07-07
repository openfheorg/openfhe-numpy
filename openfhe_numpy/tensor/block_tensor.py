from typing import TypeVar, Generic, Tuple, List
from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from openfhe_numpy.tensor.tensor import FHETensor, T


class BlockFHETensor(FHETensor[T], Generic[T]):
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
