"""Tensor implementations for homomorphic encryption operations."""

# Import tensor classes
from .tensor import BaseTensor, FHETensor
from .ptarray import PTArray
from .ctarray import CTArray
from .block_tensor import BlockFHETensor
from .block_ctarray import BlockCTArray


# Import tensor constructors
from .constructors import array

# Define public API
__all__ = [
    "BaseTensor",
    "FHETensor",
    "PTArray",
    "CTArray",
    "BlockFHETensor",
    "BlockCTArray",
    "array",
]

# [TANGO] I don't understand this????
def _register_all_operations():
    import openfhe_numpy.operations.matrix_arithmetic
    # import operations.matrix_api

_register_all_operations()
