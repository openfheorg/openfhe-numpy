"""Tensor implementations for homomorphic encryption operations."""

# Import tensor classes
from .block_ctarray import BlockCTArray
from .block_tensor import BlockFHETensor
from .constructors import (
    array,
    pack,
    ravel_matrix,
    ravel_vector,
)
from .ctarray import CTArray
from .ptarray import PTArray
from .tensor import BaseTensor, FHETensor, copy_tensor, T
#######################################################################################################################
__all__ = [
    # block_ctarray
    "BlockCTArray",
    # block_tensor
    "BlockFHETensor",
    # constructors
    "array",
    "pack",
    "ravel_matrix",
    "ravel_vector",
    # ctarray
    "CTArray",
    # ptarray
    "PTArray",
    # tensor
    "BaseTensor",
    "FHETensor",
    "copy_tensor",
    "T",
]

def _register_all_operations():
    import openfhe_numpy.operations.matrix_arithmetic
    # import operations.matrix_api

_register_all_operations()
