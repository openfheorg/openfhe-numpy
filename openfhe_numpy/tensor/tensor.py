# -----------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import io
import sys
import logging


# -----------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------
import numpy as np
import openfhe
from openfhe import *

from .. import _openfhe_numpy  # Import from parent package
from openfhe_numpy.utils.log import ONP_ERROR
from openfhe_numpy.utils.utils import is_power_of_two, next_power_of_two, MatrixOrder

# -----------------------------------------------------------
# Ultilities Imports
# -----------------------------------------------------------
T = TypeVar("T")


# -----------------------------------------------------------
# BaseTensor - Abstract Interface
# -----------------------------------------------------------
class BaseTensor(ABC, Generic[T]):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def original_shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def ncols(self) -> int: ...

    @property
    @abstractmethod
    def order(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> str: ...

    @property
    @abstractmethod
    def meta(self) -> dict: ...

    @abstractmethod
    def clone(self, data: T = None) -> "BaseTensor[T]": ...

    @abstractmethod
    def decrypt(self, *args, **kwargs): ...


# -----------------------------------------------------------
# FHETensor - Generic Tensor with Metadata
# -----------------------------------------------------------
class FHETensor(BaseTensor[T], Generic[T]):
    """
    Concrete base class for tensors in FHE computation.

    Parameters
    ----------
    data : T
        Underlying encrypted or encoded data.
    original_shape : Tuple[int, int]
        Shape before any padding.
    batch_size : int
        Total number of packed slots.
    ncols : int
        Number of logical columns after padding.
    order : int
        Packing order, e.g., row- or column-major.
    """

    __slots__ = ("_data", "_original_shape", "_batch_size", "_ncols", "_ndim", "_order")

    def __init__(
        self,
        data: T,
        original_shape: Tuple[int, int],
        batch_size: int,
        ncols: int = 1,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        self._data = data
        self._original_shape = original_shape
        self._batch_size = batch_size
        self._ncols = ncols if is_power_of_two(ncols) else next_power_of_two(ncols)
        if len(original_shape) == 2:
            self._ndim = 2 if original_shape[1] else 1
        elif len(original_shape) == 1:
            self._ndim = 1
        else:
            ONP_ERROR("Don't support high dimension")
        self._order = order
        self._dtype = self.__class__.__name__  # e.g., "CTArray", "BlockCTArray"
        self.extra = {}

    ###
    ### Properties
    ###

    @property
    def dtype(self):
        return self._dtype

    @property
    def data(self) -> T:
        """Underlying encrypted/plaintext payload."""
        return self._data

    @property
    def shape(self) -> Tuple[int, int]:
        """Logical 2-D shape after packing."""
        rows = self._batch_size // self._ncols
        return (rows, self._ncols)

    @property
    def original_shape(self) -> Tuple[int, int]:
        """Shape before any padding was applied."""
        return self._original_shape

    @property
    def ndim(self) -> int:
        """Dimensionality of the original tensor."""
        return self._ndim

    @property
    def batch_size(self) -> int:
        """Total number of packed slots."""
        return self._batch_size

    @property
    def ncols(self) -> int:
        """Number of columns in the packed representation."""
        return self._ncols

    @property
    def order(self) -> int:
        """Packing order constant (row- or column-major)."""
        return self._order

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata dict for serialization or inspection."""
        return {
            "type": self.dtype,
            "shape": self.shape,
            "original_shape": self.original_shape,
            "ncols": self.ncols,
            "batch_size": self.batch_size,
            "order": self.order,
            "extra": self.extra,
        }

    @property
    def info(self) -> Tuple:
        """
        Tuple of shape and encoding metadata.

        Returns
        -------
        Tuple
            Contains [None, original_shape, batch_size, ncols, order]
        """
        return [None, self.original_shape, self.batch_size, self.ncols, self.order]

    @property
    def is_encrypted(self) -> bool:
        return "CT" in self.dtype

    ###
    ### Setter
    ###

    def set_batch_size(self, value: int):
        """Set batch size with validation."""
        if not isinstance(value, int):
            raise TypeError(f"Batch size must be integer, got {type(value)}")
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        self._batch_size = value

    def set_ncols(self, value: int):
        """
        Set the number of columns in the packed representation.

        Parameters
        ----------
        value : int
            New number of columns value. Should be a power of two.
        """
        if not isinstance(value, int):
            raise TypeError(f"Batch size must be integer, got {type(value)}")
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        self._ncols = value

    def clone(self, data: Optional[T] = None) -> "BaseTensor[T]":
        """
        Copy the tensor, optionally replacing the data payload.
        """
        return type(self)(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

    def __eq__(self, other) -> bool:
        """
        Structural comparison of shape and layout.

        Parameters
        ----------
        other : object
            Object to compare with

        Returns
        -------
        bool
            True if other is the same type and has identical metadata
        """
        return (
            isinstance(other, type(self))
            and self.original_shape == other.original_shape
            and self.batch_size == other.batch_size
            and self.ncols == other.ncols
            and self.order == other.order
        )

    ###
    ### Operators
    ###
    # Replace all these methods in FHETensor class
    def __add__(self, other):
        return self.__tensor_function__("add", (self, other))

    def __sub__(self, other):
        return self.__tensor_function__("subtract", (self, other))

    def __mul__(self, other):
        return self.__tensor_function__("multiply", (self, other))

    def __matmul__(self, other):
        return self.__tensor_function__("matmul", (self, other))

    def __pow__(self, exp):
        return self.__tensor_function__("power", (self, exp))

    # Replace these methods too
    def sum(self, axis=0):
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"Invalid axis {axis} for tensor with {self.ndim} dimensions.")
        return self.__tensor_function__("sum", (self,), {"axis": axis})

    def reduce(self, axis=0):
        return self.__tensor_function__("reduce", (self,), {"axis": axis})

    def transpose(self):
        return self.__tensor_function__("transpose", (self,))

    def __tensor_function__(self, func_name, args, kwargs=None):
        """Dispatch tensor operations via the registry."""
        from openfhe_numpy.operations.dispatch import dispatch_tensor_function

        return dispatch_tensor_function(func_name, args, kwargs or {})

    def __getitem__(self, key):
        """
        Extract a slice from the encrypted tensor.

        Parameters
        ----------
        key : int, tuple, or slice
            Indices to extract

        Returns
        -------
        CTArray
        """
        raise NotImplementedError()

    # def ensure_compatible_packing(self, other):
    #     """
    #     Ensure tensors have compatible packing for operations.

    #     Returns a version of 'other' with matching packing order.
    #     """
    #     if not isinstance(other, FHETensor):
    #         return other

    #     if self.order == other.order:
    #         return other

    #     return other.convert_packing_order(self.order)

    # def convert_packing_order(self, target_order):
    #     """
    #     Convert tensor to a different packing order.

    #     Parameters
    #     ----------
    #     target_order : int
    #         Desired packing order (ROW_MAJOR or COL_MAJOR)

    #     Returns
    #     -------
    #     FHETensor
    #         New tensor with converted packing order
    #     """
    #     if self.order == target_order:
    #         return self.clone()

    #     # Perform conversion
    #     if self.dtype == "CTArray":
    #         # For ciphertexts, use transpose operation
    #         transposed = self._transpose()
    #         # Update order flag
    #         transposed._order = target_order
    #         return transposed
    #     else:
    #         pass


def copy_tensor(tensor: "FHETensor") -> "FHETensor":
    """
    Generic copy constructor for FHETensor and subclasses.

    Parameters
    ----------
    tensor : FHETensor
        Tensor to be copied.

    Returns
    -------
    FHETensor
        A new instance with the same metadata and (optionally deep-copied) data.
    """
    import copy

    return type(tensor)(
        data=copy.deepcopy(tensor.data),
        original_shape=tensor.original_shape,
        batch_size=tensor.batch_size,
        ncols=tensor.ncols,
        order=tensor.order,
    )
