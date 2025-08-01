# Standard Library Imports
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import overload, Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import numpy as np

# Internal C++ module Imports
from openfhe_numpy.utils.errors import ONP_ERROR
from openfhe_numpy.utils.constants import *

# -----------------------------------------------------------
# Ultilities Imports
T = TypeVar("T")


# BaseTensor - Abstract Interface
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
    def info(self) -> dict: ...

    @abstractmethod
    def clone(self, data: T = None) -> "BaseTensor[T]": ...

    @abstractmethod
    def decrypt(self, *args, **kwargs): ...


# -----------------------------------------------------------
# FHETensor - Generic Tensor with Metadata
# -----------------------------------------------------------
@dataclass
class PackedArrayInformation:
    data: list | np.ndarray | T
    original_shape: tuple[int, int]
    ndim: int
    batch_size: int
    shape: tuple[int, int]
    order: int


class FHETensor(BaseTensor[T], Generic[T]):
    """
    Concrete base class for tensors in FHE computation.

    Parameters
    ----------
    data : T
        Underlying encrypted or encoded of a packed encoding array.
    original_shape : Tuple[int, int]
        Shape before any padding.
    batch_size : int
        Total number of packed slots.
    new_shape : Tuple[int, int]
        Since the shape may change after some operations, we need to store the new information.
    order : int
        Packing order: only support row-major or column-major.
    """

    __slots__ = (
        "_data",
        "_original_shape",
        "_shape",
        "_batch_size",
        "_ndim",
        "_order",
        "_dtype",
        "extra",
    )

    @overload
    def __init__(
        self,
        data: T,
        original_shape: Tuple[int, int],
        batch_size: int,
        new_shape: Tuple[int, int],
        order: int = 0,
    ) -> None: ...

    @overload
    def __init__(self, info: PackedArrayInformation) -> None: ...

    def __init__(
        self,
        data: Union[list, np.ndarray, PackedArrayInformation],
        original_shape: Tuple[int, int],
        batch_size: int,
        new_shape: Tuple[int, int],
        order: int = 0,
    ) -> None:
        if isinstance(data, PackedArrayInformation):
            self._data = data.data
            self._original_shape = data.original_shape
            self._shape = data.shape
            self._batch_size = data.batch_size
            self._ndim = data.ndim
            self._order = data.order
            self._dtype = self.__class__.__name__
            self.extra = {}
        else:
            if None in (original_shape, batch_size, new_shape):
                ONP_ERROR(
                    "Raw form requires (data, original_shape, ndim, batch_size, shape[, order])"
                )
            self._data = data
            self._original_shape = original_shape
            self._shape = new_shape
            self._batch_size = batch_size

            self._ndim = len(original_shape)
            if self._ndim > 2 or self._ndim < 0:
                ONP_ERROR("Dimension is invalid!!!")
            self._order = order
            # dtype in ["CTArray", "BlockCTArray"]
            self._dtype = self.__class__.__name__
            self.extra = {}

    ###
    ### Properties
    ###
    @property
    # Total size of a packed encoded array
    def size(self):
        if self.ndim == 1:
            return self.shape[0]
        elif self.ndim == 2:
            return self.shape[0] * self.shape[1]
        return 0

    @property
    # Determine if the tensor is Ciphertext or Plaintext
    def dtype(self):
        return self._dtype

    @property
    def data(self) -> T:
        """Underlying encrypted/plaintext payload."""
        return self._data

    @data.setter
    def data(self, data):
        import openfhe

        if isinstance(data, openfhe.Ciphertext):
            self._dtype = "CTArray"
        elif isinstance(data, openfhe.Plaintext):
            self._dtype = "PTArray"
        else:
            ONP_ERROR(
                "Object data is incorrect. \
                      Only support FHETensor only supports Ciphertext or Plaintext"
            )
        self._data = data

    @property
    def original_shape(self) -> Tuple[int, int]:
        """Original shape before any padding was applied."""
        return self._original_shape

    @original_shape.setter
    def original_shape(self, original_shape):
        self._original_shape = original_shape

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape after padding."""
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, int]):
        self._shape = value

    @property
    def ndim(self) -> int:
        """Dimensionality of the original tensor."""
        return self._ndim

    @property
    def batch_size(self) -> int:
        """Total number of packed slots."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        """Set batch size with validation."""
        if not isinstance(size, int):
            raise TypeError(f"Batch size must be integer, got {type(size)}")
        if size <= 0:
            raise ValueError(f"Batch size must be positive, got {size}")
        self._batch_size = size

    @property
    def ncols(self) -> int:
        """Number of columns after padding"""
        if self.ndim == 2:
            return self._shape[1]
        return None

    @property
    def nrows(self) -> int:
        """Number of rows after padding"""
        return self._shape[0]

    @property
    def order(self) -> int:
        """Packing order constant (row-major or column-major)."""
        return self._order

    @order.setter
    def order(self, order: int):
        if order in ["R", "C"]:
            self._order = order
        else:
            ONP_ERROR("Not support order [{order}]")

    @property
    def is_encrypted(self) -> int:
        return "CT" in self.dtype

    @property
    def info(self) -> Dict[str, Any]:
        """Metadata dict for serialization or inspection."""
        return {
            "type": self.dtype,
            "shape": self.shape,
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "order": self.order,
            "extra": self.extra,
            "ndim": self.ndim,
        }

    @property
    def T(self):
        return self.transpose()

    ###
    ### Update properties in some specific cases
    ###

    def clone(self, data: Optional[T] = None) -> "BaseTensor[T]":
        """
        Copy the tensor, optionally replacing the data payload.
        """
        return type(self)(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.shape,
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
            raise ValueError(
                f"Invalid axis {axis} for tensor with {self.ndim} dimensions."
            )
        return self.__tensor_function__("sum", (self,), {"axis": axis})

    def reduce(self, axis=0):
        return self.__tensor_function__("reduce", (self,), {"axis": axis})

    def transpose(self):
        return self.__tensor_function__("transpose", (self,))

    def __tensor_function__(
        self, func_name, args, kwargs=None, verbose: bool = False
    ):
        """Dispatch tensor operations via the registry."""
        if verbose:
            print(
                f"DEBUG: tensor.__tensor_function__ called for '{func_name}' with {len(args)} args"
            )
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
        shape=tensor.shape,
        order=tensor.order,
    )
