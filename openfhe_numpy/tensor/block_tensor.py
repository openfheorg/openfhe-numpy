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

from math import prod
from typing import Any, Generic, Literal, TypeVar
from operator import index as operator_index

from openfhe_numpy.openfhe_numpy import ArrayEncodingType
from .tensor import BaseTensor


TPL = TypeVar("TPL")

# =============================================================================
# Block tensor base class
# =============================================================================


class BlockFHETensor(BaseTensor, Generic[TPL]):
    """Base class for block-encoded FHE tensors.

    A block tensor stores a logical 1-D or 2-D tensor as a flat row-major list of
    encoded blocks. Each block stores a local chunk of the tensor, and the block
    grid describes how those chunks tile the padded logical shape.

    Parameters
    ----------
    data : list[Any] | tuple[Any, ...]
        Flat row-major sequence of encoded blocks. Nested block lists are not
        accepted here;  construction should use ``block_array(...)``.
    block_shape : tuple[int, ...]
        Logical shape of each encoded block. Only 1-D and 2-D blocks are
        currently supported.
    original_shape : tuple[int, ...]
        Shape requested by the user before block padding.
    batch_size : int
        Number of plaintext slots available to each encoded block.
    order : ArrayEncodingType, default=ArrayEncodingType.ROW_MAJOR
        Packing order used by the encoded blocks.
    grid_shape : tuple[int, ...] | None, default=None
        Shape of the block grid. For 1-D block tensors, this is inferred from
        ``len(data)`` when omitted. For 2-D block tensors, it must be provided.

    Notes
    -----
    The padded tensor shape is computed as ``grid_shape * block_shape`` elementwise.
    Thus, mathematically, if ``grid_shape = g`` and ``block_shape = b``, then
    ``shape[i] = g[i] * b[i]``. The original logical shape must fit inside this
    padded shape.

    Limitations
    -----------
    - Only 1-D vectors and 2-D matrices are supported.
    - Logical slicing/rotation/broadcasting is not implemented yet.

    Subclasses declare encryption status using the class attribute:
    ``is_encrypted = True`` for ciphertext block arrays and ``False`` for
    plaintext block arrays.
    """

    tensor_priority = 30
    # Ensure NumPy scalar operators defer to the tensor reverse operators.
    __array_priority__ = 1000
    is_encrypted: bool = False
    __hash__ = None

    def __init__(
        self,
        data: list[Any] | tuple[Any, ...],
        block_shape: tuple[int, ...],
        original_shape: tuple[int, ...],
        batch_size: int,
        order: ArrayEncodingType = ArrayEncodingType.ROW_MAJOR,
        grid_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize a low-level block tensor from flat block storage.
        This constructor validates only structural metadata.
        Raises
        ------
        ValueError
            If the block list is empty, nested, dimensionally inconsistent, or
            cannot fit ``original_shape`` into the padded block layout.
        """
        if not isinstance(data, (list, tuple)):
            raise ValueError("BlockFHETensor data must be a flat list/tuple of blocks.")

        data = list(data)

        if len(data) == 0:
            raise ValueError("Block storage cannot be empty.")

        if any(isinstance(block, (list, tuple)) for block in data):
            raise ValueError(
                "BlockFHETensor expects flat row-major block data. "
                "Use block_array(...) for nested/user-facing input."
            )

        block_shape = tuple(block_shape)
        original_shape = tuple(original_shape)

        if grid_shape is None:
            if len(block_shape) == 1:
                grid_shape = (len(data),)
            else:
                raise ValueError("grid_shape is required for matrix BlockFHETensor.")

        grid_shape = tuple(grid_shape)

        self._validate_metadata(
            data=data,
            block_shape=block_shape,
            original_shape=original_shape,
            batch_size=batch_size,
            grid_shape=grid_shape,
        )

        padded_shape = tuple(
            grid_dim * block_dim for grid_dim, block_dim in zip(grid_shape, block_shape)
        )

        self._data = data
        self._original_shape = original_shape
        self._batch_size = batch_size
        self._shape = padded_shape
        self._order = order
        self._dtype = self.__class__.__name__
        self._block_shape = block_shape
        self._grid_shape = grid_shape

    @staticmethod
    def _validate_metadata(
        data: list[Any],
        block_shape: tuple[int, ...],
        original_shape: tuple[int, ...],
        batch_size: int,
        grid_shape: tuple[int, ...],
    ) -> None:
        """Validate block-grid metadata.

        Checks that ranks agree, dimensions are positive, each block fits in the
        available slot capacity, the number of blocks matches ``grid_shape``, and
        ``original_shape`` fits inside the padded shape.
        """
        if len(block_shape) not in (1, 2):
            raise ValueError(f"Only 1D/2D block tensors are supported; got {block_shape}.")
        if len(original_shape) not in (1, 2):
            raise ValueError(f"Only 1D/2D tensors are supported; got {original_shape}.")
        if len(block_shape) != len(original_shape):
            raise ValueError(
                f"block_shape rank must equal original_shape rank; "
                f"got {block_shape} and {original_shape}."
            )
        if len(grid_shape) != len(block_shape):
            raise ValueError(
                f"grid_shape rank must equal block_shape rank; got {grid_shape} and {block_shape}."
            )
        if any(dim <= 0 for dim in block_shape):
            raise ValueError(f"block_shape must be positive; got {block_shape}.")
        if any(dim <= 0 for dim in original_shape):
            raise ValueError(f"original_shape must be positive; got {original_shape}.")
        if any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"grid_shape must be positive; got {grid_shape}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")

        block_slots = prod(block_shape)
        if block_slots > batch_size:
            raise ValueError(
                f"block_shape={block_shape} requires {block_slots} slots, "
                f"but batch_size={batch_size}."
            )

        expected_blocks = prod(grid_shape)
        if len(data) != expected_blocks:
            raise ValueError(
                f"grid_shape={grid_shape} expects {expected_blocks} blocks, but got {len(data)}."
            )

        padded_shape = tuple(g * b for g, b in zip(grid_shape, block_shape))
        if any(orig > pad for orig, pad in zip(original_shape, padded_shape)):
            raise ValueError(
                f"original_shape={original_shape} cannot exceed padded_shape={padded_shape}."
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data(self) -> list[Any]:
        """Flat row-major list of encoded blocks."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """Replace the internal block list without revalidating metadata."""
        self._data = value

    @property
    def original_shape(self) -> tuple[int, ...]:
        """Logical tensor shape before block padding."""
        return self._original_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Padded logical shape covered by the block grid."""
        return self._shape

    @property
    def batch_size(self) -> int:
        """Number of plaintext slots available per encoded block."""
        return self._batch_size

    @property
    def order(self) -> ArrayEncodingType:
        """Packing order used by the encoded blocks."""
        return self._order

    @property
    def dtype(self) -> str:
        """Tensor dtype name, currently the concrete class name."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of logical tensor dimensions."""
        return len(self._original_shape)

    @property
    def ncols(self) -> int | None:
        """Number of padded columns for a matrix, or ``None`` for vectors."""
        if self.ndim == 2:
            return self._shape[1]
        return None

    @property
    def block_shape(self) -> tuple[int, ...]:
        """Logical shape of each encoded block."""
        return self._block_shape

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the block grid."""
        return self._grid_shape

    @property
    def block_ndim(self) -> int:
        """Number of dimensions inside each block."""
        return len(self._block_shape)

    @property
    def block_size(self) -> int:
        """Number of logical entries represented by one block."""
        return prod(self._block_shape)

    @property
    def num_blocks(self) -> int:
        """Total number of encoded blocks."""
        return len(self._data)

    @property
    def is_vector(self) -> bool:
        """Whether this block tensor is logically 1-D."""
        return self.ndim == 1

    @property
    def is_matrix(self) -> bool:
        """Whether this block tensor is logically 2-D."""
        return self.ndim == 2

    @property
    def logical_size(self) -> int:
        """Number of entries in the unpadded logical tensor."""
        return prod(self._original_shape)

    @property
    def padded_size(self) -> int:
        """Number of entries in the padded logical tensor."""
        return prod(self._shape)

    @property
    def total_slot_capacity(self) -> int:
        """Total slot capacity across all blocks."""
        return self.num_blocks * self._batch_size

    @property
    def slot_utilization(self) -> float:
        """Fraction of total block slot capacity used by logical entries."""
        return self.logical_size / self.total_slot_capacity

    @property
    def padding_overhead(self) -> float:
        """Fraction of padded logical entries outside ``original_shape``."""
        return 1.0 - (self.logical_size / self.padded_size)

    @property
    def info(self) -> dict[str, Any]:
        """Return a dictionary of layout and storage metadata."""
        return {
            "dtype": self._dtype,
            "original_shape": self._original_shape,
            "shape": self._shape,
            "block_shape": self._block_shape,
            "grid_shape": self._grid_shape,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "batch_size": self._batch_size,
            "logical_size": self.logical_size,
            "padded_size": self.padded_size,
            "total_slot_capacity": self.total_slot_capacity,
            "slot_utilization": self.slot_utilization,
            "padding_overhead": self.padding_overhead,
            "order": self._order,
            "encrypted": self.is_encrypted,
        }

    # ------------------------------------------------------------------
    # Block-grid indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_index(index: int, size: int, name: str) -> int:
        """Normalize a possibly negative Python index."""
        try:
            index = operator_index(index)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer; got {index!r}.") from exc

        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(f"{name} {index} out of range for size={size}.")
        return index

    def _block_offset(self, grid_index: tuple[int, ...]) -> int:
        """Convert a block-grid index to a flat row-major block offset."""
        if len(grid_index) != len(self._grid_shape):
            raise IndexError(
                f"Expected {len(self._grid_shape)} block index value(s), got {len(grid_index)}."
            )

        if len(self._grid_shape) == 1:
            i = self._normalize_index(grid_index[0], self._grid_shape[0], "Block index")
            return i

        i, j = grid_index
        rows, cols = self._grid_shape
        i = self._normalize_index(i, rows, "Block row")
        j = self._normalize_index(j, cols, "Block column")
        return i * cols + j

    def get_block(self, *grid_index) -> Any:
        """Return a block by block-grid index.

        The index may be passed either as separate integers, e.g.
        ``get_block(i, j)``, or as one tuple, e.g. ``get_block((i, j))``.
        """
        if len(grid_index) == 1 and isinstance(grid_index[0], tuple):
            grid_index = grid_index[0]
        return self._data[self._block_offset(tuple(grid_index))]

    def get_block_row(self, i: int) -> list[Any]:
        """Return all blocks in block row ``i``."""
        if self.ndim != 2:
            raise ValueError("get_block_row is only valid for block matrices.")
        rows, cols = self._grid_shape
        i = self._normalize_index(i, rows, "Block row")
        start = i * cols
        return self._data[start : start + cols]

    def get_block_col(self, j: int) -> list[Any]:
        """Return all blocks in block column ``j``."""
        if self.ndim != 2:
            raise ValueError("get_block_col is only valid for block matrices.")
        rows, cols = self._grid_shape
        j = self._normalize_index(j, cols, "Block column")
        return [self._data[i * cols + j] for i in range(rows)]

    def iter_block_indices(self):
        """Yield block-grid indices in row-major order."""
        if self.ndim == 1:
            for i in range(self._grid_shape[0]):
                yield (i,)
        else:
            rows, cols = self._grid_shape
            for i in range(rows):
                for j in range(cols):
                    yield (i, j)

    def block_grid(self) -> list[Any]:
        """Return a nested block view for debugging and inspection.

        The returned containers are new Python lists, but block objects are not
        copied.
        """
        if self.ndim == 1:
            return list(self._data)
        rows, cols = self._grid_shape
        return [self._data[i * cols : (i + 1) * cols] for i in range(rows)]

    # ------------------------------------------------------------------
    # Logical indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the length of the first logical dimension."""
        return self._original_shape[0]

    def __getitem__(self, key):
        """Return a logical scalar entry by integer index.

        Negative integer indices follow Python indexing convention.

        Limitations
        -----------
        - Slice indexing is not implemented.
        - Only scalar logical entries are supported.
        - Actual extraction depends on whether the encoded block implements
          ``__getitem__``.
        """
        if isinstance(key, slice) or (
            isinstance(key, tuple) and any(isinstance(k, slice) for k in key)
        ):
            raise NotImplementedError(
                "Logical slicing is not implemented yet. "
                "Use get_block_row/get_block_col for block-level access."
            )
        return self._get_logical_entry(key)

    def get_entry(self, *key):
        """Return a logical scalar entry.

        This is a convenience wrapper around ``__getitem__`` that accepts either
        ``get_entry(i)`` for vectors or ``get_entry(i, j)`` for matrices.
        """
        if len(key) == 0:
            raise IndexError("BlockFHETensor does not support scalar indexing.")
        if len(key) == 1:
            return self[key[0]]
        return self[tuple(key)]

    def _locate_index(self, key) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Map a logical index to ``(block_index, local_index)``."""
        if self.ndim == 1:
            if isinstance(key, tuple):
                if len(key) != 1:
                    raise IndexError(f"Expected one index for vector, got {key}.")
                key = key[0]

            i = self._normalize_index(key, self._original_shape[0], "Index")
            b = self._block_shape[0]
            return (i // b,), (i % b,)

        if self.ndim == 2:
            if not isinstance(key, tuple) or len(key) != 2:
                raise IndexError(f"Expected two indices for matrix, got {key}.")

            i, j = key
            nrows, ncols = self._original_shape
            i = self._normalize_index(i, nrows, "Row index")
            j = self._normalize_index(j, ncols, "Column index")

            br, bc = self._block_shape
            return (i // br, j // bc), (i % br, j % bc)

        raise ValueError(f"Unsupported ndim={self.ndim}.")

    def _get_logical_entry(self, key):
        """Extract a logical entry from the selected encoded block.

        This helper performs index mapping only. The concrete block type is
        responsible for supporting local ``__getitem__`` extraction.
        """
        block_index, local_index = self._locate_index(key)
        block = self.get_block(*block_index)
        if not hasattr(block, "__getitem__"):
            raise NotImplementedError(
                "Block object does not support __getitem__. "
                f"block_type={type(block).__name__}, local_index={local_index}."
            )
        try:
            return block[local_index]
        except Exception as exc:
            raise NotImplementedError(
                "Logical entry extraction from an encoded block failed. "
                f"Unsupported local_index={local_index}."
            ) from exc

    # ------------------------------------------------------------------
    # Operator dispatch
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Return ``self + other`` through tensor dispatch."""
        return self.__tensor_function__("add", (self, other))

    def __radd__(self, other):
        """Return ``other + self`` through tensor dispatch."""
        return self.__tensor_function__("add", (other, self))

    def __sub__(self, other):
        """Return ``self - other`` through tensor dispatch."""
        return self.__tensor_function__("subtract", (self, other))

    def __rsub__(self, other):
        """Return ``other - self`` through tensor dispatch."""
        return self.__tensor_function__("subtract", (other, self))

    def __mul__(self, other):
        """Return element-wise ``self * other`` through tensor dispatch."""
        return self.__tensor_function__("multiply", (self, other))

    def __rmul__(self, other):
        """Return element-wise ``other * self`` through tensor dispatch."""
        return self.__tensor_function__("multiply", (other, self))

    def __matmul__(self, other):
        """Return ``self @ other`` through tensor dispatch."""
        return self.__tensor_function__("matmul", (self, other))

    def __pow__(self, exp):
        """Return matrix power ``self ** exp`` through tensor dispatch."""
        return self.__tensor_function__("power", (self, exp))

    @property
    def T(self):
        """Transpose view/result, equivalent to ``self.transpose()``."""
        return self.transpose()

    def sum(self, axis=None, keepdims: bool = False):
        """Sum block tensor entries through tensor dispatch."""
        return self.__tensor_function__("sum", (self,), {"axis": axis, "keepdims": keepdims})

    def cumsum(self, axis=None):
        """Compute cumulative sums through tensor dispatch."""
        return self.__tensor_function__("cumsum", (self,), {"axis": axis})

    def transpose(self):
        """Transpose the block tensor through tensor dispatch.

        For 1-D tensors, this follows NumPy behavior and returns the tensor
        unchanged.
        """
        return self.__tensor_function__("transpose", (self,))

    def cumulative_reduce(self, axis: int = 0, keepdims: bool = False):
        """Compute backend-specific cumulative reduction through dispatch.

        This is an FHE-specific operation, not a direct NumPy API equivalent.
        """
        return self.__tensor_function__(
            "cumulative_reduce", (self,), {"axis": axis, "keepdims": keepdims}
        )

    def __tensor_function__(self, func_name, args, kwargs=None, verbose: bool = False):
        """Dispatch tensor operations via the registry."""
        from openfhe_numpy.operations.dispatch import dispatch_tensor_function

        return dispatch_tensor_function(func_name, args, kwargs or {}, verbose=verbose)

    # ------------------------------------------------------------------
    # Copy / equality / repr
    # ------------------------------------------------------------------

    def clone(self, data: list[Any] | None = None) -> BlockFHETensor:
        """Return a shallow copy, optionally with new block data."""
        blocks = list(self._data) if data is None else list(data)
        if data is not None:
            for source, result in zip(self._data, blocks):
                if hasattr(result, "extra"):
                    result.extra.update(getattr(source, "extra", {}))

        return self.__class__(
            data=blocks,
            grid_shape=self._grid_shape,
            block_shape=self._block_shape,
            original_shape=self._original_shape,
            batch_size=self._batch_size,
            order=self._order,
        )

    def same_layout(self, other: Any, mode: Literal["logical", "physical"] = "logical") -> bool:
        """Return whether two block tensors share the requested layout."""
        if mode not in ("logical", "physical"):
            raise ValueError("mode must be 'logical' or 'physical'")

        if not isinstance(other, BlockFHETensor):
            return False

        return (
            self._batch_size == other.batch_size
            and self._order == other.order
            and self._block_shape == other.block_shape
            and self._grid_shape == other.grid_shape
            and (mode == "physical" or self._original_shape == other.original_shape)
            and all(
                left.same_layout(right, mode=mode) for left, right in zip(self._data, other.data)
            )
        )

    def is_standard_layout(self) -> bool:
        """Return whether child frames use the unexpanded block shape."""
        expected = (self._block_shape[0], 1) if self.ndim == 1 else self._block_shape
        return all(tuple(child.shape) == expected for child in self._data)

    def same_metadata(self, other: Any) -> bool:
        """Return ``True`` if layout metadata, encryption status, and dtype match."""
        return (
            self.same_layout(other)
            and getattr(other, "is_encrypted", None) == self.is_encrypted
            and getattr(other, "dtype", None) == self._dtype
        )

    def __eq__(self, other: Any) -> bool:
        """Return metadata-only equality.

        This does not compare encoded block values. For encrypted tensors, value
        equality is generally not directly observable without decryption.
        """
        return self.same_metadata(other)

    def __repr__(self) -> str:
        """Return a compact representation of block tensor metadata."""
        return (
            f"{self.__class__.__name__}("
            f"original_shape={self._original_shape}, "
            f"shape={self._shape}, "
            f"block_shape={self._block_shape}, "
            f"grid_shape={self._grid_shape}, "
            f"num_blocks={self.num_blocks}, "
            f"batch_size={self._batch_size}, "
            f"slot_utilization={self.slot_utilization:.3f}, "
            f"order={self._order})"
        )
