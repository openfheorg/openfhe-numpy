from __future__ import annotations

from math import prod
from typing import Any, Generic, TypeVar

from openfhe_numpy.openfhe_numpy import ArrayEncodingType

from .tensor import BaseTensor


TPL = TypeVar("TPL")


class BlockFHETensor(BaseTensor, Generic[TPL]):
    """
    Low-level block tensor.

    `data` must be a flat row-major list of encoded blocks.
    User-facing construction should be handled by block_array(...).
    """

    tensor_priority = 30
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

    @property
    def ncols(self) -> int:
        if self.ndim == 2:
            return self._shape[1]
        return None

    @staticmethod
    def _validate_metadata(
        data: list[Any],
        block_shape: tuple[int, ...],
        original_shape: tuple[int, ...],
        batch_size: int,
        grid_shape: tuple[int, ...],
    ) -> None:
        """Validate structural metadata."""
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

        padded_shape = tuple(
            grid_dim * block_dim for grid_dim, block_dim in zip(grid_shape, block_shape)
        )

        if any(orig > pad for orig, pad in zip(original_shape, padded_shape)):
            raise ValueError(
                f"original_shape={original_shape} cannot exceed padded_shape={padded_shape}."
            )

    # ------------------------------------------------------------------
    # Storage / metadata
    # ------------------------------------------------------------------

    @property
    def data(self) -> list[Any]:
        """Flat row-major block list."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = value

    @property
    def original_shape(self) -> tuple[int, ...]:
        return self._original_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def order(self) -> ArrayEncodingType:
        return self._order

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._original_shape)

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self._block_shape

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self._grid_shape

    @property
    def block_ndim(self) -> int:
        return len(self._block_shape)

    @property
    def block_size(self) -> int:
        return prod(self._block_shape)

    @property
    def num_blocks(self) -> int:
        return len(self._data)

    @property
    def is_encrypted(self) -> bool:
        return "CT" in self._dtype

    @property
    def is_vector(self) -> bool:
        return self.ndim == 1

    @property
    def is_matrix(self) -> bool:
        return self.ndim == 2

    @property
    def logical_size(self) -> int:
        return prod(self._original_shape)

    @property
    def padded_size(self) -> int:
        return prod(self._shape)

    @property
    def total_slot_capacity(self) -> int:
        return self.num_blocks * self._batch_size

    @property
    def slot_utilization(self) -> float:
        return self.logical_size / self.total_slot_capacity

    @property
    def padding_overhead(self) -> float:
        return 1.0 - (self.logical_size / self.padded_size)

    @property
    def info(self) -> dict[str, Any]:
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
    def _unravel_block_offset(
        offset: int,
        grid_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Convert flat block offset to block-grid index."""
        if len(grid_shape) == 1:
            return (offset,)

        rows, cols = grid_shape
        return (offset // cols, offset % cols)

    def _block_offset(self, grid_index: tuple[int, ...]) -> int:
        """Convert block-grid index to flat block offset."""
        if len(grid_index) != len(self._grid_shape):
            raise IndexError(
                f"Expected {len(self._grid_shape)} block index value(s), got {len(grid_index)}."
            )

        if len(self._grid_shape) == 1:
            i = grid_index[0]
            n = self._grid_shape[0]

            if not isinstance(i, int):
                raise TypeError(f"Block index must be an integer; got {grid_index}.")

            if i < 0:
                i += n

            if i < 0 or i >= n:
                raise IndexError(
                    f"Block index {grid_index} out of range for grid_shape={self._grid_shape}."
                )

            return i

        i, j = grid_index
        rows, cols = self._grid_shape

        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError(f"Block indices must be integers; got {grid_index}.")

        if i < 0:
            i += rows

        if j < 0:
            j += cols

        if i < 0 or i >= rows or j < 0 or j >= cols:
            raise IndexError(
                f"Block index {grid_index} out of range for grid_shape={self._grid_shape}."
            )

        return i * cols + j

    def get_block(self, *grid_index):
        """Return block by block-grid index."""
        if len(grid_index) == 1 and isinstance(grid_index[0], tuple):
            grid_index = grid_index[0]

        offset = self._block_offset(tuple(grid_index))
        return self._data[offset]

    def get_block_row(self, i: int) -> list[Any]:
        """Return block row by block-grid index."""
        if self.ndim != 2:
            raise ValueError("get_block_row is only valid for block matrices.")

        rows, cols = self._grid_shape

        if i < 0:
            i += rows

        if i < 0 or i >= rows:
            raise IndexError(f"Block row {i} out of range for grid_shape={self._grid_shape}.")

        start = i * cols
        return self._data[start : start + cols]

    def get_block_col(self, j: int) -> list[Any]:
        """Return block column by block-grid index."""
        if self.ndim != 2:
            raise ValueError("get_block_col is only valid for block matrices.")

        rows, cols = self._grid_shape

        if j < 0:
            j += cols

        if j < 0 or j >= cols:
            raise IndexError(f"Block column {j} out of range for grid_shape={self._grid_shape}.")

        return [self._data[i * cols + j] for i in range(rows)]

    def iter_block_indices(self):
        """Iterate block-grid indices in row-major order."""
        if self.ndim == 1:
            for i in range(self._grid_shape[0]):
                yield (i,)
        else:
            rows, cols = self._grid_shape
            for i in range(rows):
                for j in range(cols):
                    yield (i, j)

    def block_grid(self) -> list[Any]:
        """Return nested block grid for debugging."""
        if self.ndim == 1:
            return list(self._data)

        rows, cols = self._grid_shape
        return [self._data[i * cols : (i + 1) * cols] for i in range(rows)]

    # ------------------------------------------------------------------
    # Tensor operation dispatch
    # ------------------------------------------------------------------

    def __add__(self, other):
        return self.__tensor_function__("add", (self, other))

    def __radd__(self, other):
        return self.__tensor_function__("add", (self, other))

    def __sub__(self, other):
        return self.__tensor_function__("subtract", (self, other))

    def __rsub__(self, other):
        return self.__tensor_function__("subtract", (other, self))

    def __mul__(self, other):
        return self.__tensor_function__("multiply", (self, other))

    def __rmul__(self, other):
        return self.__tensor_function__("multiply", (self, other))

    def __matmul__(self, other):
        return self.__tensor_function__("matmul", (self, other))

    def __pow__(self, exp):
        return self.__tensor_function__("pow", (self, exp))

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        return self.__tensor_function__("sum", (self,), {"axis": axis, "keepdims": keepdims})

    def transpose(self):
        return self.__tensor_function__("transpose", (self,))

    def __tensor_function__(self, func_name, args, kwargs=None, verbose: bool = False):
        from openfhe_numpy.operations.dispatch import dispatch_tensor_function

        return dispatch_tensor_function(func_name, args, kwargs or {}, verbose=verbose)

    # ------------------------------------------------------------------
    # Logical indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._original_shape[0]

    def __getitem__(self, key):
        """Return logical scalar entry."""
        if isinstance(key, slice) or (
            isinstance(key, tuple) and any(isinstance(k, slice) for k in key)
        ):
            raise NotImplementedError(
                "Logical slicing is not implemented yet. "
                "Use get_block_row/get_block_col for block-level access."
            )

        return self._get_logical_entry(key)

    def get_entry(self, *key):
        """Return logical scalar entry."""
        if len(key) == 0:
            raise IndexError("BlockFHETensor does not support scalar indexing.")

        if len(key) == 1:
            return self[key[0]]

        return self[tuple(key)]

    def _locate_index(self, key) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Map logical index to block index and local index."""
        if self.ndim == 1:
            if isinstance(key, tuple):
                if len(key) != 1:
                    raise IndexError(f"Expected one index for vector, got {key}.")
                i = key[0]
            else:
                i = key

            if not isinstance(i, int):
                raise TypeError("Only integer indexing is currently supported.")

            n = self._original_shape[0]

            if i < 0:
                i += n

            if i < 0 or i >= n:
                raise IndexError(f"Index {i} out of bounds for shape {self._original_shape}.")

            b = self._block_shape[0]
            return (i // b,), (i % b,)

        if self.ndim == 2:
            if not isinstance(key, tuple) or len(key) != 2:
                raise IndexError(f"Expected two indices for matrix, got {key}.")

            i, j = key

            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError("Only integer indexing is currently supported.")

            nrows, ncols = self._original_shape

            if i < 0:
                i += nrows

            if j < 0:
                j += ncols

            if i < 0 or i >= nrows or j < 0 or j >= ncols:
                raise IndexError(f"Index {(i, j)} out of bounds for shape {self._original_shape}.")

            br, bc = self._block_shape
            return (i // br, j // bc), (i % br, j % bc)

        raise ValueError(f"Unsupported ndim={self.ndim}.")

    def _get_logical_entry(self, key):
        """Locate block and return block-local entry."""
        block_index, local_index = self._locate_index(key)
        block = self.get_block(*block_index)
        return self._extract_entry_from_block(block, local_index)

    def _extract_entry_from_block(self, block, local_index):
        """Delegate scalar extraction to the block."""
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
    # Copy / equality / repr
    # ------------------------------------------------------------------

    def clone(self, data: list[Any] | None = None) -> "BlockFHETensor":
        """Return shallow copy."""
        return self.__class__(
            data=data if data is not None else list(self._data),
            grid_shape=self._grid_shape,
            block_shape=self._block_shape,
            original_shape=self._original_shape,
            batch_size=self._batch_size,
            order=self._order,
        )

    def same_layout(self, other) -> bool:
        """Return True if layout metadata matches."""
        return (
            hasattr(other, "block_shape")
            and hasattr(other, "grid_shape")
            and self._original_shape == other.original_shape
            and self._shape == other.shape
            and self._batch_size == other.batch_size
            and self._order == other.order
            and self._block_shape == other.block_shape
            and self._grid_shape == other.grid_shape
        )

    def same_metadata(self, other) -> bool:
        """Return True if metadata matches."""
        return self.same_layout(other) and self._dtype == other._dtype

    def __eq__(self, other) -> bool:
        """Metadata-only equality."""
        return self.same_metadata(other)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self._shape}, "
            f"original_shape={self._original_shape}, "
            f"block_shape={self._block_shape}, "
            f"grid_shape={self._grid_shape}, "
            f"num_blocks={self.num_blocks}, "
            f"batch_size={self._batch_size}, "
            f"slot_utilization={self.slot_utilization:.3f}, "
            f"padding_overhead={self.padding_overhead:.3f}, "
            f"order={self._order})"
        )
