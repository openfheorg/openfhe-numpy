from __future__ import annotations

from typing import Any

from openfhe_numpy.tensor.block_tensor import BlockFHETensor
from openfhe_numpy.tensor.block_ctarray import BlockCTArray
from openfhe_numpy.utils.typecheck import Number
from openfhe_numpy.utils.errors import (
    ONPIncompatibleShapeError,
    ONPNotImplementedError,
)


_BINARY_OPS = frozenset({"add", "subtract", "multiply"})


def _assert_same_block_layout(
    a: BlockFHETensor,
    b: BlockFHETensor,
    op_name: str,
) -> None:
    """Verify that two block tensors share identical block layout."""
    if not a.same_layout(b):
        raise ONPIncompatibleShape(
            a.original_shape,
            b.original_shape,
            f"Block {op_name} requires identical block layout.",
        )


def _resolve_result_type(
    a: BlockFHETensor,
    b: BlockFHETensor | None = None,
) -> type[BlockFHETensor]:
    """Choose result class for a block operation.

    If either operand is encrypted, the result should be encrypted.
    Otherwise, preserve the left operand type.
    """
    if b is not None and isinstance(b, BlockFHETensor):
        if getattr(a, "is_encrypted", False):
            return type(a)
        if getattr(b, "is_encrypted", False):
            return type(b)

    return type(a)


def _build_block_result(
    ref: BlockFHETensor,
    blocks: list[Any],
    result_cls: type[BlockFHETensor] | None = None,
) -> BlockFHETensor:
    """Wrap a flat block list into a new block tensor preserving layout."""
    cls = result_cls or type(ref)

    return cls(
        data=blocks,
        grid_shape=ref.grid_shape,
        block_shape=ref.block_shape,
        original_shape=ref.original_shape,
        batch_size=ref.batch_size,
        order=ref.order,
    )


def _eval_block_binary(
    a: BlockFHETensor,
    b: BlockFHETensor,
    op_name: str,
) -> BlockFHETensor:
    """Element-wise binary operation between two block tensors."""
    if op_name not in _BINARY_OPS:
        raise NotImplementedError(f"Unsupported block binary operation: {op_name!r}")

    _assert_same_block_layout(a, b, op_name)

    if op_name == "add":
        blocks = [x + y for x, y in zip(a.data, b.data)]
    elif op_name == "subtract":
        blocks = [x - y for x, y in zip(a.data, b.data)]
    else:
        blocks = [x * y for x, y in zip(a.data, b.data)]

    return _build_block_result(a, blocks, _resolve_result_type(a, b))


def _eval_block_scalar(
    a: BlockFHETensor,
    scalar: Number,
    op_name: str,
) -> BlockFHETensor:
    """Element-wise operation: block tensor op scalar."""
    if op_name not in _BINARY_OPS:
        raise NotImplementedError(f"Unsupported block-scalar operation: {op_name!r}")

    if op_name == "add":
        blocks = [blk + scalar for blk in a.data]
    elif op_name == "subtract":
        blocks = [blk - scalar for blk in a.data]
    else:
        blocks = [blk * scalar for blk in a.data]

    return _build_block_result(a, blocks)


def _eval_scalar_block(
    scalar: Number,
    a: BlockFHETensor,
    op_name: str,
) -> BlockFHETensor:
    """Element-wise operation: scalar op block tensor."""
    if op_name not in _BINARY_OPS:
        raise NotImplementedError(f"Unsupported scalar-block operation: {op_name!r}")

    if op_name == "add":
        blocks = [scalar + blk for blk in a.data]
    elif op_name == "subtract":
        blocks = [scalar - blk for blk in a.data]
    else:
        blocks = [scalar * blk for blk in a.data]

    return _build_block_result(a, blocks)


# ------------------------------------------------------------------------------
# Linear Algebra Operations
# ------------------------------------------------------------------------------
def _sum_ctarray_terms(terms):
    """Add a non-empty list of CTArray terms."""
    terms = list(terms)

    if len(terms) == 0:
        raise ONPNotImplementedError("Cannot sum an empty list of CTArray terms.")

    acc = terms[0]
    for term in terms[1:]:
        acc = acc + term

    return acc


def _assert_block_vector_dot_compatible(a, b):
    """Validate block vector dot-product compatibility."""
    if a.ndim != 1 or b.ndim != 1:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            "Block dot currently requires two block vectors.",
        )

    if a.original_shape != b.original_shape:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            "Block vector dot requires equal logical vector lengths.",
        )

    if a.block_shape != b.block_shape:
        raise ONPIncompatibleShapeError(
            a.block_shape,
            b.block_shape,
            "Block vector dot requires equal block_shape.",
        )

    if a.grid_shape != b.grid_shape:
        raise ONPIncompatibleShapeError(
            a.grid_shape,
            b.grid_shape,
            "Block vector dot requires equal grid_shape.",
        )

    if a.batch_size != b.batch_size:
        raise ONPIncompatibleShapeError(
            (a.batch_size,),
            (b.batch_size,),
            "Block vector dot requires equal batch_size.",
        )


def _eval_block_dot(a, b):
    """Compute inner product of two block vectors.

    If a and b are split into blocks,

        a = [a_0, ..., a_{t-1}]
        b = [b_0, ..., b_{t-1}]

    then

        <a, b> = sum_i <a_i, b_i>.

    Returns
    -------
    CTArray
        Scalar encrypted result.
    """
    _assert_block_vector_dot_compatible(a, b)

    terms = []

    for x_block, y_block in zip(a.data, b.data):
        # CTArray @ CTArray already performs inner product for vector blocks.
        terms.append(x_block @ y_block)

    return _sum_ctarray_terms(terms)


# [SQUARE] Matrix x Matrix Multiplication
def _assert_block_matmul_compatible(a, b):
    """Validate block matrix-matrix multiplication compatibility.

    This first implementation supports square matrix blocks only:

        block_shape = (s, s)

    because the underlying CTArray matrix multiplication uses the existing
    square packed-block multiplication path.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            "Block matrix multiplication currently requires two block matrices.",
        )

    if a.original_shape[1] != b.original_shape[0]:
        raise ONPIncompatibleShapeError(
            a.original_shape,
            b.original_shape,
            "Block matrix multiplication dimension mismatch.",
        )

    if a.grid_shape[1] != b.grid_shape[0]:
        raise ONPIncompatibleShapeError(
            a.grid_shape,
            b.grid_shape,
            "Block matrix multiplication requires matching inner block dimension.",
        )

    if a.block_shape != b.block_shape:
        raise ONPIncompatibleShapeError(
            a.block_shape,
            b.block_shape,
            "Block matrix multiplication currently requires equal block_shape.",
        )

    if len(a.block_shape) != 2:
        raise ONPIncompatibleShapeError(
            a.block_shape,
            b.block_shape,
            "Block matrix multiplication requires 2-D matrix blocks.",
        )

    br, bc = a.block_shape
    if br != bc:
        raise ONPNotImplementedError(
            "Block matrix multiplication currently supports only square blocks."
        )

    if a.batch_size != b.batch_size:
        raise ONPIncompatibleShapeError(
            (a.batch_size,),
            (b.batch_size,),
            "Block matrix multiplication requires equal batch_size.",
        )

    if a.order != b.order:
        raise ONPIncompatibleShapeError(
            (a.order,),
            (b.order,),
            "Block matrix multiplication currently requires equal packing order.",
        )


def _eval_block_matmat(a, b):
    """Compute block matrix-matrix multiplication.

    For block matrices A and B:

        C[i, j] = sum_k A[i, k] @ B[k, j]

    Returns
    -------
    BlockCTArray
        Encrypted block matrix result.
    """
    _assert_block_matmul_compatible(a, b)

    out_grid_shape = (a.grid_shape[0], b.grid_shape[1])
    out_original_shape = (a.original_shape[0], b.original_shape[1])

    out_blocks = []

    for i in range(out_grid_shape[0]):
        for j in range(out_grid_shape[1]):
            terms = []

            for k in range(a.grid_shape[1]):
                a_ik = a.get_block(i, k)
                b_kj = b.get_block(k, j)

                # Uses existing CTArray matrix multiplication on each square block.
                terms.append(a_ik @ b_kj)

            out_blocks.append(_sum_ctarray_terms(terms))

    return BlockCTArray(
        data=out_blocks,
        grid_shape=out_grid_shape,
        block_shape=a.block_shape,
        original_shape=out_original_shape,
        batch_size=a.batch_size,
        order=a.order,
    )


def _eval_block_matmul(a, b):
    """Dispatch block matmul by block tensor ranks."""
    if a.ndim == 1 and b.ndim == 1:
        return _eval_block_dot(a, b)

    if a.ndim == 2 and b.ndim == 2:
        return _eval_block_matmat(a, b)

    raise ONPNotImplementedError(
        "Block matmul currently supports only vector-vector dot "
        "and matrix-matrix multiplication. "
        "Block matrix-vector multiplication is not implemented yet."
    )
