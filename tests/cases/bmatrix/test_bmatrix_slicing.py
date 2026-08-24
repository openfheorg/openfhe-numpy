import openfhe_numpy as onp

from ._framework import BLOCK_SHAPE, BlockMatrixTestFramework


OPS = [
    ("slice_rows_from_1", lambda a: a[1:], lambda a: a[1:]),
    ("slice_submatrix", lambda a: a[1:3, 1:3], lambda a: a[1:3, 1:3]),
    ("slice_single_row", lambda a: a[0], lambda a: a[0]),
    ("slice_single_column", lambda a: a[:, 0], lambda a: a[:, 0]),
    ("slice_step_2", lambda a: a[::2, ::2], lambda a: a[::2, ::2]),
]


def _prepare_context(_cc, keys):
    onp.generate_slicing_key(
        keys.secretKey,
        BLOCK_SHAPE,
        physical_shape=BLOCK_SHAPE,
    )


class TestBlockMatrixSliceRowsFrom1(BlockMatrixTestFramework):
    def test_slice_rows_from_1(self):
        self._run_unary(*OPS[0], prepare_context=_prepare_context)


class TestBlockMatrixSliceSubmatrix(BlockMatrixTestFramework):
    def test_slice_submatrix(self):
        self._run_unary(*OPS[1], prepare_context=_prepare_context)


class TestBlockMatrixSliceSingleRow(BlockMatrixTestFramework):
    def test_slice_single_row(self):
        self._run_unary(*OPS[2], prepare_context=_prepare_context)


class TestBlockMatrixSliceSingleColumn(BlockMatrixTestFramework):
    def test_slice_single_column(self):
        self._run_unary(*OPS[3], prepare_context=_prepare_context)


class TestBlockMatrixSliceStep2(BlockMatrixTestFramework):
    def test_slice_step_2(self):
        self._run_unary(*OPS[4], prepare_context=_prepare_context)
