import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OP = (
    "matmul_matrix_vector",
    lambda matrix, vector: np.matmul(matrix, vector),
    lambda matrix, vector: onp.matmul(matrix, vector),
)


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)


def _prepare_operands(keys, matrix, _vector):
    onp.attach_block_matvec_keys(matrix, keys.secretKey)


class TestBlockMatrixVectorRowMajor(BlockMatrixTestFramework):
    def test_matmul_matrix_vector(self):
        self._run_matrix_vector(
            *OP,
            matrix_order_name="row_major",
            matrix_order=onp.ROW_MAJOR,
            vector_order_name="col_major",
            vector_order=onp.COL_MAJOR,
            prepare_context=_prepare_context,
            prepare_operands=_prepare_operands,
        )


class TestBlockMatrixVectorColMajor(BlockMatrixTestFramework):
    def test_matmul_matrix_vector(self):
        self._run_matrix_vector(
            *OP,
            matrix_order_name="col_major",
            matrix_order=onp.COL_MAJOR,
            vector_order_name="row_major",
            vector_order=onp.ROW_MAJOR,
            prepare_context=_prepare_context,
            prepare_operands=_prepare_operands,
        )
