import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OP = (
    "inner_product",
    lambda lhs, rhs: np.inner(lhs, rhs),
    lambda lhs, rhs: onp.dot(lhs, rhs),
)


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)


class TestBlockVectorInnerProductRowMajor(BlockMatrixTestFramework):
    def test_inner_product(self):
        self._run_vector_inner_product(
            *OP,
            order_name="row_major",
            order=onp.ROW_MAJOR,
            prepare_context=_prepare_context,
        )


class TestBlockVectorInnerProductColMajor(BlockMatrixTestFramework):
    def test_inner_product(self):
        self._run_vector_inner_product(
            *OP,
            order_name="col_major",
            order=onp.COL_MAJOR,
            prepare_context=_prepare_context,
        )
