import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OPS = [
    ("sum", lambda a: np.sum(a), lambda a: onp.sum(a)),
    ("sum_axis_0", lambda a: np.sum(a, axis=0), lambda a: onp.sum(a, axis=0)),
    ("sum_axis_1", lambda a: np.sum(a, axis=1), lambda a: onp.sum(a, axis=1)),
]


def _prepare_context(_cc, keys):
    onp.gen_sum_key(keys.secretKey)


def _prepare_tensor(keys, tensor):
    onp.attach_block_sum_keys(tensor, keys.secretKey)


class TestBlockMatrixSum(BlockMatrixTestFramework):
    def test_sum(self):
        self._run_unary(
            *OPS[0],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )


class TestBlockMatrixSumAxis0(BlockMatrixTestFramework):
    def test_sum_axis_0(self):
        self._run_unary(
            *OPS[1],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )


class TestBlockMatrixSumAxis1(BlockMatrixTestFramework):
    def test_sum_axis_1(self):
        self._run_unary(
            *OPS[2],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )
