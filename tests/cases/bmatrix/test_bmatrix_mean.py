import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OPS = [
    ("mean", lambda a: np.mean(a), lambda a: onp.mean(a)),
    ("mean_axis_0", lambda a: np.mean(a, axis=0), lambda a: onp.mean(a, axis=0)),
    ("mean_axis_1", lambda a: np.mean(a, axis=1), lambda a: onp.mean(a, axis=1)),
]


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)
    onp.gen_sum_key(keys.secretKey)


def _prepare_tensor(keys, tensor):
    onp.attach_block_sum_keys(tensor, keys.secretKey)


class TestBlockMatrixMean(BlockMatrixTestFramework):
    def test_mean(self):
        self._run_unary(
            *OPS[0],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )


class TestBlockMatrixMeanAxis0(BlockMatrixTestFramework):
    def test_mean_axis_0(self):
        self._run_unary(
            *OPS[1],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )


class TestBlockMatrixMeanAxis1(BlockMatrixTestFramework):
    def test_mean_axis_1(self):
        self._run_unary(
            *OPS[2],
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor,
        )
