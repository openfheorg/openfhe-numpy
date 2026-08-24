import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OPS = [
    ("cumsum", lambda a: np.cumsum(a), lambda a: onp.cumsum(a), None),
    (
        "cumsum_axis_0",
        lambda a: np.cumsum(a, axis=0),
        lambda a: onp.cumsum(a, axis=0),
        0,
    ),
    (
        "cumsum_axis_1",
        lambda a: np.cumsum(a, axis=1),
        lambda a: onp.cumsum(a, axis=1),
        1,
    ),
]


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)


def _prepare_tensor(axis):
    def prepare(keys, tensor):
        onp.gen_block_cumsum_keys(
            keys.secretKey,
            tensor,
            axis=axis,
        )

    return prepare


class TestBlockMatrixCumsum(BlockMatrixTestFramework):
    def test_cumsum(self):
        tag, np_fn, fhe_fn, axis = OPS[0]
        self._run_unary(
            tag,
            np_fn,
            fhe_fn,
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor(axis),
        )


class TestBlockMatrixCumsumAxis0(BlockMatrixTestFramework):
    def test_cumsum_axis_0(self):
        tag, np_fn, fhe_fn, axis = OPS[1]
        self._run_unary(
            tag,
            np_fn,
            fhe_fn,
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor(axis),
        )


class TestBlockMatrixCumsumAxis1(BlockMatrixTestFramework):
    def test_cumsum_axis_1(self):
        tag, np_fn, fhe_fn, axis = OPS[2]
        self._run_unary(
            tag,
            np_fn,
            fhe_fn,
            prepare_context=_prepare_context,
            prepare_tensor=_prepare_tensor(axis),
        )
