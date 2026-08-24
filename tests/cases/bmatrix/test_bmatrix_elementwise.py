import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OPS = [
    ("add", lambda a, b: np.add(a, b), lambda a, b: onp.add(a, b)),
    ("subtract", lambda a, b: np.subtract(a, b), lambda a, b: onp.subtract(a, b)),
    ("multiply", lambda a, b: np.multiply(a, b), lambda a, b: onp.multiply(a, b)),
]


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)


class TestBlockMatrixAdd(BlockMatrixTestFramework):
    def test_add(self):
        self._run_binary(*OPS[0], prepare_context=_prepare_context)


class TestBlockMatrixSubtract(BlockMatrixTestFramework):
    def test_subtract(self):
        self._run_binary(*OPS[1], prepare_context=_prepare_context)


class TestBlockMatrixMultiply(BlockMatrixTestFramework):
    def test_multiply(self):
        self._run_binary(*OPS[2], prepare_context=_prepare_context)
