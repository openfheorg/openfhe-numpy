import numpy as np

import openfhe_numpy as onp

from ._framework import BlockMatrixTestFramework


OPS = [
    ("add", np.add, onp.add),
    ("subtract", np.subtract, onp.subtract),
    ("multiply", np.multiply, onp.multiply),
]


def _prepare_context(cc, keys):
    cc.EvalMultKeyGen(keys.secretKey)


def _prepare_operands(keys, vector, matrix):
    onp.generate_block_broadcast_key(
        keys.secretKey,
        vector,
        matrix,
    )


class TestBlockMatrixBroadcastAdd(BlockMatrixTestFramework):
    def test_add(self):
        self._run_broadcast(
            *OPS[0],
            prepare_context=_prepare_context,
            prepare_operands=_prepare_operands,
        )


class TestBlockMatrixBroadcastSubtract(BlockMatrixTestFramework):
    def test_subtract(self):
        self._run_broadcast(
            *OPS[1],
            prepare_context=_prepare_context,
            prepare_operands=_prepare_operands,
        )


class TestBlockMatrixBroadcastMultiply(BlockMatrixTestFramework):
    def test_multiply(self):
        self._run_broadcast(
            *OPS[2],
            prepare_context=_prepare_context,
            prepare_operands=_prepare_operands,
        )
