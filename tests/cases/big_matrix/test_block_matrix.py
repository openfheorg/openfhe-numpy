import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SIZES = [5, 8, 16]
BLOCK_BATCH_SIZE = 4
ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES = ["zero"]


def _ensure_depth(params: dict, min_depth: int = 4) -> dict:
    p = params.copy()
    if p.get("multiplicativeDepth", 0) < min_depth:
        p["multiplicativeDepth"] = min_depth
    return p


def _make_block_matrix(
    cc,
    data,
    batch_size,
    order,
    mode,
    fhe_type,
    public_key=None,
):
    kwargs = {
        "cc": cc,
        "data": data,
        "batch_size": batch_size,
        "block_shape": None,
        "order": order,
        "mode": mode,
        "fhe_type": fhe_type,
    }

    if fhe_type == "C":
        kwargs["public_key"] = public_key

    return onp.block_array(**kwargs)


class BlockMatrixTestBase:
    """Base class for one-operation block matrix tests."""

    op_name = None
    min_depth = 4
    needs_mult_key = False
    needs_sum_key = False
    needs_global_sum_key = False

    def np_fn(self, a, b=None):
        raise NotImplementedError

    def fhe_fn(self, a, b=None):
        raise NotImplementedError

    def test_matrix_operation(self):
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            params = _ensure_depth(p, self.min_depth)
            max_batch_size = params["ringDim"] // 2
            batch_size = min(BLOCK_BATCH_SIZE, max_batch_size)

            if batch_size <= 0:
                continue

            cc, keys = gen_crypto_context(params)

            if self.needs_mult_key:
                cc.EvalMultKeyGen(keys.secretKey)

            if self.needs_sum_key:
                cc.EvalSumKeyGen(keys.secretKey)

            if self.needs_global_sum_key:
                onp.gen_sum_key(keys.secretKey)

            try:
                for size in SIZES:
                    if size * size > max_batch_size and batch_size > max_batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    B = generate_random_array(rows=size, cols=size)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            result = None
                            ctm_a = None
                            ctm_b = None
                            ctm_res = None

                            try:
                                ctm_a = _make_block_matrix(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    mode=mode,
                                    fhe_type="C",
                                    public_key=keys.publicKey,
                                )
                                ctm_b = _make_block_matrix(
                                    cc=cc,
                                    data=B,
                                    batch_size=batch_size,
                                    order=order_value,
                                    mode=mode,
                                    fhe_type="C",
                                    public_key=keys.publicKey,
                                )

                                expected = self.np_fn(A, B)
                                ctm_res = self.fhe_fn(ctm_a, ctm_b)
                                result = ctm_res.decrypt(
                                    keys.secretKey,
                                    unpack_type="original",
                                )

                                self.assertArrayClose(actual=result, expected=expected)

                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "block_matrix_operation",
                                        "op": self.op_name,
                                        "size": size,
                                        "order": order_name,
                                        "mode": mode,
                                        "batch_size": batch_size,
                                        "ringDim": params["ringDim"],
                                    },
                                    input_data={"A": A, "B": B},
                                    expected=expected,
                                    result=result,
                                )
                                raise

                            finally:
                                del ctm_a, ctm_b, ctm_res, result
                                gc.collect()

            finally:
                del cc, keys
                gc.collect()


class TestBlockMatrixAdd(BlockMatrixTestBase, MainUnittest):
    op_name = "add"
    min_depth = 2

    def np_fn(self, a, b=None):
        return a + b

    def fhe_fn(self, a, b=None):
        return onp.add(a, b)


class TestBlockMatrixSubtract(BlockMatrixTestBase, MainUnittest):
    op_name = "subtract"
    min_depth = 2

    def np_fn(self, a, b=None):
        return a - b

    def fhe_fn(self, a, b=None):
        return onp.subtract(a, b)


class TestBlockMatrixMultiply(BlockMatrixTestBase, MainUnittest):
    op_name = "multiply"
    min_depth = 4
    needs_mult_key = True

    def np_fn(self, a, b=None):
        return a * b

    def fhe_fn(self, a, b=None):
        return onp.multiply(a, b)


class TestBlockMatrixSum(BlockMatrixTestBase, MainUnittest):
    op_name = "sum"
    min_depth = 4
    needs_sum_key = True
    needs_global_sum_key = True

    def np_fn(self, a, b=None):
        return np.sum(a)

    def fhe_fn(self, a, b=None):
        return onp.sum(a)


class TestBlockMatrixMean(BlockMatrixTestBase, MainUnittest):
    op_name = "mean"
    min_depth = 4
    needs_mult_key = True
    needs_sum_key = True
    needs_global_sum_key = True

    def np_fn(self, a, b=None):
        return np.mean(a)

    def fhe_fn(self, a, b=None):
        return onp.mean(a)
