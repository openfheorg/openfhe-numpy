import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


# ckks_params_block.csv uses ringDim=512, so the CKKS slot capacity is 256.
# A vector of length 257 forces block packing because 257 > 256.
SIZES = [257]
ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES = ["zero"]
BLOCK_PARAMS_CSV = CRYPTO_PARAMS_DIR / "ckks_params_block.csv"


def _ensure_depth(params: dict, min_depth: int = 4) -> dict:
    p = params.copy()
    if p.get("multiplicativeDepth", 0) < min_depth:
        p["multiplicativeDepth"] = min_depth
    return p


def _make_block_vector(
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


class BlockVectorBinaryTestBase:
    """Base class for one-operation block vector binary tests."""

    op_name = None
    min_depth = 4
    needs_mult_key = False
    needs_sum_key = False

    def np_fn(self, a, b):
        raise NotImplementedError

    def fhe_fn(self, a, b):
        raise NotImplementedError

    def test_binary_operation(self):
        if self.__class__ is BlockVectorBinaryTestBase:
            self.skipTest("base class")

        ckks_params = load_ckks_params(BLOCK_PARAMS_CSV)

        for _, p in enumerate(ckks_params):
            params = _ensure_depth(p, self.min_depth)
            slot_capacity = params["ringDim"] // 2
            batch_size = slot_capacity

            if batch_size <= 0:
                continue

            cc, keys = gen_crypto_context(params)

            if self.needs_mult_key:
                cc.EvalMultKeyGen(keys.secretKey)

            if self.needs_sum_key:
                cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    self.assertGreater(size, slot_capacity)

                    a = generate_random_array(rows=size)
                    b = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        expected = self.np_fn(a, b)

                        for mode in MODES:
                            result = None
                            ctv_a = None
                            ctv_b = None
                            ctv_res = None

                            try:
                                ctv_a = _make_block_vector(
                                    cc=cc,
                                    data=a,
                                    batch_size=batch_size,
                                    order=order_value,
                                    mode=mode,
                                    fhe_type="C",
                                    public_key=keys.publicKey,
                                )
                                ctv_b = _make_block_vector(
                                    cc=cc,
                                    data=b,
                                    batch_size=batch_size,
                                    order=order_value,
                                    mode=mode,
                                    fhe_type="C",
                                    public_key=keys.publicKey,
                                )

                                ctv_res = self.fhe_fn(ctv_a, ctv_b)
                                result = ctv_res.decrypt(
                                    keys.secretKey,
                                    unpack_type="original",
                                )

                                self.assertArrayClose(actual=result, expected=expected)

                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "block_vector_binary",
                                        "op": self.op_name,
                                        "size": size,
                                        "vector_slots": size,
                                        "slot_capacity": slot_capacity,
                                        "order": order_name,
                                        "mode": mode,
                                        "batch_size": batch_size,
                                        "ringDim": params["ringDim"],
                                    },
                                    input_data={"a": a, "b": b},
                                    expected=expected,
                                    result=result,
                                )
                                raise

                            finally:
                                del ctv_a, ctv_b, ctv_res, result
                                gc.collect()

            finally:
                del cc, keys
                gc.collect()


class TestBlockVectorAdd(BlockVectorBinaryTestBase, MainUnittest):
    op_name = "add"
    min_depth = 2

    def np_fn(self, a, b):
        return a + b

    def fhe_fn(self, a, b):
        return onp.add(a, b)


class TestBlockVectorSubtract(BlockVectorBinaryTestBase, MainUnittest):
    op_name = "subtract"
    min_depth = 2

    def np_fn(self, a, b):
        return a - b

    def fhe_fn(self, a, b):
        return onp.subtract(a, b)


class TestBlockVectorMultiply(BlockVectorBinaryTestBase, MainUnittest):
    op_name = "multiply"
    min_depth = 4
    needs_mult_key = True

    def np_fn(self, a, b):
        return a * b

    def fhe_fn(self, a, b):
        return onp.multiply(a, b)


class TestBlockVectorDot(BlockVectorBinaryTestBase, MainUnittest):
    op_name = "dot"
    min_depth = 4
    needs_mult_key = True
    needs_sum_key = True

    def np_fn(self, a, b):
        return np.dot(a, b)

    def fhe_fn(self, a, b):
        return onp.dot(a, b)


class TestBlockVectorMatmul(BlockVectorBinaryTestBase, MainUnittest):
    op_name = "matmul"
    min_depth = 4
    needs_mult_key = True
    needs_sum_key = True

    def np_fn(self, a, b):
        return np.dot(a, b)

    def fhe_fn(self, a, b):
        return a @ b
