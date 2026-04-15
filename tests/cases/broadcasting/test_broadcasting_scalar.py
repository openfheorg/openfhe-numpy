import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SCALAR = 10
SIZES_VECTOR = [5, 16]
SIZES_MAT = [(3, 6), (1, 8), (16, 1)]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]
OPS = [
    ("add", lambda a, b: np.add(a, b), lambda a, b: onp.add(a, b)),
    ("sub", lambda a, b: np.subtract(a, b), lambda a, b: a - b),
    ("multiply", lambda a, b: np.multiply(a, b), lambda a, b: onp.multiply(a, b)),
]


def _run_scalar_vector_ops(test_case, tag, np_fn, fhe_fn):
    ckks_params = load_ckks_params()
    for p in ckks_params:
        batch_size = p["ringDim"] // 2

        cc, keys = gen_crypto_context(p)
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        try:
            for size in SIZES_VECTOR:
                if size > batch_size:
                    continue

                vector = generate_random_array(rows=size)
                onp.generate_broadcast_key(keys.secretKey, (), vector.shape)
                expected = np_fn(vector, SCALAR)

                for order_name, order_value in ORDERS:
                    for mode in MODES:
                        with test_case.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            mode=mode,
                            ringDim=p["ringDim"],
                        ):
                            result = None
                            ctv = None
                            cts = None
                            ctv_res = None

                            try:
                                ctv = onp.array(
                                    cc=cc,
                                    data=vector,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )
                                cts = onp.array(
                                    cc=cc,
                                    data=SCALAR,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )
                                ctv_res = fhe_fn(ctv, cts)
                                result = ctv_res.decrypt(keys.secretKey, unpack_type="original")
                                test_case.assertArrayClose(actual=result, expected=expected)
                            except Exception:
                                test_case._record_case(
                                    params={
                                        "case": "scalar_broadcasting",
                                        "size": size,
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"vector": vector},
                                    expected=expected,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctv, cts, ctv_res, result
                                gc.collect()
        finally:
            del cc, keys
            gc.collect()


def _run_scalar_matrix_ops(self, ops):
    tag, np_fn, fhe_fn = ops[0], ops[1], ops[2]
    ckks_params = load_ckks_params()
    for p in ckks_params:
        batch_size = p["ringDim"] // 2

        cc, keys = gen_crypto_context(p)
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        try:
            for rows, cols in SIZES_MAT:
                if onp.next_power_of_two(rows) * onp.next_power_of_two(cols) > batch_size:
                    continue

                onp.generate_broadcast_key(keys.secretKey, (), (rows, cols))
                matrix = generate_random_array(rows, cols)
                expected = np_fn(matrix, SCALAR)

                for order_name, order_value in ORDERS:
                    for mode in MODES:
                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=(rows, cols),
                            mode=mode,
                            ringDim=p["ringDim"],
                        ):
                            result = None
                            ctm = None
                            ctm_res = None

                            try:
                                ctm = onp.array(
                                    cc=cc,
                                    data=matrix,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode=mode,
                                    public_key=keys.publicKey,
                                )
                                ctm_res = fhe_fn(ctm, SCALAR)
                                result = ctm_res.decrypt(keys.secretKey, unpack_type="original")
                                self.assertArrayClose(actual=result, expected=expected)
                            except Exception:
                                self._record_case(
                                    params={
                                        "case": "scalar_broadcasting",
                                        "size": (rows, cols),
                                        "ringDim": p["ringDim"],
                                    },
                                    input_data={"matrix": matrix},
                                    expected=expected,
                                    result=result,
                                )
                                raise
                            finally:
                                del ctm, ctm_res, result
                                gc.collect()
        finally:
            del cc, keys
            gc.collect()


class TestScalarVectorOps(MainUnittest):
    def test_add(self):
        _run_scalar_vector_ops(self, *OPS[0])

    def test_sub(self):
        _run_scalar_vector_ops(self, *OPS[1])

    def test_multiply(self):
        _run_scalar_vector_ops(self, *OPS[2])


class TestScalarMatrixOps_Add(MainUnittest):
    def test_add(self):
        _run_scalar_matrix_ops(self, OPS[0])

    def test_sub(self):
        _run_scalar_matrix_ops(self, OPS[1])

    def test_multiply(self):
        _run_scalar_matrix_ops(self, OPS[2])
