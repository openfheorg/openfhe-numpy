import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SCALAR = 7.9
SIZES = [5, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES_UNARY = ["zero"]
MODES_BINARY = ["tile"]

OPS_UNARY = [
    ("transpose", lambda x: x.T, lambda x: onp.transpose(x)),
    ("scalar_mul", lambda x, s: x * s, lambda x, s: x * s),
    ("sum", lambda x: np.sum(x), lambda x: onp.sum(x)),
]
OPS_BINARY = [
    ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
    ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
    ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
    ("dot", lambda x, y: np.dot(x, y), lambda a, b: onp.dot(a, b)),
]


class TestMatrixUnaryOps(MainUnittest):
    def test_transpose(self):
        self._run(*OPS_UNARY[0])

    def test_scalar_mul(self):
        self._run(*OPS_UNARY[1])

    def test_sum(self):
        self._run(*OPS_UNARY[2])

    def _run(self, tag, np_fn, fhe_fn):

        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)

                    for order_name, order_value in ORDERS:
                        for mode in MODES_UNARY:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                size=size,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm_a = None
                                ctm_res = None
                                try:
                                    ctm_a = onp.array(
                                        cc=cc,
                                        data=A,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    if tag == "scalar_mul":
                                        expected = np_fn(A, SCALAR)
                                        ctm_res = fhe_fn(ctm_a, SCALAR)
                                    elif tag == "transpose":
                                        onp.gen_transpose_keys(keys.secretKey, ctm_a)
                                        expected = np_fn(A)
                                        ctm_res = fhe_fn(ctm_a)
                                    else:  # sum
                                        expected = np_fn(A)
                                        ctm_res = fhe_fn(ctm_a)

                                    result = ctm_res.decrypt(keys.secretKey, unpack_type="original")
                                    self.assertArrayClose(actual=result, expected=expected)
                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "matrix_unary",
                                            "op": tag,
                                            "size": size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"A": A},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise
                                finally:
                                    del ctm_a, ctm_res, result
                                    gc.collect()
            finally:
                del cc, keys
                gc.collect()


class TestMatrixBinaryOps(MainUnittest):
    def test_add(self):
        self._run(*OPS_BINARY[0])

    def test_sub(self):
        self._run(*OPS_BINARY[1])

    def test_mult(self):
        self._run(*OPS_BINARY[2])

    def test_dot(self):
        self._run(*OPS_BINARY[2])

    def _run(self, tag, np_fn, fhe_fn):

        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            try:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    B = generate_random_array(rows=size, cols=size)
                    expected = np_fn(A, B)
                    if tag == "mul":
                        onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, size)

                    for order_name, order_value in ORDERS:
                        for mode in MODES_BINARY:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                size=size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm_a = None
                                ctm_b = None
                                ctm_res = None
                                try:
                                    ctm_a = onp.array(
                                        cc=cc,
                                        data=A,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )
                                    ctm_b = onp.array(
                                        cc=cc,
                                        data=B,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )
                                    ctm_res = fhe_fn(ctm_a, ctm_b)
                                    result = ctm_res.decrypt(keys.secretKey, unpack_type="original")
                                    self.assertArrayClose(actual=result, expected=expected)
                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "matrix_binary",
                                            "op": tag,
                                            "size": size,
                                            "ringDim": p["ringDim"],
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
