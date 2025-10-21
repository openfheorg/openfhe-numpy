import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SCALAR = 7.9
SIZES = [5, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES = ["zero"]


class TestVectorUnaryOps(MainUnittest):
    """Test class for unary matrix operations"""

    def test_unary_operations(self):
        ops = [
            ("transpose", lambda x: x.T, lambda x: onp.transpose(x)),
            ("scalar_mul", lambda x, s: x * s, lambda x, s: x * s),
            ("sum", lambda x: np.sum(x), lambda x: onp.sum(x)),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    # generate vector with dimension (size)
                    a = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                size=size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None

                                try:
                                    ctv = onp.array(
                                        cc=cc,
                                        data=a,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    onp.gen_transpose_keys(keys.secretKey, ctv)

                                    if tag == "scalar_mul":
                                        expected = np_fn(a, SCALAR)
                                        ctv_res = fhe_fn(ctv, SCALAR)
                                    else:
                                        expected = np_fn(a)
                                        ctv_res = fhe_fn(ctv)

                                    # decrypt and compare
                                    result = ctv_res.decrypt(
                                        keys.secretKey, unpack_type="original"
                                    )

                                    self.assertArrayClose(
                                        actual=result, expected=expected
                                    )
                                except Exception as e:
                                    self._record_case(
                                        params={
                                            "case": "rowmajor_colmajor",
                                            "size": size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


class TestVectorBinaryOps(MainUnittest):
    def test_binary_operations(self):
        ops = [
            ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
            ("dot", lambda x, y: np.dot(x, y), lambda a, b: onp.dot(a, b)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    a = generate_random_array(rows=size)
                    b = generate_random_array(rows=size)

                    for order_name, order_value in ORDERS:
                        expected = np_fn(a, b)
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                size=size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                try:
                                    # encrypt matrices
                                    ctv_a = onp.array(
                                        cc=cc,
                                        data=a,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode="zero",
                                        public_key=keys.publicKey,
                                    )
                                    ctv_b = onp.array(
                                        cc=cc,
                                        data=b,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode="zero",
                                        public_key=keys.publicKey,
                                    )

                                    ctv_res = fhe_fn(ctv_a, ctv_b)

                                    # decrypt and compare
                                    result = ctv_res.decrypt(
                                        keys.secretKey, unpack_type="original"
                                    )

                                    self.assertArrayClose(
                                        actual=result, expected=expected
                                    )

                                except Exception as e:
                                    self._record_case(
                                        params={
                                            "case": "rowmajor_colmajor",
                                            "size": size,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"a": a, "b": b},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise


if __name__ == "__main__":
    TestVectorBinaryOps.run_test_summary()
    TestVectorUnaryOps.run_test_summary()
