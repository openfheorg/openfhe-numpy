import numpy as np
import openfhe_numpy as onp
from core import *

sizes = [5, 8, 16]
orders = [("row_major", onp.ROW_MAJOR)]


class TestMatrixUnaryOps(MainUnittest):
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
            scalar = 7.2  # for scalar_mul

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in sizes:
                    if size > batch_size:
                        continue

                    for order_name, order_value in orders:
                        # plaintext input & expected
                        A = generate_random_array(rows=size)

                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            ringDim=p["ringDim"],
                        ):
                            result = expected = None

                            try:
                                ctv_a = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )

                                onp.gen_transpose_keys(keys.secretKey, ctv_a)

                                if tag == "scalar_mul":
                                    expected = np_fn(A, scalar)
                                    ctv_res = fhe_fn(ctv_a, scalar)
                                else:
                                    expected = np_fn(A)
                                    ctv_res = fhe_fn(ctv_a)

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
                                    input_data={"A": A},
                                    expected=expected,
                                    result=result,
                                )
                                raise


class TestMatrixBinaryOps(MainUnittest):
    def test_binary_operations(self):
        ops = [
            ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            # ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            # ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
            # ("dot", lambda x, y: np.dot(x, y), lambda a, b: onp.dot(a, b)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in sizes:
                    if size > batch_size:
                        continue

                    for order_name, order_value in orders:
                        A = generate_random_array(rows=size)
                        B = generate_random_array(rows=size)

                        expected = np_fn(A, B)

                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            ringDim=p["ringDim"],
                        ):
                            result = None
                            try:
                                # encrypt matrices
                                ctv_a = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )
                                ctv_b = onp.array(
                                    cc=cc,
                                    data=B,
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
                                    input_data={"a": A, "b": B},
                                    expected=expected,
                                    result=result,
                                )
                                raise


if __name__ == "__main__":
    TestMatrixBinaryOps.run_test_summary()
    TestMatrixUnaryOps.run_test_summary()
