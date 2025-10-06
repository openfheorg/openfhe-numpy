# tests/test_matrix_ops.py
import numpy as np
import openfhe_numpy as onp
from core import *

sizes = [5]
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
                        A = generate_random_array(rows=size, cols=size)

                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            ringDim=p["ringDim"],
                        ):
                            ctm_a = ctm_res = None
                            try:
                                # encrypt A (tile mode for matrices)
                                ctm_a = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="zero",
                                    public_key=keys.publicKey,
                                )

                                # Run encrypted operation
                                if tag == "scalar_mul":
                                    expected = np_fn(A, scalar)
                                    ctm_res = fhe_fn(ctm_a, scalar)
                                elif tag == "transpose":
                                    onp.gen_transpose_keys(
                                        keys.secretKey, ctm_a
                                    )
                                    expected = np_fn(A)
                                    ctm_res = fhe_fn(ctm_a)
                                else:  # sum
                                    expected = np_fn(A)
                                    ctm_res = fhe_fn(ctm_a)

                                # decrypt and compare
                                result = ctm_res.decrypt(
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
                for size in sizes:
                    if size > batch_size:
                        continue

                    for order_name, order_value in orders:
                        A = generate_random_array(rows=size, cols=size)
                        B = generate_random_array(rows=size, cols=size)
                        expected = A @ B

                        with self.subTest(
                            op=tag,
                            order=order_name,
                            size=size,
                            ringDim=p["ringDim"],
                        ):
                            try:
                                # encrypt matrices
                                ctm_a = onp.array(
                                    cc=cc,
                                    data=A,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="tile",
                                    public_key=keys.publicKey,
                                )
                                ctm_b = onp.array(
                                    cc=cc,
                                    data=B,
                                    batch_size=batch_size,
                                    order=order_value,
                                    fhe_type="C",
                                    mode="tile",
                                    public_key=keys.publicKey,
                                )

                                onp.EvalSquareMatMultRotateKeyGen(
                                    keys.secretKey, ctm_a.ncols
                                )
                                ctm_res = ctm_a @ ctm_b

                                # decrypt and compare
                                result = ctm_res.decrypt(
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
                                    input_data={"A": A, "B": B},
                                    expected=expected,
                                    result=result,
                                )
                                raise


if __name__ == "__main__":
    TestMatrixUnaryOps.run_test_summary()
    TestMatrixBinaryOps.run_test_summary()
