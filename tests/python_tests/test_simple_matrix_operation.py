# tests/test_matrix_ops.py
import numpy as np
import openfhe_numpy as onp
from core import *


class TestMatrixSimpleOperations(MainUnittest):
    sizes = [5, 8, 16]
    orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

    def test_unary_operations(self):
        ops = [
            ("transpose", lambda x: x.T, lambda x: onp.transpose(x)),
            ("scalar_mul", lambda x, s: x * s, lambda x, s: x * s),
            ("sum", lambda x: np.sum(x), lambda x: onp.sum(x)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2
            scalar = 7.2  # used by scalar_mul

            for tag, np_fn, fhe_fn in ops:
                for size in self.sizes:
                    for order_name, order_value in self.orders:
                        if size > batch_size:
                            continue

                        A = generate_random_array(
                            rows=size, cols=size
                        )  # Square matrix for matrix ops

                        with self.subTest(tag=tag, size=size, ringDim=p["ringDim"]):
                            # Encrypt A as matrix (tile mode)
                            ctm_a = onp.array(
                                cc=cc,
                                data=A,
                                batch_size=batch_size,
                                order=order_value,
                                fhe_type="C",
                                mode="tile",  # Use tile mode for matrices
                                public_key=keys.publicKey,
                            )

                            # Run encrypted operation
                            if tag == "scalar_mul":
                                expected = np_fn(A, scalar)
                                ctm_res = fhe_fn(ctm_a, scalar)
                            elif tag == "transpose":
                                # Generate transpose keys for matrix transpose
                                onp.gen_transpose_keys(keys.secretKey, ctm_a)
                                expected = np_fn(A)
                                ctm_res = fhe_fn(ctm_a)
                            else:  # sum
                                expected = np_fn(A)
                                ctm_res = fhe_fn(ctm_a)

                            # Decrypt and compare
                            result = ctm_res.decrypt(keys.secretKey, unpack_type="original")
                            self.assertArrayClose(
                                params={
                                    "case": "matrix_unary_op_" + tag,
                                    "size": size,
                                    "ringDim": p["ringDim"],
                                    "order": order_value,
                                },
                                input_data={"A": A},
                                actual=result,
                                expected=expected,
                            )

    def test_binary_operations(self):
        ops = [
            ("add", lambda x, y: x + y, lambda a, b: onp.add(a, b)),
            ("sub", lambda x, y: x - y, lambda a, b: onp.subtract(a, b)),
            ("mul", lambda x, y: x * y, lambda a, b: onp.multiply(a, b)),
            ("dot", lambda x, y: np.dot(x, y), lambda a, b: onp.dot(a, b)),
        ]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for tag, np_fn, fhe_fn in ops:
                for size in self.sizes:
                    for order_name, order_value in self.orders:
                        if size > batch_size:
                            continue

                        A = generate_random_array(rows=size, cols=size)
                        B = generate_random_array(rows=size, cols=size)
                        expected = np_fn(A, B)

                        with self.subTest(tag=tag, size=size, ringDim=p["ringDim"]):
                            # Encrypt matrices (tile mode)
                            ctm_a = onp.array(
                                cc=cc,
                                data=A,
                                batch_size=batch_size,
                                order=onp.ROW_MAJOR,
                                fhe_type="C",
                                mode="tile",
                                public_key=keys.publicKey,
                            )
                            ctm_b = onp.array(
                                cc=cc,
                                data=B,
                                batch_size=batch_size,
                                order=onp.ROW_MAJOR,
                                fhe_type="C",
                                mode="tile",
                                public_key=keys.publicKey,
                            )

                            # Special handling for matrix multiplication
                            if tag == "dot":
                                # Generate rotation keys for matrix multiplication
                                onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_a.ncols)
                                ctm_res = ctm_a @ ctm_b  # Use @ operator for matrix multiplication
                            else:
                                ctm_res = fhe_fn(ctm_a, ctm_b)

                            # Decrypt and compare
                            result = ctm_res.decrypt(keys.secretKey, unpack_type="original")
                            self.assertArrayClose(
                                params={
                                    "case": "matrix_binary_op_" + tag,
                                    "size": size,
                                    "ringDim": p["ringDim"],
                                    "order": order_value,
                                },
                                input_data={"A": A, "B": B},
                                actual=result,
                                expected=expected,
                            )


if __name__ == "__main__":
    TestMatrixSimpleOperations.run_test_summary("Matrix Operations", verbose=True)
