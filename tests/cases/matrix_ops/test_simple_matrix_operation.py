# tests/test_matrix_ops.py
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *

SCALAR = 7.9
SIZES = [5, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR)]
MODES_UNARY = ["zero"]
MODES_BINARY = ["tile"]


# -----------------------------------------------------------
#  Matrix Unary Ops
# (transpose/scalar multiplication/sum)
# -----------------------------------------------------------


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

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    # generate square matrix with size (size x size)
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

                                    # Run encrypted operation
                                    if tag == "scalar_mul":
                                        expected = np_fn(A, SCALAR)
                                        ctm_res = fhe_fn(ctm_a, SCALAR)
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


# -----------------------------------------------------------
#  Matrix Unary Ops
# (add/sub/mul/dot)
# -----------------------------------------------------------


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
                for size in SIZES:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    B = generate_random_array(rows=size, cols=size)
                    expected = np_fn(A, B)

                    for order_name, order_value in ORDERS:
                        for mode in MODES_BINARY:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                size=size,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                try:
                                    result = None

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

                                    onp.EvalSquareMatMultRotateKeyGen(
                                        keys.secretKey, ctm_a.ncols
                                    )
                                    ctm_res = fhe_fn(ctm_a, ctm_b)

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


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    TestMatrixUnaryOps.run_test_summary()
    TestMatrixBinaryOps.run_test_summary()
