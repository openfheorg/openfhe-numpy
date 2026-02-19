import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SCALAR = 10
SIZES_VECTOR = [5, 8, 16]
SIZES_MAT = [(5, 5), (3, 6), (15, 13), (13, 1), (1, 14)]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]


class TestScalar(MainUnittest):
    """Test class for unary matrix operations"""

    def test_scalar_ops(self):
        ops = [
            ("add", lambda a, b: np.add(a, b), lambda a, b: onp.add(a, b)),
            ("sub", lambda a, b: np.subtract(a, b), lambda a, b: a - b),
            ("multiply", lambda a, b: np.multiply(a, b), lambda a, b: onp.multiply(a, b)),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for size in SIZES_VECTOR:
                    if size > batch_size:
                        continue

                    # generate vector with dimension (size)
                    vector = generate_random_array(rows=size)
                    onp.generate_broadcast_key(keys.secretKey, (size,))

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
                                        mode=mode,
                                        fhe_type="C",
                                        public_key=keys.publicKey,
                                    )

                                    onp.gen_transpose_keys(keys.secretKey, ctv)

                                    expected = np_fn(vector, SCALAR)
                                    ctv_res = fhe_fn(ctv, cts)

                                    # decrypt and compare
                                    result = ctv_res.decrypt(keys.secretKey, unpack_type="original")

                                    self.assertArrayClose(actual=result, expected=expected)
                                except Exception as e:
                                    self._record_case(
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

    def test_scalar_matrix_ops(self):
        ops = [
            ("add", lambda a, b: np.add(a, b), lambda a, b: onp.add(a, b)),
            ("sub", lambda a, b: np.subtract(a, b), lambda a, b: onp.subtract(a, b)),
            ("multiply", lambda a, b: np.multiply(a, b), lambda a, b: onp.multiply(a, b)),
        ]

        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for tag, np_fn, fhe_fn in ops:
                for rows, cols in SIZES_MAT:
                    onp.generate_broadcast_key(keys.secretKey, (rows, cols))
                    if onp.next_power_of_two(rows) * onp.next_power_of_two(cols) > batch_size:
                        continue

                    # generate vector with dimension (size)
                    matrix = generate_random_array(rows, cols)

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

                                    expected = np_fn(matrix, SCALAR)
                                    ctm_res = fhe_fn(ctm, SCALAR)

                                    # decrypt and compare
                                    result = ctm_res.decrypt(keys.secretKey, unpack_type="original")

                                    self.assertArrayClose(actual=result, expected=expected)
                                except Exception as e:
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


if __name__ == "__main__":
    TestScalar.run_test_summary()
