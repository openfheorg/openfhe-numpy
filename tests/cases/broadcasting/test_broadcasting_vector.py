import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SIZES_MAT = [(5, 5), (3, 6), (15, 13), (13, 1)]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]


class TestVectorBroadCasting(MainUnittest):
    def test_vector_matrix_ops(self):
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

            for rows, cols in SIZES_MAT:
                onp.generate_broadcast_key(keys.secretKey, (rows, cols))
                if onp.next_power_of_two(rows) * onp.next_power_of_two(cols) > batch_size:
                    continue
                for tag, np_fn, fhe_fn in ops:
                    # generate vector with dimension (size)
                    matrix = generate_random_array(rows, cols)
                    vector = generate_random_array(rows=rows, cols=1)

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

                                    ctv = onp.array(
                                        cc=cc,
                                        data=vector,
                                        batch_size=batch_size,
                                        order=order_value,
                                        fhe_type="C",
                                        mode=mode,
                                        public_key=keys.publicKey,
                                    )

                                    onp.gen_transpose_keys(keys.secretKey, ctm)

                                    expected = np_fn(vector, matrix)
                                    ctm_res = fhe_fn(ctv, ctm)

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
    # TestScalar.run_test_summary()
    TestVector.run_test_summary()
