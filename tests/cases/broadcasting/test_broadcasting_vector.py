import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SIZES_MAT = [(3, 6), (1, 8), (16, 1)]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]

OPS = [
    ("add", lambda a, b: np.add(a, b), lambda a, b: onp.add(a, b)),
    ("sub", lambda a, b: np.subtract(a, b), lambda a, b: onp.subtract(a, b)),
    ("multiply", lambda a, b: np.multiply(a, b), lambda a, b: onp.multiply(a, b)),
]


class TestVectorBroadcasting(MainUnittest):
    def _run(self, tag, np_fn, fhe_fn):
        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)
            try:
                for rows, cols in SIZES_MAT:
                    if onp.next_power_of_two(rows) * onp.next_power_of_two(cols) > batch_size:
                        continue

                    # generate vector with dimension (size)
                    matrix = generate_random_array(rows, cols)
                    vector = generate_random_array(rows=rows, cols=1)
                    expected = np_fn(vector, matrix)

                    onp.generate_broadcast_key(keys.secretKey, vector.shape, matrix.shape)

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
            finally:
                del cc
                del keys
                gc.collect()

    def test_add(self):
        self._run(*OPS[0])

    def test_sub(self):
        self._run(*OPS[1])

    def test_multiply(self):
        self._run(*OPS[2])
