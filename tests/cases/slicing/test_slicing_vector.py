import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SIZES = [5, 8]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]
OPS = [
    ("slice_1_to_end", lambda a: a[1:], lambda a: a[1:]),
    ("slice_1_to_3", lambda a: a[1:3], lambda a: a[1:3]),
    ("slice_last", lambda a: a[-1], lambda a: a[-1]),
    ("slice_step_2", lambda a: a[::2], lambda a: a[::2]),
]


class TestSlicingVector(MainUnittest):
    def _run(self, tag, np_fn, fhe_fn):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)

            try:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    vector = generate_random_array(rows=size)
                    expected = np.asarray(np_fn(vector))
                    onp.generate_slicing_key(keys.secretKey, vector.shape)

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
                                ctv = None
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
                                    ctv_res = fhe_fn(ctv)
                                    result = ctv_res.decrypt(
                                        keys.secretKey,
                                        unpack_type="original",
                                    )
                                    self.assertArrayClose(
                                        actual=np.asarray(result),
                                        expected=expected,
                                    )

                                except Exception:
                                    self._record_case(
                                        params={
                                            "case": "vector_slicing",
                                            "op": tag,
                                            "size": size,
                                            "order": order_name,
                                            "mode": mode,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"vector": vector},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise

                                finally:
                                    del ctv
                                    del ctv_res
                                    del result
                                    gc.collect()
            finally:
                del cc
                del keys
                gc.collect()

    def test_slice_1_to_end(self):
        self._run(*OPS[0])

    def test_slice_1_to_3(self):
        self._run(*OPS[1])

    def test_slice_last(self):
        self._run(*OPS[2])

    def test_slice_step_2(self):
        self._run(*OPS[3])
