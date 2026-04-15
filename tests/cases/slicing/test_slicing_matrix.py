import gc
import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *


SHAPES = [(4, 4), (5, 7)]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]
MODES = ["zero"]

OPS = [
    ("slice_row_1_to_end", lambda a: a[1:], lambda a: a[1:]),
    ("slice_row_col_1_to_3", lambda a: a[1:3, 1:3], lambda a: a[1:3, 1:3]),
    ("slice_single_row", lambda a: a[0], lambda a: a[0]),
    ("slice_single_col", lambda a: a[:, 0], lambda a: a[:, 0]),
    ("slice_step_2", lambda a: a[::2, ::2], lambda a: a[::2, ::2]),
]


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class TestSlicingMatrix(MainUnittest):
    def _run(self, tag, np_fn, fhe_fn):
        ckks_params = load_ckks_params()

        for p in ckks_params:
            batch_size = p["ringDim"] // 2
            cc, keys = gen_crypto_context(p)

            try:
                for rows, cols in SHAPES:
                    nrow = next_power_of_2(rows)
                    ncol = next_power_of_2(cols)
                    nelements = nrow * ncol

                    if nelements > batch_size:
                        continue

                    matrix = generate_random_array(rows=rows, cols=cols)

                    onp.generate_slicing_key(keys.secretKey, matrix.shape)
                    expected = np.asarray(np_fn(matrix))

                    for order_name, order_value in ORDERS:
                        for mode in MODES:
                            with self.subTest(
                                op=tag,
                                order=order_name,
                                rows=rows,
                                cols=cols,
                                mode=mode,
                                ringDim=p["ringDim"],
                            ):
                                result = None
                                ctm = None
                                ctm_res = None

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

                                    ctm_res = fhe_fn(ctm)
                                    result = ctm_res.decrypt(
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
                                            "case": "matrix_slicing",
                                            "op": tag,
                                            "rows": rows,
                                            "cols": cols,
                                            "order": order_name,
                                            "mode": mode,
                                            "ringDim": p["ringDim"],
                                        },
                                        input_data={"matrix": matrix},
                                        expected=expected,
                                        result=result,
                                    )
                                    raise

                                finally:
                                    del ctm
                                    del ctm_res
                                    del result
                                    gc.collect()
            finally:
                del cc, keys
                gc.collect()

    def test_slice_row_1_to_end(self):
        self._run(*OPS[0])

    def test_slice_row_col_1_to_3(self):
        self._run(*OPS[1])

    def test_slice_single_row(self):
        self._run(*OPS[2])

    def test_slice_single_col(self):
        self._run(*OPS[3])

    def test_slice_step_2(self):
        self._run(*OPS[4])
