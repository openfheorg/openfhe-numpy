import numpy as np
import openfhe_numpy as onp
from core import *


# ==============================================================
#   Matrix (row-major, tile) x Vector (column-major, tile)
# ==============================================================
class TestRowMajorColMajor(MainUnittest):
    sizes = [2, 3, 4, 8]

    def test_mult_matrix_vector(self):
        ckks_params = load_ckks_params()
        for p in ckks_params:
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2
            for size in self.sizes:
                if (size > 4 and p["ringDim"] < 8192) or size > batch_size:
                    continue

                A = generate_random_array(rows=size, cols=size)
                b = generate_random_array(rows=size, cols=1).flatten()
                expected = np.dot(A, b)

                with self.subTest(size=size, ringDim=p["ringDim"]):
                    result = None
                    try:
                        ctm = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=onp.ROW_MAJOR,
                            fhe_type="C",
                            mode="tile",
                            public_key=keys.publicKey,
                        )
                        ctm.extra["colkey"] = onp.sum_col_keys(
                            keys.secretKey, ctm.ncols
                        )
                        ctv = onp.array(
                            cc=cc,
                            data=b,
                            batch_size=batch_size,
                            order=onp.COL_MAJOR,
                            fhe_type="C",
                            mode="tile",
                            public_key=keys.publicKey,
                        )
                        ctv_result = ctm @ ctv
                        result = ctv_result.decrypt(
                            keys.secretKey, unpack_type="original"
                        )

                        self.assertArrayClose(result, expected)

                    except Exception as e:
                        self._record_case(
                            params={
                                "case": "rowmajor_colmajor",
                                "size": size,
                                "ringDim": p["ringDim"],
                            },
                            input_data={"A": A, "b": b},
                            expected=expected,
                            result=result,
                        )
                        raise


# ==============================================================
#   Matrix (column-major, zero) x Vector (row-major, zero)
# ==============================================================
class TestColMajorRowMajor(MainUnittest):
    sizes = [2, 3, 4, 8]

    def test_mult_matrix_vector(self):
        ckks_params = load_ckks_params()
        for p in ckks_params:
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2
            for size in self.sizes:
                if (size > 4 and p["ringDim"] < 8192) or size > batch_size:
                    continue

                A = generate_random_array(rows=size, cols=size)
                b = generate_random_array(rows=size, cols=1).flatten()
                expected = np.dot(A, b)

                with self.subTest(size=size, ringDim=p["ringDim"]):
                    result = None
                    try:
                        ctm = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=onp.COL_MAJOR,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )
                        ctv = onp.array(
                            cc=cc,
                            data=b,
                            batch_size=batch_size,
                            order=onp.ROW_MAJOR,
                            fhe_type="C",
                            mode="zero",
                            target_cols=ctm.nrows,
                            public_key=keys.publicKey,
                        )
                        ctm.extra["rowkey"] = onp.sum_row_keys(
                            keys.secretKey, ctm.nrows, ctm.batch_size
                        )
                        ctv_result = ctm @ ctv
                        result = ctv_result.decrypt(
                            keys.secretKey, unpack_type="original"
                        )

                        self.assertArrayClose(result, expected)

                    except Exception as e:
                        self._record_case(
                            params={
                                "case": "colmajor_rowmajor",
                                "size": size,
                                "ringDim": p["ringDim"],
                            },
                            input_data={"A": A, "b": b},
                            expected=expected,
                            result=result,
                        )
                        raise


# ==============================================================
#   Entry point
# ==============================================================
if __name__ == "__main__":
    TestRowMajorColMajor.run_test_summary()
    TestColMajorRowMajor.run_test_summary()
