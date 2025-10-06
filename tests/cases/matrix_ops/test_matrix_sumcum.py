import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *

SIZES = [2, 3, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]


def _ensure_depth(p: dict, min_depth: int = 3) -> dict:
    params = p.copy()
    if params.get("multiplicativeDepth", 0) < min_depth:
        params["multiplicativeDepth"] = min_depth
    return params


# -----------------------------------------------------------
#   Matrix Cumulative Row Sum
# -----------------------------------------------------------
class TestMatrixCumulativeSumRow(MainUnittest):
    def test_cumsum_rows(self):
        """Cumulative sum along rows (axis=0)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            for order_name, order_value in ORDERS:
                for size in SIZES:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.cumsum(A, axis=0)

                    with self.subTest(
                        order=order_name, size=size, ringDim=p["ringDim"]
                    ):
                        try:
                            result = None
                            # ensure enough multiplicative depth
                            params = _ensure_depth(p, len(A) + 1)
                            cc, keys = gen_crypto_context(params)
                            cc.EvalMultKeyGen(keys.secretKey)
                            cc.EvalSumKeyGen(keys.secretKey)

                            ctm_matrix = onp.array(
                                cc=cc,
                                data=A,
                                batch_size=batch_size,
                                order=order_value,
                                fhe_type="C",
                                mode="zero",
                                public_key=keys.publicKey,
                            )

                            if order_value == onp.ROW_MAJOR:
                                onp.gen_accumulate_rows_key(
                                    keys.secretKey, ctm_matrix.ncols
                                )
                            else:
                                onp.gen_accumulate_cols_key(
                                    keys.secretKey, ctm_matrix.ncols
                                )

                            ctm_result = onp.cumulative_sum(ctm_matrix, axis=0)
                            result = ctm_result.decrypt(
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


class TestMatrixCumulativeSumCol(MainUnittest):
    def test_cumsum_cols(self):
        """Cumulative sum along columns (axis=1)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            for size in SIZES:
                if size > batch_size:
                    continue

                A = generate_random_array(rows=size, cols=size)
                expected = np.cumsum(A, axis=1)

                for order_name, order_value in ORDERS:
                    with self.subTest(
                        order=order_name, size=size, ringDim=p["ringDim"]
                    ):
                        try:
                            result = None
                            # ensure enough multiplicative depth
                            params = _ensure_depth(p, len(A) + 1)
                            cc, keys = gen_crypto_context(params)
                            cc.EvalMultKeyGen(keys.secretKey)
                            cc.EvalSumKeyGen(keys.secretKey)

                            ctm_matrix = onp.array(
                                cc=cc,
                                data=A,
                                batch_size=batch_size,
                                order=order_value,
                                fhe_type="C",
                                mode="zero",
                                public_key=keys.publicKey,
                            )

                            if order_value == onp.ROW_MAJOR:
                                onp.gen_accumulate_cols_key(
                                    keys.secretKey, ctm_matrix.ncols
                                )
                            else:
                                onp.gen_accumulate_rows_key(
                                    keys.secretKey, ctm_matrix.ncols
                                )

                            ctm_result = onp.cumulative_sum(ctm_matrix, axis=1)
                            result = ctm_result.decrypt(
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


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    TestMatrixCumulativeSumCol.run_test_summary()
    TestMatrixCumulativeSumRow.run_test_summary()
