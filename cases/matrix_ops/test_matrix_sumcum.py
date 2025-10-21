import numpy as np
import openfhe_numpy as onp
from core import *

sizes = [2, 3, 8, 16]
orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]


class TestMatrixCumulativeSumRow(MainUnittest):
    def test_cumsum_rows(self):
        """Cumulative sum along rows (axis=0)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            for order_name, order_value in orders:
                for size in sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.cumsum(A, axis=0)

                    # ensure enough multiplicative depth (work on a copy)
                    params = p.copy()
                    required_depth = len(A)
                    if params.get("multiplicativeDepth", 0) < required_depth:
                        params["multiplicativeDepth"] = required_depth + 1

                    with self.subTest(
                        order=order_name, size=size, ringDim=params["ringDim"]
                    ):
                        self._record_case(
                            params={
                                "case": "cumsum_rows",
                                "size": size,
                                "ringDim": params["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            expected=expected,
                            result=None,
                        )

                        try:
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

            for order_name, order_value in orders:
                for size in sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.cumsum(A, axis=1)

                    # ensure multiplicative depth (copy)
                    params = p.copy()
                    required_depth = A.shape[1]
                    if params.get("multiplicativeDepth", 0) < required_depth:
                        params["multiplicativeDepth"] = required_depth + 1

                    with self.subTest(
                        order=order_name, size=size, ringDim=params["ringDim"]
                    ):
                        self._record_case(
                            params={
                                "case": "cumsum_cols",
                                "size": size,
                                "ringDim": params["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            expected=expected,
                            result=None,
                        )

                        try:
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
                                input_data={"A": A, "b": b},
                                expected=expected,
                                result=result,
                            )
                            raise


if __name__ == "__main__":
    TestMatrixCumulativeSumCol.run_test_summary()
    TestMatrixCumulativeSumRow.run_test_summary()
