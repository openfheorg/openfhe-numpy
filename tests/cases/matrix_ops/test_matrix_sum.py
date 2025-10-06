import numpy as np
from openfhe import *
import openfhe_numpy as onp
from core import *

###
# Note: Column-/row-wise cumulative sum may require deeper multiplicative
# depth or larger ring dimensions for accurate results. Small ring dimensions
# (<4096) might introduce approximation errors.
###


SIZES = [2, 3, 8, 16]
ORDERS = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]


def _ensure_depth(p: dict, min_depth: int = 3) -> dict:
    params = p.copy()
    if params.get("multiplicativeDepth", 0) < min_depth:
        params["multiplicativeDepth"] = min_depth
    return params


# -----------------------------------------------------------
#   Matrix Sum
# -----------------------------------------------------------
class TestMatrixSum(MainUnittest):
    """Test class for matrix sum operations."""

    def test_total_sum(self):
        """Total matrix sum (all elements)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2

            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for order_name, order_value in ORDERS:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A)

                    with self.subTest(
                        order=order_name, size=size, ringDim=params["ringDim"]
                    ):
                        try:
                            result = None
                            ctm_matrix = onp.array(
                                cc=cc,
                                data=A,
                                batch_size=batch_size,
                                order=order_value,
                                fhe_type="C",
                                mode="zero",
                                public_key=keys.publicKey,
                            )

                            onp.gen_sum_key(keys.secretKey)
                            ctm_result = onp.sum(ctm_matrix)

                            result = ctm_result.decrypt(
                                keys.secretKey, unpack_type="original"
                            )
                            self.assertArrayClose(
                                actual=result, expected=expected
                            )

                        except Exception as e:
                            self._record_case(
                                params={
                                    "case": "total_sum",
                                    "size": size,
                                    "ringDim": p["ringDim"],
                                },
                                input_data={"A": A},
                                expected=expected,
                                result=result,
                            )
                            raise


# -----------------------------------------------------------
#   Matrix Row Sum
# -----------------------------------------------------------
class TestMatrixRowSum(MainUnittest):
    """Test class for matrix sum operations."""

    def test_row_sum(self):
        """Row-wise sum (axis=0)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            params = _ensure_depth(p, 3)
            batch_size = params["ringDim"] // 2

            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            for order_name, order_value in ORDERS:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=0)

                    with self.subTest(
                        order=order_name, size=size, ringDim=params["ringDim"]
                    ):
                        try:
                            result = None
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
                                ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                    keys.secretKey,
                                    ctm_matrix.ncols,
                                    ctm_matrix.batch_size,
                                )
                            else:
                                ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                    keys.secretKey, ctm_matrix.nrows
                                )

                            ctm_result = onp.sum(ctm_matrix, axis=0)

                            result = ctm_result.decrypt(
                                keys.secretKey, unpack_type="original"
                            )

                            self.assertArrayClose(
                                actual=result, expected=expected
                            )

                        except Exception as e:
                            self._record_case(
                                params={
                                    "case": "row_sum",
                                    "size": size,
                                    "ringDim": p["ringDim"],
                                },
                                input_data={"A": A},
                                expected=expected,
                                result=result,
                            )
                            raise


# -----------------------------------------------------------
#   Matrix Col Sum
# -----------------------------------------------------------
class TestMatrixColSum(MainUnittest):
    """Test class for matrix sum operations."""

    def test_col_sum(self):
        """Column-wise sum (axis=1)."""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            params = _ensure_depth(p, 3)

            cc, keys = gen_crypto_context(params)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = params["ringDim"] // 2

            for order_name, order_value in ORDERS:
                for size in SIZES:
                    if params["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=1)

                    with self.subTest(
                        order=order_name, size=size, ringDim=params["ringDim"]
                    ):
                        try:
                            result = None

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
                                ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                    keys.secretKey, ctm_matrix.ncols
                                )
                            else:
                                ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                    keys.secretKey,
                                    ctm_matrix.nrows,
                                    ctm_matrix.batch_size,
                                )

                            ctm_result = onp.sum(ctm_matrix, axis=1)

                            result = ctm_result.decrypt(
                                keys.secretKey, unpack_type="original"
                            )
                            self.result = result
                            self.assertArrayClose(
                                actual=result, expected=expected
                            )
                        except Exception as e:
                            self._record_case(
                                params={
                                    "case": "column_sum",
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
    TestMatrixSum.run_test_summary()
    TestMatrixColSum.run_test_summary()
    TestMatrixRowSum.run_test_summary()
