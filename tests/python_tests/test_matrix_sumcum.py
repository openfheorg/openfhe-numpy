import numpy as np
import openfhe_numpy as onp
from core import *


class TestMatrixCumulativeSum(MainUnittest):
    sizes = [2, 3, 8, 16]
    # Ensure sufficient multiplicative depth

    def test_cumsum_rows(self):
        """Test cumulative sum along rows (axis=0)"""
        orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            for order_name, order_value in orders:
                for size in self.sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    required_depth = len(A)
                    if p["multiplicativeDepth"] < required_depth:
                        p["multiplicativeDepth"] = required_depth + 1
                    cc, keys = gen_crypto_context(p)
                    cc.EvalMultKeyGen(keys.secretKey)
                    cc.EvalSumKeyGen(keys.secretKey)

                    expected = np.cumsum(A, axis=0)  # Row-wise cumulative sum

                    with self.subTest(order=order_name, size=size, ringDim=p["ringDim"]):
                        # Encrypt matrix
                        ctm_matrix = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=order_value,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )

                        # Generate keys for row cumsum
                        if order_value == onp.ROW_MAJOR:
                            onp.gen_accumulate_rows_key(keys.secretKey, ctm_matrix.ncols)
                        else:  # COL_MAJOR
                            onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)

                        # Perform cumulative sum along rows
                        ctm_result = onp.cumulative_sum(ctm_matrix, axis=0)

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                        self.assertArrayClose(
                            params={
                                "case": "cumsum_rows",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )

    def test_cumsum_cols(self):
        """Test cumulative sum along columns (axis=1)"""
        orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            batch_size = p["ringDim"] // 2

            for order_name, order_value in orders:
                for size in self.sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    required_depth = len(A[0])
                    if p["multiplicativeDepth"] < required_depth:
                        p["multiplicativeDepth"] = required_depth + 1
                    cc, keys = gen_crypto_context(p)
                    cc.EvalMultKeyGen(keys.secretKey)
                    cc.EvalSumKeyGen(keys.secretKey)

                    expected = np.cumsum(A, axis=1)  # Column-wise cumulative sum

                    with self.subTest(order=order_name, size=size, ringDim=p["ringDim"]):
                        # Encrypt matrix
                        ctm_matrix = onp.array(
                            cc=cc,
                            data=A,
                            batch_size=batch_size,
                            order=order_value,
                            fhe_type="C",
                            mode="zero",
                            public_key=keys.publicKey,
                        )

                        # Generate keys for column cumsum
                        if order_value == onp.ROW_MAJOR:
                            onp.gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)
                        else:  # COL_MAJOR
                            onp.gen_accumulate_rows_key(keys.secretKey, ctm_matrix.ncols)

                        # Perform cumulative sum along columns
                        ctm_result = onp.cumulative_sum(ctm_matrix, axis=1)

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                        self.assertArrayClose(
                            params={
                                "case": "cumsum_cols",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )


if __name__ == "__main__":
    TestMatrixCumulativeSum.run_test_summary("Matrix Cumulative Sum", verbose=True)
