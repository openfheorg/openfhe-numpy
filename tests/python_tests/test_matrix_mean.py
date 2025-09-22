import numpy as np
import openfhe_numpy as onp
from core import *

"""
Note: Mean operations may require sufficient multiplicative depth
and ring dimension to accommodate division operations. Small ring
dimensions (<4096) may result in higher approximation errors.
"""


class TestMatrixMean(MainUnittest):
    """Test class for matrix mean operations."""

    sizes = [2, 3, 4]  # Smaller sizes for mean operations
    orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

    def test_total_mean(self):
        """Test total matrix mean (all elements)"""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            # Ensure sufficient multiplicative depth for division
            params_copy = p.copy()
            if params_copy["multiplicativeDepth"] < 3:
                params_copy["multiplicativeDepth"] = 3

            cc, keys = gen_crypto_context(params_copy)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = params_copy["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    # Skip tests with very small ring dimensions for stability
                    if params_copy["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.mean(A)  # Total mean

                    with self.subTest(order=order_name, size=size, ringDim=params_copy["ringDim"]):
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

                        # Generate key for total mean
                        onp.gen_sum_key(keys.secretKey)

                        # Perform total mean
                        ctm_result = onp.mean(ctm_matrix)  # No axis parameter for total mean

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                        self.assertArrayClose(
                            params={
                                "case": "total_mean",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )

    def test_row_mean(self):
        """Test row-wise mean (axis=0)"""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            # Ensure sufficient multiplicative depth for division
            params_copy = p.copy()
            if params_copy["multiplicativeDepth"] < 3:
                params_copy["multiplicativeDepth"] = 3

            cc, keys = gen_crypto_context(params_copy)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = params_copy["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    # Skip tests with very small ring dimensions for stability
                    if params_copy["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.mean(A, axis=0)  # Row-wise mean

                    with self.subTest(order=order_name, size=size, ringDim=params_copy["ringDim"]):
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

                        # Generate appropriate keys for row mean
                        if order_value == onp.ROW_MAJOR:
                            ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                keys.secretKey, ctm_matrix.ncols, ctm_matrix.batch_size
                            )
                        else:  # COL_MAJOR
                            ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                keys.secretKey, ctm_matrix.nrows
                            )

                        # Perform row mean
                        ctm_result = onp.mean(ctm_matrix, axis=0)

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                        self.assertArrayClose(
                            params={
                                "case": "row_mean",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )

    def test_col_mean(self):
        """Test column-wise mean (axis=1)"""
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            # Ensure sufficient multiplicative depth for division
            params_copy = p.copy()
            if params_copy["multiplicativeDepth"] < 3:
                params_copy["multiplicativeDepth"] = 3

            cc, keys = gen_crypto_context(params_copy)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = params_copy["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    # Skip tests with very small ring dimensions for stability
                    if params_copy["ringDim"] < 4096 and size > 2:
                        continue
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.mean(A, axis=1)  # Column-wise mean

                    with self.subTest(order=order_name, size=size, ringDim=params_copy["ringDim"]):
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

                        # Generate appropriate keys for column mean
                        if order_value == onp.ROW_MAJOR:
                            ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                keys.secretKey, ctm_matrix.ncols
                            )
                        else:  # COL_MAJOR
                            ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                keys.secretKey, ctm_matrix.nrows, ctm_matrix.batch_size
                            )

                        # Perform column mean
                        ctm_result = onp.mean(ctm_matrix, axis=1)

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

                        self.assertArrayClose(
                            params={
                                "case": "column_mean",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )


if __name__ == "__main__":
    TestMatrixMean.run_test_summary("Matrix Mean", verbose=True)
