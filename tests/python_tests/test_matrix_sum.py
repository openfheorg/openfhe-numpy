import numpy as np
import openfhe_numpy as onp
from core import *

###
# Note: Column-/row-wise cumulative sum may require deeper multiplicative
# depth or larger ring dimensions for accurate results. Small ring dimensions
# (<4096) might introduce approximation errors.
###


class TestMatrixSum(MainUnittest):
    """Test class for matrix sum operations."""

    sizes = [2, 3, 8, 16]
    orders = [("row_major", onp.ROW_MAJOR), ("col_major", onp.COL_MAJOR)]

    def test_total_sum(self):
        """Test total matrix sum (all elements)"""

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    # Skip large matrices for total sum tests
                    if size > 3:
                        continue

                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A)  # Total sum

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

                        # Generate key for total sum
                        onp.gen_sum_key(keys.secretKey)

                        # Perform total sum
                        ctm_result = onp.sum(ctm_matrix)  # No axis parameter for total sum

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

    def test_row_sum(self):
        """Test row-wise sum (axis=0)"""

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=0)  # Row-wise sum

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

                        # Generate appropriate keys for row sum
                        if order_value == onp.ROW_MAJOR:
                            ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                keys.secretKey, ctm_matrix.ncols, ctm_matrix.batch_size
                            )
                        else:  # COL_MAJOR
                            ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                keys.secretKey, ctm_matrix.nrows
                            )

                        # Perform row sum
                        ctm_result = onp.sum(ctm_matrix, axis=0)

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

    def test_col_sum(self):
        """Test column-wise sum (axis=1)"""

        ckks_params = load_ckks_params()
        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for order_name, order_value in self.orders:
                for size in self.sizes:
                    if size > batch_size:
                        continue

                    A = generate_random_array(rows=size, cols=size)
                    expected = np.sum(A, axis=1)  # Column-wise sum

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

                        # Generate appropriate keys for column sum
                        if order_value == onp.ROW_MAJOR:
                            ctm_matrix.extra["colkey"] = onp.sum_col_keys(
                                keys.secretKey, ctm_matrix.ncols
                            )
                        else:  # COL_MAJOR
                            ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                                keys.secretKey, ctm_matrix.nrows, ctm_matrix.batch_size
                            )

                        # Perform column sum
                        ctm_result = onp.sum(ctm_matrix, axis=1)

                        # Decrypt and compare
                        result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
                        self.assertArrayClose(
                            params={
                                "case": "col_mean",
                                "size": size,
                                "ringDim": p["ringDim"],
                                "order": order_value,
                            },
                            input_data={"A": A},
                            actual=result,
                            expected=expected,
                        )


if __name__ == "__main__":
    TestMatrixSum.run_test_summary("Matrix Sum", verbose=True)
