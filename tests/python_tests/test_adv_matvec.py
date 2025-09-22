import numpy as np
import openfhe_numpy as onp
from core import *


class TestMatrixVectorProduct(MainUnittest):
    """Test class for matrix-vector product operations."""

    sizes = [2, 3, 4, 8]

    def test_mat_vec_prod_rowmajor_colmajor(self):
        """
        Test matrix (row-major) vector (col-major) product:
        - Matrix: row-major, tile mode
        - Vector: column-major, tile mode
        - Result: row-major
        """
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for size in self.sizes:
                # Skip large sizes for small ring dimensions
                if size > 4 and p["ringDim"] < 8192:
                    continue
                if size > batch_size:
                    continue

                # Generate test data
                A = generate_random_array(rows=size, cols=size)  # Square matrix
                b = generate_random_array(rows=size, cols=1).flatten()  # Column vector
                expected = np.dot(A, b)

                with self.subTest(case="rowmajor_colmajor", size=size, ringDim=p["ringDim"]):
                    # Encrypt matrix in row-major, tile mode
                    ctm_matrix = onp.array(
                        cc=cc,
                        data=A,
                        batch_size=batch_size,
                        order=onp.ROW_MAJOR,
                        fhe_type="C",
                        mode="tile",
                        public_key=keys.publicKey,
                    )

                    # Generate column sum keys for matrix
                    ctm_matrix.extra["colkey"] = onp.sum_col_keys(keys.secretKey, ctm_matrix.ncols)

                    # Encrypt vector in column-major, tile mode
                    ctv_vector = onp.array(
                        cc=cc,
                        data=b,
                        batch_size=batch_size,
                        order=onp.COL_MAJOR,
                        fhe_type="C",
                        mode="tile",
                        public_key=keys.publicKey,
                    )

                    # Perform matrix-vector multiplication
                    ctv_result = ctm_matrix @ ctv_vector

                    # Decrypt and compare
                    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
                    self.assertArrayClose(
                        params={"case": "rowmajor_colmajor", "size": size, "ringDim": p["ringDim"]},
                        input_data={"A": A, "b": b},
                        actual=result,
                        expected=expected,
                    )

    def test_mat_vec_prod_colmajor_rowmajor(self):
        """
        Test matrix (column-major) vector (row-major) product:
        - Matrix: column-major, zero mode
        - Vector: row-major, zero mode
        - Result: column-major
        """
        ckks_params = load_ckks_params()

        for _, p in enumerate(ckks_params):
            cc, keys = gen_crypto_context(p)
            cc.EvalMultKeyGen(keys.secretKey)
            cc.EvalSumKeyGen(keys.secretKey)

            batch_size = p["ringDim"] // 2

            for size in self.sizes:
                # Skip large sizes for small ring dimensions
                if size > 4 and p["ringDim"] < 8192:
                    continue
                if size > batch_size:
                    continue

                # Generate test data
                A = generate_random_array(rows=size, cols=size)  # Square matrix
                b = generate_random_array(rows=size, cols=1).flatten()  # Column vector
                expected = np.dot(A, b)

                with self.subTest(case="colmajor_rowmajor", size=size, ringDim=p["ringDim"]):
                    # Encrypt matrix in column-major, zero mode
                    ctm_matrix = onp.array(
                        cc=cc,
                        data=A,
                        batch_size=batch_size,
                        order=onp.COL_MAJOR,
                        fhe_type="C",
                        mode="zero",
                        public_key=keys.publicKey,
                    )

                    # Encrypt vector in row-major, zero mode
                    ctv_vector = onp.array(
                        cc=cc,
                        data=b,
                        batch_size=batch_size,
                        order=onp.ROW_MAJOR,
                        fhe_type="C",
                        mode="zero",
                        target_cols=ctm_matrix.nrows,  # Important for row-major vector
                        public_key=keys.publicKey,
                    )

                    # Generate row sum keys for matrix
                    ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
                        keys.secretKey, ctm_matrix.nrows, ctm_matrix.batch_size
                    )

                    # Perform matrix-vector multiplication
                    ctv_result = ctm_matrix @ ctv_vector

                    # Decrypt and compare
                    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")

                    self.assertArrayClose(
                        params={"case": "colmajor_rowmajor", "size": size, "ringDim": p["ringDim"]},
                        input_data={"A": A, "b": b},
                        actual=result,
                        expected=expected,
                    )


if __name__ == "__main__":
    TestMatrixVectorProduct.run_test_summary("Matrix-Vector Product", verbose=True)
