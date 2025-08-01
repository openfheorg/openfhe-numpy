import numpy as np
import openfhe_numpy as onp

from tests.core.test_framework import MainUnittest
from tests.core.test_utils import generate_random_array, suppress_stdout
from tests.core.test_crypto_context import load_ckks_params, gen_crypto_context


def fhe_matrix_vector_product_case1(params, data):
    """
    Execute matrix-vector product with FHE - Case 1:
    - Matrix: row-major
    - Vector: column-major
    - Result: row-major
    """
    params_copy = params.copy()
    total_slots = params_copy["ringDim"] // 2

    with suppress_stdout(False):
        # Generate crypto context
        cc, keys = gen_crypto_context(params_copy)

        # Extract input data
        matrix = np.array(data[0])
        vector = np.array(data[1])

        # Encrypt matrix in row-major format
        ctm_matrix = onp.array(
            cc=cc,
            data=matrix,
            batch_size=total_slots,
            order=onp.ROW_MAJOR,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

        # Generate column sum keys
        ctm_matrix.extra["colkey"] = onp.sum_col_keys(keys.secretKey)

        # Encrypt vector in column-major format
        ctv_vector = onp.array(
            cc=cc,
            data=vector,
            batch_size=total_slots,
            order=onp.COL_MAJOR,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

        # Perform matrix-vector multiplication
        ctv_result = ctm_matrix @ ctv_vector

        # Decrypt result
        result = ctv_result.decrypt(keys.secretKey, unpack_type="original")

    return result


def fhe_matrix_vector_product_case2(params, data):
    """
    Execute matrix-vector product with FHE - Case 2:
    - Matrix: column-major
    - Vector: row-major
    - Result: column-major
    """
    params_copy = params.copy()
    total_slots = params_copy["ringDim"] // 2

    with suppress_stdout(False):
        # Generate crypto context
        cc, keys = gen_crypto_context(params_copy)

        # Extract input data
        matrix = np.array(data[0])
        vector = np.array(data[1])

        # Encrypt matrix in column-major format
        ctm_matrix = onp.array(
            cc=cc,
            data=matrix,
            batch_size=total_slots,
            order=onp.COL_MAJOR,
            fhe_type="C",
            mode="zero",
            public_key=keys.publicKey,
        )

        # Encrypt vector in row-major format
        ctv_vector = onp.array(
            cc=cc,
            data=vector,
            batch_size=total_slots,
            order=onp.ROW_MAJOR,
            fhe_type="C",
            mode="zero",
            target_cols=ctm_matrix.nrows,  # Important for row-major vector
            public_key=keys.publicKey,
        )

        # Generate row sum keys
        ctm_matrix.extra["rowkey"] = onp.sum_row_keys(
            keys.secretKey, ctm_matrix.nrows, ctm_matrix.batch_size
        )

        # Perform matrix-vector multiplication
        ctv_result = ctm_matrix @ ctv_vector

        # Decrypt result
        result = ctv_result.decrypt(keys.secretKey, unpack_type="original")

    return result


class TestMatrixVectorProduct(MainUnittest):
    """Test class for matrix-vector product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix-vector product."""
        # Define the two test cases
        operations = [
            ("case1_rowmajor_colmajor", fhe_matrix_vector_product_case1),
            ("case2_colmajor_rowmajor", fhe_matrix_vector_product_case2),
        ]

        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 4, 8]
        test_counter = 1

        for op_name, func in operations:
            for param in ckks_param_list:
                for size in matrix_sizes:
                    # Skip large sizes for small ring dimensions
                    if size > 4 and param["ringDim"] < 8192:
                        continue

                    # Generate random test data
                    A = generate_random_array(size)
                    b = generate_random_array(size, 1)  # 1D vector

                    # Calculate expected result directly
                    expected = np.dot(A, b)

                    # Create test name with descriptive format
                    test_name = f"matvec_{op_name}_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                    # Generate the test case
                    cls.generate_test_case(
                        func=func,
                        test_name=test_name,
                        params=param,
                        input_data=[A, b],
                        expected=expected,
                        compare_fn=onp.check_equality,
                        debug=True,
                    )

                    test_counter += 1


if __name__ == "__main__":
    TestMatrixVectorProduct.run_test_summary(
        "Matrix-Vector Product", debug=True
    )
