import numpy as np

# Direct imports from main_unittest
from tests.main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

from tensor.constructors import array
from operations.crypto_context import gen_accumulate_rows_key
from operations.matrix_api import cumsum

def fhe_matrix_sumcum_rows(params, input):
    """Execute matrix row summation with FHE."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout(False):  # Allow output for debugging
        matrix = np.array(input[0])

        if params["multiplicativeDepth"] < len(matrix):
            params["multiplicativeDepth"] = len(matrix) + 1

        cc, keys = gen_crypto_context(params)
        public_key = keys.publicKey

        matrixA = np.array(input[0])

        # Encrypt matrix
        ct_matrixA = array(cc, matrixA, total_slots, public_key=public_key)
        nrows = ct_matrixA.original_shape[0]

        # Generate accumulation keys
        gen_accumulate_rows_key(keys.secretKey, ct_matrixA.ncols)

        # Perform row-wise cumulative sum
        ct_result = cumsum(ct_matrixA, axis=0)

        # Decrypt result
        result = ct_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixSumCumRows(MainUnittest):
    """Test class for matrix row summation operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix row summation."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrix
                A = generate_random_array(size)

                # Calculate expected result directly
                expected = np.cumsum(A, axis=0)

                # Create test with descriptive name
                name = "TestMatrixSumCumRows"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_sumcum_rows, name, test_name, param, [A], expected, debug=True
                )

                # Register test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixSumCumRows._generate_test_cases()


if __name__ == "__main__":
    TestMatrixSumCumRows.run_test_summary("Matrix Cumulative Sum Rows", debug=True)
