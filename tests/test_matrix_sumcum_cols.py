import numpy as np

# Import directly from main_unittest - aligned with new framework
from tests.main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

from tensor.constructors import array
from operations.crypto_context import gen_accumulate_cols_key
from operations.matrix_api import cumsum

"""
Note: Column-wise cumulative sum requires sufficient multiplicative 
depth and ring dimension to accommodate the computational complexity.
Small ring dimensions (<4096) may result in high approximation errors.
"""


def fhe_matrix_sumcum_cols(original_params, input):
    """Execute matrix column summation with suppressed output."""
    params = original_params.copy()

    with suppress_stdout(False):
        matrix = np.array(input[0])

        if params["multiplicativeDepth"] < len(matrix[0]):
            params["multiplicativeDepth"] = len(matrix[0]) + 1

        # Use gen_crypto_context for consistency with new framework
        cc, keys = gen_crypto_context(params)

        total_slots = params["ringDim"] // 2

        public_key = keys.publicKey
        ctm_matrix = array(cc, matrix, total_slots, public_key=public_key)

        gen_accumulate_cols_key(keys.secretKey, ctm_matrix.ncols)
        ctm_result = cumsum(ctm_matrix, 1, True)
        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixSumCumCols(MainUnittest):
    """Test class for matrix column summation operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix column summation."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrix
                A = generate_random_array(size)

                # Calculate expected result directly
                expected = np.cumsum(A, axis=1)

                # Create test name with descriptive format
                name = "TestMatrixSumCumCols"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate the test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_sumcum_cols,
                    name,
                    test_name,
                    param,
                    [A],
                    expected,
                    debug=True,
                )

                # Register the test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixSumCumCols._generate_test_cases()


if __name__ == "__main__":
    TestMatrixSumCumCols.run_test_summary("Matrix Cumulative Sum Columns", debug=True)
