import numpy as np
import openfhe_numpy as onp

# Import directly from main_unittest - aligned with new framework
from tests.main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

from tensor.constructors import array
from operations.crypto_context import sum_col_keys, sum_row_keys
from operations.matrix_api import mean

"""
Note: Column-wise cumulative mean requires sufficient multiplicative 
depth and ring dimension to accommodate the computational complexity.
Small ring dimensions (<4096) may result in high approximation errors.
"""


def fhe_matrix_mean(original_params, input):
    """Execute matrix column meanmation with suppressed output."""
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

        if input[1] is None:
            cc.EvalSumKeyGen(keys.secretKey)
            ctm_result = mean(ctm_matrix)
        elif input[1] == 0:
            sum_row_keys(keys.secretKey, ctm_matrix.ncols)
            ctm_result = mean(ctm_matrix, 0, True)
        elif input[1] == 1:
            sum_col_keys(cc, keys.secretKey, ctm_matrix.ncols)
            ctm_result = mean(ctm_matrix, 1, True)
        else:
            ctm_result = None
        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixmean(MainUnittest):
    """Test class for matrix column meanmation operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix column meanmation."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for mean_type in ["mean", "meancols", "meanrows"]:
            for param in ckks_param_list:
                for size in matrix_sizes:
                    # Generate random test matrix
                    A = generate_random_array(size)

                    if mean_type == "mean":
                        name = "TestMatrixmean"
                        expected = np.mean(A)
                        _input = [A, None]
                    elif mean_type == "meancols":
                        name = "TestMatrixmeanCols"
                        expected = np.mean(A, axis=1)
                        _input = [A, 1]
                    else:
                        name = "TestMatrixmeanRows"
                        expected = np.mean(A, axis=0)
                        _input = [A, 0]

                    # Calculate expected result directly

                    # Create test name with descriptive format
                    test_name = f"test_id_{mean_type}_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                    # Generate the test case with debug output
                    test_method = MainUnittest.generate_test_case(
                        fhe_matrix_mean,
                        name,
                        test_name,
                        param,
                        _input,
                        expected,
                        debug=True,
                    )

                    # Register the test method
                    setattr(cls, test_name, test_method)
                    test_counter += 1


TestMatrixmean._generate_test_cases()


if __name__ == "__main__":
    TestMatrixmean.run_test_meanmary("Matrix Cumulative mean Columns", debug=True)
