import numpy as np
import openfhe_numpy as onp

# Import directly from main_unittest - aligned with new framework
from main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

"""
Note: Column-wise cumulative sum requires sufficient multiplicative 
depth and ring dimension to accommodate the computational complexity.
Small ring dimensions (<4096) may result in high approximation errors.
"""


def fhe_matrix_sum(original_params, input):
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
        ctm_matrix = onp.array(cc, matrix, total_slots, public_key=public_key)

        if input[1] is None:
            cc.EvalSumKeyGen(keys.secretKey)
            ctm_result = onp.sum(ctm_matrix)
        elif input[1] == 0:
            onp.sum_row_keys(keys.secretKey, ctm_matrix.ncols)
            ctm_result = onp.sum(ctm_matrix, 0, True)
        elif input[1] == 1:
            onp.sum_col_keys(keys.secretKey, ctm_matrix.ncols)
            ctm_result = onp.sum(ctm_matrix, 1, True)
        else:
            ctm_result = None
        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixSum(MainUnittest):
    """Test class for matrix column summation operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix column summation."""
        ckks_param_list = load_ckks_params()
        # matrix_sizes = [2, 3, 8, 16]
        matrix_sizes = [2]
        test_counter = 1

        for sum_type in ["sum", "sumcols", "sumrows"]:
            for param in ckks_param_list:
                for size in matrix_sizes:
                    # Generate random test matrix
                    A = generate_random_array(size)

                    if sum_type == "sum":
                        name = "TestMatrixSum"
                        expected = np.sum(A)
                        _input = [A, None]
                    elif sum_type == "sumcols":
                        name = "TestMatrixSumCols"
                        expected = np.sum(A, axis=1)
                        _input = [A, 1]
                    else:
                        name = "TestMatrixSumRows"
                        expected = np.sum(A, axis=0)
                        _input = [A, 0]

                    # Calculate expected result directly

                    # Create test name with descriptive format
                    test_name = (
                        f"test_{sum_type}_{test_counter}_ring_{param['ringDim']}_size_{size}"
                    )

                    # Generate the test case with debug output
                    test_method = MainUnittest.generate_test_case(
                        fhe_matrix_sum,
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


TestMatrixSum._generate_test_cases()


if __name__ == "__main__":
    TestMatrixSum.run_test_summary("Matrix Sum", debug=True)
