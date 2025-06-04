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
from operations.crypto_context import gen_transpose_keys
from operations.matrix_api import transpose

def fhe_matrix_transpose(params, input):
    """Execute matrix transpose with FHE."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout(False):  # Allow output for debugging
        cc, keys = gen_crypto_context(params)
        public_key = keys.publicKey

        matrixA = np.array(input[0])

        # Encrypt matrix
        ct_matrixA = array(cc, matrixA, total_slots, public_key=public_key)

        # Generate transpose keys
        gen_transpose_keys(keys.secretKey, ct_matrixA)

        # Perform transpose operation
        ct_result = transpose(ct_matrixA)

        # Decrypt result
        result = ct_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixTranspose(MainUnittest):
    """Test class for matrix transpose operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix transpose."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrix
                A = generate_random_array(size)

                # Calculate expected result
                expected = A.T

                # Create test with descriptive name
                name = "TestMatrixTranspose"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_transpose, name, test_name, param, [A], expected, debug=True
                )

                # Register test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixTranspose._generate_test_cases()


if __name__ == "__main__":
    TestMatrixTranspose.run_test_summary("Matrix Transpose", debug=True)
