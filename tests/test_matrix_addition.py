import numpy as np
import openfhe_numpy as onp

# Import directly from main_unittest instead of test_imports
# to ensure we're using the latest optimized functionality
from test_imports import *
from tensor.constructors import array
from operations.matrix_api import add


def fhe_matrix_addition(params, input):
    """Execute matrix addition with FHE."""
    # Calculate slot count
    total_slots = params["ringDim"] // 2

    # Get crypto context with optimized function
    cc, keys = gen_crypto_context(params)

    # Parse input matrices
    matrixA = np.array(input[0])
    matrixB = np.array(input[1])

    # Encrypt matrices
    ctm_A = array(cc, matrixA, total_slots, public_key=keys.publicKey)
    ctm_B = array(cc, matrixB, total_slots, public_key=keys.publicKey)

    # Perform homomorphic addition
    ctm_sum = add(ctm_A, ctm_B)

    # Decrypt and format result
    result = ctm_sum.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixAddition(MainUnittest):
    """Test class for matrix addition operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix addition."""
        # Load parameters and define test dimensions
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrices
                A = generate_random_array(size)
                B = generate_random_array(size)

                # Calculate expected result directly
                expected = np.array(A) + np.array(B)

                # Create test name with descriptive format
                name = "TestMatrixAddition"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate the test case with debug output
                test_method = MainUnittest.generate_test_case(
                    func=fhe_matrix_addition,
                    name=name,
                    test_name=test_name,
                    params=param,
                    input_data=[A, B],
                    expected=expected,
                    debug=True,  # Enable debug output
                )

                # Register the test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixAddition._generate_test_cases()

if __name__ == "__main__":
    # Interactive test execution mode
    # TestMatrixAddition.setUpClass()
    TestMatrixAddition.run_test_summary("Matrix Addition", debug=True)
