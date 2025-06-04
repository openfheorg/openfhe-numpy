import numpy as np
import openfhe_numpy as onp

# Direct imports from main_unittest
from main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

from tensor.constructors import array

def fhe_square_matrix_product(params, input):
    """Execute square matrix product with FHE."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout(False):  # Allow output for debugging
        cc, keys = gen_crypto_context(params)
        public_key = keys.publicKey

        A = np.array(input[0])
        B = np.array(input[1])

        ctm_matA = array(cc, A, total_slots, public_key=keys.publicKey)
        ctm_matB = array(cc, B, total_slots, public_key=keys.publicKey)

        # Generate rotation keys for matrix multiplication
        onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_matA.ncols)

        # Perform matrix multiplication
        ctm_result = ctm_matA @ ctm_matB

        # Decrypt and reshape result
        result = ctm_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestSquareMatrixProduct(MainUnittest):
    """Test class for square matrix product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for square matrix product."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test matrices
                matrixA = generate_random_array(size)
                matrixB = generate_random_array(size)

                # Calculate expected result
                expected = np.array(matrixA) @ np.array(matrixB)

                # Create test with descriptive name
                name = "TestSquareMatrixProduct"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_square_matrix_product,
                    name,
                    test_name,
                    param,
                    [matrixA, matrixB],
                    expected,
                    debug=True,
                )

                # Register test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestSquareMatrixProduct._generate_test_cases()


if __name__ == "__main__":
    TestSquareMatrixProduct.run_test_summary("Square Matrix Product", debug=True)
