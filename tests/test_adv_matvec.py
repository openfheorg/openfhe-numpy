import numpy as np

# Direct imports from main_unittest
from main_unittest import (
    generate_random_array,
    gen_crypto_context,
    load_ckks_params,
    suppress_stdout,
    MainUnittest,
)

from tensor.constructors import array
from operations.crypto_context import sum_col_keys
from utils.utils import next_power_of_two, pack_vec_row_wise

def fhe_matrix_vector_product(params, input):
    """Execute matrix-vector product with FHE."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout(False):  # Allow output for debugging
        cc, keys = gen_crypto_context(params)
        public_key = keys.publicKey

        matrix = np.array(input[0])
        vector = np.array(input[1])

        # Encrypt the matrix
        ctm_matrix = array(cc, matrix, total_slots, public_key=keys.publicKey)
        ncols = ctm_matrix.ncols

        # Generate column sum keys
        sumkey = sum_col_keys(cc, keys.secretKey)
        ctm_matrix.extra["colkey"] = sumkey

        # Encrypt the vector in column-major format
        ctv_vector = array(cc, vector, total_slots, ncols, "C", public_key=keys.publicKey)

        # Perform matrix-vector multiplication
        ctv_result = ctm_matrix @ ctv_vector

        # Decrypt result
        result = ctv_result.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestMatrixVectorProduct(MainUnittest):
    """Test class for matrix-vector product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for matrix-vector product."""
        ckks_param_list = load_ckks_params()
        matrix_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in matrix_sizes:
                # Generate random test data
                A = generate_random_array(size)
                b = generate_random_array(size, 1)

                # Calculate expected result
                size = next_power_of_two(size)
                expected = pack_vec_row_wise((A @ b), size, param["ringDim"] // 2)

                # Create test with descriptive name
                name = "TestMatrixVectorProduct"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Generate test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_matrix_vector_product, name, test_name, param, [A, b], expected, debug=True
                )

                # Register test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestMatrixVectorProduct._generate_test_cases()


if __name__ == "__main__":
    TestMatrixVectorProduct.run_test_summary("Matrix-Vector Product", debug=True)
