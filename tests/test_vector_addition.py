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
from operations.matrix_api import add

def fhe_addition(params, input):
    """Execute vector addition with FHE."""
    total_slots = params["ringDim"] // 2

    with suppress_stdout(False):  # Allow output for debugging
        cc, keys = gen_crypto_context(params)

        vector1 = np.array(input[0])
        vector2 = np.array(input[1])

        tensor1 = array(cc, vector1, total_slots, public_key=keys.publicKey)
        tensor2 = array(cc, vector2, total_slots, public_key=keys.publicKey)

        tensor_sum = add(tensor1, tensor2)
        result = tensor_sum.decrypt(keys.secretKey, format_type="reshape")

    return result


class TestVectorAddition(MainUnittest):
    """Test class for vector addition operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for vector addition."""
        ckks_param_list = load_ckks_params()
        vector_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in vector_sizes:
                # Generate random test vectors
                A = generate_random_array(rows=size, cols=1)
                B = generate_random_array(rows=size, cols=1)

                # Calculate expected result
                expected = np.array(A) + np.array(B)

                # Create test with descriptive name
                name = "TestVectorAddition"
                test_name = f"test_id_{test_counter}_ring_{param['ringDim']}_size_{size}"

                # Generate test case with debug output
                test_method = MainUnittest.generate_test_case(
                    fhe_addition, name, test_name, param, [A, B], expected, debug=True
                )

                # Register test method
                setattr(cls, test_name, test_method)
                test_counter += 1


TestVectorAddition._generate_test_cases()


if __name__ == "__main__":
    TestVectorAddition.run_test_summary("Vector Addition", debug=True)
