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
from operations.matrix_api import dot


def fhe_vector_dot(params, input):
    """Execute vector dot product with FHE."""
    total_slots = params["ringDim"] // 2

    # Use debug parameter for controlled output
    with suppress_stdout(False):  # Allow output
        cc, keys = gen_crypto_context(params)
        public_key = keys.publicKey

        input_a = np.array(input[0])
        input_b = np.array(input[1])

        if input_a.ndim == 1:
            ctm_input_a = array(cc, input_a, total_slots, public_key=keys.publicKey)
            ctm_input_b = array(cc, input_b, total_slots, public_key=keys.publicKey)
        else:
            ctm_input_a = array(cc, input_a, total_slots, public_key=keys.publicKey)
            ctm_input_b = array(cc, input_b, total_slots, public_key=keys.publicKey)

        ctm_dot = dot(ctm_input_a, ctm_input_b)
        result = ctm_dot.decrypt(keys.secretKey, format_type="reshape", new_shape=(1,))

    return result


class TestVectorInnerProduct(MainUnittest):
    """Test class for vector inner product operations."""

    @classmethod
    def _generate_test_cases(cls):
        """Generate test cases for vector inner product."""
        ckks_param_list = load_ckks_params()
        vector_sizes = [2, 3, 8, 16]
        test_counter = 1

        for param in ckks_param_list:
            for size in vector_sizes:
                # Generate random test vectors
                input_a = generate_random_array(size, 1)
                input_b = generate_random_array(size, 1)

                # Calculate expected result
                expected = np.dot(input_a, input_b)

                # Generate test case with descriptive name
                name = "TestVectorInnerProduct"
                test_name = f"test_id_{test_counter:03d}_ring_{param['ringDim']}_size_{size}"

                # Create test with debug enabled
                test_method = MainUnittest.generate_test_case(
                    fhe_vector_dot,
                    name,
                    test_name,
                    param,
                    [input_a, input_b],
                    [expected],
                    debug=True,
                )

                setattr(cls, test_name, test_method)
                test_counter += 1


TestVectorInnerProduct._generate_test_cases()


if __name__ == "__main__":
    TestVectorInnerProduct.run_test_summary("Vector Inner Product", debug=True)
