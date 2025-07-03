import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print vector results."""
    print("\n" + "*" * 60)
    print(f"{operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")
    if expected.ndim == 0:
        is_match, error = onp.check_single_equality(computed, expected)
    elif expected.ndim == 1:
        is_match, error = onp.check_equality_vector(computed, expected)
    else:
        is_match, error = onp.check_equality_matrix(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")
    return is_match, error


def main():
    """
    Run a demonstration of homomorphic vector operations using OpenFHE-NumPy:
      • addition
      • subtraction
      • transpose
      • elementwise multiplication
      • inner product via *
    """
    # Cryptographic setup
    mult_depth = 4
    params = CCParamsCKKSRNS()

    cc = GenCryptoContext(params)
    params.SetMultiplicativeDepth(mult_depth)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    total_slots = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {total_slots}")

    # Sample input vectors
    vector_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector_b = [4.0, 0.0, 1.0, 3.0, 6.0]

    print("\nInput vectors")
    print("vector_a:", vector_a)
    print("vector_b:", vector_b)

    # Encrypt vector_a directly to ciphertext
    ctv_a = onp.array(cc, vector_a, total_slots, public_key=keys.publicKey)
    ctv_b = onp.array(cc, vector_b, total_slots, public_key=keys.publicKey)

    # 1) Addition
    ctv_add = ctv_a + ctv_b
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_add, np.add(vector_a, vector_b), f"Vector Addition  \n{vector_a} \n{vector_b}")

    # 2) Subtraction
    ctv_sub = ctv_a - ctv_b
    res_sub = ctv_sub.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_sub, np.subtract(vector_a, vector_b), f"Vector Subtraction  \n{vector_a} and \n{vector_b}"
    )

    # 3) Transpose
    onp.gen_transpose_keys(keys.secretKey, ctv_a)
    ctv_a_T = onp.transpose(ctv_a)
    res_T = ctv_a_T.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_T, np.transpose(vector_a), f"Tranpose \n{vector_a}")

    # 4) Elementwise multiplication
    ctv_mul = ctv_a * ctv_b
    res_mul = ctv_mul.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_mul, np.multiply(vector_a, vector_b), f"Elementwise multiplication \n{vector_a} \n{vector_b} "
    )

    # 5) Elementwise multiplication
    ctv_mul_scalar = ctv_a * 7
    res_mul_scalar = ctv_mul_scalar.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_mul_scalar, np.multiply(vector_a, 7), f"Scalar Multiplcation \n{vector_a} and 7")

    # 6) Inner product
    # We can use onp.dot(ctv_a, ctv_b) as well
    ctv_inner = ctv_a @ ctv_b
    res_inner_decrypted = ctv_inner.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_inner_decrypted, np.dot(vector_a, vector_b), f"Inner product of \n{vector_a} \n{vector_b}"
    )

    # 7) Sum
    ctv_sum = onp.sum(ctv_a)
    res_sum_decrypted = ctv_sum.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(res_sum_decrypted, np.sum(vector_a), "Sum of vector\n" + str(vector_a))


if __name__ == "__main__":
    main()
