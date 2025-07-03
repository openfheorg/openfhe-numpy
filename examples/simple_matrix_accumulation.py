import numpy as np
from openfhe import *

import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name} *")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")
    is_match, error = onp.check_equality_matrix(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")
    return is_match, error


def run_row_accumulation_example(cc, keys, ctm_a, matrix):
    """Run homomorphic row accumulation example."""

    # Generate rotation keys for row operations
    onp.gen_accumulate_rows_key(keys.secretKey, ctm_a.ncols)

    # Perform homomorphic row accumulation
    ctm_result = onp.cumsum(ctm_a, axis=0)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=0)

    # Validate and print results
    validate_and_print_results(result, expected, "Row Accumulation")


def run_column_accumulation_example(cc, keys, ctm_a, matrix):
    """Run homomorphic column accumulation example."""

    # Generate rotation keys for column operations
    onp.gen_accumulate_cols_key(keys.secretKey, ctm_a.ncols)

    # Perform homomorphic column accumulation
    ctm_result = onp.cumsum(ctm_a, axis=1)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    expected = np.cumsum(matrix, axis=1)

    # Validate and print results
    validate_and_print_results(result, expected, "Column Accumulation")


def main():
    """
    Run a demonstration of homomorphic matrix accumulation using OpenFHE-NumPy.
    """

    # Cryptographic parameters
    mult_depth = 8
    scale_mod_size = 59

    # Setup CKKS parameters
    parameters = CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(scale_mod_size)
    parameters.SetFirstModSize(60)
    parameters.SetScalingTechnique(FIXEDAUTO)

    # Generate crypto context
    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    # Generate keys
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    # Get system parameters
    ring_dim = cc.GetRingDimension()
    total_slots = ring_dim // 2

    print(f"\nCrypto Info:")
    print(f"  - Used slots: {total_slots}")
    print(f"  - Ring dimension: {cc.GetRingDimension()}")

    # Sample input matrix (3x8)
    matrix = [
        [0, 7, 8, 10, 1, 2, 7, 6],
        [0, 1, 1, 9, 7, 5, 1, 7],
        [8, 8, 4, 5, 8, 2, 6, 1],
    ]

    print(f"\nInput Matrix\n{np.array(matrix)}")

    # Encrypt matrix
    # We can use the 'array' function to encode or encrypt a matrix:
    # - type="C" - encrypt to ciphertext - default
    # - type="P" - encode to plaintext
    # Before processing, the matrix is flattened to a list using:
    # - row-major concatenation (order="ROW_MAJOR") - default
    # - column-major concatenation (order="COL_MAJOR")

    ctm_a = onp.array(cc, matrix, total_slots, order=onp.ROW_MAJOR, public_key=keys.publicKey)

    # Run accumulation examples
    run_row_accumulation_example(cc, keys, ctm_a, matrix)
    run_column_accumulation_example(cc, keys, ctm_a, matrix)


if __name__ == "__main__":
    main()
