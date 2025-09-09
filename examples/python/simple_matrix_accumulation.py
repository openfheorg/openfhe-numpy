import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")
    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


def run_row_accumulation_example(cc, keys, ctm_x, matrix):
    """Run homomorphic row accumulation example."""

    # Generate rotation keys for row operations
    if ctm_x.order == onp.ROW_MAJOR:
        onp.gen_accumulate_rows_key(keys.secretKey, ctm_x.ncols)
    elif ctm_x.order == onp.COL_MAJOR:
        onp.gen_accumulate_cols_key(keys.secretKey, ctm_x.nrows)
    else:
        raise ValueError("Invalid order.")

    # Perform homomorphic row accumulation
    ctm_result = onp.cumulative_sum(ctm_x, axis=0)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    # NumPy >= 2.0: please use numpy.cumulative_sum instead of numpy.cumsum.
    expected = np.cumsum(matrix, axis=0)

    # Validate and print results
    validate_and_print_results(result, expected, "Row Accumulation")


def run_column_accumulation_example(cc, keys, ctm_x, matrix):
    """Run homomorphic column accumulation example."""

    # Generate rotation keys for column operations
    if ctm_x.order == onp.ROW_MAJOR:
        onp.gen_accumulate_cols_key(keys.secretKey, ctm_x.ncols)
    elif ctm_x.order == onp.COL_MAJOR:
        onp.gen_accumulate_rows_key(keys.secretKey, ctm_x.nrows)
    else:
        raise ValueError("Invalid order.")

    # Perform homomorphic column accumulation
    ctm_result = onp.cumulative_sum(ctm_x, axis=1)
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Compare with plain result
    # NumPy >= 2.0: please use numpy.cumulative_sum instead of numpy.cumsum.
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
    batch_size = ring_dim // 2

    print(f"\nCrypto Info:")
    print(f"  - Used slots: {batch_size}")
    print(f"  - Ring dimension: {cc.GetRingDimension()}")

    # Sample input matrix (3x8)
    # matrix = [
    #     [0.0, 7.2, 8.5, 10.0, 1.5, 2.3333, 7.125, 6.0],
    #     [0.0, 1.0, 1.414, 9.1, 7.07, 5.5, 1.25, 7.9],
    #     [8.08, 8.0, 4.55, 5.625, 8.125, 2.0, 6.618, 0.33231],
    # ]

    # matrix = [
    #     [1, 3, 1, 1],
    #     [2, 2, 2, 2],
    #     [3, 3, 3, 3],
    #     [3, 3, 3, 3],
    #     [3, 3, 3, 3],
    # ]

    matrix = [[1.26475507, 2.15868416], [1.16980177, 9.97609032]]

    print(f"\nInput Matrix\n{np.array(matrix)}")

    # Encrypt matrix
    # We can use the 'array' function to encode or encrypt a matrix:
    # - type="C" - encrypt to ciphertext - default
    # - type="P" - encode to plaintext
    # Before processing, the matrix is flattened to a list using:
    # - row-major concatenation (order="ROW_MAJOR") - default
    # - column-major concatenation (order="COL_MAJOR")

    ctm_x = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        # order=onp.COL_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Run accumulation examples
    run_row_accumulation_example(cc, keys, ctm_x, matrix)
    run_column_accumulation_example(cc, keys, ctm_x, matrix)


if __name__ == "__main__":
    main()
