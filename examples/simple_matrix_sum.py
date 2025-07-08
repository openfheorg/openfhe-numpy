import time
import numpy as np
from openfhe import *

import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name} ")
    print("*" * 60)

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected)
    print(
        f"\nMatch: {is_match} within tolerance: {onp.EPSILON}, Total Error: {error}"
    )
    return is_match, error


def run_total_sum_example(crypto_context, keys, ctm_x, matrix):
    """Run homomorphic total sum example."""
    # Generate rotation keys for total sum operations
    onp.gen_accumulate_rows_key(keys.secretKey, ctm_x.ncols)

    # Perform homomorphic total sum
    ctm_result = onp.sum(ctm_x)

    # Perform decryption
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Validate and print results
    validate_and_print_results(result, np.sum(matrix), "Total Sum")


def run_row_sum_example(crypto_context, keys, ctm_x, matrix):
    """Run homomorphic row sum example."""
    # Generate rotation keys for row sum operations
    ctm_x.extra["rowkey"] = onp.sum_row_keys(
        keys.secretKey, ctm_x.ncols, ctm_x.batch_size
    )

    # Perform homomorphic row sum
    ctm_result = onp.sum(ctm_x, axis=0)

    # Perform decryption
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Validate and print results
    validate_and_print_results(result, np.sum(matrix, axis=0), "Row Sum")


def run_column_sum_example(crypto_context, keys, ctm_x, matrix):
    """Run homomorphic column sum example."""
    # Generate rotation keys for column sum operations
    ctm_x.extra["colkey"] = onp.sum_col_keys(keys.secretKey, ctm_x.ncols)

    # Perform homomorphic column sum
    ctm_result = onp.sum(ctm_x, axis=1)

    # Perform decryption
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")

    # Validate and print results
    validate_and_print_results(result, np.sum(matrix, axis=1), "Column Sum")


def main():
    """
    Run a demonstration of homomorphic matrix sum using OpenFHE-NumPy.
    """

    # Setup CKKS parameters
    params = CCParamsCKKSRNS()

    # Generate crypto context
    crypto_context = GenCryptoContext(params)
    crypto_context.Enable(PKESchemeFeature.PKE)
    crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)
    crypto_context.Enable(PKESchemeFeature.ADVANCEDSHE)

    # Generate keys
    keys = crypto_context.KeyGen()
    crypto_context.EvalMultKeyGen(keys.secretKey)
    crypto_context.EvalSumKeyGen(keys.secretKey)

    # Get system parameters
    ring_dim = crypto_context.GetRingDimension()
    batch_size = (
        params.GetBatchSize() if params.GetBatchSize() else ring_dim // 2
    )

    print("\nCrypto Info:")
    print(f"  - Used slots: {batch_size}")
    print(f"  - Ring dimension: {crypto_context.GetRingDimension()}")

    # Sample input matrix - using a simple 3x2 matrix for demonstration
    # matrix = np.array(
    #     [
    #         [1.80521609, 0.46970757, 1.0],
    #         [7.82405472, 8.52768494, 2.0],
    #         [2.0, 1.0, 3.0],
    #     ]
    # )

    matrix = np.array(
        [
            [1, 1],
            [2, 1],
            [3, 1],
            [2, 6],
        ]
    )

    print(f"\nInput Matrix:\n{matrix}")

    # We can use the 'array' function to encode or encrypt a matrix:
    # 1. Before processing, the matrix is flattened to a list using:
    #        - row-major concatenation (order="ROW_MAJOR") - default
    #        - column-major concatenation (order="COL_MAJOR")
    # 2. Use - type="C" - encrypt the array to ciphertext - default
    #        - type="P" - encode the array to plaintext

    ctm_x = onp.array(
        cc=crypto_context,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Run sum examples
    run_total_sum_example(crypto_context, keys, ctm_x, matrix)
    run_row_sum_example(crypto_context, keys, ctm_x, matrix)
    run_column_sum_example(crypto_context, keys, ctm_x, matrix)


if __name__ == "__main__":
    main()
