import time

# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
import openfhe_numpy as onp


def demo():
    """
    Demonstrate square and rectangular homomorphic matrix transpose.
    """

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(3)
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()

    # Sample input matrix (8x8)
    matrix = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            [1, 0, 0, 1, 10, 3, 1, 7],
            [7, 8, 2, 5, 3, 2, 10, 9],
            [0, 3, 4, 10, 10, 5, 2, 5],
            [2, 5, 0, 2, 8, 8, 5, 9],
            [5, 1, 10, 6, 2, 8, 6, 3],
        ]
    )

    print("Matrix:\n", matrix)

    batch_size = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    ctm_x = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )
    print("\n" + "*" * 60)
    print(f"* HOMOMORPHIC MATRIX TRANSPOSE ")
    print("*" * 60)

    # Perform matrix tranpose on ciphertexts
    onp.gen_transpose_keys(keys.secretKey, ctm_x)
    ctm_result = onp.transpose(ctm_x)  # ctm_x.T

    # Decrypt the result
    result = ctm_result.decrypt(keys.secretKey)
    # Compare with plain result
    expected = matrix.T
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")

    rectangular_matrix = np.arange(1, 16, dtype=float).reshape(3, 5)
    print("\n" + "*" * 60)
    print("* RECTANGULAR TRANSPOSE + ORDER TRANSFORM + ADDITION")
    print("*" * 60)
    print(f"\nRectangular matrix:\n{rectangular_matrix}")

    ctm_rectangular = onp.array(
        cc=cc,
        data=rectangular_matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    onp.gen_transpose_keys(keys.secretKey, ctm_rectangular)
    ctm_rectangular_transposed = ctm_rectangular.transpose()

    # Transform changes only the packed order. The logical matrix is unchanged.
    onp.gen_transform_keys(keys.secretKey, ctm_rectangular_transposed)
    ctm_col_major = ctm_rectangular_transposed.transform(onp.COL_MAJOR)
    ctm_combined = ctm_col_major + 2.0

    combined_result = ctm_combined.decrypt(keys.secretKey)
    combined_expected = rectangular_matrix.T + 2.0
    print(f"\nExpected:\n{combined_expected}")
    print(f"\nDecrypted Result:\n{combined_result}")

    is_match, error = onp.check_equality(combined_result, combined_expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")


if __name__ == "__main__":
    demo()
