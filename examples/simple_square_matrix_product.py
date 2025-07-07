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


def run_matmul_example(cc, keys, A, B, description):
    """Run a homomorphic matrixxmatrix multiplication example."""
    print(f"\n--- {description} ---")
    print("Input A:\n", A)
    print("Input B:\n", B)

    # Encrypt A and B
    ctm_A = onp.array(
        cc, A, cc.GetRingDimension() // 2, public_key=keys.publicKey
    )
    ctm_B = onp.array(
        cc, B, cc.GetRingDimension() // 2, public_key=keys.publicKey
    )

    # Generate rotation keys for matrix multiplication
    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_A.ncols)

    # Perform homomorphic multiplication
    ct_res = ctm_A @ ctm_B

    # Decrypt result
    res = ct_res.decrypt(keys.secretKey, unpack_type="original")

    # Validate
    validate_and_print_results(res, A @ B, description)


def main():
    """
    Demonstrate homomorphic matrix multiplication for two separate cases:
      1) power-of-two dimensions
      2) non–power-of-two dimensions
    """
    # Cryptographic setup
    mult_depth = 4
    scale_mod_size = 59

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(FIXEDAUTO)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    print(f"\nCrypto context: ring_dim={ring_dim}, slots={ring_dim // 2}")

    # Case 1: power-of-two (8x8)
    A8 = np.array(
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
    B8 = np.array(
        [
            [6, 5, 4, 3, 2, 1, 0, 7],
            [7, 1, 1, 2, 7, 5, 9, 3],
            [4, 8, 8, 10, 8, 2, 1, 6],
            [7, 0, 0, 5, 10, 3, 4, 2],
            [9, 3, 2, 8, 3, 2, 1, 0],
            [5, 2, 4, 1, 10, 5, 8, 2],
            [9, 8, 0, 2, 8, 8, 7, 5],
            [3, 6, 10, 1, 2, 8, 4, 0],
        ]
    )
    run_matmul_example(cc, keys, A8, B8, "8x8 Matrix Product (power-of-two)")

    # Case 2: non–power-of-two (3x3)
    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B3 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    run_matmul_example(
        cc, keys, A3, B3, "3x3 Matrix Product (non-power-of-two)"
    )


if __name__ == "__main__":
    main()
