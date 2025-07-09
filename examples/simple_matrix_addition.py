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


def main():
    """
    Run a demonstration of homomorphic matrix addition using OpenFHE-NumPy,
    for both power-of-two and non–power-of-two dimensions.
    """
    # Cryptographic setup
    scale_mod_size = 59

    params = CCParamsCKKSRNS()
    params.SetScalingModSize(scale_mod_size)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {batch_size}")

    # --- Case 1: power-of-two matrices (44) ---
    matrix_a = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
    ]
    matrix_b = [
        [2, 5, 7, 8],
        [1, 4, 1, 2],
        [1, 2, 3, 3],
        [0, 1, 2, 0],
    ]

    print("\n### POWER-OF-TWO MATRICES (4x4) ###")
    print("Matrix A:\n", np.array(matrix_a))
    print("Matrix B:\n", np.array(matrix_b))

    # Encrypt A directly to ciphertext
    ctm_a = onp.array(
        cc=cc,
        data=matrix_a,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Encode B as plaintext then encrypt
    ptm_b = onp.array(
        cc=cc,
        data=matrix_b,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="P",
        mode="zero",
        public_key=keys.publicKey,
    )

    ctm_b = ptm_b.encrypt(cc, keys.publicKey)

    # Homomorphic operations
    ctm_add_ab = ctm_a + ctm_b
    ctm_sub_ab = ctm_a - ctm_b
    ctm_add_mix = onp.add(ctm_a, ptm_b)

    # Decrypt (restore original shape)
    res_add_ab = ctm_add_ab.decrypt(keys.secretKey, unpack_type="original")
    res_sub_ab = ctm_sub_ab.decrypt(keys.secretKey, unpack_type="original")
    res_mix_ab = ctm_add_mix.decrypt(keys.secretKey, unpack_type="original")

    # Validate
    validate_and_print_results(
        res_add_ab, np.add(matrix_a, matrix_b), "Encrypt(A) + Encrypt(B)"
    )
    validate_and_print_results(
        res_sub_ab, np.subtract(matrix_a, matrix_b), "Encrypt(A) - Encrypt(B)"
    )
    validate_and_print_results(
        res_mix_ab, np.add(matrix_a, matrix_b), "Encrypt(A) + Encode(B)"
    )

    # --- Case 2: non–power-of-two matrices (33) ---
    matrix_c = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ]
    matrix_d = [
        [2, 5, 7],
        [1, 4, 2],
        [1, 2, 3],
    ]

    print("\n### NON–POWER-OF-TWO MATRICES (33) ###")
    print("Matrix C:\n", np.array(matrix_c))
    print("Matrix D:\n", np.array(matrix_d))

    # Encrypt C and D
    ctm_c = onp.array(
        cc=cc,
        data=matrix_c,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )
    ctm_d = onp.array(
        cc=cc,
        data=matrix_d,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )
    # Homomorphic operations
    ctm_add_cd = onp.add(ctm_c, ctm_d)
    ctm_sub_cd = onp.subtract(ctm_c, ctm_d)

    # Decrypt
    res_add_cd = ctm_add_cd.decrypt(keys.secretKey, unpack_type="original")
    res_sub_cd = ctm_sub_cd.decrypt(keys.secretKey, unpack_type="original")

    # Validate
    validate_and_print_results(
        res_add_cd, np.add(matrix_c, matrix_d), "Encrypt(C) + Encrypt(D)"
    )
    validate_and_print_results(
        res_sub_cd, np.subtract(matrix_c, matrix_d), "Encrypt(C) - Encrypt(D)"
    )


if __name__ == "__main__":
    main()
