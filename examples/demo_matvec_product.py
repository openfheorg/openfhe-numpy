# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *

from openfhe_numpy import array, sum_col_keys, pack_vec_row_wise, check_equality_vector


def gen_crypto_context(mult_depth):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetKeySwitchTechnique(HYBRID)
    params.SetSecretKeyDist(UNIFORM_TERNARY)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, params, keys


def demo():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """
    mult_depth = 10
    cc, params, keys = gen_crypto_context(mult_depth)

    total_slots = cc.GetRingDimension() // 2
    print("\n****** CRYPTO PARAMETERS ******")
    print(f"Total Slots: {total_slots}")
    print("*******************************")

    # Sample input matrices (8x8)
    A = np.array(
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

    b = np.array(
        [7, 0, 1, 3, 5, 0, 1, 8],
    )

    print("\nInput")
    print("Matrix A:\n", A)
    print("Vector b:\n", b)

    # Encrypt both matrices
    ctm_matrix = array(cc, A, total_slots, public_key=keys.publicKey)
    ncols = ctm_matrix.ncols
    sumkey = sum_col_keys(cc, keys.secretKey)
    ctm_matrix.extra["colkey"] = sumkey
    ctv_vector = array(cc, b, total_slots, ncols, "C", public_key=keys.publicKey)

    print("\n********** Homomorphic Matrix Vector Product **********")
    ctv_result = ctm_matrix @ ctv_vector

    result = ctv_result.decrypt(keys.secretKey, format_type="reshape")
    expected = pack_vec_row_wise((A @ b), ncols, total_slots)

    print(f"\nExpected:\n{A @ b}")
    print(f"\nPacked Expected:\n{expected[:32]}")
    print(f"\nDecrypted Result:\n{result[:32]}")

    is_match, error = check_equality_vector(result[:32], expected[:32])
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
