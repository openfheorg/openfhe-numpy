import time
import numpy as np
from openfhe import *

from tensor.constructors import array
from utils.matlib import check_equality_matrix


def gen_crypto_context(mult_depth):
    """
    Generate a CryptoContext and key pair for CKKS encryption.
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

    return cc, keys


def demo():
    """
    Run a demonstration of homomorphic matrix addition using OpenFHE-NumPy.
    """
    cc, keys = gen_crypto_context(4)
    ring_dim = cc.GetRingDimension()
    total_slots = ring_dim // 2

    # Sample input matrices (4x4)
    matA = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    matB = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    print("Matrix A:\n", matA)
    print("Matrix B:\n", matB)

    # Encrypt both matrices
    ctm_matA = array(cc, matA, total_slots, public_key=keys.publicKey)
    ctm_matB = array(cc, matB, total_slots, public_key=keys.publicKey)

    print("\n********** HOMOMORPHIC ADDITIONS **********")
    print("1. Matrix Additions...")

    # Timing the homomorphic addition
    start_add = time.time()
    ctm_result = ctm_matA + ctm_matB
    end_add = time.time()
    print(f"Time for homomorphic addition: {end_add - start_add:.4f} seconds")

    # Timing the decryption
    start_dec = time.time()
    result = ctm_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Time for decryption: {end_dec - start_dec:.4f} seconds")

    # Compare with plain result
    expected = matA + matB

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
