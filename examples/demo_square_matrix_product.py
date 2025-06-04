"""
Demo of matrix-vector product using homomorphic encryption with OpenFHE-NumPy.
"""

import time
import numpy as np
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
    HEStd_NotSet,
)

from tensor.constructors import array
from utils.matlib import check_equality_matrix
import openfhe_numpy as onp


def gen_crypto_context(mult_depth, ring_dim=0):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    ring_dim : int
        Ring dimension (must be power of two).
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """
    params = CCParamsCKKSRNS()
    if ring_dim != 0:
        params.SetRingDim(ring_dim)
        params.SetBatchSize(ring_dim // 2)
        params.SetSecurityLevel(HEStd_NotSet)
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
    Run a demonstration of homomorphic matrix-vector multiplication using OpenFHE-NumPy.
    """
    ring_dim = 2**15
    mult_depth = 4
    total_slots = ring_dim // 2

    # Initialize crypto context
    start_setup = time.time()
    cc, keys = gen_crypto_context(mult_depth, ring_dim)
    end_setup = time.time()
    print(f"Crypto context setup time: {(end_setup - start_setup) * 1000:.2f} ms")

    # Sample input matrices (8x8) and vector (8)
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

    B = np.array(
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

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # Encrypt matrix
    start_enc_matrix = time.time()
    ctm_matA = array(cc, A, total_slots, public_key=keys.publicKey)

    enc_time = time.time()
    print(f"Matrix [2] encryption time: {(enc_time - start_enc_matrix) * 1000:.2f} ms")

    ncols = ctm_matA.ncols

    # Generate keys for sum operations
    start_key_gen = time.time()
    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, ctm_matA.ncols)
    end_key_gen = time.time()
    print(f"Sum keys generation time: {(end_key_gen - start_key_gen) * 1000:.2f} ms")

    print("\n********** HOMOMORPHIC Matrix Matrix Product **********")

    # Perform homomorphic matrix-vector multiplication
    start_matmul = time.time()
    ct_result = ctm_matA @ ctm_matB
    end_matmul = time.time()
    print(
        f"Homomorphic matrix-matrix multiplication time: {(end_matmul - start_matmul) * 1000:.2f} ms"
    )

    # Decrypt result
    start_dec = time.time()
    result = ct_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()
    print(f"Decryption time: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = A @ B
    print(f"\nExpected:\n{A @ B}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
