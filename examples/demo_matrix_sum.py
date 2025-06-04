import time
import numpy as np
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    FIXEDAUTO,
    HYBRID,
    UNIFORM_TERNARY,
)

from tensor.constructors import array
from operations.crypto_context import (gen_accumulate_rows_key, sum_row_keys, sum_col_keys)
from operations.matrix_api import sum
from utils.matlib import check_equality_matrix


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
        (CryptoContext, CCParamsCKKSRNS, KeyPair)
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
    Run a demonstration of homomorphic matrix accumulation using OpenFHE-NumPy.
    """
    mult_depth = 8
    cc, params, keys = gen_crypto_context(mult_depth)

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
    slots = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    tensor = array(cc, matrix, slots, public_key=keys.publicKey)

    print(f"slots = {slots}, dim = {cc.GetRingDimension()}, ncols = {tensor.ncols}")

    print("\n********** HOMOMORPHIC SUM BY ALL ENTRIES **********")
    #  Generate rotation keys for column operations
    start_keygen = time.time()
    gen_accumulate_rows_key(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    tensor_result = sum(tensor)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = tensor_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Row Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Row Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.sum(matrix)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC SUM BY ROWS **********")
    #  Generate rotation keys for column operations
    start_keygen = time.time()
    tensor.extra["rowkey"] = sum_row_keys(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    tensor_result = sum(tensor, axis=0)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = tensor_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Row Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Row Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.sum(matrix, axis=0)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")

    print("\n********** HOMOMORPHIC SUM BY COLUMNS **********")

    #  Generate rotation keys for column operations
    start_keygen = time.time()
    tensor.extra["colkey"] = sum_col_keys(keys.secretKey, tensor.ncols)
    end_keygen = time.time()

    # Perform homomorphic column accumulation
    start_acc = time.time()
    tensor_result = sum(tensor, axis=1)
    end_acc = time.time()

    # Perform decryption
    start_dec = time.time()
    result = tensor_result.decrypt(keys.secretKey, format_type="reshape")
    end_dec = time.time()
    result = np.round(result, decimals=1)
    print(f"Col Accumulation Time (KeyGen): {(end_keygen - start_keygen) * 1000:.2f} ms")
    print(f"Col Accumulation Time (Eval): {(end_acc - start_acc) * 1000:.2f} ms")
    print(f"Time for decryption: {(end_dec - start_dec) * 1000:.2f} ms")

    # Compare with plain result
    expected = np.sum(matrix, axis=1)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = check_equality_matrix(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
