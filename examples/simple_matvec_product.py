import time

# Import OpenFHE and Numpy
import numpy as np
from openfhe import *

# Import OpenFHE NumPy-style interface
import openfhe_numpy as onp


def demo():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """
    mult_depth = 10

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

    batch_size = cc.GetRingDimension() // 2
    print("\n****** CRYPTO PARAMETERS ******")
    print(f"Total Slots: {batch_size}")
    print("*******************************")

    # Sample input matrices (8x8)
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

    vector = np.array(
        [7, 0, 1, 3, 5, 0, 1, 8],
    )

    print("\nInput")
    print("Matrix:\n", matrix)
    print("Vector:\n", vector)

    # Encryption
    ctm_matrix = onp.array(cc, matrix, batch_size, onp.ROW_MAJOR, public_key=keys.publicKey)
    ncols = ctm_matrix.ncols
    sumkey = onp.sum_col_keys(keys.secretKey)
    ctm_matrix.extra["colkey"] = sumkey
    ctv_vector = onp.array(cc, vector, batch_size, onp.COL_MAJOR, public_key=keys.publicKey, target_cols=ncols)

    print("\n********** Homomorphic Matrix Vector Product **********")
    ctv_result = ctm_matrix @ ctv_vector

    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
    expected = matrix @ vector

    print(f"\nExpected:\n{matrix @ vector}")
    print(f"\nDecrypted Result:\n{result}")

    is_match, error = onp.check_equality_vector(result, expected)
    print(f"\nMatch: {is_match}, Total Error: {error:.6f}")


if __name__ == "__main__":
    demo()
