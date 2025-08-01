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

    # Sample input
    # matrix = np.array(
    #     [
    #         [0, 7, 8, 10, 1, 2, 7],
    #         [0, 1, 1, 9, 7, 5, 1],
    #         [8, 8, 4, 5, 8, 2, 6],
    #         [1, 0, 0, 1, 10, 3, 1],
    #         [7, 8, 2, 5, 3, 2, 10],
    #         [0, 3, 4, 10, 10, 5, 2],
    #         [2, 5, 0, 2, 8, 8, 5],
    #         [5, 1, 10, 6, 2, 8, 6],
    #     ]
    # )

    # vector = np.array(
    #     [7, 0, 1, 3, 5, 0, 1],
    # )

    matrix = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]])

    vector = np.array(
        [1, 1, 1],
    )

    print("\nInput")
    print("\nMatrix:\n", matrix)
    print("\nVector:\n", vector)

    print("\n" + "*" * 60)
    print(f"* Homomorphic Matrix Vector Product")
    print("*" * 60)

    expected = matrix @ vector
    print(f"\nExpected:\n{expected}")

    # Case 1:
    #   - Matrix Packing Style: row-major
    #   - Vector Packing Style: col-major
    #   - Product Packing Style: row-major

    ctm_m_rm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        public_key=keys.publicKey,
    )
    ctm_m_rm.extra["colkey"] = onp.sum_col_keys(keys.secretKey)
    ctv_v_cm = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_result_rm = ctm_m_rm @ ctv_v_cm
    result_rm = ctv_result_rm.decrypt(keys.secretKey, unpack_type="original")
    is_match_rm, error_rm = onp.check_equality(result_rm, expected)

    print("Case 1:")
    print("  - Matrix Packing Style: row-major")
    print("  - Vector Packing Style: col-major")
    print("  - Product Packing Style: row-major")
    print(f"\nDecrypted Result:\n{result_rm}")
    print(f"\nMatch: {is_match_rm}, Total Error: {error_rm}")

    # Case 2:
    #   - Matrix Packing Style: col-major
    #   - Vector Packing Style: row-major
    #   - Product Packing Style: col-major
    ctm_m_cm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_v_rm = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        target_cols=ctm_m_cm.nrows,
        public_key=keys.publicKey,
    )

    ctm_m_cm.extra["rowkey"] = onp.sum_row_keys(
        keys.secretKey, ctm_m_cm.ncols, ctm_m_cm.batch_size
    )

    ctv_result_cm = ctm_m_cm @ ctv_v_rm
    result_cm = ctv_result_cm.decrypt(keys.secretKey, unpack_type="original")
    is_match_cm, error_cm = onp.check_equality(result_cm, expected)

    print("Case 2:")
    print("  - Matrix Packing Style: col-major")
    print("  - Vector Packing Style: row-major")
    print("  - Product Packing Style: col-major")
    print(f"\nDecrypted Result:\n{result_cm}")
    print(f"\nMatch: {is_match_cm}, Total Error: {error_cm}")


if __name__ == "__main__":
    demo()
