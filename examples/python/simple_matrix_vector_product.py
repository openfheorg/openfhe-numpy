import numpy as np
from openfhe import *
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

    # ---------------------------------------------
    # Case 1:
    #   - Matrix Packing Style: row-major
    #   - Vector Packing Style: col-major
    #   - Product Packing Style: row-major
    # ---------------------------------------------

    # When mode = "tile", the matrix is repeated (tiled) across all slots.
    #
    # Example:
    #   Matrix:
    #     1 2 3
    #     4 5 6
    #
    #   Row-wise packing:
    #     1 2 3 0 4 5 6 0  1 2 3 0 4 5 6 0
    #
    #   Column-wise packing:
    #     1 4 2 5 3 6 0 0  1 4 2 5 3 6 0 0
    #
    # For a vector [1 2 3], both row-wise and column-wise packing
    # produce the same result:
    #   1 2 3 0 1 2 3 0 1 2 3 0

    ctm_m_rm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="tile",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    ctm_m_rm.extra["colkey"] = onp.sum_col_keys(keys.secretKey)
    ctv_v_cm = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
        mode="tile",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_result_rm = ctm_m_rm @ ctv_v_cm
    print(f"dim = {ctv_result_rm.ndim}")
    print(f"shape = {ctv_result_rm.shape}")
    print(f"original_shape = {ctv_result_rm.original_shape}")

    print(f"result.info = {ctv_result_rm.info}")
    result_rm = ctv_result_rm.decrypt(keys.secretKey, unpack_type="original")
    is_match_rm, error_rm = onp.check_equality(result_rm, expected)

    print("\n--------------------------------\n")
    print("Case 1:")
    print("  - Matrix Packing Style: row-major")
    print("  - Vector Packing Style: col-major")
    print("  - Product Packing Style: row-major")
    print(f"\nDecrypted Result:\n{result_rm}")
    print(f"\nMatch: {is_match_rm}, Total Error: {error_rm}")

    # ---------------------------------------------
    # Case 2:
    #   - Matrix Packing Style: col-major
    #   - Vector Packing Style: row-major
    #   - Product Packing Style: col-major
    # ---------------------------------------------
    ctm_m_cm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # In this function, when you choose target_cols = a,
    # the original vector of shape (n, 1) is expanded into
    # a matrix of shape (n, a), and then packed accordingly.
    #
    # For matrix-vector multiplication, target_cols is set
    # to the number of columns in the matrix.

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

    ctm_m_cm.extra["rowkey"] = onp.sum_row_keys(keys.secretKey, ctm_m_cm.ncols, ctm_m_cm.batch_size)
    ctv_result_cm = ctm_m_cm @ ctv_v_rm
    result_cm = ctv_result_cm.decrypt(keys.secretKey, unpack_type="original")
    is_match_cm, error_cm = onp.check_equality(result_cm, expected)

    print("\n--------------------------------\n")
    print("Case 2:")
    print("  - Matrix Packing Style: col-major")
    print("  - Vector Packing Style: row-major")
    print("  - Product Packing Style: col-major")
    print(f"\nDecrypted Result:\n{result_cm}")
    print(f"\nMatch: {is_match_cm}, Total Error: {error_cm}")


if __name__ == "__main__":
    demo()
