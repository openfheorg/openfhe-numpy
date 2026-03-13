import numpy as np
from openfhe import *
import openfhe_numpy as onp


def next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def demo_matrix():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(10)
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()

    # Sample input matrix (8x8)
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    np.set_printoptions(linewidth=np.inf)

    print("Matrix:\n", matrix)

    batch_size = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    ctm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    onp.generate_slicing_key(keys.secretKey, ctm.original_shape)

    ctm_result, expected = ctm[2][1], matrix[2][1]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n1. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[1:], matrix[1:]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n2. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[:, 1], matrix[:, 1]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n3. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[1:3, 1:3], matrix[1:3, 1:3]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n4. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[1:3], matrix[1:3]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n5. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[:, 0:2], matrix[:, 0:2]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n6. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[-1], matrix[-1]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n7. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[:, -1], matrix[:, -1]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n8. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[::2], matrix[::2]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n9. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[:, ::2], matrix[:, ::2]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n10. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[::2, ::2], matrix[::2, ::2]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n11. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[:, :], matrix[:, :]
    result = ctm_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\n12. Match: {is_match}, Total Error: {error}")

    ctm_result, expected = ctm[0:0], matrix[0:0]
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    if ctm_result is None:
        print(f"\n14. Match: True")


def demo_vector():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(10)
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()

    # Sample input matrix (8x8)
    vector = np.array([1, 2, 3, 4, 5])

    np.set_printoptions(linewidth=np.inf)

    print("Vector:\n", vector)

    batch_size = params.GetBatchSize() if params.GetBatchSize() else cc.GetRingDimension() // 2

    # Encrypt matrix A
    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    onp.generate_slicing_key(keys.secretKey, ctv.original_shape)

    ctv_result, expected = ctv[1:], vector[1:]
    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\nMatch: {is_match}, Total Error: {error}")

    ctv_result, expected = ctv[1:3], vector[1:3]
    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\nMatch: {is_match}, Total Error: {error}")

    ctv_result, expected = ctv[-1], vector[-1]
    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\nMatch: {is_match}, Total Error: {error}")

    ctv_result, expected = ctv[::2], vector[::2]
    result = ctv_result.decrypt(keys.secretKey, unpack_type="original")
    is_match, error = onp.check_equality(result, expected)
    print(f"\n\nresult\n{result} \nexpected\n{expected}")
    print(f"\nMatch: {is_match}, Total Error: {error}")


if __name__ == "__main__":
    demo_matrix()
    demo_vector()
