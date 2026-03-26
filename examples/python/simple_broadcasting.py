import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print vector results."""
    print("\n" + "*" * 60)
    print(f"{operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")
    is_match, error = onp.check_equality(expected, computed)
    print(f"\nMatch: {is_match}, Total Error: {error}")

    if not is_match:
        print("\n>>> ERROR: Pattern does not match")


def example_broadcasting(order=onp.ROW_MAJOR):
    # Cryptographic setup
    mult_depth = 5
    params = CCParamsCKKSRNS()
    params.SetScalingModSize(59)
    params.SetMultiplicativeDepth(mult_depth)

    cc = GenCryptoContext(params)

    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {batch_size}")

    scalar = 5
    vector = np.array([10, 20, 30])

    # A object can be broadcast to a vector (m, 1) or a matrix (m, n)
    # by duplicating its value across all slots. This is achieved using
    # repeated-doubling rotations.
    #
    # Cost:
    #   - O(log m) rotations for a vector of length m
    #   - O(log m + log n) rotations for an (m x n) matrix
    # assuming dimensions are padded to the next power of two.

    # Generate keys for broadcasting a scalar to a (3 x 5) matrix
    onp.generate_broadcast_key(keys.secretKey, (), (3, 5))

    # Generate keys for broadcasting a row vector to a (5 x 3) matrix
    onp.generate_broadcast_key(keys.secretKey, vector.shape, (5, 3))

    column_vector = [[10], [20], [30]]
    matrix = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]

    print("\nInputs")
    print("scalar:", scalar)
    print("vector:", vector)
    print("column_vector:", column_vector)
    print("matrix:", matrix)

    cts = onp.array(
        cc=cc,
        data=scalar,
        batch_size=batch_size,
        order=order,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=order,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_col = onp.array(
        cc=cc,
        data=column_vector,
        batch_size=batch_size,
        order=order,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Broadcast scalar to scalar
    # Example:
    #   scalar = 5
    #   target shape = ()
    #   result = 5
    cts_s = onp.broadcast_to(cts, ())
    res = cts_s.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Scalar:\n{res}")

    # Broadcast scalar to vector
    # Example:
    #   scalar = 5
    #   target shape = (3,)
    #   result = [5 5 5]
    cts_vec = onp.broadcast_to(cts, (3,))
    res = cts_vec.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Vector:\n{res}")

    # Broadcast scalar to matrix using ROW_MAJOR packing
    # Example:
    #   scalar = 5
    #   target shape = (3, 5)
    #   result =
    #     [[5 5 5 5 5]
    #      [5 5 5 5 5]
    #      [5 5 5 5 5]]
    cts_mat1 = onp.broadcast_to(cts, (3, 5), onp.ROW_MAJOR)
    res = cts_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Matrix:\n{res} (row-wise)")

    # Broadcast scalar to matrix using COL_MAJOR packing
    # Example:
    #   scalar = 5
    #   target shape = (3, 5)
    #   result =
    #     [[5 5 5 5 5]
    #      [5 5 5 5 5]
    #      [5 5 5 5 5]]
    #   The logical output is the same as ROW_MAJOR, but the internal packing differs.
    cts_mat2 = onp.broadcast_to(cts, (3, 5), onp.COL_MAJOR)
    res = cts_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Matrix:\n{res} (col-wise)")

    # Broadcast vector to matrix using ROW_MAJOR packing
    # Example:
    #   vector = [10 20 30]
    #   target shape = (5, 3)
    #   result =
    #     [[10 20 30]
    #      [10 20 30]
    #      [10 20 30]
    #      [10 20 30]
    #      [10 20 30]]
    ctv_mat1 = onp.broadcast_to(ctv, (5, 3), onp.ROW_MAJOR)
    res = ctv_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Vector -> Matrix:\n{res} (row-wise)")

    # Broadcast vector to matrix using COL_MAJOR packing
    # Example:
    #   vector = [10 20 30]
    #   target shape = (5, 3)
    #   result =
    #     [[10 20 30]
    #      [10 20 30]
    #      [10 20 30]
    #      [10 20 30]
    #      [10 20 30]]
    #   The logical output is the same as ROW_MAJOR, but the internal packing differs.
    ctv_mat2 = onp.broadcast_to(ctv, (5, 3), onp.COL_MAJOR)
    res = ctv_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Vector -> Matrix :\n{res} (col-wise)")

    # Broadcast column vector to matrix using ROW_MAJOR packing
    # Example:
    #   column_vector =
    #     [[10]
    #      [20]
    #      [30]]
    #   target shape = (3, 5)
    #   result =
    #     [[10 10 10 10 10]
    #      [20 20 20 20 20]
    #      [30 30 30 30 30]]
    ctv_mat1 = onp.broadcast_to(ctv_col, (3, 5), onp.ROW_MAJOR)
    res = ctv_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Column Vector-> Matrix (ROW_MAJOR):\n{res}")

    # Broadcast column vector to matrix using COL_MAJOR packing
    # Example:
    #   column_vector =
    #     [[10]
    #      [20]
    #      [30]]
    #   target shape = (3, 5)
    #   result =
    #     [[10 10 10 10 10]
    #      [20 20 20 20 20]
    #      [30 30 30 30 30]]
    #   The logical output is the same as ROW_MAJOR, but the internal packing differs.
    ctv_mat2 = onp.broadcast_to(ctv_col, (3, 5), onp.COL_MAJOR)
    res = ctv_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Column Vector-> Matrix (COL_MAJOR):\n{res}")


def example_broadcasting_with_operations(ord=onp.ROW_MAJOR):
    """
    Example for scalar broadcasting
    """
    # Cryptographic setup
    mult_depth = 5
    params = CCParamsCKKSRNS()
    params.SetScalingModSize(59)
    params.SetMultiplicativeDepth(mult_depth)

    cc = GenCryptoContext(params)

    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    # onp.gen_rotation_keys(keys.secretKey, list(range(-32, 32)))

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {batch_size}")

    scalar = np.array(5)
    vector = np.array([101, 210, 310, 140, 150])
    column_vector = np.array([[10], [20], [30]])
    matrix = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1], [1.2, 2.2, 3.2, 4.2, 5.2]]

    onp.generate_broadcast_key(keys.secretKey, scalar.shape, (3, 5))
    onp.generate_broadcast_key(keys.secretKey, vector.shape, (3, 5))
    onp.generate_broadcast_key(keys.secretKey, column_vector.shape, (3, 5))

    print("\nInputs")
    print("scalar:", scalar)
    print("vector:", vector)
    print("column_vector:", column_vector)
    print("matrix:", matrix)

    cts = onp.array(
        cc=cc,
        data=scalar,
        batch_size=batch_size,
        order=ord,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=ord,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_col = onp.array(
        cc=cc,
        data=column_vector,
        batch_size=batch_size,
        order=ord,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=ord,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Example for scalar + vector
    # Example:
    #   scalar = 5
    #   vector = [101 210 310 140 150]
    #   result = [106 215 315 145 155]
    ctv_add = ctv + cts

    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(scalar, vector),
        f"Scalar + Vector  \n{scalar} \n{vector}",
    )

    # Example for scalar + matrix
    # Example:
    #   scalar = 5
    #   matrix =
    #     [[1.0 2.0 3.0 4.0 5.0]
    #      [1.1 2.1 3.1 4.1 5.1]
    #      [1.2 2.2 3.2 4.2 5.2]]
    #   result =
    #     [[6.0 7.0 8.0 9.0 10.0]
    #      [6.1 7.1 8.1 9.1 10.1]
    #      [6.2 7.2 8.2 9.2 10.2]]
    ctv_add = cts + ctm
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(scalar, matrix),
        f"Scalar + Vector  \n{scalar} \n{matrix}",
    )

    # Example for vector + matrix
    # Example:
    #   vector = [101 210 310 140 150]
    #   matrix =
    #     [[1.0 2.0 3.0 4.0 5.0]
    #      [1.1 2.1 3.1 4.1 5.1]
    #      [1.2 2.2 3.2 4.2 5.2]]
    #   result =
    #     [[102.0 212.0 313.0 144.0 155.0]
    #      [102.1 212.1 313.1 144.1 155.1]
    #      [102.2 212.2 313.2 144.2 155.2]]
    ctv_add = ctv + ctm
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(vector, matrix),
        f"Vector + Matrix  \n{scalar} \n{matrix}",
    )

    # Example for column vector + matrix
    # Example:
    #   column_vector =
    #     [[10]
    #      [20]
    #      [30]]
    #   matrix =
    #     [[1.0 2.0 3.0 4.0 5.0]
    #      [1.1 2.1 3.1 4.1 5.1]
    #      [1.2 2.2 3.2 4.2 5.2]]
    #   result =
    #     [[11.0 12.0 13.0 14.0 15.0]
    #      [21.1 22.1 23.1 24.1 25.1]
    #      [31.2 32.2 33.2 34.2 35.2]]
    ctv_add = ctm + ctv_col
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(column_vector, matrix),
        f"Column Vector + Matrix  \n{column_vector} \n{matrix}",
    )


if __name__ == "__main__":
    example_broadcasting()
    example_broadcasting_with_operations()
