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

    onp.generate_broadcast_key(keys.secretKey, (), (3, 5))
    onp.generate_broadcast_key(keys.secretKey, vector.shape, (3, 5))

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

    # Broadcasting Scalar
    cts_s = onp.broadcast_to(cts, ())
    res = cts_s.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Scalar:\n{res}")

    cts_vec = onp.broadcast_to(cts, (3,))
    res = cts_vec.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Vector:\n{res}")

    cts_mat1 = onp.broadcast_to(cts, (3, 5), onp.ROW_MAJOR)
    res = cts_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Matrix:\n{res}")

    cts_mat2 = onp.broadcast_to(cts, (3, 5), onp.COL_MAJOR)
    res = cts_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Scalar -> Matrix:\n{res}")

    # Broadcasting Vector
    ctv_mat1 = onp.broadcast_to(ctv, (5, 3), onp.ROW_MAJOR)
    res = ctv_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Vector -> Matrix (ROW_MAJOR):\n{res}")

    ctv_mat2 = onp.broadcast_to(ctv, (5, 3), onp.COL_MAJOR)
    res = ctv_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Vector -> Matrix (COL_MAJOR) :\n{res}")

    # Broadcasting Vector Column
    ctv_mat1 = onp.broadcast_to(ctv_col, (3, 5), onp.ROW_MAJOR)
    res = ctv_mat1.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Column Vector-> Matrix (ROW_MAJOR):\n{res}")

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
    ctv_add = ctv + cts

    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(scalar, vector),
        f"Scalar + Vector  \n{scalar} \n{vector}",
    )
    # Example for scalar + matrix
    ctv_add = cts + ctm
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(scalar, matrix),
        f"Scalar + Vector  \n{scalar} \n{matrix}",
    )
    # Example for vector + matrix
    ctv_add = ctv + ctm
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(vector, matrix),
        f"Vector + Matrix  \n{scalar} \n{matrix}",
    )

    # Example for column vector + matrix
    ctv_add = ctm + ctv_col
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    validate_and_print_results(
        res_add,
        np.add(column_vector, matrix),
        f"Column Vector + Matrix  \n{column_vector} \n{matrix}",
    )


if __name__ == "__main__":
    # example_broadcasting()
    example_broadcasting_with_operations()
