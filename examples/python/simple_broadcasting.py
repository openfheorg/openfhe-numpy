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

    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


def example_broadcasting():
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
    onp.gen_rotation_keys(keys.secretKey, list(range(-32, 32)))

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {batch_size}")

    scalar = 5
    vector = [10, 20, 30]
    matrix = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]

    print("\nInputs")
    print("vector_a:", scalar)
    print("vector_b:", vector)
    print("vector_c:", matrix)

    cts = onp.array(
        cc=cc,
        data=scalar,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    cta_rm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    cta_cm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
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
    print(f"\n Broadcast Vector -> Matrix:\n{res}")

    ctv_mat2 = onp.broadcast_to(ctv, (5, 3), onp.COL_MAJOR)
    res = ctv_mat2.decrypt(keys.secretKey, unpack_type="original")
    print(f"\n Broadcast Vector -> Matrix:\n{res}")


def example_broadcasting_with_operations():
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
    onp.gen_rotation_keys(keys.secretKey, list(range(-32, 32)))

    ring_dim = cc.GetRingDimension()
    batch_size = ring_dim // 2
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots:    {batch_size}")

    scalar = 5
    vector = [10, 20, 30, 40, 50]
    matrix = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]

    print("\nInputs")
    print("vector_a:", scalar)
    print("vector_b:", vector)
    print("vector_c:", matrix)

    cts = onp.array(
        cc=cc,
        data=scalar,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Example for scalar + vector
    ctv_add = cts + ctv
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


if __name__ == "__main__":
    example_broadcasting_with_operations()
