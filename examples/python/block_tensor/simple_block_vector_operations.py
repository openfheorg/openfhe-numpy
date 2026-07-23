import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print vector results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


def print_block_metadata(name, block_array):
    """Print block tensor metadata."""
    print(f"\n{name} metadata")
    print(f"  original_shape : {block_array.original_shape}")
    print(f"  padded shape   : {block_array.shape}")
    print(f"  block_shape    : {block_array.block_shape}")
    print(f"  grid_shape     : {block_array.grid_shape}")
    print(f"  num_blocks     : {block_array.num_blocks}")
    print(f"  batch_size     : {block_array.batch_size}")


def main():
    """
    Demonstrate homomorphic block vector operations using OpenFHE-NumPy:

      - block vector addition       (CT + CT)
      - block vector subtraction    (CT - CT)
      - block vector elementwise multiplication (CT * CT)
      - block vector scalar multiplication      (CT * scalar)
      - block mixed addition        (CT + PT)
      - block vector dot product via onp.dot
      - block vector dot product via @
    """

    # Cryptographic setup
    # Length-5 vectors with batch_size=4 force two ciphertext blocks each.
    mult_depth = 4
    scale_mod_size = 59
    batch_size = 4

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(FIXEDAUTO)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Block batch size:    {batch_size}")

    # Sample input vectors.
    # Length 5 with batch_size=4 forces two ciphertext blocks.
    vector_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    vector_b = np.array([4.0, 0.0, 1.0, 3.0, 6.0], dtype=float)

    print("\nInput vectors")
    print("vector_a:", vector_a)
    print("vector_b:", vector_b)

    # Create encrypted block vectors.
    # block_shape=None lets the library choose the block size from batch_size.
    ctv_a = onp.block_array(
        cc=cc,
        data=vector_a,
        batch_size=batch_size,
        block_shape=None,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctv_b = onp.block_array(
        cc=cc,
        data=vector_b,
        batch_size=batch_size,
        block_shape=None,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Also create a plaintext block vector for the mixed CT+PT test.
    ptv_b = onp.block_array(
        cc=cc,
        data=vector_b,
        batch_size=batch_size,
        block_shape=None,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="P",
    )

    print_block_metadata("Block encrypted vector_a", ctv_a)
    print_block_metadata("Block encrypted vector_b", ctv_b)
    print_block_metadata("Block plaintext  vector_b", ptv_b)

    all_ok = True

    # 1) Block vector addition (CT + CT)
    ctv_add = ctv_a + ctv_b
    res_add = ctv_add.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_add,
        vector_a + vector_b,
        f"Block vector addition (CT + CT)\n{vector_a}\n+\n{vector_b}",
    )
    all_ok = all_ok and is_match

    # 2) Block vector subtraction (CT - CT)
    ctv_sub = ctv_a - ctv_b
    res_sub = ctv_sub.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_sub,
        vector_a - vector_b,
        f"Block vector subtraction (CT - CT)\n{vector_a}\n-\n{vector_b}",
    )
    all_ok = all_ok and is_match

    # 3) Block vector elementwise multiplication (CT * CT)
    ctv_mul = ctv_a * ctv_b
    res_mul = ctv_mul.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_mul,
        vector_a * vector_b,
        f"Block vector elementwise multiplication (CT * CT)\n{vector_a}\n*\n{vector_b}",
    )
    all_ok = all_ok and is_match

    # 4) Block vector scalar multiplication (CT * scalar)
    ctv_scalar_mul = ctv_a * 7.0
    res_scalar_mul = ctv_scalar_mul.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_scalar_mul,
        vector_a * 7.0,
        f"Block vector scalar multiplication (CT * 7.0)\n{vector_a} * 7.0",
    )
    all_ok = all_ok and is_match

    # 5) Mixed block vector addition (CT + PT)
    ctv_add_plain = ctv_a + ptv_b
    res_add_plain = ctv_add_plain.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_add_plain,
        vector_a + vector_b,
        f"Block vector mixed addition (CT + PT)\n{vector_a}\n+\n{vector_b}",
    )
    all_ok = all_ok and is_match

    # 6) Block vector dot product via onp.dot
    #
    # Expected: 1*4 + 2*0 + 3*1 + 4*3 + 5*6 = 49
    ctv_dot = onp.dot(ctv_a, ctv_b)
    res_dot = ctv_dot.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_dot,
        np.dot(vector_a, vector_b),
        f"Block vector dot product via onp.dot\n{vector_a}\n·\n{vector_b}",
    )
    all_ok = all_ok and is_match

    # 7) Block vector dot product via @
    ctv_dot_op = ctv_a @ ctv_b
    res_dot_op = ctv_dot_op.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_dot_op,
        np.dot(vector_a, vector_b),
        f"Block vector dot product via @\n{vector_a}\n@\n{vector_b}",
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
