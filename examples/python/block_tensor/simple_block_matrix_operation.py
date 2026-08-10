import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print matrix results."""
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
    We deliberately pick a tiny CKKS ring dimension (2**6 = 64), which gives only
    32 usable slots per ciphertext. Our 8x8 matrix has 64 entries -- more than 32 --
    so it *cannot* be packed into a single ciphertext. openfhe-numpy tiles it into a
    grid of small blocks, each of which fits in one ciphertext.

    Operations shown:
      - block matrix addition (CT + CT)
      - block matrix subtraction (CT - CT)
      - block matrix elementwise multiplication (CT * CT)
      - block matrix scalar multiplication (CT * scalar)
      - block matrix mixed addition (CT + PT)
      - block matrix transpose with rectangular blocks
      - packed-order transform followed by addition
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**6  # 64
    mult_depth = 3  # transpose + transform + logical-slot arithmetic
    scale_mod_size = 50

    block_shape = (2, 4)

    params = CCParamsCKKSRNS()
    params.SetRingDim(ring_dim)
    params.SetSecurityLevel(HEStd_NotSet)
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

    batch_size = cc.GetRingDimension() // 2

    # --- Why we need blocks --------------------------------------------------
    matrix_a = np.arange(1, 65, dtype=float).reshape(8, 8)
    matrix_b = 65.0 - matrix_a

    n_entries = matrix_a.size
    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix_a.shape}  ({n_entries} entries)")
    print(
        f"\n{n_entries} entries > {batch_size} slots  =>  the matrix does NOT fit in a\n"
        f"single ciphertext, so openfhe-numpy tiles it into blocks that each do."
    )

    print("\nInput matrices")
    print("matrix_a:")
    print(matrix_a)
    print("\nmatrix_b:")
    print(matrix_b)

    # --- Build encrypted block matrices --------------------------------------
    # The 8x8 matrix is tiled into a 4x2 grid of (2,4) blocks. Each block is one
    # ciphertext. Non-divisible sizes would zero-pad the edge blocks.
    ctm_a = onp.block_array(
        cc=cc,
        data=matrix_a,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctm_b = onp.block_array(
        cc=cc,
        data=matrix_b,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # A plaintext block matrix for the mixed CT + PT test.
    ptm_b = onp.block_array(
        cc=cc,
        data=matrix_b,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="P",
    )

    print(
        f"\nSelected block_shape = {ctm_a.block_shape}, "
        f"grid_shape = {ctm_a.grid_shape}  ({ctm_a.num_blocks} ciphertext blocks)"
    )

    print_block_metadata("Encrypted block matrix_a", ctm_a)
    print_block_metadata("Encrypted block matrix_b", ctm_b)
    print_block_metadata("Plaintext block matrix_b", ptm_b)

    all_ok = True

    # 1) Block matrix addition (CT + CT)
    res_add = (ctm_a + ctm_b).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_add, matrix_a + matrix_b, "Block matrix addition (CT + CT)"
    )
    all_ok = all_ok and is_match

    # 2) Block matrix subtraction (CT - CT)
    res_sub = (ctm_a - ctm_b).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_sub, matrix_a - matrix_b, "Block matrix subtraction (CT - CT)"
    )
    all_ok = all_ok and is_match

    # 3) Block matrix elementwise multiplication (CT * CT)
    res_mul = (ctm_a * ctm_b).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_mul, matrix_a * matrix_b, "Block matrix elementwise multiplication (CT * CT)"
    )
    all_ok = all_ok and is_match

    # 4) Block matrix scalar multiplication (CT * scalar)
    res_scalar_mul = (ctm_a * 3.0).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_scalar_mul, matrix_a * 3.0, "Block matrix scalar multiplication (CT * 3.0)"
    )
    all_ok = all_ok and is_match

    # 5) Mixed block matrix addition (CT + PT)
    res_add_plain = (ctm_a + ptm_b).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_add_plain, matrix_a + matrix_b, "Block matrix mixed addition (CT + PT)"
    )
    all_ok = all_ok and is_match

    # 6) Transpose a block matrix whose individual blocks are rectangular.
    onp.gen_block_transpose_keys(keys.secretKey, ctm_a)
    ctm_transposed = ctm_a.transpose()
    res_transposed = ctm_transposed.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(res_transposed, matrix_a.T, "Block matrix transpose")
    all_ok = all_ok and is_match
    print_block_metadata("Transposed block matrix", ctm_transposed)

    # 7) Change the packing order, then continue with ordinary arithmetic.
    onp.gen_transform_keys(keys.secretKey, ctm_transposed)
    ctm_col_major = ctm_transposed.transform(onp.COL_MAJOR)
    res_transformed_add = (ctm_col_major + 2.0).decrypt(
        keys.secretKey,
        unpack_type="original",
    )
    is_match, _ = validate_and_print_results(
        res_transformed_add,
        matrix_a.T + 2.0,
        "Block transpose + COL_MAJOR transform + scalar addition",
    )
    all_ok = all_ok and is_match
    print_block_metadata("Column-major transposed block matrix", ctm_col_major)

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
