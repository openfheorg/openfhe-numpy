import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


def print_block_metadata(name, block_array):
    """Print block matrix metadata."""
    print(f"\n{name} metadata")
    print(f"original_shape: {block_array.original_shape}")
    print(f"padded shape:   {block_array.shape}")
    print(f"block_shape:    {block_array.block_shape}")
    print(f"grid_shape:     {block_array.grid_shape}")
    print(f"num_blocks:     {block_array.num_blocks}")
    print(f"batch_size:     {block_array.batch_size}")


def gen_block_matmul_keys(keys, block_shape):
    """
    Generate rotation keys for square block matrix multiplication.

    For block matrix multiplication, each small encrypted block is multiplied
    using the existing packed square-matrix multiplication routine. Therefore
    the key size should match the block dimension, not the full matrix dimension.
    """
    block_rows, block_cols = block_shape

    if block_rows != block_cols:
        raise ValueError(
            f"Block matrix multiplication currently expects square blocks; "
            f"got block_shape={block_shape}."
        )

    block_size = block_rows

    # Prefer the Python helper if available.
    if hasattr(onp, "gen_square_matmult_key"):
        onp.gen_square_matmult_key(keys.secretKey, block_size)

    # Fallback to the direct backend function if exposed.
    elif hasattr(onp, "EvalSquareMatMultRotateKeyGen"):
        onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, block_size)

    else:
        raise RuntimeError(
            "No square-matrix multiplication key-generation function found. "
            "Expected onp.gen_square_matmult_key or "
            "onp.EvalSquareMatMultRotateKeyGen."
        )


def run_block_matmul_example(cc, keys, A, B, block_shape, description):
    """Run a homomorphic block matrix x block matrix multiplication example."""
    print(f"\n--- {description} ---")
    print("Input A:\n", A)
    print("Input B:\n", B)

    batch_size = int(np.prod(block_shape))

    # Encrypt A and B as block matrices.
    ctm_A = onp.block_array(
        cc=cc,
        data=A,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    ctm_B = onp.block_array(
        cc=cc,
        data=B,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    print_block_metadata("Block encrypted A", ctm_A)
    print_block_metadata("Block encrypted B", ctm_B)

    # Generate rotation keys for each small square block.
    gen_block_matmul_keys(keys, block_shape)

    # Perform block homomorphic matrix multiplication.
    #
    # This should compute:
    #
    #     C[i, j] = sum_k A[i, k] @ B[k, j]
    #
    # where each A[i, k] and B[k, j] is a small CTArray block.
    ct_res = ctm_A @ ctm_B

    print_block_metadata("Block encrypted result", ct_res)

    # Decrypt result.
    res = ct_res.decrypt(keys.secretKey, unpack_type="original")

    # Validate.
    validate_and_print_results(res, A @ B, description)


def main():
    """
    Demonstrate homomorphic block matrix multiplication for two cases:

      1) power-of-two dimensions
      2) non-power-of-two dimensions

    The block shape is fixed to (2, 2). Therefore each encrypted block
    stores a 2x2 matrix using 4 CKKS slots.
    """

    # Cryptographic setup
    mult_depth = 8
    scale_mod_size = 59

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
    print(f"\nCrypto context: ring_dim={ring_dim}, slots={ring_dim // 2}")

    # Each encrypted block is a 2x2 matrix.
    block_shape = (2, 2)

    # ============================================================
    # Case 1: power-of-two dimensions, 8x8 @ 8x8
    # ============================================================

    A8 = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            [1, 0, 0, 1, 10, 3, 1, 7],
            [7, 8, 2, 5, 3, 2, 10, 9],
            [0, 3, 4, 10, 10, 5, 2, 5],
            [2, 5, 0, 2, 8, 8, 5, 9],
            [5, 1, 10, 6, 2, 8, 6, 3],
        ],
        dtype=float,
    )

    B8 = np.array(
        [
            [6, 5, 4, 3, 2, 1, 0, 7],
            [7, 1, 1, 2, 7, 5, 9, 3],
            [4, 8, 8, 10, 8, 2, 1, 6],
            [7, 0, 0, 5, 10, 3, 4, 2],
            [9, 3, 2, 8, 3, 2, 1, 0],
            [5, 2, 4, 1, 10, 5, 8, 2],
            [9, 8, 0, 2, 8, 8, 7, 5],
            [3, 6, 10, 1, 2, 8, 4, 0],
        ],
        dtype=float,
    )

    run_block_matmul_example(
        cc,
        keys,
        A8,
        B8,
        block_shape,
        "Block 8x8 Matrix Product",
    )

    # ============================================================
    # Case 2: non-power-of-two dimensions, 3x3 @ 3x3
    # ============================================================
    #
    # With block_shape=(2,2), this becomes a padded block grid:
    #
    #   original_shape = (3,3)
    #   padded shape   = (4,4)
    #   grid_shape     = (2,2)
    #
    # The decrypt path crops back to original_shape.
    # Your previous constructor/decrypt tests already showed this
    # padded block shape reconstruction works for 3x3 matrices.
    # ============================================================

    A3 = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )

    B3 = np.array(
        [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1],
        ],
        dtype=float,
    )

    run_block_matmul_example(
        cc,
        keys,
        A3,
        B3,
        block_shape,
        "Block 3x3 Matrix Product",
    )


if __name__ == "__main__":
    main()
