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
    """Print block tensor metadata."""
    print(f"\n{name} metadata")
    print(f"  original_shape : {block_array.original_shape}")
    print(f"  padded shape   : {block_array.shape}")
    print(f"  block_shape    : {block_array.block_shape}")
    print(f"  grid_shape     : {block_array.grid_shape}")
    print(f"  num_blocks     : {block_array.num_blocks}")
    print(f"  batch_size     : {block_array.batch_size}")


def run_block_matmul_example(cc, keys, A, B, block_shape, description):
    """Run one homomorphic block matrix multiplication example.

    Both operands must use ROW_MAJOR packing and identical square block_shape.
    The rotation keys are generated from the block size (not the full matrix),
    since EvalSquareMatMultRotateKeyGen operates on a single block.
    """
    print(f"\n--- {description} ---")
    print("Input A:\n", A)
    print("Input B:\n", B)

    # Each ciphertext block stores one block_shape tile;
    # batch_size equals the number of slots consumed by one block.
    batch_size = int(np.prod(block_shape))

    ctm_A = onp.block_array(
        cc=cc,
        data=A,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctm_B = onp.block_array(
        cc=cc,
        data=B,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    print_block_metadata("Block encrypted A", ctm_A)
    print_block_metadata("Block encrypted B", ctm_B)

    # Generate rotation keys for one square block of size block_shape[0].
    # The keys cover all rotations needed by EvalMatMulSquare on each block.
    block_size = block_shape[0]
    onp.gen_square_matmult_key(keys.secretKey, block_size)

    # Compute C[i, j] = sum_k A[i, k] @ B[k, j],
    # where each A[i, k] and B[k, j] is an encrypted CTArray block.
    ctm_C = ctm_A @ ctm_B

    print_block_metadata("Block encrypted result C", ctm_C)

    res = ctm_C.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(res, A @ B, description)
    return is_match


def main():
    """
    Demonstrate homomorphic block matrix multiplication using OpenFHE-NumPy:

      1) power-of-two dimensions:     8x8 @ 8x8
      2) non-power-of-two dimensions: 3x3 @ 3x3

    The block_shape is fixed to (2, 2). Each encrypted block stores a 2x2
    sub-matrix in 4 CKKS slots. Non-power-of-two matrices are zero-padded
    to a 4x4 block grid and the result is cropped back to the original shape.
    Both operands must use ROW_MAJOR packing for block matrix multiplication.
    """

    # Cryptographic setup
    # Block matmul calls EvalMatMulSquare on each 2x2 block, which requires
    # depth 2 per multiply-accumulate step; mult_depth=8 is sufficient for
    # an 8x8 matrix decomposed into 2x2 blocks (4x4 block grid, depth ~4*2).
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
    block_shape = (2, 2)
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Block shape:         {block_shape}")

    all_ok = True

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

    all_ok = all_ok and run_block_matmul_example(
        cc=cc,
        keys=keys,
        A=A8,
        B=B8,
        block_shape=block_shape,
        description="Block 8x8 Matrix Product (power-of-two)",
    )

    # ============================================================
    # Case 2: non-power-of-two dimensions, 3x3 @ 3x3
    # ============================================================
    # With block_shape=(2, 2), the 3x3 matrix is internally padded to a
    # 4x4 block layout (2x2 grid of 2x2 blocks) and the result is
    # cropped back to the original 3x3 shape on decryption.
    A3 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    B3 = np.array(
        [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=float,
    )

    all_ok = all_ok and run_block_matmul_example(
        cc=cc,
        keys=keys,
        A=A3,
        B=B3,
        block_shape=block_shape,
        description="Block 3x3 Matrix Product (non-power-of-two)",
    )

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
