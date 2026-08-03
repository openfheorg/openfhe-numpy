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


def main():
    """
    Demonstrate homomorphic block matrix-vector multiplication using OpenFHE-NumPy.

    Two encoding combinations are shown:

      Case 1: ROW_MAJOR matrix  @  COL_MAJOR vector  ->  ROW_MAJOR result
      Case 2: COL_MAJOR matrix  @  ROW_MAJOR vector  ->  COL_MAJOR result

    The block_shape=(2, 2) splits the matrix across multiple ciphertext
    blocks, each holding a 2x2 sub-matrix in batch_size=4 slots.
    compact=True on the vector produces the duplicated, square-compatible
    packing required for block matvec; do NOT use compact for plain
    vector arithmetic (add/sub/dot).
    """

    # Cryptographic setup
    mult_depth = 4
    scale_mod_size = 59
    block_shape = (2, 2)
    # Each 2x2 block uses exactly 4 slots.
    batch_size = int(np.prod(block_shape))

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
    print(f"Block shape:         {block_shape}")

    # Sample input
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )

    vector = np.array([1.0, 1.0, 1.0], dtype=float)

    print("\nInput")
    print("\nMatrix:\n", matrix)
    print("\nVector:\n", vector)

    expected = matrix @ vector
    print(f"\nExpected:\n{expected}")

    all_ok = True

    # =========================================================
    # Case 1: ROW_MAJOR matrix @ COL_MAJOR vector
    # =========================================================
    #
    # Row-wise packing of a 2-row x 3-col matrix with pad-to-power-of-2:
    #
    #   1 2 3 0 | 4 5 6 0   (two 2x2 blocks packed as [row0|row1])
    #
    # The vector is encoded with compact=True (duplicated packing):
    #
    #   1 1 1 1 | 0 0 0 0   (compact COL_MAJOR block vector)
    #
    # The result lands in ROW_MAJOR order, one entry per row of the matrix.

    ctm_rm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # compact=True: produce duplicated block packing required by block matvec.
    ctv_cm = onp.block_array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        block_shape=None,
        order=onp.COL_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
        compact=True,
    )

    # Attach summation keys to every matrix block before the product.
    # ROW_MAJOR matrix requires extra["colkey"] for EvalSumCols.
    onp.attach_block_matvec_keys(ctm_rm, keys.secretKey)

    print_block_metadata("Block matrix (ROW_MAJOR)", ctm_rm)
    print_block_metadata("Block vector (COL_MAJOR, compact)", ctv_cm)

    ctv_result_rm = ctm_rm @ ctv_cm
    res_rm = ctv_result_rm.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_rm,
        expected,
        "Block matvec Case 1: ROW_MAJOR matrix @ COL_MAJOR vector",
    )
    all_ok = all_ok and is_match

    # =========================================================
    # Case 2: COL_MAJOR matrix @ ROW_MAJOR vector
    # =========================================================
    #
    # Column-wise packing interleaves the matrix columns across slots.
    # The vector uses compact=True (ROW_MAJOR), producing duplicated
    # per-element blocks compatible with the COL_MAJOR matrix blocks.
    #
    # The result lands in COL_MAJOR order, one entry per row of the matrix.

    ctm_cm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.COL_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # compact=True: produce duplicated block packing required by block matvec.
    ctv_rm = onp.block_array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        block_shape=None,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
        compact=True,
    )

    # Attach summation keys to every matrix block before the product.
    # COL_MAJOR matrix requires extra["rowkey"] for EvalSumRows.
    onp.attach_block_matvec_keys(ctm_cm, keys.secretKey)

    print_block_metadata("Block matrix (COL_MAJOR)", ctm_cm)
    print_block_metadata("Block vector (ROW_MAJOR, compact)", ctv_rm)

    ctv_result_cm = ctm_cm @ ctv_rm
    res_cm = ctv_result_cm.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_cm,
        expected,
        "Block matvec Case 2: COL_MAJOR matrix @ ROW_MAJOR vector",
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
