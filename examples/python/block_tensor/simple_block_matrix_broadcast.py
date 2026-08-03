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


def main():
    """
    Broadcasting for block matrices.

        (m, n) + (n,)    -> add the row vector to each row
        (m, n) + (m, 1)  -> add the column vector to each column

    Key points for block broadcasting:
      1. The vector operand must be tiled so its shared axis lines up with the
         matrix blocks. Pass block_shape=(block_cols,) for a row vector and
         block_shape=(block_rows, 1) for a column vector, reading block_cols /
         block_rows from the matrix's ``block_shape``.
      2. Each source block is expanded into a matrix block with rotations, so the
         required rotation keys must be generated first via
         ``generate_block_broadcast_key``.
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**7  # 128
    mult_depth = 3  # broadcast mask + one element-wise multiply
    scale_mod_size = 50

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
    cc.EvalSumKeyGen(keys.secretKey)

    batch_size = cc.GetRingDimension() // 2

    # --- Inputs --------------------------------------------------------------
    # A 12x12 matrix needs 144 entries > 64 slots, so it is tiled into a 2x2 grid
    # of (8, 8) blocks. The vectors are aligned to that block layout below.
    rows, cols = 12, 12
    matrix = np.arange(1, rows * cols + 1, dtype=float).reshape(rows, cols)
    row_vector = np.arange(1, cols + 1, dtype=float)  # shape (n,)
    col_vector = np.arange(1, rows + 1, dtype=float).reshape(rows, 1)  # shape (m, 1)

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix.shape}  ({matrix.size} entries)")

    # --- Build the encrypted block matrix ------------------------------------
    ctm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    block_rows, block_cols = ctm.block_shape
    print(
        f"\nAuto-selected block_shape = {ctm.block_shape}, "
        f"grid_shape = {ctm.grid_shape}  ({ctm.num_blocks} ciphertext blocks)"
    )

    # Tile the vectors so their shared axis aligns with the matrix blocks.
    ctv_row = onp.block_array(
        cc=cc,
        data=row_vector,
        batch_size=batch_size,
        block_shape=(block_cols,),
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    ctv_col = onp.block_array(
        cc=cc,
        data=col_vector,
        batch_size=batch_size,
        block_shape=(block_rows, 1),
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Rotation keys for the per-block expansion (both row and column directions).
    onp.generate_block_broadcast_key(keys.secretKey, ctm)

    all_ok = True

    # 1) Add a row vector to every row: (m, n) + (n,)
    res = (ctm + ctv_row).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res, matrix + row_vector, "Row broadcast add: matrix (m, n) + vector (n,)"
    )
    all_ok = all_ok and is_match

    # 2) Add a column vector to every column: (m, n) + (m, 1)
    res = (ctm + ctv_col).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res, matrix + col_vector, "Column broadcast add: matrix (m, n) + column (m, 1)"
    )
    all_ok = all_ok and is_match

    # 3) Multiply every row by a row vector (element-wise): (m, n) * (n,)
    res = (ctm * ctv_row).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res, matrix * row_vector, "Row broadcast multiply: matrix (m, n) * vector (n,)"
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
