import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print reduction results."""
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
    Sum and mean of an encrypted block matrix over each axis.

    - axis=None: reduce over all entries (one scalar).
    - axis=0:    reduce down rows, returning one value per column.
    - axis=1:    reduce across columns, returning one value per row.

    axis=0/1 accumulate partial reductions ACROSS block rows/columns and read
    per-block EvalSumRows / EvalSumCols keys; attach them with
    ``attach_block_sum_keys``. axis=None also needs the full-sum key from
    ``gen_sum_key``. ``mean`` divides the corresponding sum by the element count.
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**6  # 64
    mult_depth = 6
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
    onp.gen_sum_key(keys.secretKey)  # needed by axis=None

    batch_size = cc.GetRingDimension() // 2

    # --- Inputs --------------------------------------------------------------
    # 64 entries > 32 slots, so the matrix is tiled into a 2x2 grid of (4, 4)
    # blocks, and axis=0/1 must accumulate across block rows/columns.
    matrix = np.arange(1, 65, dtype=float).reshape(8, 8) / 8.0

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix.shape}  ({matrix.size} entries)")

    ctm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    # Per-block EvalSumRows / EvalSumCols keys for the axis=0/1 reductions.
    onp.attach_block_sum_keys(ctm, keys.secretKey)

    print_block_metadata("Encrypted block matrix", ctm)

    all_ok = True

    for axis in (None, 0, 1):
        res_sum = onp.sum(ctm, axis=axis).decrypt(keys.secretKey, unpack_type="original")
        is_match, _ = validate_and_print_results(
            res_sum, np.sum(matrix, axis=axis), f"Block sum (axis={axis})"
        )
        all_ok = all_ok and is_match

        res_mean = onp.mean(ctm, axis=axis).decrypt(keys.secretKey, unpack_type="original")
        is_match, _ = validate_and_print_results(
            res_mean, np.mean(matrix, axis=axis), f"Block mean (axis={axis})"
        )
        all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
