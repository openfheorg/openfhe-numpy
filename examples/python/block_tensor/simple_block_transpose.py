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
    Transpose an encrypted block matrix.

    ``A.T`` transposes each ciphertext block and swaps the block grid: a matrix
    with grid_shape (gr, gc) of (br, bc) blocks becomes grid_shape (gc, gr) of
    (bc, br) blocks. The rectangular matrix below makes the grid swap visible
    (grid (2, 3) -> (3, 2)).

    Each per-block transpose uses EvalTranspose, a linear transform whose keys
    must be sized to the block (not the padded tensor). Generate them with
    ``gen_block_transpose_keys`` before calling transpose.
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**6  # 64
    mult_depth = 4
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
    # 96 entries > 32 slots, so the 8x12 matrix is tiled into a 2x3 grid of
    # (4, 4) blocks; the transpose returns a 12x8 matrix on a 3x2 grid.
    matrix = np.arange(1, 97, dtype=float).reshape(8, 12)

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

    # Linear-transform keys for the per-block transpose.
    onp.gen_block_transpose_keys(keys.secretKey, ctm)

    print_block_metadata("Encrypted block matrix", ctm)

    ctm_t = ctm.transpose()
    print_block_metadata("Transposed block matrix", ctm_t)

    res = ctm_t.decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(res, matrix.T, "Block matrix transpose (A.T)")

    print("\n" + "=" * 60)
    print(f"All checks passed: {is_match}")
    print("=" * 60)


if __name__ == "__main__":
    main()
