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
    Element-wise integer power of an encrypted block tensor.

    ``A ** k`` raises every entry to the integer power ``k`` by repeated
    homomorphic multiplication, applied independently to each ciphertext block.
    It works for any block shape (no square or grid constraint).

    The multiplicative depth required is ceil(log2(k)); k=3 needs depth 2.
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**6  # 64
    mult_depth = 4  # ceil(log2(k)) for k=3 is 2, plus headroom
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

    batch_size = cc.GetRingDimension() // 2

    # --- Inputs --------------------------------------------------------------
    # 64 entries > 32 slots, so the matrix is tiled into a 2x2 grid of (4, 4)
    # blocks. Values are kept small so the cubes stay in a comfortable range.
    exponent = 3
    matrix = np.arange(1, 65, dtype=float).reshape(8, 8) / 8.0

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix.shape}  ({matrix.size} entries)")
    print(f"Exponent            : {exponent}")

    ctm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    print_block_metadata("Encrypted block matrix", ctm)

    # A ** exponent, element-wise across every block.
    res = (ctm**exponent).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res, matrix**exponent, f"Block matrix element-wise power (A ** {exponent})"
    )

    print("\n" + "=" * 60)
    print(f"All checks passed: {is_match}")
    print("=" * 60)


if __name__ == "__main__":
    main()
