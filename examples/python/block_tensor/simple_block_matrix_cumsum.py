import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print cumulative-sum results."""
    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected, eps=1e-6)
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
    Demonstrate homomorphic cumulative sums on encrypted block tensors.

    Operations shown:
      - 1-D cumulative sum across multiple ciphertext blocks
      - 2-D cumulative sum across matrix block rows along axis=0
      - 2-D cumulative sum across matrix block columns along axis=1

    In both cases, the terminal cumulative value from one block is carried into
    the next block along the cumulative axis.
    """

    # --- Cryptographic setup -------------------------------------------------
    # These small parameters keep the tutorial fast. They are not secure and
    # must not be used in production.
    ring_dim = 2**6  # 64
    mult_depth = 12
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
    # Cumsum uses plaintext masks when carrying totals between blocks.
    cc.EvalMultKeyGen(keys.secretKey)

    batch_size = cc.GetBatchSize()

    # --- Inputs --------------------------------------------------------------
    # Both inputs contain more entries than one ciphertext can hold.
    vector = np.arange(1, 41, dtype=float)
    matrix = np.arange(1, 65, dtype=float).reshape(8, 8)

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Vector shape        : {vector.shape}  ({vector.size} entries)")
    print(f"Matrix shape        : {matrix.shape}  ({matrix.size} entries)")
    print(
        f"\nThe vector has {vector.size} entries and the matrix has "
        f"{matrix.size} entries, both exceeding the {batch_size} available slots.\n"
        "They must therefore be represented using multiple ciphertext blocks."
    )

    print("\nInput vector:")
    print(vector)
    print("\nInput matrix:")
    print(matrix)

    # --- Build encrypted block tensors ---------------------------------------
    vector_block_shape = (8,)
    matrix_block_shape = (4, 4)

    ctv = onp.block_array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        block_shape=vector_block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    ctm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        block_shape=matrix_block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    onp.attach_block_cumsum_keys(ctv, keys.secretKey)
    onp.attach_block_cumsum_keys(ctm, keys.secretKey)

    print_block_metadata("Encrypted block vector", ctv)
    print_block_metadata("Encrypted block matrix", ctm)

    all_ok = True

    # 1) Cumsum across all ciphertext blocks of the vector.
    res_vector = onp.cumsum(ctv).decrypt(
        keys.secretKey,
        unpack_type="original",
    )
    is_match, _ = validate_and_print_results(
        res_vector,
        np.cumsum(vector),
        "Block vector cumulative sum",
    )
    all_ok = all_ok and is_match

    # 2) Matrix cumsum down each column, across both block rows.
    # axis=0 means that values accumulate from top to bottom.
    res_matrix = onp.cumsum(ctm, axis=0).decrypt(
        keys.secretKey,
        unpack_type="original",
    )
    is_match, _ = validate_and_print_results(
        res_matrix,
        np.cumsum(matrix, axis=0),
        "Block matrix cumulative sum (axis=0)",
    )
    all_ok = all_ok and is_match

    # 3) Matrix cumsum across each row, across both block columns.
    # axis=1 means that values accumulate from left to right.
    res_matrix = onp.cumsum(ctm, axis=1).decrypt(
        keys.secretKey,
        unpack_type="original",
    )
    is_match, _ = validate_and_print_results(
        res_matrix,
        np.cumsum(matrix, axis=1),
        "Block matrix cumulative sum (axis=1)",
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
