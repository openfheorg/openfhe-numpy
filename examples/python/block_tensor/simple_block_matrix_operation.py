import numpy as np
from openfhe import *
import openfhe_numpy as onp


def validate_and_print_results(computed, expected, operation_name):
    """Helper function to validate and print matrix results."""
    print("\n" + "*" * 60)
    print(f"{operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected)
    print(f"\nMatch: {is_match}, Total Error: {error}")
    return is_match, error


def print_block_metadata(name, block_array):
    print(f"\n{name} metadata")
    print(f"original_shape: {block_array.original_shape}")
    print(f"padded shape:   {block_array.shape}")
    print(f"block_shape:    {block_array.block_shape}")
    print(f"grid_shape:     {block_array.grid_shape}")
    print(f"num_blocks:     {block_array.num_blocks}")
    print(f"batch_size:     {block_array.batch_size}")


def main():
    """
    Demonstrate homomorphic block matrix operations using OpenFHE-NumPy:

      - block matrix addition
      - block matrix subtraction
      - block matrix elementwise multiplication
      - block matrix scalar multiplication
      - block ciphertext matrix + block plaintext matrix

    This example uses elementwise operations only.
    It does not test block matrix multiplication with @.
    """

    # Cryptographic setup
    mult_depth = 4
    batch_size = 4
    block_shape = (2, 2)

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)

    try:
        params.SetBatchSize(batch_size)
    except Exception:
        pass

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
    print(f"Matrix block shape:  {block_shape}")

    # Sample input matrices
    matrix_a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    matrix_b = np.array(
        [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=float,
    )

    print("\nInput matrices")
    print("matrix_a:")
    print(matrix_a)
    print("\nmatrix_b:")
    print(matrix_b)

    # Create encrypted block matrices
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

    # Also create a plaintext block matrix
    ptm_b = onp.block_array(
        cc=cc,
        data=matrix_b,
        batch_size=batch_size,
        block_shape=block_shape,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="P",
    )

    print_block_metadata("Block encrypted matrix_a", ctm_a)
    print_block_metadata("Block encrypted matrix_b", ctm_b)

    # 1) Block matrix addition
    ctm_add = ctm_a + ctm_b
    res_add = ctm_add.decrypt(keys.secretKey, unpack_type="original")

    validate_and_print_results(
        res_add,
        matrix_a + matrix_b,
        "Block matrix addition",
    )

    # 2) Block matrix subtraction
    ctm_sub = ctm_a - ctm_b
    res_sub = ctm_sub.decrypt(keys.secretKey, unpack_type="original")

    validate_and_print_results(
        res_sub,
        matrix_a - matrix_b,
        "Block matrix subtraction",
    )

    # 3) Block matrix elementwise multiplication
    ctm_mul = ctm_a * ctm_b
    res_mul = ctm_mul.decrypt(keys.secretKey, unpack_type="original")

    validate_and_print_results(
        res_mul,
        matrix_a * matrix_b,
        "Block matrix elementwise multiplication",
    )

    # 4) Block matrix scalar multiplication
    ctm_scalar_mul = ctm_a * 3.0
    res_scalar_mul = ctm_scalar_mul.decrypt(keys.secretKey, unpack_type="original")

    validate_and_print_results(
        res_scalar_mul,
        matrix_a * 3.0,
        "Block matrix scalar multiplication",
    )

    # 5) Ciphertext block matrix + plaintext block matrix
    ctm_add_plain = ctm_a + ptm_b
    res_add_plain = ctm_add_plain.decrypt(keys.secretKey, unpack_type="original")

    validate_and_print_results(
        res_add_plain,
        matrix_a + matrix_b,
        "BlockCTArray + BlockPTArray matrix addition",
    )


if __name__ == "__main__":
    main()
