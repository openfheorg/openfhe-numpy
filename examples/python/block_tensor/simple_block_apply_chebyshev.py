import numpy as np
from openfhe import *
import openfhe_numpy as onp


def sigmoid(x):
    """Logistic sigmoid, evaluated by the Chebyshev approximation below."""
    return 1.0 / (1.0 + np.exp(-x))


def validate_and_print_results(computed, expected, operation_name, eps):
    """Validate an approximate (Chebyshev) result against the plaintext reference."""
    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    is_match, error = onp.check_equality(computed, expected, eps)
    print(f"\nMatch (tol={eps}): {is_match}, Total Error: {error}")
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
    Apply an arbitrary ciphertext-level function to every block with ``apply``.

    ``BlockCTArray.apply(func, ...)`` runs ``func`` on each block's underlying
    ciphertext and rewraps the results with the original block metadata, so any
    OpenFHE ciphertext-to-ciphertext routine can be evaluated element-wise across
    a tiled tensor.

    This example evaluates smooth activation functions homomorphically via
    Chebyshev approximation:
      - sigmoid over an 8x8 block matrix (a 2x2 grid of ciphertext blocks)
      - tanh over a 40-entry block vector (5 ciphertext blocks)

    ``cc.EvalChebyshevFunction(func, ct, a, b, degree)`` fits ``func`` on the
    interval ``[a, b]`` with a degree-``degree`` polynomial and evaluates it on
    the ciphertext. It takes the ciphertext as its second argument, so it is
    wrapped in a lambda for ``apply`` (which passes the ciphertext first). The
    result is a polynomial approximation, so it is compared with a looser
    tolerance than the exact block operations.
    """

    # --- Cryptographic setup -------------------------------------------------
    ring_dim = 2**12  # 4096
    mult_depth = 8  # degree-13 Chebyshev evaluation
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
    cc.EvalMultKeyGen(keys.secretKey)  # Chebyshev needs the relinearization key

    batch_size = cc.GetRingDimension() // 2

    # --- Chebyshev approximation settings ------------------------------------
    # Inputs lie in [-3, 3]; approximate over the slightly wider [-4, 4].
    lower_bound, upper_bound = -4.0, 4.0
    poly_degree = 13
    tol = 1e-2  # approximation tolerance (exact block ops match far tighter)

    # --- Inputs --------------------------------------------------------------
    matrix = np.linspace(-3.0, 3.0, 64).reshape(8, 8)
    vector = np.linspace(-3.0, 3.0, 40)

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix.shape}  ({matrix.size} entries)")
    print(f"Vector shape        : {vector.shape}  ({vector.size} entries)")

    # --- Build encrypted block tensors ---------------------------------------
    # Explicit block_shape forces a multi-block grid so apply runs over several
    # ciphertext blocks rather than one.
    ctm = onp.block_array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        block_shape=(4, 4),
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    ctv = onp.block_array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        block_shape=(8,),
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    print_block_metadata("Encrypted block matrix", ctm)
    print_block_metadata("Encrypted block vector", ctv)

    all_ok = True

    # 1) Sigmoid over every block of the matrix.
    # apply passes the block ciphertext first, so wrap EvalChebyshevFunction
    # (which expects the ciphertext second) in a lambda.
    res_matrix = ctm.apply(
        lambda ct: cc.EvalChebyshevFunction(sigmoid, ct, lower_bound, upper_bound, poly_degree)
    ).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_matrix, sigmoid(matrix), "Block matrix sigmoid via apply", tol
    )
    all_ok = all_ok and is_match

    # 2) tanh over every block of the vector.
    res_vector = ctv.apply(
        lambda ct: cc.EvalChebyshevFunction(np.tanh, ct, lower_bound, upper_bound, poly_degree)
    ).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_vector, np.tanh(vector), "Block vector tanh via apply", tol
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
