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


def main():
    """
    Apply an arbitrary ciphertext-level function to a CTArray with ``apply``.

    ``CTArray.apply(func, ...)`` runs ``func`` on the underlying ciphertext and
    rewraps the result with the original shape/metadata, so any OpenFHE
    ciphertext-to-ciphertext routine can be evaluated element-wise. Here the whole
    tensor fits in a single ciphertext (unlike the block-tensor apply example).

    This evaluates smooth activation functions homomorphically via Chebyshev
    approximation:
      - sigmoid over a 4x4 matrix
      - tanh over an 8-entry vector

    ``cc.EvalChebyshevFunction(func, ct, a, b, degree)`` fits ``func`` on the
    interval ``[a, b]`` with a degree-``degree`` polynomial and evaluates it on
    the ciphertext. It takes the ciphertext as its second argument, so it is
    wrapped in a lambda for ``apply`` (which passes the ciphertext first). The
    result is a polynomial approximation, so it is compared with a looser
    tolerance than exact operations.
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
    tol = 1e-2  # approximation tolerance (exact ops match far tighter)

    # --- Inputs --------------------------------------------------------------
    matrix = np.linspace(-3.0, 3.0, 16).reshape(4, 4)
    vector = np.linspace(-3.0, 3.0, 8)

    print(f"\nCKKS ring dimension : {cc.GetRingDimension()}")
    print(f"Slots per ciphertext: {batch_size}")
    print(f"Matrix shape        : {matrix.shape}")
    print(f"Vector shape        : {vector.shape}")

    # Each tensor fits in one ciphertext, so onp.array gives a plain CTArray.
    ctm = onp.array(
        cc=cc,
        data=matrix,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )
    ctv = onp.array(
        cc=cc,
        data=vector,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        mode="zero",
        fhe_type="C",
        public_key=keys.publicKey,
    )

    all_ok = True

    # 1) Sigmoid over the matrix ciphertext.
    # apply passes the ciphertext first, so wrap EvalChebyshevFunction
    # (which expects the ciphertext second) in a lambda.
    res_matrix = ctm.apply(
        lambda ct: cc.EvalChebyshevFunction(
            sigmoid, ct, lower_bound, upper_bound, poly_degree
        )
    ).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_matrix, sigmoid(matrix), "Matrix sigmoid via apply", tol
    )
    all_ok = all_ok and is_match

    # 2) tanh over the vector ciphertext.
    res_vector = ctv.apply(
        lambda ct: cc.EvalChebyshevFunction(
            np.tanh, ct, lower_bound, upper_bound, poly_degree
        )
    ).decrypt(keys.secretKey, unpack_type="original")
    is_match, _ = validate_and_print_results(
        res_vector, np.tanh(vector), "Vector tanh via apply", tol
    )
    all_ok = all_ok and is_match

    print("\n" + "=" * 60)
    print(f"All checks passed: {all_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
