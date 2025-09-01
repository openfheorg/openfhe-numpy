import numpy as np
from openfhe import CCParamsCKKSRNS, FIXEDAUTO, \
  HYBRID, GenCryptoContext, PKESchemeFeature
import openfhe_numpy as onp


def get_next_power_of_two(n: int) -> int:
    """
    Find the next power of two for an integer. If \
        the integer is already a power of two, \
        it returns the integer itself.

    Args:
        n: The input integer.

    Returns:
        The next power of two.
    """

    if n <= 0:
        return 1

    p = 1
    while p < n:
        p <<= 1
    return p


def numpy_conv1d(signal, kernel, mode='valid'):
    """
    Compute 1D convolution using NumPy.

    Args:
        signal: Input signal (1D array)
        kernel: Convolution kernel (1D array)
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Convolved signal
    """
    return np.convolve(signal, kernel, mode=mode)


def validate_and_print_results(computed, expected, operation_name):
    """
    Validates the result of a homomorphic operation against \
        an expected value and prints a summary.

    Args:
        computed (np.ndarray): The decrypted result of \
            the homomorphic operation.
        expected (np.ndarray): The expected result, \
            computed in plaintext.
        operation_name (str): Operation being validated.
    """

    print("\n" + "*" * 60)
    print(f"* {operation_name}")
    print("*" * 60)
    print(f"\nExpected (NumPy):\n{expected}")
    print(f"\nDecrypted Result (OpenFHE):\n{computed}")
    difference = expected - computed
    squared_difference = difference**2
    mse = np.mean(squared_difference)
    print(f"\nMean Squared Error (MSE): " f"{mse}")


def openfhe_conv1d(signal, kernel, cc, keys, batch_size, mode='valid'):
    """
    Compute 1D convolution using OpenFHE-NumPy.

    Args:
        signal: Input signal (1D array)
        kernel: Convolution kernel (1D array)
        cc: Crypto context
        keys: Key pair
        batch_size: Batch size for CKKS
        mode: Convolution mode

    Returns:
        Decrypted convolved signal
    """
    if mode != 'valid':
        raise NotImplementedError("Only 'valid' mode implemented \
          for OpenFHE convolution")

    signal_len = len(signal)
    kernel_len = len(kernel)
    output_size = signal_len - kernel_len + 1

    # Pad the signal to the batch size
    padded_signal = np.zeros(batch_size)
    padded_signal[:signal_len] = signal

    # Use the reversed kernel for convolution
    reversed_kernel = kernel[::-1]
    padded_reversed_kernel = np.zeros(batch_size)
    padded_reversed_kernel[:kernel_len] = reversed_kernel

    # Encrypt the signal as an onp.array ciphertext object
    onp_ct_signal = onp.array(
        cc=cc,
        data=padded_signal,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Encode the kernel as an onp.array ciphertext object
    onp_ct_reversed_kernel = onp.array(
        cc=cc,
        data=padded_reversed_kernel,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Initialize the result with zeros as an encrypted array
    onp_ct_convolution_result = onp.array(
        cc=cc,
        data=np.zeros(batch_size),
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="zero",
        public_key=keys.publicKey,
    )

    # Perform convolution using onp.roll and \
    # element-wise multiplication/addition
    # output[i] = sum_{j=0}^{kernel_len-1} signal[i+j] * reversed_kernel[j]
    for j in range(output_size):
        # Rotate the kernel by `j` positions to the right
        onp_ct_rotated_reversed_kernel = onp.roll(onp_ct_reversed_kernel, j)

        # Multiply (point-wise) the rotated kernel by the signal
        term_j = onp_ct_signal * onp_ct_rotated_reversed_kernel

        # Compute total sum
        sum_term_j = onp.sum(term_j)

        # Mask unnecessary components and
        # Add this term to the accumulated result
        mask = np.zeros(batch_size, dtype=int)
        mask[j] = 1
        onp_pt_mask = onp.array(
            cc=cc,
            data=mask,
            batch_size=batch_size,
            order=onp.ROW_MAJOR,
            fhe_type="P",
            mode="zero",
            public_key=keys.publicKey,
        )
        onp_ct_convolution_result = onp_ct_convolution_result + \
            onp_pt_mask * sum_term_j

    # Decryption
    decrypted_array = onp_ct_convolution_result.decrypt(keys.secretKey)

    # Return the first `output_size` elements
    return decrypted_array[:output_size]


def main():
    """
    Run a demonstration of 1D convolution using NumPy and OpenFHE-NumPy.
    """

    np.random.seed(42)

    # Generate random data
    signal_length = 20
    kernel_length = 3

    signal = np.random.randn(signal_length).astype(np.float64)
    kernel = np.random.randn(kernel_length).astype(np.float64)

    print("=== 1D Convolution: NumPy vs OpenFHE-NumPy ===\n")
    print(f"Signal (first 10 values): {signal[:10]}")
    print(f"Kernel (first  5 values): {kernel[:5]}")

    # Cryptographic setup for OpenFHE
    scale_mod_size = 55

    params = CCParamsCKKSRNS()
    params.SetScalingModSize(scale_mod_size)
    params.SetFirstModSize(60)
    params.SetMultiplicativeDepth(2)
    params.SetBatchSize(get_next_power_of_two(signal_length))
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetKeySwitchTechnique(HYBRID)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    rotations = list(range(-(signal_length-kernel_length+1), 1, 1))
    cc.EvalRotateKeyGen(keys.secretKey, rotations)

    ring_dim = cc.GetRingDimension()
    batch_size = cc.GetBatchSize()
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots: {batch_size}")

    if signal_length > batch_size:
        print(f"Warning: Signal length {signal_length} \
            is larger than available slots {batch_size}.")
        print("The signal will be truncated or \
            wrapped around by openfhe-numpy.")

    # Convolution
    try:
        # Compute ground truth result using numpy (unencrypted)
        expected_conv = numpy_conv1d(signal, kernel, mode='valid')

        # Compute convolution with OpenFHE (encrypted)
        openfhe_result = openfhe_conv1d(signal, kernel, cc,
                                        keys, batch_size, mode='valid')

        validate_and_print_results(openfhe_result, expected_conv,
                                   "OpenFHE Convolution")

    except Exception as e:
        print(f"Error in OpenFHE convolution: {e}")


if __name__ == "__main__":
    main()
