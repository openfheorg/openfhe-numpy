import numpy as np
from openfhe import *
import openfhe_numpy as onp

def validate_and_print_results(computed, expected, operation_name):
    ### Helper function to validate and print results ###

    print("\n" + "*" * 60)
    print(f"* {operation_name} ")
    print("*" * 60)

    print(f"\nExpected:\n{expected}")
    print(f"\nDecrypted Result:\n{computed}")

    _, error = onp.check_equality(computed, expected)
    print(f"\nTotal Error: {error}")
    return


def sigmoid(x):
    """
    Sigmoid activation function for output layer.
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function for hidden layer.
    """
    return np.maximum(0, x)


def np_forward_pass(X, W1, b1, W2, b2):
    """
    Performs the forward pass (inference) of the 2-layer MLP using numpy.
    This is the reference implementation.

    Args:
        X (np.ndarray): Input data of shape (num_samples, num_features).
        W1 (np.ndarray): Weights for the hidden layer.
        b1 (np.ndarray): Biases for the hidden layer.
        W2 (np.ndarray): Weights for the output layer.
        b2 (np.ndarray): Biases for the output layer.

    Returns:
        np.ndarray: The predicted probabilities for each sample.
    """
    
    print(f"Evaluating NP Forward Pass Started ... \n")
    
    # Hidden Layer
    # Z1 = W1 * X + b1
    Z1 = np.dot(X, W1) + b1
    # A1 = activation(Z1)
    A1 = relu(Z1)

    # Output Layer
    # Z2 = W2 * A1 + b2
    Z2 = np.dot(A1, W2) + b2
    # A2 = sigmoid(Z2)
    A2 = sigmoid(Z2)
    
    print(f"Evaluating NP Forward Pass Ended Successfully\n")
    
    return A2


def make_square_power_of_2(arr, target_dim=None):
    """
    Convert a numpy array to a square matrix with dimensions that are a power of 2.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Input array of 1D or 2D shape
    target_dim : int, optional
        Desired dimension (must be power of 2). If None, uses the smallest 
        power of 2 that fits the input array.
    
    Returns:
    --------
    numpy.ndarray
        Square matrix with dimensions that are a power of 2, zero-padded as needed
    
    Raises:
    -------
    ValueError
        If target_dim is not a power of 2 or if target_dim is smaller than required
    """
    
    # Convert to numpy array if not already
    arr = np.array(arr)
    
    # Handle 1D arrays by converting to row vector
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        raise ValueError("Input array must be 1D or 2D")
    
    rows, cols = arr.shape
    
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    def next_power_of_2(n):
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()
    
    if target_dim is not None:
        # Validate target_dim
        if not is_power_of_2(target_dim):
            raise ValueError(f"target_dim ({target_dim}) must be a power of 2")
        
        # Check if target_dim is large enough
        min_required = max(rows, cols)
        if target_dim < min_required:
            raise ValueError(f"target_dim ({target_dim}) must be at least {min_required} to fit the input array")
        
        dim = target_dim
    else:
        # Find the smallest power of 2 that can contain the input
        max_dim = max(rows, cols)
        dim = next_power_of_2(max_dim)
    
    # Create the output matrix
    result = np.zeros((dim, dim), dtype=arr.dtype)   
    # Embed input in the output array (upper-left corner) 
    result[:rows, :cols] = arr
    return result


def onp_forward_pass(X, W1, b1, W2, b2, cc, keys, batch_size):
    """
    Performs the forward pass (inference) of the 2-layer MLP.

    Args:
        X: Input sample of shape (1, num_features).
        W1: Weights for the hidden layer.
        b1: Biases for the hidden layer.
        W2: Weights for the output layer.
        b2: Biases for the output layer.
        cc: Crypto context
        keys: Key pair
        batch_size: Batch size for CKKS

    Returns:
        decrypted_A2_masked: The predicted probabilities for the input sample.
    """
    
    print(f"\nEvaluating ONP Forward Pass (Encrypted) Started ...")
    
    # Hidden Layer
    # Z1 = W1 * X + b1
    
    X_square_zero_padded = make_square_power_of_2(X, 4)
    onp_ct_X = onp.array(
        cc=cc,
        data=X_square_zero_padded,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="tile",
        public_key=keys.publicKey,
    )    
   
    W1_square_zero_padded = make_square_power_of_2(W1)
    onp_ct_W1 = onp.array(
        cc=cc,
        data=W1_square_zero_padded,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="tile",
        public_key=keys.publicKey,
    )
    
    b1_square_zero_padded = make_square_power_of_2(b1)
    onp_ct_b1 = onp.array(
        cc=cc,
        data=b1_square_zero_padded,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="tile",
        public_key=keys.publicKey,
    )
    
    onp.EvalSquareMatMultRotateKeyGen(keys.secretKey, onp_ct_X.ncols)
     
    onp_ct_Z1 = onp_ct_X @ onp_ct_W1 + onp_ct_b1

    # Evaluate RELU
    lower_bound = -1
    upper_bound = 1
    poly_degree = 13
    # Get Ciphertext object from the onp.array pbject
    ct_A1 = cc.EvalChebyshevFunction(relu, onp_ct_Z1.data, lower_bound, upper_bound, poly_degree)
    # Set the result ciphertext object in the onp.array pbject
    onp_ct_A1 = onp_ct_Z1
    onp_ct_A1.data = ct_A1
        
    # Output layer
    W2_square_zero_padded = make_square_power_of_2(W2)
    onp_ct_W2 = onp.array(
        cc=cc,
        data=W2_square_zero_padded,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="C",
        mode="tile",
        public_key=keys.publicKey,
    )
    b2_square_zero_padded = make_square_power_of_2(b2, 4)
    onp_ct_b2 = onp.array(
        cc=cc,
        data=b2_square_zero_padded,
        batch_size=batch_size,
        order=onp.COL_MAJOR,
        fhe_type="C",
        mode="tile",
        public_key=keys.publicKey,
    )
    
    onp_ct_Z2 = onp_ct_A1 @ onp_ct_W2 + onp_ct_b2

    # Evaluate sigmoid
    ct_A2 = cc.EvalChebyshevFunction(sigmoid, onp_ct_Z2.data, lower_bound, upper_bound, poly_degree)
    onp_ct_A2 = onp_ct_Z2
    onp_ct_A2.data = ct_A2
    
    # mask out any unnecessary results around the borders of the output array
    mask = [1]
    mask_square_zero_padded = make_square_power_of_2(mask, 4)
    onp_pt_mask = onp.array(
        cc=cc,
        data=mask_square_zero_padded,
        batch_size=batch_size,
        order=onp.ROW_MAJOR,
        fhe_type="P",
        mode="tile",
        public_key=keys.publicKey,
    )
    
    onp_ct_A2_masked = onp_ct_A2 * onp_pt_mask
    decrypted_A2_masked = onp_ct_A2_masked.decrypt(keys.secretKey)
    
    print(f"Evaluating ONP Forward Pass (Encrypted) Ended Successfully\n")
    
    return decrypted_A2_masked


if __name__ == '__main__':
    # Network architecture: 2 input features, 4 hidden neurons, 1 output neuron

    # Weights and biases for the hidden layer
    W1 = np.array([[0.1, -0.5, 0.3, -0.2],
                   [0.4, 0.6, -0.7, 0.1]
                  ])
    b1 = np.array([0.1, 0.2, 0.3, 0.4])

    # Weights and biases for the output layer
    W2 = np.array([[-0.5], 
                   [0.8], 
                   [-0.4], 
                   [0.6]])
    b2 = np.array([-0.3])

    # Test sample 
    X_test = np.array([
        [0.1, 0.2],
    ])

    print("2-Layer MLP Inference with NumPy and OpenFHE-Numpy")
    print(f"Input data:\n{X_test}\n")

    np_probabilities = np_forward_pass(X_test, W1, b1, W2, b2)
    print(f"NP Predicted Probabilities:\n{np_probabilities.flatten()}")
    
    # openfhe-numpy implementation
    # Cryptographic setup for OpenFHE
    batch_size = 16
    scale_mod_size = 35
    params = CCParamsCKKSRNS()
    params.SetScalingModSize(scale_mod_size)
    params.SetFirstModSize(40)
    params.SetMultiplicativeDepth(15)
    params.SetBatchSize(batch_size)
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetKeySwitchTechnique(HYBRID)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ring_dim = cc.GetRingDimension()
    batch_size = cc.GetBatchSize()
    print(f"\nCKKS ring dimension: {ring_dim}")
    print(f"Available slots: {batch_size}")
    
    onp_probabilities = onp_forward_pass(X_test, W1, b1, W2, b2, cc, keys, batch_size)
    print(f"ONP Predicted Probabilities:\n{onp_probabilities.flatten()}")
    
    validate_and_print_results(onp_probabilities[0][0], np_probabilities[0][0], "MLP")
    