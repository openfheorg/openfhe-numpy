# ==================================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==================================================================================

import math
import numpy as np
from .constants import EPSILON


def next_power_of_two(x):
    """
    Compute the smallest power of two greater than or equal to x.

    Parameters
    ----------
    x : int
        The input number.

    Returns
    -------
    int
        The next power of two ≥ x.
    """
    return 2 ** math.ceil(math.log2(x))


def is_power_of_two(x):
    """
    Check if a number is a power of two.

    Parameters
    ----------
    x : int
        Number to check.

    Returns
    -------
    bool
        True if x is a power of two, False otherwise.
    """
    return (x & (x - 1) == 0) and x != 0


def check_single_equality(a, b, eps=EPSILON):
    """
    Compare two floating-point values within a tolerance.

    Parameters
    ----------
    a, b : float
        Values to compare.
    eps : float, optional
        Tolerance threshold (default: EPSILON).

    Returns
    -------
    tuple
        (is_equal: bool, error: float)
    """
    error = abs(a - b)
    return error <= eps, error


def check_equality_vector(a, b, vector_size=None, eps=EPSILON):
    """
    Compare two vectors element-wise, ensuring the maximum absolute error does not exceed eps.
    The total error is computed using l2 norm sqrt(sum_ (a_i-b_i)^2)

    Parameters
    ----------
    a, b : list
        Input vectors.
    vector_size : int, optional
        Number of entries to compare (default: full length).
    eps : float
        Comparison tolerance.

    Returns
    -------
    tuple
        (is_equal: bool, total_error: float)
    """
    if vector_size is None:
        vector_size = len(a)
    error = 0
    is_equal = True
    for i in range(vector_size):
        f, e = check_single_equality(a[i], b[i], eps)
        error += e**2
        if not f:
            is_equal = False
    return is_equal, math.sqrt(error)


def check_equality(a, b, eps=EPSILON):
    """
    Compare two matrices element-wise with tolerance.

    Parameters
    ----------
    a, b : list of list
        Matrices to compare.
    eps : float
        Tolerance threshold.

    Returns
    -------
    tuple
        (is_equal: bool, total_error: float)
    """
    a = np.asarray(a)
    b = np.asarray(b)

    error, is_equal = 0, True

    if np.isscalar(a) or (hasattr(a, "shape") and a.shape == ()):
        error = abs(a - b)
        return error <= eps, error

    if a.ndim == 1:
        for i in range(len(a)):
            f, e = check_single_equality(a[i], b[i], eps)
            error += e**2
            if not f:
                is_equal = False
        return is_equal, error

    rows, cols = len(a), len(a[0])

    for i in range(rows):
        for j in range(cols):
            f, e = check_single_equality(a[i][j], b[i][j], eps)
            error += e**2
            if not f:
                is_equal = False
    return is_equal, error


def _rotate_vector(vec, k):
    """Rotate a vector by k positions.

    Parameters
    ----------
    vec : list or ndarray
        The input vector to rotate.
    k : int
        Number of positions to rotate the vector: left rotation if k > 0, right rotation if k < 0.

    Returns
    -------
    list
        Rotated vector.
    """
    n = len(vec)
    # avoid division by zero if vec is empty
    if n == 0:
        return []

    k %= n
    return vec[k:] + vec[:k]
