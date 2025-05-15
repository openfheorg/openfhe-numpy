import sys
import numpy as np
from openfhe_numpy.matlib import *
from openfhe_numpy.config import *


def format(array, ndim, original_shape, new_shape):
    """Reshape a flattened array to its original matrix shape.

    Parameters
    ----------
    array : array_like
        The flattened array to reshape.
    ndim : int
        Number of dimensions of the original matrix.
    original_shape : tuple
        Original shape of the matrix before flattening.
    new_shape : tuple
        Intermediate reshaping dimensions.

    Returns
    -------
    ndarray
        Reshaped matrix with original dimensions.
    """
    reshaped_matrix = np.reshape(array, new_shape)
    if ndim == 2:
        return reshaped_matrix[: original_shape[0], : original_shape[1]]
    return reshaped_matrix[0]


def get_shape(data):
    """Determine the shape and dimension of a given matrix-like structure.

    Parameters
    ----------
    data : list, tuple, or ndarray
        The input matrix or array.

    Returns
    -------
    tuple
        A tuple containing the number of rows, number of columns, and dimensionality.

    Raises
    ------
    ValueError
        If input is neither list, tuple, nor ndarray.
    """
    if isinstance(data, (list, tuple)):
        rows = len(data)
        cols = len(data[0]) if isinstance(data[0], (list, tuple)) else 1
        ndim = 2 if cols > 1 else 1
        return rows, cols, ndim

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return data.shape[0], 0, 1
        return data.shape[0], data.shape[1], 2

    raise ValueError("Invalid data type provided. Must be list, tuple, or ndarray.")


def rotate_vector(vec, k):
    """Rotate a vector by k positions.

    Parameters
    ----------
    vec : list or ndarray
        The input vector to rotate.
    k : int
        Number of positions to rotate the vector.

    Returns
    -------
    list
        Rotated vector.
    """
    n = len(vec)
    new_vec = vec[:]
    return [new_vec[(i + k) % n] for i in range(n)]


def pack_vec_row_wise(v, block_size, num_slots):
    """
    Clone a vector v to fill num_slots
    1 -> 1111 2222 3333
    2
    3
    """
    n = len(v)
    assert is_power_of_two(block_size)
    assert is_power_of_two(num_slots)
    if num_slots < n:
        sys.exit("ERROR ::: [row_wise_vector] vector is longer than total   slots")
    if num_slots == n:
        if num_slots // block_size > 1:
            sys.exit("ERROR ::: [row_wise_vector] vector is too longer, can't duplicate")
        return v

    # print data
    assert num_slots % block_size == 0
    total_blocks = num_slots // block_size
    free_slots = num_slots - n * block_size

    # compute padding
    packed = np.zeros(num_slots)
    k = 0
    for i in range(n):
        for j in range(block_size):
            packed[k] = v[i]
            k += 1
    return packed


def pack_vec_col_wise(v, block_size, num_slots):
    """
    Clone a vector v to fill num_slots
    1 -> 1230 1230 1230
    2
    3
    """
    n = len(v)
    assert is_power_of_two(block_size)
    assert is_power_of_two(num_slots)
    if block_size < n:
        sys.exit(
            f"ERROR ::: [col_wise_vector] vector of size ({n}) is longer than size of a slot ({block_size})"
        )
    if num_slots < n:
        sys.exit("ERROR ::: [col_wise_vector] vector is longer than total slots")
    if num_slots == n:
        return v

    packed = np.zeros(num_slots)

    # print data
    assert num_slots % block_size == 0
    total_blocks = num_slots // block_size
    free_slots = num_slots - n * total_blocks

    k = 0  # index into vector to write
    for i in range(total_blocks):
        for j in range(n):
            packed[k] = v[j]
            k += 1
        k += block_size - n

    return packed


# convert a vector of an packed_rw_mat to its original matrix
def reoriginal_shape(vec, total_slots, ncols):
    n_slots = len(vec)
    row = []
    mat = []
    for k in range(n_slots):
        row.append(vec[k])
        if (k + 1) % ncols == 0 and k >= 1:
            mat.append(row)
            row = []
    return mat


def convert_cw_rw(v, block_size, num_slots):
    org_v = v[:block_size]
    vv = pack_vec_row_wise(org_v, block_size, num_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert \n  {wnice_org}\n->{wnice}")
        print(f"{wnice}")
    return vv


def convert_rw_cw(v, block_size, num_slots):
    org_v = []
    # print(len(v), block_size, num_slots)
    for k in range(block_size):
        org_v.append(v[k * block_size])

    vv = pack_vec_col_wise(org_v, block_size, num_slots)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert {org_v} to {vv[:block_size]}")
        print(f"{wnice}")
    return vv


def print_matrix(matrix, rows):
    """
    Print a matrix in a nicely formatted way.

    Parameters
    ----------
    matrix : array_like
        A 2D matrix (list of lists or ndarray) to print.
    rows : int
        Number of rows to print from the matrix.
    """
    for i in range(rows):
        row_str = "\t".join(f"{val:.2f}" for val in matrix[i])
        print(f"[{row_str}]")


def pack_mat_row_wise(matrix, ncols, total_slots, pad_cols=False):
    """Pack a matrix into a flat array row-wise with zero padding.

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    ncols : int
        Target row size after padding; must be a power of two.
    total_slots : int
        Total number of slots available in the output array; must be a power of two and divisible by ncols.
    pad_rows : bool, optional
        If True, pad the number of rows to the next power of two. Default is False.

    Returns
    -------
    flat_array : ndarray
        Flat array containing the packed and padded elements of the input matrix.

    Raises
    ------
    ValueError
        If ncols or total_slots are not powers of two, or total_slots is insufficient.
    """

    rows, cols = len(matrix), len(matrix[0])
    ncols = next_power_of_two(ncols)

    if not is_power_of_two(total_slots):
        raise ValueError(f"total_slots [{total_slots}] must be a power of two")
    if total_slots % ncols != 0:
        raise ValueError("total_slots must be divisible by ncols")

    padded_cols = next_power_of_two(rows) if pad_cols else rows
    required_size = padded_cols * ncols

    if total_slots < required_size:
        raise ValueError("Total slots insufficient for the given matrix and padding.")

    flat_array = np.zeros(total_slots)

    index = 0
    repeats = total_slots // required_size

    for _ in range(repeats):
        for i in range(rows):
            flat_array[index : index + cols] = matrix[i]
            index += ncols

        index += (padded_cols - rows) * ncols

    return flat_array


# def pack_mat_row_wise(matrix, ncols, total_slots, reps=0, debug=0):
#     """
#     Packing Matrix M using row-wise
#     [[1 2 3] -> [1 2 3 0 4 5 6 0 7 8 9 0]
#     [4 5 6]
#     [7 8 9]]
#     """
#     assert is_power_of_two(ncols)
#     assert is_power_of_two(total_slots)
#     assert total_slots % ncols == 0
#     n, m = len(matrix), len(matrix[0])
#     col_size = len(matrix)
#     if reps > 0:
#         col_size = next_power_of_two(col_size)
#     size = col_size * ncols

#     if total_slots < size:
#         Exception("encrypt_matrix ::: Matrix is too big compared with num_slots")

#     flat = np.zeros(total_slots)

#     k = 0
#     for t in range(total_slots // size):
#         for i in range(n):
#             for j in range(m):
#                 flat[k] = matrix[i][j]
#                 k += 1
#             for j in range(m, ncols):
#                 k += 1

#         for i in range(n, col_size):
#             k += 1

#     return flat


def pack_mat_col_wise(matrix, block_size, num_slots, verbose=0):
    """
    Packing Matric M using row-wise
    [[1 2 3] -> [1 4 7 0 2 5 8 0 3 6 9 0]
     [4 5 6]
     [7 8 9]]
    """
    assert is_power_of_two(block_size)
    assert is_power_of_two(num_slots)
    assert num_slots % block_size == 0
    cols = len(matrix)
    rows = len(matrix[0])
    total_blocks = num_slots // block_size
    free_slots = num_slots - cols * block_size

    if verbose:
        print(
            "#\t [enc. matrix] n = %d, m = %d, #slots = %d, bs = %d, blks = %d, #freeslots = %d, used <= %.3f"
            % (
                cols,
                rows,
                num_slots,
                block_size,
                total_blocks,
                free_slots,
                (num_slots - free_slots) / num_slots,
            )
        )

    if num_slots < cols * rows:
        Exception("encrypt_matrix ::: Matrix is too big compared with num_slots")

    packed = np.zeros(num_slots)
    k = 0  # index into vector to write

    for col in range(cols):
        for row in range(block_size):
            if row < rows:
                packed[k] = matrix[row][col]
            k = k + 1

    return packed


def gen_comm_mat(m, n, opt=1):
    """
    Generate a commutation matrix https://en.wikipedia.org/wiki/Commutation_matrix
    """
    d = m * n
    vec_commutation = [0] * (d**2)
    matrix = np.zeros((m * n, m * n), dtype=int)
    # matrix = [[0] * d for _ in range(d)]
    for i in range(m):
        for j in range(n):
            vec_commutation[(i * n + j) * d + (j * m + i)] = 1
            matrix[i * n + j, j * m + i] = 1
    if opt == 0:
        return matrix
    return vec_commutation


# Function to generate a random square matrix of size n x n
def generate_random_matrix(n):
    return [[random.randint(0, 9) for _ in range(n)] for _ in range(n)]


# Function to multiply two matrices A and B in Plain
def matrix_multiply(A, B, precision=2):
    """
    Multiply two square matrices A and B.

    Parameters
    ----------
    A : list of list of float
        The left-hand matrix.
    B : list of list of float
        The right-hand matrix.
    precision : int, optional
        Number of decimal places to round the result to. Default is 2.

    Returns
    -------
    result : list of list of float
        The resulting matrix after multiplication.
    """
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return [[round(result[i][j], precision) for j in range(n)] for i in range(n)]
