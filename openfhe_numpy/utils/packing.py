"""Packing utilities for OpenFHE-NumPy.

This module provides functions to pack vectors and matrices into flat arrays
with specific layouts (row-wise or column-wise) and padding. These functions
are essential for homomorphic encryption operations that require specific
data layouts for efficient computation.

The module is organized into sections:
1. Shape and utility functions
2. Vector packing functions
3. Matrix packing functions
4. Conversion functions between different packing formats
"""

# from pydantic import validate_call, PositiveInt
from typing import Union, List, Tuple, Dict, Any

import numpy as np

from openfhe_numpy.utils.errors import ONP_ERROR
from openfhe_numpy.utils.matlib import is_power_of_two, next_power_of_two
from openfhe_numpy.utils.typecheck import *
from openfhe_numpy._onp_cpp import *


# === Shape and Utility Functions ==
# # # @validate_call
# def _get_shape(data: ArrayNumeric) -> Tuple[int, int, int]:
#     """Determine the shape and dimension of a given array.

#     Parameters
#     ----------
#     data : list, tuple, or ndarray
#         The input matrix or array.

#     Returns
#     -------
#     tuple
#         A tuple containing the number of rows, number of columns, and dimensionality.

#     Raises
#     ------
#     ONP_ERROR
#         If input is neither list, tuple, nor ndarray.
#     """
#     if isinstance(data, (list, tuple)):
#         rows = len(data)
#         cols = len(data[0]) if isinstance(data[0], (list, tuple)) else 1
#         ndim = 2 if cols > 1 else 1
#         return rows, cols, ndim

#     if isinstance(data, np.ndarray):
#         if data.ndim == 1:
#             return data.shape[0], 0, 1
#         return data.shape[0], data.shape[1], 2

#     ONP_ERROR(f"Invalid data type ({type(data)}) provided. Must be list, tuple, or ndarray.")


# === Vector Packing Functions ==


def _pack_vector_row_wise(
    vector: ArrayNumeric,
    batch_size: int,
    target_cols: int,
    expand: str = "repeat",
    tile: str = "repeat",
    pad_to_power_of_2: bool = True,
    pad_value: str = "repeat",
):
    """Pack a vector by repeating it multiple times in a row-wise pattern.

    This function takes a vector, pads it to a specified size (pad_columns - optionally rounding to
    the next power of two), and then repeats it to fill a target batch size.

    Parameters
    ----------
    vector : ArrayNumeric
        The input 1D vector to be packed.
    batch_size : int
        Total number of slots in the output array; must be a power of two.
    target_cols : int
        Number of columns to expand each row to.
    expand : {"repeat", "zero"}, optional
        How to expand the rows: repeat values or place in first column with zeros.
    tile : {"repeat", "zero"}, optional
        Whether to repeat the entire matrix to fill the batch size ("repeat"), or leave rest as zero ("zero").
    pad_to_power_of_2 : bool, optional
        If True, expand nrows/ncols to next power of 2.
    pad_value : {"repeat", "zero"}, optional
        For "repeat" expand mode, determines how to fill each row.

    Returns
    -------
    tuple
        A tuple containing:
        - ndarray: The packed array of length batch_size
        - tuple: The logical shape of the expanded matrix

    Raises
    ------
    ONP_ERROR
        - If batch_size is not a power of two
        - If expanded vector size exceeds batch_size
        - If expand mode is invalid
        - If tile mode is invalid
        - If final size exceeds batch_size

    Examples
    --------
    >>> _pack_vector_row_wise([1,2,3], 32, 3, tile="zero" )
    (array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0 ,0, 0, 0, 0 ,0, 0, 0, 0 ,0]), (4, 4))

    >>> _pack_vector_row_wise([1,2,3], 32, 3, tile="repeat")
    (array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0,]), (4, 4))

    """

    if not is_power_of_two(batch_size):
        ONP_ERROR(f"Batch size [{batch_size}] must be a power of two")

    n = len(vector)
    nrows = next_power_of_two(n) if pad_to_power_of_2 else n
    if target_cols is None:
        ncols = 1
    else:
        ncols = next_power_of_two(target_cols) if pad_to_power_of_2 else target_cols

    shape = (nrows, ncols)
    expanded_size = nrows * ncols

    if batch_size < (expanded_size):
        ONP_ERROR(f"Padded vector [{nrows} x{ncols}] is longer than the batch size [{batch_size}]")

    flattened = np.zeros(expanded_size, dtype=np.asarray(vector).dtype)
    if expand == "repeat":
        if pad_value == "zero":
            for i in range(n):
                flattened[i * ncols : i * ncols + target_cols] = vector[i]
        elif pad_value == "repeat":
            for i in range(n):
                flattened[i * ncols : (i + 1) * ncols] = vector[i]
        else:
            ONP_ERROR(f"Invalid pad_value: '{pad_value}'. Valid options are 'zero' or 'repeat'.")
    elif expand == "zero":
        flattened = np.zeros(expanded_size, dtype=np.asarray(vector).dtype)
        for i in range(n):
            flattened[i * target_cols] = vector[i]
    else:
        ONP_ERROR(f"Invalid expand mode: '{expand}'. Valid options are 'zero' or 'repeat'.")

    if tile == "repeat":
        repeats = batch_size // expanded_size
    elif tile == "zero":
        repeats = 1
    else:
        ONP_ERROR(f"Invalid tile mode: '{tile}'. Valid options are 'zero' or 'repeat'.")

    if batch_size < (expanded_size * repeats):
        ONP_ERROR(f"Padded vector [{expanded_size} x {repeats}] is longer than the batch size [{batch_size}]")

    total_len = repeats * expanded_size
    output = np.zeros(batch_size, dtype=flattened.dtype)
    output[:total_len] = np.tile(flattened, repeats)

    return output, shape


# # @validate_call
def _pack_vector_col_wise(
    vector: ArrayNumeric,
    batch_size: int,
    target_cols: int,
    expand: str = "repeat",
    tile: str = "repeat",
    pad_to_power_of_2: bool = True,
    pad_value: str = "repeat",
):
    """Pack a vector by expanding it in a column-wise 2D layout and filling a target batch size.
    Parameters
    ----------
    vector : ArrayNumeric
        The input vector to pack (any array-like numeric object).
    batch_size : int
        Total number of slots available in the output array; must be a power of two.
    target_rows : int
        Number of rows to use to expand the vector into 2D.
    expand : {"repeat", "zero"}, optional
        How to expand the columns: repeat each value down, or pad with zeros.
    tile : {"repeat", "zero"}, optional
        Whether to repeat the full expanded array to fill the batch.
    pad_to_power_of_2 : bool, optional
        If True, pad rows/columns to next power of two.
    pad_value : {"repeat", "zero"}, optional
        For "repeat" expand mode, determines how values fill each column.

    Returns
    -------
    tuple
        - ndarray: The packed array of length batch_size.
        - tuple: The logical shape of the matrix (nrows, ncols).

    Raises
    ------
    ONP_ERROR
        - If batch_size is not a power of two
        - If expanded vector size exceeds batch_size
        - If tile_mode is invalid
        - If final size exceeds batch_size

    Examples
    --------
    >>> _pack_vector_col_wise([1,2,3], 32, 3, tile_mode="zero" )
    (array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0 ,0, 0, 0, 0 ,0, 0, 0, 0 ,0, 0, 0, 0 ,0]), (4, 4))

    >>> _pack_vector_col_wise([1,2,3], 32, 3, tile_mode="repeat")
    (array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]), (4, 4))

    """
    if not is_power_of_two(batch_size):
        ONP_ERROR(f"Batch size [{batch_size}] must be a power of two")

    n = len(vector)
    nrows = next_power_of_two(n) if pad_to_power_of_2 else n
    if target_cols is None:
        ncols = 1
    else:
        ncols = next_power_of_two(target_cols) if pad_to_power_of_2 else target_cols
    shape = (nrows, ncols)
    expanded_size = nrows * ncols

    if batch_size < (expanded_size):
        ONP_ERROR(f"Padded vector [{nrows} x{ncols}] is longer than the batch size [{batch_size}]")

    padded = np.zeros(nrows, dtype=vector.dtype)
    padded[:n] = vector
    flattened = np.zeros(expanded_size, dtype=padded.dtype)

    if expand == "repeat":
        if pad_value == "repeat":
            flattened = np.tile(padded, ncols)
        elif pad_value == "zero":
            for col in range(target_cols):
                flattened[col::ncols] = padded
    elif expand == "zero":
        flattened[: n * ncols : ncols] = vector
    else:
        ONP_ERROR(f"Invalid expand mode: '{expand}'. Valid options are 'zero' or 'repeat'.")

    if tile == "repeat":
        repeats = batch_size // expanded_size
    elif tile == "zero":
        repeats = 1
    else:
        ONP_ERROR(f"Invalid tile mode: '{tile}'. Valid options are 'zero' or 'repeat'.")

    total_len = repeats * expanded_size
    output = np.zeros(batch_size, dtype=flattened.dtype)
    output[:total_len] = np.tile(flattened, repeats)
    return output, shape


# === Matrix Packing Functions ==


# # @validate_call
def _pack_matrix_row_wise(matrix, batch_size, pad_to_power_of_2=True, mode="repeat", **kwargs):
    """Pack a matrix into a flat array row-wise with zero padding.

    This function packs a matrix row by row, padding each row to a power of two length
    if specified. For example:
    [[1,2,3],  ->  [1,2,3,0,4,5,6,0,7,8,9,0]
     [4,5,6],
     [7,8,9]]

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    batch_size : int
        Total number of slots available in the output array; must be a power of two.
    pad_to_power_of_2 : bool, optional
        If True, pad the number of rows and columns to the next power of two. Default is True.
    repeats : int, optional
        Number of times to repeat the matrix. If 0, calculated automatically. Default is 0.

    Returns
    -------
    tuple
        A tuple containing:
        - flat_array: ndarray containing the packed and padded elements of the input matrix
        - shape: tuple (nrows, ncols) indicating the logical shape of the packed matrix

    Raises
    ------
    ONP_ERROR
        If batch_size is not a power of two, or if batch_size is not divisible by
        the required size, or if batch_size is insufficient for the padded matrix.
    """
    if not is_power_of_two(batch_size):
        ONP_ERROR(f"batch_size [{batch_size}] must be a power of two")

    rows, cols = len(matrix), len(matrix[0])
    nrows, ncols = rows, cols

    if pad_to_power_of_2:
        ncols = next_power_of_two(cols)
        nrows = next_power_of_two(rows)

    required_size = nrows * ncols
    shape = nrows, ncols

    if mode == "repeat":
        if "repeats" in kwargs and isinstance(kwargs["repeats"], int) and kwargs["repeats"] > 0:
            repeats = kwargs["repeats"]
        else:
            repeats = batch_size // required_size
    elif mode == "zero":
        repeats = 1
    else:
        ONP_ERROR(f"Invalid padding mode: '{mode}'. Valid options are 'zero' or 'repeat'.")

    if batch_size % required_size != 0:
        ONP_ERROR(f"batch_size [{batch_size}] must be divisible by required_size")

    if batch_size < required_size:
        ONP_ERROR(f"batch_size [{batch_size}] is insufficient for padding  [{required_size * repeats}].")

    flat_array = np.zeros(batch_size)
    index = 0

    for _ in range(repeats):
        for i in range(rows):
            flat_array[index : index + cols] = matrix[i]
            index += ncols
        index += (nrows - rows) * ncols

    return flat_array, shape


# @validate_call
def _pack_matrix_col_wise(matrix, batch_size, pad_to_power_of_2=True, mode="repeat", **kwargs):
    """Pack a matrix into a flat array column-wise with zero padding.

    This function packs a matrix column by column, padding each column to a power of two length
    if specified. For example:
    [[1,2,3],  ->  [1,4,7,0,2,5,8,0,3,6,9,0]
     [4,5,6],
     [7,8,9]]

    Parameters
    ----------
    matrix : array_like
        The input 2D matrix to be packed.
    batch_size : int
        Total number of slots available in the output array; must be a power of two.
    pad_to_power_of_2 : bool, optional
        If True, pad the number of rows and columns to the next power of two. Default is True.
    repeats : int, optional
        Number of times to repeat the matrix. If 0, calculated automatically. Default is 0.

    Returns
    -------
    tuple
        A tuple containing:
        - flat_array: ndarray containing the packed and padded elements of the input matrix
        - shape: tuple (nrows, ncols) indicating the logical shape of the packed matrix

    Raises
    ------
    ONP_ERROR
        If batch_size is not a power of two, or if batch_size is not divisible by
        the required size, or if batch_size is insufficient for the padded matrix.
    """
    if not is_power_of_two(batch_size):
        ONP_ERROR(f"batch_size [{batch_size}] must be a power of two")

    rows, cols = len(matrix), len(matrix[0])
    nrows, ncols = rows, cols

    if pad_to_power_of_2:
        ncols = next_power_of_two(cols)
        nrows = next_power_of_two(rows)

    required_size = ncols * nrows
    shape = nrows, ncols

    if mode == "repeat":
        if "repeats" in kwargs and isinstance(kwargs["repeats"], int) and kwargs["repeats"] > 0:
            repeats = kwargs["repeats"]
        else:
            repeats = batch_size // required_size
    elif mode == "zero":
        repeats = 1
    else:
        ONP_ERROR(f"Invalid padding mode: '{mode}'. Valid options are 'zero' or 'repeat'.")

    if batch_size % required_size != 0:
        ONP_ERROR(f"batch_size [{batch_size}] must be divisible by required_size")

    if batch_size < required_size:
        ONP_ERROR(f"batch_size [{batch_size}] is insufficient for padding  [{required_size * repeats}].")

    flat_array = np.zeros(batch_size)
    index = 0

    for c in range(ncols):
        for r in range(nrows):
            if r < rows and c < cols:
                flat_array[index] = matrix[r][c]
            index += 1

    return flat_array, shape


# === Conversion Functions ===


# @validate_call
def reoriginal_shape(vec, batch_size, ncols):
    """Convert a packed vector back to its original matrix shape.

    Parameters
    ----------
    vec : array_like
        The packed vector to reshape into a matrix.
    batch_size : int
        Total number of slots in the packed vector.
    ncols : int
        Number of columns in the target matrix.

    Returns
    -------
    list
        A 2D list representing the reshaped matrix.
    """
    n_slots = len(vec)
    row = []
    mat = []
    for k in range(n_slots):
        row.append(vec[k])
        if (k + 1) % ncols == 0 and k >= 1:
            mat.append(row)
            row = []
    return mat


# @validate_call
def vector_col_major_2_row_major(v, block_size, batch_size):
    """Convert a column-wise packed vector to row-wise packing.

    Parameters
    ----------
    v : array_like
        The column-wise packed vector to convert.
    block_size : int
        Size of each block in the packed vector.
    batch_size : int
        Total number of slots in the output vector.

    Returns
    -------
    ndarray
        The vector repacked in row-wise format.
    """
    org_v = v[:block_size]
    vv = _pack_vector_row_wise(org_v, block_size, batch_size)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert \n  {wnice_org}\n->{wnice}")
        print(f"{wnice}")
    return vv


# @validate_call
def vector_row_major_2_col_major(v, block_size, batch_size):
    """Convert a row-wise packed vector to column-wise packing.

    Parameters
    ----------
    v : array_like
        The row-wise packed vector to convert.
    block_size : int
        Size of each block in the packed vector.
    batch_size : int
        Total number of slots in the output vector.

    Returns
    -------
    ndarray
        The vector repacked in column-wise format.
    """
    org_v = []
    # print(len(v), block_size, batch_size)
    for k in range(block_size):
        org_v.append(v[k * block_size])

    vv = _pack_vector_col_wise(org_v, block_size, batch_size)

    if 0:
        wnice_org = [round(x, 3) for x in v[: 2 * block_size]]
        vv_b = vv[: 2 * block_size]
        wnice = [round(x, 3) for x in vv_b]
        print(f"convert {org_v} to {vv[:block_size]}")
        print(f"{wnice}")
    return vv


# Utilities functions for unpacking


# @validate_call
def _extract_matrix(data, info):
    ncols = info["shape"][1]
    nrows = info["batch_size"] // ncols
    reshaped = np.reshape(data, (nrows, ncols))

    if info["order"] == ArrayEncodingType.ROW_MAJOR:
        return reshaped[: info["original_shape"][0], : info["original_shape"][1]]
    elif info["order"] == ArrayEncodingType.COL_MAJOR:
        tranposed = np.transpose(reshaped)
        return tranposed[: info["original_shape"][0], : info["original_shape"][1]]
    else:
        ONP_ERROR("Order is not supported!!!")
        return None


# @validate_call
def _extract_vector(data, info):
    if info["ndim"] == 1:
        original_row = info["original_shape"][0]

        ncols = info["shape"][1]
        nrows = info["batch_size"] // ncols
        reshaped = np.reshape(data, (nrows, ncols))

        if info["order"] == ROW_MAJOR:
            return reshaped[:original_row, 0]
        elif info["order"] == COL_MAJOR:
            return reshaped[0, :original_row]
        else:
            ONP_ERROR("Order is not supported!!!")
            return None
    else:
        return data[0]


# @validate_call
def process_packed_data(
    data: np.ndarray,
    info: Dict[str, Any],
) -> np.ndarray:
    """
    Reshape a flattened array to its original matrix shape or a new shape.

    Parameters
    ----------
    array : array_like
        The flattened array to reshape.
    info: dictionary
        Array's metadata

    Returns
    -------
    ndarray
        Reshaped array to its desired shape
    """

    if info["ndim"] == 2:
        return _extract_matrix(data, info)
    else:
        return _extract_vector(data, info)
