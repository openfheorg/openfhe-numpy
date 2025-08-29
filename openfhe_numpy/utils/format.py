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

import numpy as np
from typing import Union
from constants import UnpackType


# ===  Visual Format Array ==
def format_array(
    data: np.ndarray,
    unpack_type: Union[UnpackType, str],
    tensor_ndim: int,
    original_shape: tuple,
    tensor_shape: tuple,
    **format_options,
) -> np.ndarray:
    """Format decrypted result according to specified format type.

    Parameters
    ----------
    result : np.ndarray
        Raw decrypted data
    unpack_type : UnpackType or str
        Format type to apply
    tensor_ndim : int
        Number of dimensions in the original tensor
    original_shape : tuple
        Original shape of the tensor
    tensor_shape : tuple
        Current shape of the tensor
    **format_options : dict
        Additional formatting options

    Returns
    -------
    np.ndarray
        Formatted result
    """
    # Convert string format type to enum if needed
    if isinstance(unpack_type, str):
        try:
            unpack_type = UnpackType(unpack_type.lower())
        except ValueError:
            print(f"Warning: Unrecognized unpack_type '{unpack_type}'. Using 'raw' instead.")
            unpack_type = UnpackType.RAW

    # Return raw result if requested
    if unpack_type == UnpackType.RAW:
        return data

    # Apply reshape if needed
    if unpack_type in (UnpackType.RESHAPE, UnpackType.ROUND):
        if "new_shape" in format_options:
            new_shape = format_options["new_shape"]
            if isinstance(new_shape, int):
                data = data.reshape(new_shape)
            elif isinstance(new_shape, tuple) and len(new_shape) == 1:
                data = data[: new_shape[0]]
            else:
                data = np.reshape(data, new_shape)
        else:
            data = _format_array(data, tensor_ndim, original_shape, tensor_shape)

    # Apply rounding if needed
    if unpack_type == UnpackType.ROUND:
        precision = format_options.get("precision", 0)
        data = np.round(data, precision)

    # Apply clipping if requested
    if "clip_range" in format_options:
        min_val, max_val = format_options["clip_range"]
        data = np.clip(data, min_val, max_val)

    return data


def _format_array(array, ndim, original_shape, new_shape):
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
        Reshaping dimensions.

    Returns
    -------
    ndarray
        Reshaped matrix with original dimensions.
    """
    reshaped = np.reshape(array, new_shape)
    if ndim == 2:
        return reshaped[: original_shape[0], : original_shape[1]]
    return np.array(reshaped.flatten()[: original_shape[0]])


# ===  Printing Utilities ==
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
