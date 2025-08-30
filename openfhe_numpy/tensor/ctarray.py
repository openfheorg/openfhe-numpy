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

import io
from typing import Optional, Tuple, Union
import numpy as np
import openfhe


from ..openfhe_numpy import (
    ArrayEncodingType,
    EvalSumCumCols,
    EvalSumCumRows,
    EvalTranspose,
)
from ..utils.constants import UnpackType, DataType
from ..utils.errors import ONP_ERROR
from ..utils.packing import process_packed_data

from .tensor import FHETensor


class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    Encrypted tensor class for OpenFHE ciphertexts.
    Represents encrypted matrices or vectors.
    """

    tensor_priority = 10

    def decrypt(
        self,
        secret_key: openfhe.PrivateKey,
        unpack_type: UnpackType = UnpackType.ORIGINAL,
        new_shape: Optional[Union[Tuple[int, ...], int]] = None,
    ) -> np.ndarray:
        """
        Decrypt the ciphertext and format the output.

        Parameters
        ----------
        secret_key : openfhe.PrivateKey
            Secret key for decryption.
        unpack_type : UnpackType
            - RAW: raw data, no reshape
            - ORIGINAL: reshape to original dimensions
            - ROUND: reshape and round to integers (not support now)
            - AUTO: auto-detect best format (not support now)
        new_shape : tuple or int, optional
            Custom shape for the output array. If None, uses original shape.

        Returns
        -------
        np.ndarray
            The decrypted data, formatted by 'unpack_type'.
        """
        if secret_key is None:
            ONP_ERROR("Secret key is missing.")

        cc = self.data.GetCryptoContext()
        plaintext = cc.Decrypt(self.data, secret_key)
        if plaintext is None:
            ONP_ERROR("Decryption failed.")

        plaintext.SetLength(self.batch_size)
        result = plaintext.GetRealPackedValue()

        if isinstance(unpack_type, str):
            unpack_type = UnpackType(unpack_type.lower())

        if unpack_type == UnpackType.RAW:
            return result
        if unpack_type == UnpackType.ORIGINAL:
            return process_packed_data(result, self.info)

        return result

    def serialize(self) -> dict:
        """
        Serialize ciphertext and metadata to a dictionary.
        """
        stream = io.BytesIO()
        if not openfhe.Serialize(self.data, stream):
            ONP_ERROR("Failed to serialize ciphertext.")

        return {
            "type": self.type,
            "original_shape": self.original_shape,
            "batch_size": self.batch_size,
            "ncols": self.ncols,
            "order": self.order,
            "ciphertext": stream.getvalue().hex(),
        }

    @classmethod
    def deserialize(cls, obj: dict) -> "CTArray":
        """
        Deserialize a dictionary back into a CTArray.
        """
        required_keys = [
            "ciphertext",
            "original_shape",
            "batch_size",
            "ncols",
            "order",
        ]
        for key in required_keys:
            if key not in obj:
                ONP_ERROR(f"Missing required key '{key}' in serialized object.")

        stream = io.BytesIO(bytes.fromhex(obj["ciphertext"]))
        ciphertext = openfhe.Ciphertext()
        if not openfhe.Deserialize(ciphertext, stream):
            ONP_ERROR("Failed to deserialize ciphertext.")

        return cls(
            ciphertext,
            tuple(obj["original_shape"]),
            obj["batch_size"],
            obj["ncols"],
            obj["order"],
        )

    def __repr__(self) -> str:
        return f"CTArray(metadata={self.metadata})"

    def _sum(self) -> "CTArray":
        # TODO: implement sum over encrypted data
        pass

    def _transpose(self) -> "CTArray":
        """Internal function to evaluate transpose of an encrypted array."""
        if self.ndim == 2:
            ciphertext = EvalTranspose(self.data, self.ncols)
            pre_padded_shape = (
                self.original_shape[1],
                self.original_shape[0],
            )
            padded_shape = (self.shape[1], self.shape[0])
        elif self.ndim == 1:
            return self
        else:
            raise NotImplementedError("This function is not implemented with dimension > 2")
        return CTArray(
            ciphertext,
            pre_padded_shape,
            self.batch_size,
            padded_shape,
            self.order,
        )

    def cumulative_sum(self, axis: int = 0) -> "CTArray":
        """
        Compute the cumulative sum of tensor elements along a given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which the cumulative sum is computed. Default is 0.

        Returns
        -------
        CTArray
            A new tensor with cumulative sums along the specified axis.
        """

        if self.ndim != 1 and self.ndim != 2:
            ONP_ERROR(f"Dimension of array {self.ndim} is illegal ")

        if self.ndim != 1 and axis is None:
            ONP_ERROR("axis=None not allowed for >1D")

        if self.ndim == 2 and axis not in (0, 1):
            ONP_ERROR("Axis must be 0 or 1 for cumulative sum operation")

        order = self.order
        shape = self.shape
        original_shape = self.original_shape

        if axis is None:
            ciphertext = EvalSumCumRows(self.data, self.ncols, self.original_shape[1])

        # cumulative_sum over rows
        elif axis == 0:
            if self.order == ArrayEncodingType.ROW_MAJOR:
                ciphertext = EvalSumCumRows(self.data, self.ncols, self.original_shape[1])

            elif self.order == ArrayEncodingType.COL_MAJOR:
                ciphertext = EvalSumCumCols(self.data, self.nrows)

            else:
                raise ONP_ERROR(f"Not support this packing order [{self.order}].")

        # cumulative_sum over cols
        elif axis == 1:
            if self.order == ArrayEncodingType.ROW_MAJOR:
                ciphertext = EvalSumCumCols(self.data, self.ncols)

            elif self.order == ArrayEncodingType.COL_MAJOR:
                ciphertext = EvalSumCumRows(self.data, self.nrows, self.original_shape[0])

            else:
                raise ONP_ERROR(f"Not support this packing order[{self.order}].")
        else:
            raise ONP_ERROR(f"Invalid axis [{axis}].")
        return CTArray(ciphertext, original_shape, self.batch_size, shape, order)

    def gen_sum_row_key(self, secret_key: openfhe.PrivateKey) -> openfhe.EvalKey:
        context = secret_key.GetCryptoContext()
        if self.order == ArrayEncodingType.ROW_MAJOR:
            sum_rows_key = context.EvalSumRowsKeyGen(secret_key, self.ncols, self.batch_size)
        elif self.order == ArrayEncodingType.COL_MAJOR:
            sum_rows_key = context.EvalSumColsKeyGen(secret_key)
        else:
            raise ValueError("Invalid order.")

        return sum_rows_key
