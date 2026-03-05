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


from ..openfhe_numpy import EvalSumCumCols, EvalSumCumRows, EvalTranspose, ArrayEncodingType
from ..utils.matlib import next_power_of_two
from ..utils.constants import UnpackType
from ..utils.errors import ONP_ERROR
from ..utils.packing import process_packed_data
from ..utils._helper_slots_ops import _get_single_element

from .tensor import FHETensor


class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    Encrypted tensor class for OpenFHE ciphertexts.
    Represents encrypted matrices or vectors.
    """

    tensor_priority = 10

    @property
    def crypto_context(self):
        """Get the underlying crypto context"""
        return self.data.GetCryptoContext()

    @property
    def zeros(self):
        """Get the zeros ciphertext"""
        if self._zeros is None:
            self._zeros = self.crypto_context.EvalMult(self.data, 0)
        return self._zeros

    def __getitem__(self, key):
        if self.shape == ():
            raise TypeError("'int' object is not subscriptable")
        if self.ndim == 1:
            return self._get_1d(key)
        else:
            return self._get_2d(key)

    def _get_1d(self, key):
        cc = self.crypto_context

        if isinstance(key, int):
            return self._cta_from_scalar(self._get_element_1D(key))

        if isinstance(key, slice):
            start, stop, step = key.indices(self.original_shape[0])
            indices = list(range(start, stop, step))

            if len(indices) == 0:
                raise IndexError("slice results in empty array")

            # Extract each selected element and pack them into slots [0..N-1]
            cts = [_get_single_element(cc, self.data, idx, self.batch_size) for idx in indices]
            return self._cta_from_1d(cts)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def _get_2d(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))  # a[0] -> a[0, :]

        if len(key) > 2:
            raise IndexError("too many indices for array")

        row_key = key[0] if len(key) > 0 else slice(None)
        col_key = key[1] if len(key) > 1 else slice(None)

        row_indices, row_collapsed = self._resolve_key(row_key, axis=0)
        col_indices, col_collapsed = self._resolve_key(col_key, axis=1)

        if len(row_indices) == 0 or len(col_indices) == 0:
            return None

        rows = [[self._get_element_2D(r, c) for c in col_indices] for r in row_indices]

        if row_collapsed and col_collapsed:
            return self._cta_from_scalar(rows[0][0])
        if row_collapsed:
            return self._cta_from_1d(rows[0])
        if col_collapsed:
            return self._cta_from_1d([r[0] for r in rows])
        return self._cta_from_2d(rows)

    def _resolve_key(self, key, axis):
        """
        Get indices using Python's builtin function
        """
        size = self.original_shape[axis]
        if isinstance(key, int):
            idx = key if key >= 0 else size + key
            if not (0 <= idx < size):
                raise IndexError(f"index {key} out of bounds for axis {axis} with size {size}")
            return [idx], True
        if isinstance(key, slice):
            s, e, step = key.indices(size)
            return range(s, e, step), False
        raise TypeError(f"invalid index type: {type(key)}")

    def _cta_from_scalar(self, ct):
        """Wrap a single ciphertext as a scalar CTArray"""
        return CTArray(
            data=ct,
            original_shape=(),
            batch_size=self.batch_size,
            new_shape=(),
            order=self.order,
        )

    def _cta_from_1d(self, cts):
        """Combine a list of single ciphertexts into one 1D CTArray"""
        cc = self.crypto_context
        N = len(cts)
        NN = next_power_of_two(N)

        ct_res = cts[0]
        if N == 1:
            return CTArray(
                data=ct_res,
                original_shape=(N,),
                batch_size=self.batch_size,
                new_shape=(NN,),
                order=self.order,
            )

        for i in range(1, N):
            ct_res = cc.EvalAdd(ct_res, cc.EvalRotate(cts[i], -i))

        return CTArray(
            data=ct_res,
            original_shape=(N,),
            batch_size=self.batch_size,
            new_shape=(NN,),
            order=self.order,
        )

    def _cta_from_2d(self, matrix):
        """Combine a matrix of single ciphertexts into one 2D CTArray"""
        cc = self.crypto_context
        nrow = len(matrix)
        ncol = len(matrix[0])

        power_2_r = next_power_of_two(nrow)
        power_2_c = next_power_of_two(ncol)

        ct_res = self.zeros

        if self.order == ArrayEncodingType.ROW_MAJOR:
            for r in range(nrow):
                for c in range(ncol):
                    k = power_2_c * r + c
                    ct_res = cc.EvalAdd(ct_res, cc.EvalRotate(matrix[r][c], -k))
        elif self.order == ArrayEncodingType.COL_MAJOR:
            for r in range(nrow):
                for c in range(ncol):
                    k = power_2_r * c + r
                    ct_res = cc.EvalAdd(ct_res, cc.EvalRotate(matrix[r][c], -k))

        return CTArray(
            data=ct_res,
            original_shape=(nrow, ncol),
            batch_size=self.batch_size,
            new_shape=(power_2_r, power_2_c),
            order=self.order,
        )

    def _get_element_1D(self, key):
        n = self.original_shape[0]
        if not (-n <= key < n):
            raise IndexError(f"index {key} is out of bounds for axis 0 with size {n}")
        key = key if key >= 0 else key + n
        return _get_single_element(self.crypto_context, self.data, key, self.batch_size)

    def _get_element_2D(self, r, c):
        if self.order == ArrayEncodingType.ROW_MAJOR:
            idx = r * self.shape[1] + c
        elif self.order == ArrayEncodingType.COL_MAJOR:
            idx = c * self.shape[0] + r
        else:
            raise TypeError(f"Unsupported packing type: {self.order}")
        return _get_single_element(self.crypto_context, self.data, idx, self.batch_size)

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
