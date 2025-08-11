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
from ..utils.constants import *
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
            - ROUND: reshape and round to integers
            - AUTO: auto-detect best format
        new_shape : tuple or int, optional
            Custom shape for the output array. If None, uses original shape.

        Returns
        -------
        np.ndarray
            The decrypted data, formatted per `unpack_type`.
        """
        if secret_key is None:
            ONP_ERROR("Secret key is missing.")

        cc = self.data.GetCryptoContext()
        plaintext = cc.Decrypt(self.data, secret_key)
        if plaintext is None:
            ONP_ERROR("Decryption failed.")

        plaintext.SetLength(self.batch_size)
        result = plaintext.GetRealPackedValue()

        # print("DEBUG ::: result = ", result[:32])

        if isinstance(unpack_type, str):
            unpack_type = UnpackType(unpack_type.lower())

        if unpack_type == UnpackType.RAW:
            return result
        if unpack_type == UnpackType.ORIGINAL:
            return process_packed_data(result, self.info)

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
        # """
        # Transpose the encrypted matrix.
        # """
        # from openfhe_numpy.utils.matlib import next_power_of_two

        # ciphertext = EvalTranspose(self.data, self.ncols)
        # shape = (self.original_shape[1], self.original_shape[0])
        # ncols = next_power_of_two(shape[1])
        # return CTArray(ciphertext, shape, self.batch_size, ncols, self.order)

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
            raise NotImplementedError(
                "This function is not implemented with dimension > 2"
            )
        return CTArray(
            ciphertext,
            pre_padded_shape,
            self.batch_size,
            padded_shape,
            self.order,
        )

    def cumulative_sum(self, axis: int) -> "CTArray":
        """
        Compute the cumulative sum of tensor elements along a given axis.

        Parameters
        ----------
        tensor : CTArray
            Input encrypted tensor.
        axis : int, optional
            Axis along which the cumulative sum is computed. Default is 0.
        keepdims : bool, optional
            Whether to keep the dimensions of the original tensor. Default is True.

        Returns
        -------
        CTArray
            A new tensor with cumulative sums along the specified axis.
        """

        if self.ndim != 1 or self.ndim != 2:
            ONP_ERROR(f"Dimension of array {self.ndim} is illegal ")

        if self.ndim != 1 and axis is None:
            ONP_ERROR("axis=None not allowed for >1D")

        if self.ndim == 2 and axis not in (0, 1):
            ONP_ERROR("Axis must be 0 or 1 for cumulative sum operation")

        order = self.order
        shape = self.shape
        original_shape = self.original_shape

        # cumulative_sum for vector
        if axis is None:
            ciphertext = EvalSumCumRows(
                self.data, self.ncols, self.original_shape[1]
            )

        # cumulative_sum over rows
        if axis == 0:
            if self.order == ArrayEncodingType.ROW_MAJOR:
                ciphertext = EvalSumCumRows(
                    self.data, self.ncols, self.original_shape[1]
                )

            elif self.order == ArrayEncodingType.COL_MAJOR:
                ciphertext = EvalSumCumCols(self.data, self.nrows)

                # shape = self.shape[1], self.shape[0]
                # original_shape = self.original_shape[1], self.original_shape[0]
            else:
                raise ValueError(
                    f"Not support this packing order [{self.order}]."
                )

        # cumulative_sum over cols
        elif axis == 1:
            if self.order == ArrayEncodingType.ROW_MAJOR:
                ciphertext = EvalSumCumCols(self.data, self.ncols)

            elif self.order == ArrayEncodingType.COL_MAJOR:
                ciphertext = EvalSumCumRows(
                    self.data, self.nrows, self.original_shape[0]
                )

                # shape = self.shape[1], self.shape[0]
                # original_shape = self.original_shape[1], self.original_shape[0]
            else:
                raise ValueError(f"Invalid axis [{self.order}].")
        else:
            raise ValueError(f"Invalid axis [{axis}].")
        return CTArray(
            ciphertext, original_shape, self.batch_size, shape, order
        )
