import openfhe as openfhe
import io
from typing import Optional, Tuple, Union
import numpy as np
import openfhe


from ..openfhe_numpy import EvalTranspose
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
        """
        Transpose the encrypted matrix.
        """
        from openfhe_numpy.utils.matlib import next_power_of_two

        ciphertext = EvalTranspose(self.data, self.ncols)
        shape = (self.original_shape[1], self.original_shape[0])
        ncols = next_power_of_two(shape[1])
        return CTArray(ciphertext, shape, self.batch_size, ncols, self.order)
