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
from .tensor import FHETensor  # Use relative import
from .ctarray import CTArray
from ..utils.constants import UnpackType
from ..utils.packing import process_packed_data
from openfhe import Plaintext, CryptoContext, PublicKey


# -----------------------------------------------------------
# PTArray - Plaintext Tensor
# -----------------------------------------------------------
class PTArray(FHETensor[Plaintext]):
    """Concrete tensor class for OpenFHE plaintexts."""

    is_encrypted = False

    def clone(self, data=None):
        return super().clone(data)

    def encrypt(self, crypto_context: CryptoContext, public_key: PublicKey):
        ciphertext = crypto_context.Encrypt(public_key, self.data)
        return CTArray(
            ciphertext,
            self.original_shape,
            self.batch_size,
            self.shape,
            self.order,
        )

    def decrypt(self, *args, **kwargs):
        raise NotImplementedError("Decrypt not implemented for plaintext")

    def decode(self, unpack_type: UnpackType = UnpackType.ORIGINAL) -> np.ndarray:
        """Decode plaintext packed slots into a NumPy array."""
        self.data.SetLength(self.batch_size)
        result = self.data.GetRealPackedValue()
        if isinstance(unpack_type, str):
            unpack_type = UnpackType(unpack_type.lower())
        if unpack_type == UnpackType.RAW:
            return np.asarray(result)
        if unpack_type == UnpackType.ORIGINAL:
            return process_packed_data(result, self.info)
        return np.asarray(result)

    def __repr__(self) -> str:
        return f"PTArray(meta={self.info})"

    def serialize(self) -> dict:
        raise NotImplementedError("Serialize not implemented for plaintext")

    @classmethod
    def deserialize(cls, obj: dict) -> "PTArray":
        raise NotImplementedError("Deserialize not implemented for plaintext")
