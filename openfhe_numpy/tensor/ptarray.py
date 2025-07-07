from .tensor import FHETensor  # Use relative import
from .ctarray import CTArray
from openfhe import Plaintext, CryptoContext, PublicKey


# -----------------------------------------------------------
# PTArray - Plaintext Tensor
# -----------------------------------------------------------
class PTArray(FHETensor[Plaintext]):
    """Concrete tensor class for OpenFHE plaintexts."""

    def clone(self, data=None):
        return PTArray(
            data or self.data,
            self.original_shape,
            self.batch_size,
            self.ncols,
            self.order,
        )

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

    def __repr__(self) -> str:
        return f"PTArray(meta={self.metadata})"

    def serialize(self) -> dict:
        raise NotImplementedError("Serialize not implemented for plaintext")

    @classmethod
    def deserialize(cls, obj: dict) -> "PTArray":
        raise NotImplementedError("Deserialize not implemented for plaintext")
