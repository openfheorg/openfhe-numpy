import openfhe_numpy.openfhe as openfhe
import io
import numpy as np
import openfhe_numpy as onp
from openfhe_numpy.tensor.tensor import FHETensor
from openfhe_numpy.utils.log import ONP_ERROR
from openfhe_numpy.utils.utils import format_array


class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    Encrypted tensor class for OpenFHE ciphertexts.

    This class represents encrypted tensors that can be manipulated
    using homomorphic operations. It supports standard operations
    like addition, multiplication, and matrix operations.

    Examples
    --------
    >>> # Create and encrypt a matrix
    >>> cc, keys = gen_crypto_context(4)
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> encrypted = onp.array(cc, matrix, slots, keys.publicKey)
    >>> result = encrypted + encrypted  # Homomorphic addition
    >>> decrypted = result.decrypt(keys.secretKey)
    """

    tensor_priority = 10

    def decrypt(
        self, secret_key: openfhe.PrivateKey, format_type: str = "raw", **format_options
    ) -> np.ndarray:
        """Decrypt ciphertext using given secret key with flexible formatting options.

        Parameters
        ----------
        secret_key : openfhe.PrivateKey
            Secret key used for decryption
        format_type : str, optional
            Formatting option to apply:
            - "raw": Return raw decrypted data without reshaping
            - "reshape": Reshape to original dimensions
            - "round": Reshape and round values to integers
            - "auto": Auto-detect best format based on data (default)
        format_options : dict
            Additional formatting options:
            - precision: int, number of decimal places for rounding
            - dtype: numpy dtype for output array
            - new_shape: tuple (min, max) to clip values

        Returns
        -------
        np.ndarray
            Decrypted data with requested formatting applied
        """

        if secret_key is None:
            ONP_ERROR("Secret Key is missing!!!")

        cc = self.data.GetCryptoContext()
        plaintext = cc.Decrypt(self.data, secret_key)

        if plaintext is None:
            ONP_ERROR("Decryption failed")

        plaintext.SetLength(self.batch_size)
        result = plaintext.GetRealPackedValue()

        # Define valid format types
        return format_array(
            result, format_type, self.ndim, self.original_shape, self.shape, **format_options
        )

    def serialize(self) -> dict:
        """Serialize ciphertext and metadata to dictionary."""
        stream = io.BytesIO()
        if not openfhe.Serialize(self.data, stream):
            ONP_ERROR("Failed to serialize Ciphertext")
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
        """Deserialize from dictionary back to CTArray."""
        required_keys = ["ciphertext", "original_shape", "batch_size", "ncols", "order"]
        for key in required_keys:
            if key not in obj:
                ONP_ERROR(f"Missing required key '{key}' in serialized object")

        stream = io.BytesIO(bytes.fromhex(obj["ciphertext"]))
        ciphertext = openfhe.Ciphertext()
        if not openfhe.Deserialize(ciphertext, stream):
            ONP_ERROR("Failed to deserialize ciphertext")
        return cls(
            ciphertext,
            tuple(obj["original_shape"]),
            obj["batch_size"],
            obj["ncols"],
            obj["order"],
        )

    def __repr__(self) -> str:
        return f"CTArray(meta={self.meta})"

    # def _add(self, other) -> "CTArray":
    #     """Element-wise addition with packing compatibility."""
    #     other = self.ensure_compatible_packing(other)

    #     if self.shape != other.shape:
    #         ONP_ERROR("Shape does not match for element-wise addition")
    #     crypto_context = self.data.GetCryptoContext()
    #     ciphertext = crypto_context.EvalAdd(self.data, other.data)
    #     return self.clone(ciphertext)

    # def _add_scalar(self, scalar) -> "CTArray":
    #     """Add scalar value to all elements."""
    #     crypto_context = self.data.GetCryptoContext()

    #     # Create plaintext encoding of scalar
    #     plaintext = crypto_context.MakeCKKSPackedPlaintext([scalar] * self.batch_size)

    #     # Perform addition
    #     result = crypto_context.EvalAdd(self.data, plaintext)
    #     return self.clone(result)

    # # Add these methods for consistency with _add_scalar:

    # def _sub_scalar(self, scalar) -> "CTArray":
    #     """Subtract scalar value from all elements."""
    #     # Implementation

    # def _sub(self, other) -> "CTArray":
    #     if self.shape != other.shape:
    #         ONP_ERROR("Shape does not match for element-wise subtraction")
    #     crypto_context = self.data.GetCryptoContext()
    #     ciphertext = crypto_context.EvalSub(self.data, other.data)
    #     return self.clone(ciphertext)

    # def _multiply(self, other) -> "CTArray":
    #     if self.shape != other.shape:
    #         ONP_ERROR(f"Shape mismatch for multiplication: {self.shape} vs {other.shape}")

    #     crypto_context = self.data.GetCryptoContext()
    #     ciphertext = crypto_context.EvalMul(self.data, other.data)
    #     return self.clone(ciphertext)

    # def _multiply_scalar(self, scalar) -> "CTArray":
    #     """Multiply all elements by scalar."""
    #     # Implementation

    # def _matvec(self, other, sumkey) -> "CTArray":
    #     crypto_context = self.data.GetCryptoContext()

    #     if not isinstance(other, FHETensor):
    #         ONP_ERROR("NOT_IMPLEMENTED")

    #     if self.ndim == 1 and other.ndim == 2:
    #         tensor_matrix = other.clone()
    #         tensor_vector = self.clone()
    #     elif self.ndim == 2 and other.ndim == 1:
    #         tensor_matrix = self.clone()
    #         tensor_vector = other.clone()
    #     else:
    #         ONP_ERROR("NOT_IMPLEMENTED")
    #     if tensor_matrix.original_shape[1] != tensor_vector.original_shape[0]:
    #         ONP_ERROR(
    #             f"Matrix dimension [{tensor_matrix.original_shape}] mismatch with vector dimension [{tensor_vector.shape}]"
    #         )
    #     if (
    #         tensor_matrix.order == MatrixOrder.ROW_MAJOR
    #         and tensor_vector.order == MatrixOrder.COL_MAJOR
    #     ):
    #         ciphertext = EvalMultMatVec(
    #             sumkey,
    #             MatVecEncoding.MM_CRC,
    #             tensor_matrix.ncols,
    #             tensor_vector.data,
    #             tensor_matrix.data,
    #         )
    #         return CTArray(
    #             ciphertext,
    #             (tensor_matrix.original_shape[0], 1),
    #             tensor_matrix.batch_size,
    #             tensor_matrix.ncols,
    #             MatrixOrder.COL_MAJOR,
    #         )

    #     elif (
    #         tensor_matrix.order == MatrixOrder.COL_MAJOR
    #         and tensor_vector.order == MatrixOrder.ROW_MAJOR
    #     ):
    #         ct_product = EvalMultMatVec(
    #             crypto_context,
    #             sumkey,
    #             MatVecEncoding.MM_RCR,
    #             tensor_matrix.ncols,
    #             tensor_vector.data,
    #             tensor_matrix.data,
    #         )
    #         return CTArray(
    #             ct_product,
    #             (tensor_matrix.original_shape[0], 1),
    #             tensor_matrix.batch_size,
    #             tensor_matrix.ncols,
    #             MatrixOrder.ROW_MAJOR,
    #         )
    #     else:
    #         ONP_ERROR(
    #             "Encoding styles of matrix and vector must be complementary (ROW_MAJOR/COL_MAJOR or vice versa)."
    #         )

    # def _matmul(self, other: "CTArray") -> "CTArray":
    #     if self.shape != other.shape:
    #         if isinstance(other, FHETensor):
    #             if other.ndim == 1:
    #                 return self._matvec(other, self.ncols)
    #             return self.clone(EvalMatMulSquare(self.data, other.data, self.ncols))

    #     else:
    #         ONP_ERROR(
    #             f"Matrix dimension mismatch for multiplication: {self.shape} and {other.shape}"
    #         )

    # def _dot(self, other: "CTArray") -> "CTArray":
    #     if self.ndim == 1 and other.ndim == 1:
    #         crypto_context = self.data.GetCryptoContext()
    #         ciphertext = crypto_context.EvalInnerProduct(self.data, other.data, self.ncols)
    #         return self.clone(ciphertext)
    #     else:
    #         return self._matmul(other)

    # def _pow(self, exp: int) -> "CTArray":
    #     """Exponentiate a matrix to power k using homomorphic multiplication."""
    #     if not isinstance(exp, int):
    #         ONP_ERROR(f"Exponent must be integer, got {type(exp).__name__}")

    #     if exp < 0:
    #         ONP_ERROR("Negative exponent not supported in homomorphic encryption")

    #     if exp == 0:
    #         # return algebra.eye(self))
    #         pass

    #     if exp == 1:
    #         return self.clone()

    #     # Binary exponentiation implementation
    #     base = self.clone()
    #     result = None

    #     while exp:
    #         if exp & 1:
    #             result = base if result is None else base @ result
    #         base = base @ base
    #         exp >>= 1
    #     return result

    # def _sum(self, axis=0) -> "CTArray":
    #     """
    #     Compute the cumulative sum of tensor elements along a given axis.

    #     Parameters
    #     ----------
    #     axis : int, optional
    #         Axis along which the cumulative sum is computed. Default is 0.

    #     Returns
    #     -------
    #     CTArray
    #         A new tensor with cumulative sums along the specified axis.
    #     """
    #     if axis == 0:
    #         ciphertext = onp.EvalSumCumRows(
    #             self.data, self.ncols, self.original_shape[1]
    #         )
    #     else:
    #         ciphertext = onp.EvalSumCumCols(self.data, self.ncols)
    #     return self.clone(ciphertext)

    # def _reduce(self, axis=0) -> "CTArray":
    #     if axis == 0:
    #         ciphertext = onp.EvalReduceCumRows(
    #             self.data, self.ncols, self.original_shape[1]
    #         )
    #     else:
    #         ciphertext = onp.EvalReduceCumCols(self.data, self.ncols)
    #     return self.clone(ciphertext)

    def _transpose(self) -> "CTArray":
        """Transpose the encrypted matrix."""
        from openfhe_numpy.utils.matlib import next_power_of_two

        ciphertext = onp.EvalTranspose(self.data, self.ncols)
        shape = (self.original_shape[1], self.original_shape[0])
        ncols = next_power_of_two(shape[1])
        return CTArray(ciphertext, shape, self.batch_size, ncols, self.order)

    # def _trace(self) -> "CTArray":
    #     """Sum along the main diagonal of a 2-D array:"""
    #     ONP_ERROR("Trace operation not implemented for CTArray.")
