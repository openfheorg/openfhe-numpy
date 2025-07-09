# class FHEDebugger:
#     """Tools for debugging FHE operations."""

#     def __init__(self, secret_key=None):
#         """Initialize with optional secret key for decryption."""
#         self.secret_key = secret_key
#         self.log = []

#     def inspect(self, ct_array: CTArray, name: str = ""):
#         """
#         Log metadata and optionally decrypt for inspection.

#         Parameters
#         ----------
#         ct_array : CTArray
#             Tensor to inspect
#         name : str
#             Identifier for this tensor in logs
#         """
#         info = {
#             "name": name,
#             "shape": ct_array.original_shape,
#             "meta": ct_array.metadata,
#             "time": time.time(),
#         }

#         # Add noise estimate if available
#         try:
#             crypto_context = ct_array.data.GetCryptoContext()
#             info["noise_bits"] = crypto_context.GetNoiseEstimate(ct_array.data)
#         except:
#             pass

#         # Decrypt if secret key available and requested
#         if self.secret_key:
#             try:
#                 info["sample_values"] = ct_array.decrypt(self.secret_key).flatten()[:5]
#                 info["decrypted_mean"] = np.mean(ct_array.decrypt(self.secret_key))
#             except:
#                 info["decrypt_error"] = "Failed to decrypt"

#         self.log.append(info)
#         return info

#     def compare(self, ct_array: CTArray, expected: np.ndarray, name: str = ""):
#         """
#         Compare encrypted result with expected plaintext result.

#         Returns
#         -------
#         dict
#             Comparison metrics
#         """
#         if self.secret_key is None:
#             return {"error": "No secret key provided for decryption"}

#         decrypted = ct_array.decrypt(self.secret_key)

#         metrics = {
#             "name": name,
#             "shape_match": decrypted.shape == expected.shape,
#             "mean_diff": np.mean(np.abs(decrypted - expected)),
#             "max_diff": np.max(np.abs(decrypted - expected)),
#             "relative_error": np.mean(np.abs((decrypted - expected) / (expected + 1e-10))),
#         }

#         self.log.append(metrics)
#         return metrics
