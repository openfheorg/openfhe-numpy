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

from typing import Optional, Tuple, Union, Callable, Any
import numpy as np
import openfhe


from ..openfhe_numpy import EvalCumSum, EvalTranspose
from ..utils.matlib import is_power_of_two, next_power_of_two
from ..utils.constants import UnpackType
from ..utils.errors import ONPDimensionError, ONPError
from ..utils.packing import process_packed_data
from ..utils._helper_slots_ops import _get_elements_at_slots, _get_slot_index

from .tensor import FHETensor, FramePacking


class CTArray(FHETensor[openfhe.Ciphertext]):
    """
    Encrypted tensor class for OpenFHE ciphertexts.
    Represents encrypted matrices or vectors.
    """

    tensor_priority = 10
    is_encrypted = True

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
        from ..operations.slicing import ctarray_getitem

        return ctarray_getitem(self, key)

    def _cta_from_1d(self, cts, *, frame_rows=None):
        """Combine single-slot ciphertexts into one packed 1-D CTArray."""
        cc = self.crypto_context
        N = len(cts)

        if N == 0:
            raise ONPError("Cannot assemble an empty encrypted vector.")

        NN = next_power_of_two(N) if frame_rows is None else frame_rows

        if not is_power_of_two(NN) or N > NN:
            raise ONPError(f"N={N} does not fit frame_rows={NN}.")
        if NN > self.batch_size:
            raise ONPError(f"frame_rows={NN} exceeds batch_size={self.batch_size}.")

        ct_res = cts[0]
        for i in range(1, N):
            ct_res = cc.EvalAdd(ct_res, cc.EvalRotate(cts[i], -i))

        return CTArray(
            data=ct_res,
            original_shape=(N,),
            batch_size=self.batch_size,
            new_shape=(NN, 1),
            order=self.order,
            geometry=FramePacking(
                active=(N, 1),
                padding="zero",
                repeats=1,
            ),
        )

    def _get_element_2D(self, r, c):
        idx = _get_slot_index(r, c, self.shape, self.order)
        return _get_elements_at_slots(
            self.data,
            (idx,),
            self.batch_size,
            idx,
        )

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
            raise ONPError("Secret key is missing.")

        cc = self.data.GetCryptoContext()
        plaintext = cc.Decrypt(self.data, secret_key)
        if plaintext is None:
            raise ONPError("Decryption failed.")

        plaintext.SetLength(self.batch_size)
        result = plaintext.GetRealPackedValue()

        if isinstance(unpack_type, str):
            unpack_type = UnpackType(unpack_type.lower())

        if unpack_type == UnpackType.RAW:
            return result
        if unpack_type == UnpackType.ORIGINAL:
            return process_packed_data(result, self.info)

        return result

    def __neg__(self) -> "CTArray":
        """Return the homomorphic negation of this ciphertext array."""
        cc = self.data.GetCryptoContext()
        return self.clone(cc.EvalNegate(self.data))

    def _transpose(self) -> "CTArray":
        """Internal function to evaluate transpose of an encrypted array."""
        if self.ndim == 2:
            ciphertext = EvalTranspose(self.data, self.ncols)
            pre_padded_shape = (
                self.original_shape[1],
                self.original_shape[0],
            )
            padded_shape = (self.shape[1], self.shape[0])
            geometry = (
                None
                if self.geometry is None
                else FramePacking(
                    active=(self.geometry.active[1], self.geometry.active[0]),
                    padding=self.geometry.padding,
                    repeats=self.geometry.repeats,
                )
            )
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
            geometry=geometry,
        )

    def cumsum(self, axis=None) -> "CTArray":
        """Compute cumulative sums using the logical tensor geometry.

        Parameters
        ----------
        axis : int, optional
            Axis along which the cumulative sum is computed. ``None`` scans a
            vector directly and flattens a matrix in logical C order.

        Returns
        -------
        CTArray
            A new tensor with cumulative sums along the specified axis.
        """

        from ..operations.arithmetic_utils import _normalize_axis

        from ..operations.block_cumsum import (
            _can_flatten_ctarray_without_slot_moves,
            _get_cumsum_lane_parameters,
        )

        if self.geometry is None:
            raise ONPError("cumsum requires packed geometry; construct the tensor with array().")

        if self.ndim not in (0, 1, 2):
            raise ONPDimensionError(
                f"cumsum requires a scalar, vector, or matrix; got {self.ndim}D."
            )

        axis_ndim = 1 if self.ndim <= 1 else self.ndim
        normalized_axis = _normalize_axis(
            "cumsum",
            axis,
            axis_ndim,
        )
        if self.ndim <= 1 and normalized_axis is None:
            normalized_axis = 0

        if self.ndim == 2 and normalized_axis is None:
            if _can_flatten_ctarray_without_slot_moves(self):
                logical_size = self.original_shape[0] * self.original_shape[1]
                frame_rows, frame_cols = self.shape
                flattened = CTArray(
                    data=self.data,
                    original_shape=(logical_size,),
                    batch_size=self.batch_size,
                    new_shape=(frame_rows * frame_cols, 1),
                    order=self.order,
                    geometry=FramePacking(
                        active=(logical_size, 1),
                        padding="zero",
                        repeats=self.geometry.repeats,
                    ),
                )
                return flattened.cumsum(axis=0)

            ciphertexts = [
                self._get_element_2D(row, col)
                for row in range(self.original_shape[0])
                for col in range(self.original_shape[1])
            ]
            return self._cta_from_1d(
                ciphertexts,
                frame_rows=next_power_of_two(len(ciphertexts)),
            ).cumsum(axis=0)

        lane_size, num_lanes, _ = _get_cumsum_lane_parameters(
            self,
            normalized_axis,
        )
        if normalized_axis == 0:
            active_rows = lane_size
            participating_cols = num_lanes
        else:
            active_rows = num_lanes
            participating_cols = lane_size

        frame_rows, frame_cols = self.shape
        ciphertext = EvalCumSum(
            self.data,
            frame_rows,
            frame_cols,
            active_rows,
            participating_cols,
            self.geometry.repeats,
            normalized_axis,
            self.order,
            self.batch_size,
        )

        if self.ndim == 0:
            return CTArray(
                data=ciphertext,
                original_shape=(1,),
                batch_size=self.batch_size,
                new_shape=(1, 1),
                order=self.order,
                geometry=self.geometry,
            )
        return self.clone(data=ciphertext)

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> "CTArray":
        """Apply a ciphertext-level function to the underlying ciphertext.

        The function must accept ``self.data`` as its first argument and return an
        OpenFHE ciphertext with the same logical packing layout.

        Parameters
        ----------
        func : Callable
            Function applied to the underlying ciphertext.
        *args : Any
            Additional positional arguments passed to ``func``.
        **kwargs : Any
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        CTArray
            A new CTArray with the same shape/metadata but the transformed ciphertext.

        Examples
        --------
        Bootstrap to refresh noise level:

        ``result = a.apply(cc.EvalBootstrap)``

        Chebyshev-style functions can be wrapped when the ciphertext is not the
        first argument:

        ``result = a.apply(lambda ct: cc.EvalChebyshevSeries(ct, coeffs, -8, 8))``
        """
        if not callable(func):
            raise TypeError(f"apply expects a callable, got {type(func).__name__}.")

        ct_result = func(self.data, *args, **kwargs)
        return self.clone(data=ct_result)
