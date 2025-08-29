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

from typing import Union, List
import numpy as np

Number = Union[np.generic, int, float, bool]
ArrayNumeric = Union[np.ndarray, List[Number]]


def is_numeric_scalar(x) -> bool:
    if isinstance(x, (int, float, complex, bool, np.generic)):
        return True
    return np.isscalar(x) and isinstance(x, Number)


def is_numeric_arraylike(x) -> bool:
    """Check if x can be converted to a numeric array.

    Parameters
    ----------
    x : Any
        The value to check

    Returns
    -------
    bool
        True if x can be converted to a numeric array, False otherwise
    """

    if isinstance(x, (int, float, complex, bool, np.number)):
        return True
    try:
        if isinstance(x, (str, bytes)):
            return False
        arr = np.asarray(x)
        return arr.dtype.kind in "iufc"  # integers, unsigned, float, complex, timedelta, datetime
    except Exception:
        return False
