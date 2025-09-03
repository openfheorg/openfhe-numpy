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

"""Operations for homomorphic encryption tensors.

This module provides arithmetic operations and cryptographic utilities
for working with homomorphically encrypted matrix/vector using OpenFHE.
"""

from . import matrix_arithmetic

# Import arithmetic operations
from .matrix_api import (
    add,
    subtract,
    multiply,
    dot,
    matmul,
    transpose,
    pow,
    cumulative_sum,
    cumulative_reduce,
    sum,
    roll,
    mean,
)

# Import crypto context utilities
from .crypto_helper import (
    sum_row_keys,
    sum_col_keys,
    gen_sum_key,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matmult_key,
    gen_accumulate_rows_key,
    gen_accumulate_cols_key,
)

# Define public API
__all__ = [
    # Arithmetic and matrix operations
    "add",
    "subtract",
    "multiply",
    "dot",
    "matmul",
    "transpose",
    "pow",
    "cumulative_sum",
    "cumulative_reduce",
    "sum",
    "roll",
    "mean",
    # Key generation utilities
    "sum_row_keys",
    "sum_col_keys",
    "gen_sum_key",
    "gen_rotation_keys",
    "gen_lintrans_keys",
    "gen_transpose_keys",
    "gen_square_matmult_key",
    "gen_accumulate_rows_key",
    "gen_accumulate_cols_key",
]
