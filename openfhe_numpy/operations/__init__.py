"""Operations for homomorphic encryption tensors."""

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
    decrypt_ciphertext,
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
    "decrypt_ciphertext",
]
