"""Operations for homomorphic encryption tensors."""

# Import arithmetic operations
from .matrix_arithmetic import *
from .matrix_api import (
    add,
    multiply,
    dot,
    matmul,
    transpose,
    power,
    cumsum,
    cumreduce,
)

# Import crypto context utilities
from .crypto_context import (
    sum_row_keys,
    sum_col_keys,
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
    "multiply",
    "dot",
    "matmul",
    "transpose",
    "power",
    "cumsum",
    "cumreduce",
    # Key generation utilities
    "sum_row_keys",
    "sum_col_keys",
    "gen_rotation_keys",
    "gen_lintrans_keys",
    "gen_transpose_keys",
    "gen_square_matmult_key",
    "gen_accumulate_rows_key",
    "gen_accumulate_cols_key",
]
