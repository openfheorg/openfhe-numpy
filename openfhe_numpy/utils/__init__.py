"""Utility functions for OpenFHE-NumPy."""

# Import utility functions from modules
from .utils import (
    check_equality_matrix,
    pack_vec_row_wise,
)


# Import matlib functions
from .matlib import (
    is_power_of_two,
    next_power_of_two,
)

"""Logging utilities for OpenFHE-NumPy."""
from .log import (
    get_logger,
    ONPError,
    InvalidAxisError,
    ONPNotImplementedError,
)


# Define public API
__all__ = [
    # Math utilities
    "is_power_of_two",
    "next_power_of_two",
    "pack_vec_row_wise",
    # Validation utilities
    "check_equality_matrix",
    "next_power_of_two",
    # Enums
    "get_logger",
    "ONPError",
    "InvalidAxisError",
    "ONPNotImplementedError",
]
