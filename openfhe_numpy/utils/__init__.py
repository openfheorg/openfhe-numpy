"""Utility functions for OpenFHE-NumPy.

This module provides math utilities, validation functions, and logging tools
for the OpenFHE-NumPy package.
"""

from .matlib import (
    is_power_of_two,
    next_power_of_two,
    check_equality_matrix,
    check_single_equality,
    check_equality_vector,
)
from .errors import ONPError


__all__ = [
    "is_power_of_two",
    "next_power_of_two",
    "check_equality_matrix",
    "check_single_equality",
    "check_equality_vector",
    "ONPError",
]
