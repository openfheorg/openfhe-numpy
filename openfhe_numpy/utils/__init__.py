"""Utility functions for OpenFHE-NumPy.

This module provides math utilities, validation functions, and logging tools
for the OpenFHE-NumPy package.
"""

from .errors import ONPError
from .matlib import (
    is_power_of_two,
    next_power_of_two,
    check_equality,
)

from .constants import EPSILON, EPSILON_HIGH, DataType, UnpackType

__all__ = [
    "is_power_of_two",
    "next_power_of_two",
    "check_equality",
    "ONPError",
    "EPSILON",
    "EPSILON_HIGH",
    "DataType",
    "UnpackType",
]
