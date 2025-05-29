"""
OpenFHE-NumPy: A NumPy-inspired framework for homomorphic encryption operations.

This module provides a NumPy-style API for homomorphic encryption using OpenFHE.

Example:
    ```python
    from openfhe_numpy import array, add
    ct = array([1, 2, 3])
    result = add(ct, ct)
    ```
"""

import importlib
import warnings
from importlib.metadata import version, PackageNotFoundError
from typing import Any, List, Dict

# Version information
try:
    # First try to import from _version.py (development mode)
    from ._version import __version__
except ImportError:
    try:
        # Next try version.py (CMake-generated version)
        from .version import __version__
    except ImportError:
        try:
            # Last, try package metadata (pip installed)
            from importlib.metadata import version, PackageNotFoundError

            __version__ = version(__name__)
        except (ImportError, PackageNotFoundError):
            __version__ = "0.0.1"  # Fallback version
            warnings.warn(
                f"Using fallback version {__version__}; metadata for package '{__name__}' not found.",
                UserWarning,
                stacklevel=2,
            )

# Mapping of public names to their defining submodule
_export_map: Dict[str, str] = {
    # Tensor classes
    "BaseTensor": "tensor",
    "FHETensor": "tensor",
    "PTArray": "tensor",
    "CTArray": "tensor",
    "BlockFHETensor": "tensor",
    "BlockCTArray": "tensor",
    "array": "tensor",
    # Matrix operations
    "add": "operations.matrix_api",
    "multiply": "operations.matrix_api",
    "dot": "operations.matrix_api",
    "matmul": "operations.matrix_api",
    "transpose": "operations.matrix_api",
    "power": "operations.matrix_api",
    "cumsum": "operations.matrix_api",
    "cumreduce": "operations.matrix_api",
    # Crypto context utilities
    "sum_row_keys": "operations.crypto_context",
    "sum_col_keys": "operations.crypto_context",
    "gen_rotation_keys": "operations.crypto_context",
    "gen_lintrans_keys": "operations.crypto_context",
    "gen_transpose_keys": "operations.crypto_context",
    "gen_square_matmult_key": "operations.crypto_context",
    "gen_accumulate_rows_key": "operations.crypto_context",
    "gen_accumulate_cols_key": "operations.crypto_context",
    # OpenFHE core types
    "LinTransType": "openfhe_numpy",
    "MatVecEncoding": "openfhe_numpy",
    "MulDepthAccumulation": "openfhe_numpy",
    "EvalLinTransKeyGen": "openfhe_numpy",
    "EvalSquareMatMultRotateKeyGen": "openfhe_numpy",
    "EvalSumCumRowsKeyGen": "openfhe_numpy",
    "EvalSumCumColsKeyGen": "openfhe_numpy",
    "EvalMultMatVec": "openfhe_numpy",
    "EvalMatMulSquare": "openfhe_numpy",
    "EvalTranspose": "openfhe_numpy",
    "EvalSumCumRows": "openfhe_numpy",
    "EvalSumCumCols": "openfhe_numpy",
    # Utility functions
    "is_power_of_two": "utils.utils",
    "next_power_of_two": "utils.utils",
    "check_equality_matrix": "utils.utils",
    "pack_vec_row_wise": "utils.utils",
    # Constants and types
    "MatrixOrder": "config",
    "DataType": "config",
    "EPSILON": "config",
    "EPSILON_HIGH": "config",
    "FormatType": "config",
    # Logging
    "ONP_WARNING": "utils.log",
    "ONP_DEBUG": "utils.log",
    "ONP_ERROR": "utils.log",
    "ONPNotImplementedError": "utils.log",
}

# Public API names driven from _export_map
__all__: List[str] = sorted(_export_map.keys())

# Sanity check (only in debug mode) to catch mismatches
if __debug__:
    assert set(__all__) == set(_export_map), (
        f"API mismatch between __all__ and _export_map: {set(__all__) ^ set(_export_map)}"
    )

# Module cache to avoid repeated imports
# Thread-safe under CPython's import lock
_module_cache: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """
    Lazy-load public symbols from submodules on first access.
    Raises a detailed AttributeError listing valid exports.
    """
    if name not in _export_map:
        raise AttributeError(
            f"{__name__!r} has no attribute {name!r}. Valid exports are: {', '.join(__all__)}"
        )
    module_path = _export_map[name]
    if module_path not in _module_cache:
        _module_cache[module_path] = importlib.import_module(f"{__name__}.{module_path}")
    value = getattr(_module_cache[module_path], name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """
    Include standard dunder names and public API names sorted alphabetically.
    """
    return sorted(__all__ + [d for d in globals() if d.startswith("__")])
