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
