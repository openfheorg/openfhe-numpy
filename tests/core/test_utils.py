import os
import contextlib
import numpy as np


# ===============================
# Utility: Suppress stdout
# ===============================
@contextlib.contextmanager
def suppress_stdout(enabled: bool = True):
    if not enabled:
        yield
    else:
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
        ):
            yield


def generate_random_array(rows, cols=None, low=0, high=10, seed=None):
    """Generate random array with given shape and range."""
    rng = np.random.default_rng(seed)
    if cols is None:
        cols = rows
    return rng.uniform(low, high, size=(rows, cols) if cols > 1 else rows)
