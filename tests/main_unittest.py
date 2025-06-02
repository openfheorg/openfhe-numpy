"""
OpenFHE-NumPy Test Framework (Consolidated Version)
"""

# ===============================
# Imports
# ===============================
import os
import csv
import traceback
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from utils.constants import EPSILON

import numpy as np
import unittest

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    UNIFORM_TERNARY,
    FIXEDAUTO,
    FLEXIBLEAUTOEXT,
    FLEXIBLEAUTO,
    FIXEDMANUAL,
    HYBRID,
    BV,
    HEStd_128_classic,
    HEStd_192_classic,
    HEStd_256_classic,
    HEStd_NotSet,
)
import openfhe_numpy as onp

# ===============================
# Globals and Paths
# ===============================
TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV = TESTS_DIR / "ckks_params_auto.csv"
LOG_DIR = TESTS_DIR / "logs"
DEBUG_LOG_DIR = TESTS_DIR / "debug_logs"
_crypto_context_cache = {}

# Optional mapping (you can add more)
SECURITY_LEVEL_MAP = {
    "HEStd_128_classic": HEStd_128_classic,
    "HEStd_192_classic": HEStd_192_classic,
    "HEStd_256_classic": HEStd_256_classic,
    "HEStd_NotSet": HEStd_NotSet,
}

# Mapping enums to strings safely
SECRET_KEY_DIST_MAP = {
    "UNIFORM_TERNARY": UNIFORM_TERNARY,
    # Add other distributions here if needed
}

SCALING_TECHNIQUE_MAP = {
    "FIXEDAUTO": FIXEDAUTO,
    "FLEXIBLEAUTOEXT": FLEXIBLEAUTOEXT,
    "FLEXIBLEAUTO": FLEXIBLEAUTO,
    "FIXEDMANUAL": FIXEDMANUAL,
}

KEY_SWITCH_TECHNIQUE_MAP = {
    "HYBRID": HYBRID,
    "BV": BV,
}


# ===============================
# Utility Functions
# ===============================
@contextlib.contextmanager
def suppress_stdout(suppress=True):
    """Suppress stdout when suppress=True."""
    if not suppress:
        yield
    else:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            yield


def ensure_dirs():
    """Ensure log directories exist."""
    LOG_DIR.mkdir(exist_ok=True)
    DEBUG_LOG_DIR.mkdir(exist_ok=True)


def generate_random_array(rows, cols=None, low=0, high=10, seed=None):
    """Generate random array with given shape and range."""
    rng = np.random.default_rng(seed)
    if cols is None:
        cols = rows
    return rng.uniform(low, high, size=(rows, cols) if cols > 1 else rows)


# ===============================
# Parameter Loading
# ===============================
def load_ckks_params() -> List[Dict[str, Any]]:
    """Load CKKS parameters from CSV file."""
    try:
        with PARAMS_CSV.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            return [
                {
                    key: (
                        int(value)
                        if key
                        not in {
                            "standardDeviation",
                            "secretKeyDist",
                            "ksTech",
                            "scalTech",
                            "securityLevel",
                        }
                        else float(value)
                        if key == "standardDeviation"
                        else value
                    )
                    for key, value in row.items()
                }
                for row in reader
            ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file {PARAMS_CSV} not found")


# ===============================
# Crypto Context Management
# ===============================
def gen_crypto_context(params):
    """Generate crypto context and keys from parameters."""
    p = CCParamsCKKSRNS()
    p.SetRingDim(params["ringDim"])
    p.SetMultiplicativeDepth(params["multiplicativeDepth"])
    p.SetScalingModSize(params["scalingModSize"])
    p.SetBatchSize(params["batchSize"])
    p.SetFirstModSize(params["firstModSize"])
    p.SetStandardDeviation(params["standardDeviation"])
    p.SetSecretKeyDist(SECRET_KEY_DIST_MAP[params["secretKeyDist"]])
    p.SetScalingTechnique(SCALING_TECHNIQUE_MAP[params["scalTech"]])
    p.SetKeySwitchTechnique(KEY_SWITCH_TECHNIQUE_MAP[params["ksTech"]])
    p.SetSecurityLevel(SECURITY_LEVEL_MAP[params["securityLevel"]])
    p.SetNumLargeDigits(params["numLargeDigits"])
    p.SetMaxRelinSkDeg(params["maxRelinSkDeg"])
    p.SetDigitSize(params["digitSize"])

    cc = GenCryptoContext(p)
    for feature in [
        PKESchemeFeature.PKE,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
    ]:
        cc.Enable(feature)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def get_cached_crypto_context(params, use_cache=True):
    """Return cached or newly generated crypto context."""
    if not use_cache:
        return gen_crypto_context(params)

    key = frozenset(params.items())
    if key not in _crypto_context_cache:
        with suppress_stdout():
            _crypto_context_cache[key] = gen_crypto_context(params)
    return _crypto_context_cache[key]


# ===============================
# Logging Utilities
# ===============================
def format_test_data(
    test_name,
    params=None,
    input_data=None,
    expected=None,
    result=None,
    error_info=None,
    status=None,
    error_size=None,
):
    """Format test metadata and arrays for logging."""
    lines = [f"Test Name: {test_name}\n"]
    if params:
        lines.append("CKKS Parameters:")
        lines += [f"  {k} = {v}" for k, v in params.items()]
        lines.append("")
    if status:
        lines.append(f"Status: {status}")
        if error_size is not None:
            lines.append(f"Error size: {error_size}")
        lines.append("")
    if input_data:
        lines.append("Input:")
        for i, x in enumerate(input_data):
            lines.append(f"Input {i}:")
            lines.append(np.array2string(np.array(x), separator=", ") + "\n")
    if expected is not None:
        lines.append("Expected:")
        lines.append(np.array2string(np.array(expected), separator=", ") + "\n")
    if result is not None:
        lines.append("Result:")
        lines.append(np.array2string(np.array(result), separator=", ") + "\n")
    if error_info:
        lines.append("Error:")
        lines.append(str(error_info))
        lines.append("\nTraceback:")
        lines.append(traceback.format_exc())
    return "\n".join(lines)


def log_test_result(name, test_name, input_data, expected, result, error_size, passed):
    """Log test result to a timestamped file."""
    ensure_dirs()
    status = "PASS" if passed else "FAIL"
    content = format_test_data(
        test_name=test_name,
        input_data=input_data,
        expected=expected,
        result=result,
        status=status,
        error_size=error_size,
    )
    content += f"\nTest executed at: {datetime.now().isoformat()}"
    with open(LOG_DIR / f"{name}.log", "a") as f:
        f.write(f"--- {datetime.now().isoformat()} ---\n{content}\n\n")


def log_exception(test_name, params, input_data, expected, result, error):
    """Log detailed exception to debug file."""
    ensure_dirs()
    content = format_test_data(
        test_name=test_name,
        params=params,
        input_data=input_data,
        result=result,
        expected=expected,
        error_info=error,
    )
    with open(DEBUG_LOG_DIR / f"{test_name}_exception.log", "w") as f:
        f.write(content)


# ===============================
# Base Test Class
# ===============================
class MainUnittest(unittest.TestCase):
    """Base class for OpenFHE-NumPy tests with dynamic test generation."""

    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, "_test_cases_generated"):
            if not hasattr(cls, "_generate_test_cases"):
                raise NotImplementedError(f"{cls.__name__} must implement _generate_test_cases()")
            cls._generate_test_cases()
            cls._test_cases_generated = True

    @classmethod
    def run_test_summary(cls, test_name="", debug=False):
        """Run all tests with summary output."""
        print(f"Running {test_name} tests...")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        start_time = datetime.now()
        runner = unittest.TextTestRunner(verbosity=2 if debug else 0)

        with suppress_stdout(not debug):
            result = runner.run(suite)

        duration = datetime.now() - start_time
        print("\n" + "=" * 50)
        print(f"{test_name} Test Summary:")
        print(f"  Total tests:  {result.testsRun}")
        print(f"  Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"  Failed:       {len(result.failures)}")
        print(f"  Errors:       {len(result.errors)}")
        print(f"  Duration:     {duration.total_seconds():.2f} seconds")
        print("=" * 50)

        if result.failures or result.errors:
            print("\nFailed tests:")
            for test, _ in result.failures:
                print(f"  - {test.id()}")
            for test, _ in result.errors:
                print(f"  - {test.id()} (ERROR)")

        return 0 if result.wasSuccessful() else 1

    @classmethod
    def generate_test_case(
        cls, func, name, test_name, params, input_data, expected, eps=EPSILON, debug=False
    ):
        """Generate a test case function."""

        def test(self):
            result = None
            try:
                with suppress_stdout(not debug):
                    result = func(params, input_data)
                    flag, error_size = onp.check_equality_matrix(result, expected, eps)
                log_test_result(
                    name, test_name, input_data, expected, result, error_size, passed=flag
                )
                if not flag:
                    raise AssertionError(f"Matrix mismatch with error {error_size}")
            except Exception as e:
                log_exception(test_name, params, input_data, expected, result, e)
                raise

        return test


# ===============================
# Optional CLI Entry
# ===============================
if __name__ == "__main__":
    print("This is a utility module. Use it to define and run tests.")
