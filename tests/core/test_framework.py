"""
OpenFHE-NumPy Test Framework
"""

import sys, logging, pprint
from pathlib import Path
from typing import Any, Dict, Callable
from datetime import datetime

# third-party
import unittest
import openfhe_numpy as onp

# local
from .test_utils import suppress_stdout
# from .test_crypto_context import get_cached_crypto_context

# ===============================
# Paths and Directories
# ===============================
TESTS_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = TESTS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
ERROR_DIR = PROJECT_ROOT / "errors"

for f in (LOG_DIR / "results.log", ERROR_DIR / "errors.log"):
    try:
        f.unlink()
    except FileNotFoundError:
        pass

LOG_DIR.mkdir(exist_ok=True, parents=True)
ERROR_DIR.mkdir(exist_ok=True, parents=True)
# ===============================
# Logging Setup
# ===============================
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# File handlers
for level, path in (
    (logging.INFO, LOG_DIR / "results.log"),
    (logging.ERROR, ERROR_DIR / "errors.log"),
):
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(ch)


# ===============================
# Logging Helpers
# ===============================
def configure_logging():
    """Ensure log directories exist."""
    for d in (LOG_DIR, ERROR_DIR):
        d.mkdir(exist_ok=True, parents=True)


def log_test_success(test_id: str):
    logger.info(f"PASS: {test_id}")


def log_test_failure(
    test_id: str,
    err_msg: str,
    params: Dict[str, Any],
    input_data: Any,
    expected: Any,
    result: Any,
):
    """Log assertion failures with full context in a multi-line, indented format."""
    msg = (
        f"FAIL: {test_id}\n"
        f"    Error: {err_msg}\n"
        f"    Params: {pprint.pformat(params)}\n"
        f"    Input: {pprint.pformat(input_data)}\n"
        f"    Expected: {pprint.pformat(expected)}\n"
        f"    Result: {pprint.pformat(result)}"
    )
    logger.error(msg)


def log_test_error(
    test_id: str,
    err: Exception,
    params: Dict[str, Any],
    input_data: Any,
    expected: Any,
    result: Any,
):
    """Log unexpected exceptions with full context in a multi-line, indented format."""
    msg = (
        f"ERROR: {test_id}\n"
        f"    Exception: {err}\n"
        f"    Params: {pprint.pformat(params)}\n"
        f"    Input: {pprint.pformat(input_data)}\n"
        f"    Expected: {pprint.pformat(expected)}\n"
        f"    Result: {pprint.pformat(result)}"
    )
    logger.error(msg)


# ===============================
# Custom TestResult
# ===============================
class LoggingTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        log_test_success(test.id())

    # Failure logging handled in test_method
    def addFailure(self, test, err):
        super().addFailure(test, err)

    # Error logging handled in test_method
    def addError(self, test, err):
        super().addError(test, err)


# ===============================
# Base Test Class
# ===============================
class MainUnittest(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # right after the class is created, generate its tests
        if hasattr(cls, "_generate_test_cases"):
            cls._generate_test_cases()
            cls._tests_generated = True

    @classmethod
    def setUpClass(cls):
        # print(f"Setting up {cls.__name__}")
        if not getattr(cls, "_tests_generated", False) and hasattr(
            cls, "_generate_test_cases"
        ):
            cls._generate_test_cases()
            cls._tests_generated = True

    @classmethod
    def run_test_summary(cls, name: str = "", debug: bool = False) -> int:
        print(f"Running {name} tests...")
        start = datetime.now()
        runner = unittest.TextTestRunner(
            verbosity=2 if debug else 1,
            resultclass=LoggingTestResult,
        )
        result = runner.run(
            unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        )
        duration = (datetime.now() - start).total_seconds()
        total = result.testsRun
        fails = len(result.failures)
        errs = len(result.errors)
        passed = total - fails - errs
        print("=" * 60)
        print(f"{name} Summary:")
        print(f"  Total:   {total}")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {fails}")
        print(f"  Errors:  {errs}")
        print(f"  Duration: {duration:.2f}s")
        return 0 if result.wasSuccessful() else 1

    @classmethod
    def generate_test_case(
        cls,
        func: Callable[[Any, Any, Any], Any],
        params: Dict[str, Any],
        input_data: Any,
        expected: Any,
        test_name: str = None,
        compare_fn: Callable = None,
        tolerance: float = onp.EPSILON,
        debug: bool = False,
    ) -> None:
        compare_function = compare_fn or onp.check_equality
        orig_params = params

        def test_method(self):
            result = None
            # change later
            # new_params = get_cached_crypto_context(orig_params)
            try:
                with suppress_stdout(not debug):
                    result = func(orig_params, input_data)
                passed, err = compare_function(result, expected, tolerance)
                if not passed:
                    raise AssertionError(f"{test_name} mismatch: error={err}")
            except AssertionError as ae:
                log_test_failure(
                    self.id(),
                    str(ae),
                    orig_params,
                    input_data,
                    expected,
                    result,
                )
                raise
            except Exception as ex:
                log_test_error(
                    self.id(), ex, params, input_data, expected, result
                )
                raise

        setattr(cls, f"test_{test_name}", test_method)
        return test_method  # Return the created method for chaining


# ===============================
# CLI Entry
# ===============================
configure_logging()
if __name__ == "__main__":
    sys.exit(MainUnittest.run_test_summary("All", debug="-v" in sys.argv))
