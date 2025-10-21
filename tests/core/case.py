# ==============================================================================
#  BSD 2-Clause License
#
#  Copyright (c) 2014-2025, NJIT, Duality Technologies Inc. and other contributors
#
#  All rights reserved.
#
#  Author TPOC: contact@openfhe.org
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
import gc
import os
import sys
import time
import unittest
from typing import Any, Iterable, Union
import functools
import numpy as np

from .config import EPSILON
from .result import MainTextTestResult


class MainUnittest(unittest.TestCase):
    """Base class for OpenFHE-NumPy tests"""

    @staticmethod
    def _dispose(obj: Any) -> None:
        """Try to dispose resources by calling common cleanup methods."""
        if obj is None:
            return

        for meth in (
            "close",
            "release",
            "free",
            "clear",
            "dispose",
            "shutdown",
        ):
            m = getattr(obj, meth, None)
            if callable(m):
                try:
                    m()
                except Exception:
                    pass

    # ---- Test case recording for debug ---------------------------------------
    def _record_case(
        self,
        *,
        params: Any = None,
        input_data: Any = None,
        expected: Any = None,
        result: Any = None,
    ) -> None:
        """Record test case data for debugging and reporting."""
        self.params = params
        self.input_data = input_data
        self.expected = expected
        self.result = result

    # ---- Assertion helpers --------------------------------------------------
    def assertArrayClose(
        self,
        actual: Union[Iterable, np.ndarray],
        expected: Union[Iterable, np.ndarray],
        *,
        rtol: float = EPSILON,
        atol: float = 0.0,
        msg: str = "",
    ) -> None:
        """Assert that two arrays are close within the specified tolerance."""
        a = np.asarray(actual)
        e = np.asarray(expected)
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol, err_msg=msg)

    def tearDown(self) -> None:
        """Clean up resources after each test."""
        gc.collect()
        super().tearDown()

    # ---- Class-level test execution -----------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize class-level setup and start timing."""
        super().setUpClass()

    # ---- Summary helper -----------------------------------------------------
    @classmethod
    def run_test_summary(cls, name: str = "") -> int:
        """Run all tests in this class with optional detailed output.

        Args:
            name: Optional custom name for the test class.

        Returns:
            0 on success, 1 on failure.
        """
        details_enabled = os.getenv("DETAILS", "0") == "1"
        failfast_enabled = os.getenv("FAILFAST", "0") == "1"

        if details_enabled:
            print(f"Running {name or cls.__name__} tests with details...")

        start = time.perf_counter()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)

        # Configure the test runner with custom result handler
        result_factory = functools.partial(
            MainTextTestResult, debug=details_enabled
        )

        runner = unittest.TextTestRunner(
            resultclass=result_factory,
            verbosity=(2 if details_enabled else 1),
            stream=sys.stdout,
            buffer=not details_enabled,
            failfast=failfast_enabled,
        )

        result: MainTextTestResult = runner.run(suite)
        duration = time.perf_counter() - start

        if result.testsRun == 0:
            print("NO TESTS RAN!")
            return 1

        # Collect and display statistics
        total, passed, failed, errors, skipped = result.case_counts()
        st_total, st_ok, st_fail, st_err = result.subtest_counts()

        if details_enabled:
            print("-" * 60)
            print(f"{name or cls.__name__} Summary:")
            print(f"  Total:    {total}")
            print(f"  Passed:   {passed}")
            print(f"  Failed:   {failed}")
            print(f"  Errors:   {errors}")
            print(f"  Skipped:  {skipped}")
            print(
                f"  Subtests: {st_total}  (ok={st_ok}, fail={st_fail}, error={st_err})"
            )
            print(f"  Duration: {duration:.2f}s")

            try:
                import psutil

                mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                print(f"  Memory:   {mem_mb:.1f} MB")
            except Exception:
                pass

            print("-" * 60)

        return 0 if result.wasSuccessful() else 1


# ---- Find test classes helper ------------------------------------------------
def find_test_classes(module):
    """Find all classes in a module that inherit from MainUnittest."""
    import inspect

    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, MainUnittest) and obj is not MainUnittest
    ]
