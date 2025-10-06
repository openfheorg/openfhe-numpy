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
#     this list of conditions and the disclaimer in the documentation
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
"""
Enhanced test result reporting for OpenFHE-NumPy test suite.

This module extends unittest's TextTestResult to provide detailed error reporting
with test parameter information, subtest tracking, and optional debug output.
"""

import os
from typing import Any, Optional, Tuple
from unittest import TextTestResult
import numpy as np


class MainTextTestResult(TextTestResult):
    """
    Enhanced test result handler with detailed error reporting and subtest tracking.
    """

    def __init__(self, *args, debug: bool = False, **kwargs):
        """Initialize the test result handler.

        Args:
            debug: Enable debug output (overridden by DETAILS env var).
            *args, **kwargs: Passed to parent TextTestResult.
        """
        super().__init__(*args, **kwargs)

        # Enable detail (verbose) via environment variable or parameter
        self.detail_mode = os.getenv("DETAILS", "0") == "1" or debug

        # Initialize subtest counters
        self.subtests_total = 0
        self.subtests_pass = 0
        self.subtests_failures = 0
        self.subtests_errors = 0

    # --- Test Event Handlers -------------------------------------------------
    def addSuccess(self, test) -> None:
        """Handle successful test completion."""
        super().addSuccess(test)

    def addFailure(self, test, err: Tuple[type, Exception, Any]) -> None:
        """Handle test failure with optional detailed output."""
        super().addFailure(test, err)
        if self.detail_mode:
            self._print_details("FAIL", test, err)

    def addError(self, test, err: Tuple[type, Exception, Any]) -> None:
        """Handle test error with optional detailed output."""
        super().addError(test, err)
        if self.detail_mode:
            self._print_details("ERROR", test, err)

    def addSubTest(
        self, test, subtest, err: Optional[Tuple[type, Exception, Any]]
    ) -> None:
        """Handle subtest completion and track statistics."""
        super().addSubTest(test, subtest, err)
        self.subtests_total += 1

        if err is None:
            self.subtests_pass += 1
        else:
            # Determine if this is a failure or error
            failure_exc = getattr(test, "failureException", AssertionError)
            try:
                is_failure = issubclass(err[0], failure_exc)
            except Exception:
                is_failure = False

            if is_failure:
                self.subtests_failures += 1
                tag = "SUBTEST FAIL"
            else:
                self.subtests_errors += 1
                tag = "SUBTEST ERROR"

            if self.detail_mode:
                self._print_details(tag, test, err, subtest=subtest)

    # --- Debug Output Methods -------------------------------------------------
    def _print_details(
        self, tag: str, test, err: Tuple[type, Exception, Any], subtest=None
    ) -> None:
        """
        Print detailed information for a failed/errored test or subtest.
        """
        exc_class, exc, _ = err

        # Extract test information
        test_name = getattr(test, "_testMethodName", "<unknown>")
        params = getattr(test, "params", {})
        input_data = getattr(test, "input_data", None)
        expected = getattr(test, "expected", None)
        result = getattr(test, "result", None)

        # Extract subtest information if available
        subtest_params = getattr(subtest, "params", None) if subtest else None
        subtest_desc = (
            getattr(subtest, "_subDescription", None) if subtest else None
        )

        # Print formatted error report
        print("\n" + "=" * 70)
        print(f"[{tag}] Test: {test_name}")
        print(f"Error: {exc_class.__name__}: {exc}")

        if subtest_desc:
            print(f"SubTest: {subtest_desc}")
        if subtest_params:
            print(f"SubTest params: {subtest_params}")

        if params:
            print(f"Test params: {params}")
            # Highlight common important parameters
            for key in ["size", "ringDim", "order", "mode"]:
                if key in params:
                    print(f"  {key}: {params[key]}")

        if input_data is not None:
            print("Input data:")
            for key, value in input_data.items():
                print(f"  {key}: {self._format_value(value)}")

        if expected is not None:
            print(f"Expected: {self._format_value(expected)}")

        if result is not None:
            print(f"Actual: {self._format_value(result)}")

        print("=" * 70 + "\n")

    def _format_value(self, value: Any) -> str:
        """Format a value for display, truncating large or complex outputs."""

        # --- Handle numpy arrays ---
        if isinstance(value, np.ndarray):
            cls_name = value.__class__.__name__

            if value.size <= 8:
                return np.array2string(value, separator=", ")
            else:
                # Show first few elements as preview
                preview = ", ".join(map(str, value.flat[:3]))
                return (
                    f"{cls_name}(shape={value.shape}, preview=[{preview}, ...])"
                )

        # --- Fallback for all other values ---
        str_val = str(value)
        if len(str_val) > 30:
            str_val = str_val[:27] + "..."
        return str_val

    # --- Statistics and Summary Methods ---------------------------------------
    def subtest_counts(self) -> Tuple[int, int, int, int]:
        """
        Get subtest statistics.

        Returns:
            Tuple of (total, passed, failed, errors) for subtests.
        """
        return (
            self.subtests_total,
            self.subtests_pass,
            self.subtests_failures,
            self.subtests_errors,
        )

    def case_counts(
        self, *, count_skips_as_fail: bool = False
    ) -> Tuple[int, int, int, int, int]:
        """
        Get main test case statistics.

        Args:
            count_skips_as_fail: Whether to count skipped tests as failures.

        Returns:
            Tuple of (total, passed, failed, errors, skipped) for main test cases.
        """
        total = self.testsRun
        errors = len(self.errors)
        failed = len(self.failures)
        skipped = len(getattr(self, "skipped", []))

        if count_skips_as_fail:
            failed += skipped
            passed = total - failed - errors
        else:
            passed = total - failed - errors - skipped

        return total, passed, failed, errors, skipped
