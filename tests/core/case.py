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

    @classmethod
    def tearDownClass(cls) -> None:
        """Report RSS to stderr for the test runner to parse.
        Uses /proc/self/status (Linux). Silently skipped on other platforms.
        No external dependencies required.
        """
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_bytes = int(line.split()[1]) * 1024  # kB → bytes
                        print(f"__RSS__:{rss_bytes}", file=sys.stderr, flush=True)
                        break
        except Exception:
            pass
        super().tearDownClass()

    # ---- Class-level test execution -----------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        """Initialize class-level setup and start timing."""
        super().setUpClass()
