# ==================================================================================
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
# ==================================================================================
import gc
import time
import unittest
from typing import Any, Union, Iterable
import numpy as np

from .config import EPSILON
from .result import MainTextTestResult


def _dispose(obj: Any) -> None:
    """Try to disposal resources"""
    if obj is None:
        return
    for meth in ("close", "release", "free", "clear", "dispose", "shutdown"):
        m = getattr(obj, meth, None)
        if callable(m):
            try:
                m()
            except Exception:
                pass


class MainUnittest(unittest.TestCase):
    _class_start_time: float | None = None

    # ---- helpers ----------------------------------------------------
    def _record_case(
        self,
        *,
        params: Any = None,
        input_data: Any = None,
        expected: Any = None,
        result: Any = None,
    ) -> None:
        self.params = params
        self.input_data = input_data
        self.expected = expected
        self.result = result

    def assertArrayClose(
        self,
        params: Any,
        input_data: Union[Iterable, np.ndarray],
        actual: Union[Iterable, np.ndarray],
        expected: Union[Iterable, np.ndarray],
        *,
        rtol: float = EPSILON,
        atol: float = 0.0,
        msg: str = "",
    ) -> None:
        """
        Assert that two array are close.

        """
        # Record context for the result logger (only emitted on fail/error)
        self._record_case(params=params, input_data=input_data, expected=expected, result=actual)
        a = np.asarray(actual)
        e = np.asarray(expected)
        np.testing.assert_allclose(a, e, rtol=rtol, atol=atol, err_msg=msg)

    def tearDown(self) -> None:
        for name in (
            "cc",
            "keys",
            "params",
            "context",
            "key",
            "tensor",
            "ctv_a",
            "ctv_b",
            "ctv_res",
        ):
            if hasattr(self, name):
                try:
                    _dispose(getattr(self, name))
                finally:
                    setattr(self, name, None)
        gc.collect()
        super().tearDown()

    # ---- run helpers ------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_start_time = time.perf_counter()

    @classmethod
    def run_test_summary(cls, name: str = "", verbose: bool = False) -> int:
        """
        Run this TestCase with our SimpleResult and print a brief summary.
        """
        print(f"Running {name or cls.__name__} tests...")
        start = time.perf_counter()

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(
            resultclass=MainTextTestResult,
            verbosity=(2 if verbose else 1),
        )
        result: MainTextTestResult = runner.run(suite)

        duration = time.perf_counter() - start
        total = result.testsRun
        fails = len(result.failures)
        errs = len(result.errors)
        passed = total - fails - errs

        sub_total = getattr(result, "subtests_total", 0)
        sub_ok = getattr(result, "subtests_pass", 0)
        sub_fail = getattr(result, "subtests_failures", 0)
        sub_err = getattr(result, "subtests_errors", 0)

        print("=" * 60)
        print(f"{name or cls.__name__} Summary:")
        print(f"  Total:    {total}")
        print(f"  Passed:   {passed}")
        print(f"  Failed:   {fails}")
        print(f"  Errors:   {errs}")
        if sub_total:
            print(f"  Subtests: {sub_total}  (ok={sub_ok}, fail={sub_fail}, error={sub_err})")
        print(f"  Duration: {duration:.2f}s")

        try:
            import psutil

            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"  Memory:   {mem_mb:.1f} MB")
        except Exception:
            pass
        print("=" * 60)

        return 0 if result.wasSuccessful() else 1

    @staticmethod
    def find_test_classes(module):
        """Find all classes in a module that inherit from MainUnittest."""
        import inspect

        return [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, MainUnittest) and obj is not MainUnittest
        ]
