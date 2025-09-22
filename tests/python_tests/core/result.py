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
import sys
import logging
import pprint
from typing import Any
from unittest import TextTestResult
import os
import shutil

from .config import *

logger = logging.getLogger(LOGGER_NAME)


class MainTextTestResult(TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # subtest counters
        self.subtests_total = 0
        self.subtests_pass = 0
        self.subtests_failures = 0
        self.subtests_errors = 0

    # ---- core hooks -------------------------------------------------
    def addSuccess(self, test):
        super().addSuccess(test)  # intentionally no logging

    def addFailure(self, test, err):
        super().addFailure(test, err)
        exc_class, exc, _ = err
        self._log_case(
            tag="FAIL",
            test_id=test.id(),
            err_msg=f"{exc_class.__name__}: {exc}",
            src=test,
        )

    def addError(self, test, err):
        super().addError(test, err)
        exc_class, exc, _ = err
        self._log_case(
            tag="ERROR",
            test_id=test.id(),
            err_msg=f"{exc_class.__name__}: {exc}",
            src=test,
        )

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        self.subtests_total += 1

        if err is None:
            self.subtests_pass += 1
            return  # no logging for passing subtests

        exc_class, exc, _ = err
        if issubclass(exc_class, self.failureException):
            self.subtests_failures += 1
            tag = "SUBTEST FAIL"
        else:
            self.subtests_errors += 1
            tag = "SUBTEST ERROR"

        # prefer attributes on the subtest, then fall back to the parent test
        self._log_case(
            tag=tag,
            test_id=test.id(),
            err_msg=f"{exc_class.__name__}: {exc}",
            src=subtest if subtest is not None else test,
        )

    def stopTestRun(self):
        print("stopTestRun reached!", file=sys.stderr)

        logger.info(
            "SUMMARY: tests=%d pass=%d fail=%d error=%d | subtests=%d pass=%d fail=%d error=%d",
            self.testsRun,
            self.testsRun
            - len(self.failures)
            - len(self.errors)
            - len(self.skipped)
            - len(self.expectedFailures)
            - len(self.unexpectedSuccesses),
            len(self.failures),
            len(self.errors),
            self.subtests_total,
            self.subtests_pass,
            self.subtests_failures,
            self.subtests_errors,
        )
        return super().stopTestRun()

    # ---- tiny helper ------------------------------------------------
    def _log_case(self, *, tag: str, test_id: str, err_msg: str, src: Any):
        """
        Build a compact message that only includes info that exist and aren't None.
        Looks for: params, input_data, expected, result
        """
        info = {
            "Params": getattr(src, "params", None),
            "Input": getattr(src, "input_data", None),
            "Expected": getattr(src, "expected", None),
            "Result": getattr(src, "result", None),
        }

        lines = [f"{tag}: {test_id}", f"    Error: {err_msg}"]
        for label, val in info.items():
            if val is not None:
                lines.append(f"    {label}: {pprint.pformat(val)}")

        logger.error("\n".join(lines))

    @staticmethod
    def setup_logging(verbose: bool = True, clear_files: bool = True) -> logging.Logger:
        """
        Idempotent logging setup for console + 2 file handlers.
        - console level honors `verbose` (INFO if True else WARNING)
        - clears old handlers to avoid duplication
        - optionally truncates existing log files
        """

        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        logger.handlers.clear()  # clear existing handlers

        if clear_files:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            if os.path.exists(ERROR_DIR):
                shutil.rmtree(ERROR_DIR)

        LOG_DIR.mkdir(exist_ok=True, parents=True)
        ERROR_DIR.mkdir(exist_ok=True, parents=True)

        fh_results = logging.FileHandler(LOG_PATH)
        fh_results.setLevel(logging.INFO)
        fh_results.setFormatter(FORMATTER)

        fh_errors = logging.FileHandler(ERROR_PATH)
        fh_errors.setLevel(logging.ERROR)
        fh_errors.setFormatter(FORMATTER)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO if verbose else logging.WARNING)
        ch.setFormatter(FORMATTER)

        logger.addHandler(fh_results)
        logger.addHandler(fh_errors)
        logger.addHandler(ch)

        return logger
