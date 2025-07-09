import sys
import unittest
from tests.core.test_framework import LoggingTestResult

if __name__ == "__main__":
    # sys.exit(unittest.main(module=None))
    sys.exit(
        unittest.main(
            testRunner=lambda *args, **kwargs: unittest.TextTestRunner(
                *args, resultclass=LoggingTestResult, **kwargs
            )
        )
    )
