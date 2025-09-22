from unittest import TextTestRunner
from tests.python_tests.core.result import MainTextTestResult


class MainTextTestRunner(TextTestRunner):
    """Custom test runner that always uses MainTextTestResult"""

    resultclass = MainTextTestResult
