import unittest
from pathlib import Path
import sys
from python_tests import core
# import MainTextTestResult

if __name__ == "__main__":
    curr_dir = Path(__file__).parent.resolve()
    parent_dir = curr_dir.parent

    logger = core.MainTextTestResult.setup_logging(verbose=True, clear_files=True)

    suite = unittest.defaultTestLoader.discover(start_dir=str(curr_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=core.MainTextTestResult,
        stream=sys.stdout,
    )
    runner.run(suite)
