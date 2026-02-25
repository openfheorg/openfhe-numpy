import unittest


class QuietRunner(unittest.TextTestRunner):
    """Run tests without printing the final summary lines."""

    def run(self, test):
        result = self._makeResult()
        test(result)
        return result
