import sys
import os
import unittest
import gc
import inspect
import importlib
from datetime import datetime
from .core.test_framework import configure_logging, MainUnittest


def find_test_classes(module):
    """Find all classes in module that inherit from MainUnittest"""
    return [
        obj
        for name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, MainUnittest) and obj != MainUnittest
    ]


def find_modules(suite_or_test):
    """Extract module names from test suite"""
    modules = set()

    def _extract_modules(item):
        if hasattr(item, "_tests"):
            for test in item._tests:
                _extract_modules(test)
        else:
            module_name = item.__class__.__module__
            if module_name != "unittest.suite" and module_name != "unittest.case":
                modules.add(module_name)

    _extract_modules(suite_or_test)
    return modules


if __name__ == "__main__":
    verbose_mode = "-v" in sys.argv
    configure_logging(verbose=verbose_mode)
    overall_start = datetime.now()

    # First discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test_*.py")

    # Extract module names
    modules = find_modules(suite)

    # Track stats
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    error_tests = 0

    # Run each class individually
    for module_name in sorted(modules):
        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Find test classes in this module
            test_classes = find_test_classes(module)

            for test_class in test_classes:
                class_name = test_class.__name__
                result_code = test_class.run_test_summary(class_name, verbose=verbose_mode)

                if result_code == 0:
                    passed_tests += 1
                else:
                    failed_tests += 1

                gc.collect()

        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
            error_tests += 1

    # Print final summary
    duration = (datetime.now() - overall_start).total_seconds()
    total_tests = passed_tests + failed_tests + error_tests

    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"  Total Tests:  {total_tests}")
    print(f"  Passed:       {passed_tests}")
    print(f"  Failed:       {failed_tests}")
    print(f"  Errors:       {error_tests}")
    if total_tests > 0:
        print(f"  Success Rate: {passed_tests / total_tests * 100:.1f}% ")
    print(f"  Total Time:   {duration:.2f}s")
    print("=" * 60)

    # Exit with appropriate code
    os._exit(0 if failed_tests == 0 and error_tests == 0 else 1)
