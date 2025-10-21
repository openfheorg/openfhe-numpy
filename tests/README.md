# Running Tests for OpenFHE-NumPy

This document explains how to run and view tests for the OpenFHE-NumPy project.
Our tests use a custom framework built on top of Python's `unittest`
with additional customizations designed to improve debugging and
provide summary results.

## Prerequisites

* Python 3.10+
* openfhe and openfhe_numpy installed
* No additional test frameworks required - we use Python's built-in `unittest`

## Quick start

* To run tests, you should run them from inside the tests directory so that Python does not mistake the local `openfhe_numpy` folder for the installed package.
* The runner `run_tests.py` locates test files under `tests/cases` using the given pattern.

```bash
cd tests

# Run all tests
python3 run_tests.py

# Run tests in a subfolder under cases/
python3 run_tests.py matrix_ops

# Run a single tests
python3 run_tests.py matrix_ops/test_matrix_sum.py

# List all possible tests
python3 run_tests.py --list

# Run with verbose
python3 run_tests.py -v

# Stop on first failure or timeout:
python3 run_tests.py -x

# Change timeout (e.g., 60 seconds per class):
python3 run_tests.py -t 60

# Run tests with pattern
python3 run_tests.py -p "test_*matrix*.py"
```

## Commandline Guide
```bash
python3 run_tests.py [targets...] [-p PATTERN] [-t TIMEOUT] [-x] [-l] [-v]
```

### Arguments
* `targets`: Folders/files under ./cases to run. If omitted, runs all matching tests.
* `-p, --pattern`: Glob for test files under cases/ (default: test_*.py)
* `-t, --timeout`: Timeout per test class in seconds (default: 1800)
* `-x, --exitfirst`: Exit on the first failure or timeout
* `-l, --list`: List discovered test files and exit
* `-v, --details`: Verbose mode (prints debug information)

## Guidelines for Writing Tests

Here is some remark when designing tests

- **Import the core module** which contains the necessary test utilities:
  ```python
  from core import *
  ```
- **Use `MainUnittest` as base class** instead of `unittest.TestCase`:
  ```python
  class TestMyFeature(MainUnittest):
      def test_something(self):
          # Your test code here
          pass
  ```

- **Use the `run_test_summary` method** to execute the test suite.
  This will output a concise summary for each test class.

- **Array comparison**: The test framework compares two arrays for equality using `np.testing.assert_allclose`.
  This function checks whether two arrays are element-wise equal within a certain tolerance.
  The tolerance values are defined in `core/config.py` by default.


## Common Problems

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'openfhe_numpy.openfhe_numpy'`

Make sure you run the script from inside the "test" directory
```bash
  cd tests
  python3 run_tests.py
```

**Problem**: Missing openfhe library `ImportError: libOPENFHEbinfhe.so.1:`

Include openfhe in your test file.
```python
  from openfhe import *
```

### Target Not Found

**Problem**: `Error: Target 'cases/target_folder' not found`

When passing targets, do not prefix them with "cases/" as the script already locates tests under `cases` directory

```bash
# Correct

# run tests in matrix_ops
python3 run_tests.py target_folder
python3 run_tests.py matrix_ops

# run a single tests in matrix_ops
python3 run_tests.py matrix_ops/test_matrix_mean.py

# Incorrect
python3 run_tests.py cases/target_folder
```
