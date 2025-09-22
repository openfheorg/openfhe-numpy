# Running Tests for OpenFHE-NumPy

This document explains how to run and view tests for the OpenFHE-NumPy project.
Our tests use a custom framework built on top of Python's `unittest`
with additional customizations designed to improve debugging and
provide summary results.

## Prerequisites

* Python 3.10+
* openfhe and openfhe_numpy installed
* No additional test frameworks required - we use Python's built-in `unittest`

## Running Tests

To run tests, you should run them from inside the tests directory so that Python does not mistake the local openfhe_numpy folder for the installed package.

### Run All Tests

```bash
cd tests
python3 -m python_tests
```


### Run a Single Test File

```bash
cd tests/python_tests
python3 -m test_matrix_sum
```
or
```bash
python3 tests/python_tests/test_matrix_sum.py
```


## Viewing Test Results

Test results are written to log files and displayed on the console:

* **`logs/results.log`**: Contains PASS/FAIL records for all tests
* **`errors/errors.log`**: Contains detailed information for failed tests including:
  - Test parameters used
  - Input data
  - Expected output
  - Actual result
  - Error details

## Guidelines for Writing Tests
Here is some remark when designing tests :
- First, import the core module which contains the necessary test utilities:
  ```python
  from core import *
  ```
- Use the function **`run_test_summary`** to execute the test suite.
This will output a concise summary for each test class

- The test framework compares two arrays for equality using `np.testing.assert_allclose`.
  This function checks whether two arrays (or array-like objects) are element-wise equal within a certain tolerance.
  The tolerance values are defined in `core/config.py` by default.
