# Running Tests for OpenFHE-NumPy

This document explains how to run and view tests for the OpenFHE-NumPy project.
Our tests use a custom framework built on top of Python's `unittest`

## Prerequisites

* Python 3.8+
* openfhe and openfhe_numpy installed
* No additional test frameworks required - we use Python's built-in `unittest`

## Running Tests

To run tests, you should run them from inside the tests directory so that Python does not mistake the local openfhe_numpy folder for the installed package.

### Run All Tests

```bash
cd tests
python3 -m unittest discover -v
```


### Run a Single Test File

```bash
cd tests
python3 -m unittest python.test_matrix_sum
```
or
```bash
 python3 tests/python/test_matrix_sum.py
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

## Unittest Quick Guide

* **Verbose Output**: Add `-v` for detailed information:
  ```bash
  python3 -m unittest discover -s tests -v
  ```

* **Stop on First Failure**: Use `--failfast`:
  ```bash
  python3 -m unittest discover -s tests --failfast
  ```

* **Pattern Filtering**: Run tests matching a pattern:
  ```bash
  python3 -m unittest discover -s tests -p "test_matrix_*.py"
  ```
