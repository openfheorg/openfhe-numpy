# Running Tests for OpenFHE-NumPy

This document explains how to run and view tests for the OpenFHE-NumPy project.
Our tests use a custom framework built on top of Python's `unittest`

## Prerequisites

* Python 3.8+
* openfhe and openfhe_numpy installed
* No additional test frameworks required - we use Python's built-in `unittest`

## Running Tests

### Run All Tests

To discover and run all tests:


```bash
python3 -m unittest discover -s tests 
```


### Run a Single Test File

```bash
python3 -m unittest tests.test_matrix_addition
```
or
```bash
 python3 tests/test_matrix_addition.py
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

## Customizing Test Runs

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
