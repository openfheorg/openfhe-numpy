"""
Test runner for OpenFHE-NumPy test suite.

This script discovers and runs unittest-based test files with support for:
- Individual test class execution for better isolation
- Timeout handling for long-running FHE operations
- Detailed output and timing information
- Early exit on first failure
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from core import find_test_classes

try:
    import resource  # not available on Windows
except Exception:
    resource = None

# --- Configuration ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
CASES_DIR = PROJECT_ROOT / "cases"
DEFAULT_PATTERN = "test_*.py"
DEFAULT_TIMEOUT = 1800  # seconds (30 minutes)


# --- Command Line Interface ---------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run unittest-style test files for OpenFHE-NumPy.",
        epilog="Examples:\n"
        "  %(prog)s                    # Run all tests\n"
        "  %(prog)s matrix_ops         # Run tests in matrix_ops folder\n"
        "  %(prog)s --details -x  # Run with details, exit on first failure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "targets",
        nargs="*",
        help="Folders/files under ./cases to test (empty = all tests)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for test files (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per test in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "-x",
        "--exitfirst",
        action="store_true",
        help="Stop on first failure or timeout",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available test files and exit",
    )
    parser.add_argument(
        "-v",
        "--details",
        action="store_true",
        help="Enable verbose output with test summaries and timing",
    )

    return parser


# --- Resource Debug -----------------------------------------------------------
def _fmt_bytes(n_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    v = float(n_bytes)
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.2f} {units[i]}"


# --- Test Discovery -----------------------------------------------------------
def find_tests(
    target: Optional[str] = None, pattern: str = DEFAULT_PATTERN
) -> List[Path]:
    """
    Find test files matching the given pattern.

    Args:
        target: Target directory or file path (None for all tests).
        pattern: Glob pattern to match test files.

    Returns:
        List of Path objects for matching test files.

    Raises:
        SystemExit: If target is specified but not found.
    """
    if target is None:
        return sorted(CASES_DIR.rglob(pattern))

    target_path = CASES_DIR / target
    if target_path.is_dir():
        return sorted(target_path.glob(pattern))
    if target_path.is_file():
        return [target_path]

    # Try as absolute path
    alt_path = PROJECT_ROOT / target
    if alt_path.is_file():
        return [alt_path]

    print(
        f"Error: Target '{target}' not found in {CASES_DIR} or {PROJECT_ROOT}"
    )
    sys.exit(1)


# --- Get module helper --------------------------------------------------------
def module_from_path(pyfile: Path) -> str:
    """Convert a Python file path to a module import path."""
    relative_path = pyfile.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(relative_path.parts)


# --- Test Execution -----------------------------------------------------------
def _run_command(
    cmd: List[str], timeout: int, env: Dict[str, str]
) -> Tuple[int, float]:
    """
    Execute a command with timeout and return results.

    Args:
        cmd: Command and arguments to execute.
        timeout: Timeout in seconds.
        env: Environment variables for the subprocess.

    Returns:
        Tuple of (exit_code, duration_seconds).
    """
    start_time = time.time()

    try:
        proc = subprocess.run(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration = time.time() - start_time
        return proc.returncode, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"Command timed out after {timeout}s")
        return 2, duration  # Exit code 2 indicates timeout

    except Exception as e:
        duration = time.time() - start_time
        print(f"Command failed with error: {e}")
        return 1, duration  # Exit code 1 indicates error


def run_test_file(
    pyfile: Path,
    *,
    timeout: int,
    current: int,
    total: int,
    details: bool,
    exit_first: bool,
) -> List[Tuple[str, int, float]]:
    print(f"\n\n=== Running {pyfile.name} ({current}/{total}) ===")
    module_name = module_from_path(pyfile)

    # Discover classes using your project's helper
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"[discover] Failed to import '{module_name}': {e}")
        return []

    class_names = sorted([cls.__name__ for cls in find_test_classes(module)])
    if details:
        print(
            f"Found {len(class_names)} class(es) in {module_name}: {', '.join(class_names)}"
        )

    if not class_names:
        print(f"No test classes found in file (module={module_name})")
        return []

    # Env for child processes
    env = os.environ.copy()
    env["DETAILS"] = "1" if details else "0"
    env["FAILFAST"] = "1" if exit_first else "0"

    results = []

    def run_one_test(label: str, cmd: List[str]) -> int:
        code, duration = _run_command(cmd, timeout, env)
        results.append((label, code, duration))
        return code

    # Run each test class in an isolated subprocess
    for idx, class_name in enumerate(class_names, 1):
        print(f"\n{class_name} ({idx}/{len(class_names)})")
        label = f"{pyfile.name}:{class_name}"

        # IMPORTANT: use module_name here (not 'module')
        one_liner = (
            "import importlib, sys; "
            f"m = importlib.import_module('{module_name}'); "
            f"c = getattr(m, '{class_name}', None); "
            "sys.exit(c.run_test_summary() if c else 1)"
        )

        cmd = [sys.executable, "-c", one_liner]
        exit_code = run_one_test(label, cmd)
        if exit_first and exit_code != 0:
            break

    return results


# --- Main Function ------------------------------------------------------------
def main() -> None:
    """Main entry point for the test runner."""
    args = build_parser().parse_args()

    t_start = time.perf_counter()

    # Discover all test files to run
    all_tests = []
    targets = args.targets or [None]

    for target in targets:
        all_tests.extend(find_tests(target, args.pattern))

    # Remove duplicates and sort
    all_tests = sorted(set(all_tests))

    # Handle list mode
    if args.list:
        print(f"Available test files ({len(all_tests)}):")
        for test_file in all_tests:
            try:
                relative_path = test_file.relative_to(CASES_DIR)
                print(f"  {relative_path}")
            except ValueError:
                print(f"  {test_file}")
        sys.exit(0)

    # Check if any tests were found
    print(f"Found {len(all_tests)} test files")
    if not all_tests:
        print("No test files found matching criteria")
        sys.exit(0)

    # Run all tests
    failed_count = 0
    timeout_count = 0
    all_results = []

    for i, test_file in enumerate(all_tests, 1):
        file_results = run_test_file(
            test_file,
            timeout=args.timeout,
            current=i,
            total=len(all_tests),
            details=args.details,
            exit_first=args.exitfirst,
        )

        # Process results from this file
        for test_name, exit_code, duration in file_results:
            all_results.append((test_name, exit_code, duration))

            if exit_code == 0:
                print(f"{test_name}: PASSED")
            elif exit_code == 2:
                print(f"{test_name}: TIMED OUT")
                timeout_count += 1
                if args.exitfirst:
                    break
            else:
                print(f"{test_name}: FAILED")
                failed_count += 1
                if args.exitfirst:
                    break

        # Exit early if requested and there were failures
        if args.exitfirst and (failed_count > 0 or timeout_count > 0):
            break

    # Print summary
    total_tests = len(all_results)
    passed_count = sum(1 for _, code, _ in all_results if code == 0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total:    {total_tests}")
    print(f"  Passed:   {passed_count}")
    print(f"  Failed:   {failed_count}")
    print(f"  Timeouts: {timeout_count}")

    # Print detailed results if requested
    if args.details:
        print("\nDetailed Results:")
        print("-" * 60)
        for test_name, exit_code, duration in all_results:
            if exit_code == 0:
                status = "PASS"
            elif exit_code == 2:
                status = "TIMEOUT"
            else:
                status = "FAIL"
            print(f"  {test_name:<40} {status:<8} {duration:6.2f}s")

        t_end = time.perf_counter() - t_start
        sum_class_time = sum(d for _, _, d in all_results)

        print("\n" + "-" * 60)
        print("TIMING")
        print("-" * 60)
        print(f"  Total time:          {t_end:.2f}s")
        print(f"  Sum of test times:   {sum_class_time:.2f}s")

        if resource is not None:
            try:
                # ru_maxrss units: Linux = KiB, macOS = bytes
                peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                if sys.platform == "darwin":
                    peak_bytes = float(peak)  # bytes on macOS
                else:
                    peak_bytes = float(peak) * 1024.0  # KiB -> bytes on Linux
                print(f"  Peak RSS (children): {_fmt_bytes(peak_bytes)}")
            except Exception as e:
                print(f"  Peak RSS (children): error reading ({e})")
        else:
            print("  Peak RSS (children): unavailable (no 'resource' module)")

    # Exit with appropriate code
    exit_code = 1 if (failed_count > 0 or timeout_count > 0) else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
