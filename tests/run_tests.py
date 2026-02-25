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
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import unittest

try:
    import resource  # not available on Windows
except Exception:
    resource = None

# --- Configuration ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
CASES_DIR = PROJECT_ROOT / "cases"
DEFAULT_PATTERN = "test_*.py"
DEFAULT_TIMEOUT = 7200  # seconds (30 minutes)


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
    return f"{v:.1f} {units[i]}"


# --- Test Discovery -----------------------------------------------------------
def find_tests(target: Optional[str] = None, pattern: str = DEFAULT_PATTERN) -> List[Path]:
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

    print(f"Error: Target '{target}' not found in {CASES_DIR} or {PROJECT_ROOT}")
    sys.exit(1)


# --- Get module helper --------------------------------------------------------
def module_from_path(pyfile: Path) -> str:
    """Convert a Python file path to a module import path."""
    relative_path = pyfile.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(relative_path.parts)


# --- Unittest ID helpers ------------------------------------------------------
def get_test_from_module(module_name: str) -> List[str]:
    """
    Return full unittest IDs like:
      package.module.ClassName.test_method
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    ids: List[str] = []

    def walk(suite_obj):
        for t in suite_obj:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            else:
                ids.append(t.id())

    walk(suite)
    return sorted(ids)


def split_test_id(test_id: str) -> Tuple[str, str, str]:
    """
    Split:
      package.module.ClassName.test_method
    into:
      (package.module, ClassName, test_method)
    """
    mod, cls, meth = test_id.rsplit(".", 2)
    return mod, cls, meth


# --- Test Execution -----------------------------------------------------------
def _run_command(cmd: List[str], timeout: int, env: Dict[str, str]) -> Tuple[int, float]:
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration = time.time() - start_time

        # Only print output on failure — passing tests stay silent
        if proc.returncode != 0:
            if proc.stdout:
                sys.stdout.write(proc.stdout)
            if proc.stderr:
                sys.stderr.write(proc.stderr)

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
) -> Tuple[List[Tuple[str, int, float]], Dict]:

    # if details:
    print(f"\n\n=== Running {pyfile.name} ({current}/{total}) ===")

    module_name = module_from_path(pyfile)

    # Find all class in this module
    try:
        test_ids = get_test_from_module(module_name)
    except Exception as e:
        print(f"[discover] Failed to discover tests in '{module_name}': {e}")
        empty_stats: Dict = {}
        return [(f"{pyfile.name}:<discover>", 1, 0.0)], empty_stats

    if not test_ids:
        print(f"No tests found in file (module={module_name})")
        return [], {}

    if details:
        by_class: Dict[str, int] = defaultdict(int)
        for tid in test_ids:
            _, cls, _ = split_test_id(tid)
            by_class[cls] += 1
        print(f"Found {len(test_ids)} test(s) in {module_name}")
        for cls in sorted(by_class):
            print(f"  {cls} ({by_class[cls]})")

    # Env for child processes
    env = os.environ.copy()
    env["DETAILS"] = "1" if details else "0"
    env["FAILFAST"] = "1" if exit_first else "0"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    results: List[Tuple[str, int, float]] = []
    class_stats: Dict = defaultdict(lambda: {"pass": 0, "fail": 0, "timeout": 0, "time": 0.0})

    if details:
        print("\n...testing...\n")

    for idx, test_id in enumerate(test_ids, 1):
        _, cls, meth = split_test_id(test_id)
        label = f"{pyfile.name}:{cls}.{meth}"

        if details:
            print(f"{label} ({idx}/{len(test_ids)})")

        cmd = [sys.executable, "-m", "unittest", "-q", test_id]
        code, duration = _run_command(cmd, timeout, env)
        results.append((label, code, duration))

        class_stats[cls]["time"] += duration
        if code == 0:
            class_stats[cls]["pass"] += 1
        elif code == 2:
            class_stats[cls]["timeout"] += 1
        else:
            class_stats[cls]["fail"] += 1

        if exit_first and code != 0:
            break

    if details:
        print("\nModule Summary:", module_name)
        for cls in sorted(class_stats):
            s = class_stats[cls]
            status = "PASS" if (s["fail"] == 0 and s["timeout"] == 0) else "FAIL"
            print(
                f"  {cls:<35} {status:<4} "
                f"pass={s['pass']} fail={s['fail']} timeout={s['timeout']} "
                f"time={s['time']:.2f}s"
            )

    return results, class_stats


# --- Main Function ------------------------------------------------------------
def main() -> None:
    """Main entry point for the test runner."""
    args = build_parser().parse_args()

    # Discover all test files to run
    all_tests = []
    targets = args.targets or [None]

    for target in targets:
        all_tests.extend(find_tests(target, args.pattern))

    # Remove duplicates and sort
    all_tests = sorted(set(all_tests))
    n_modules = len(all_tests)

    # Handle list mode
    if args.list:
        print(f"Available test files ({n_modules}):")
        for test_file in all_tests:
            try:
                relative_path = test_file.relative_to(CASES_DIR)
                print(f"  {relative_path}")
            except ValueError:
                print(f"  {test_file}")
        sys.exit(0)

    # Check if any tests were found
    if not all_tests:
        print("No test files found matching criteria")
        sys.exit(0)

    print(f"Found {n_modules} test file(s)")
    if args.details:
        for test_file in all_tests:
            print(f"  {test_file.relative_to(CASES_DIR)}")

    # Run all tests
    total_tests = 0
    failed_count = 0
    timeout_count = 0
    passed_count = 0
    test_time = 0.0

    for i, test_file in enumerate(all_tests, 1):
        file_results, class_stats = run_test_file(
            test_file,
            timeout=args.timeout,
            current=i,
            total=n_modules,
            details=args.details,
            exit_first=args.exitfirst,
        )

        for cls in class_stats:
            s = class_stats[cls]
            passed_count += s["pass"]
            timeout_count += s["timeout"]
            failed_count += s["fail"]
            test_time += s["time"]

        total_tests += len(file_results)

        if args.exitfirst and (failed_count > 0 or timeout_count > 0):
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total module:        {n_modules}")
    print(f"  Total tests:         {total_tests}")
    print(f"  Passed:              {passed_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Timeouts:            {timeout_count}")
    print(f"  Total time:          {test_time:.2f}s")
    if resource is not None:
        try:
            # ru_maxrss units: Linux = KiB, macOS = bytes
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            if sys.platform == "darwin":
                peak_bytes = float(peak)
            else:
                peak_bytes = float(peak) * 1024.0
            print(f"  Peak RSS (children): {_fmt_bytes(peak_bytes)}")
        except Exception as e:
            print(f"  Peak RSS (children): error reading ({e})")
    else:
        print("  Peak RSS (children): unavailable (no 'resource' module)")

    exit_code = 1 if (failed_count > 0 or timeout_count > 0) else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
