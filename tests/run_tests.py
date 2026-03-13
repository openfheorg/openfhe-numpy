"""
Test runner for OpenFHE-NumPy test suite.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import unittest

try:
    import resource  # not available on Windows
except Exception:
    resource = None

# --- Configuration ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
CASES_DIR = PROJECT_ROOT / "cases"
DEFAULT_PATTERN = "test_*.py"
DEFAULT_TIMEOUT = 7200  # seconds (2 hours)

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_TIMEOUT = 2
EXIT_KILLED = 3


# --- Command Line Interface ---------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run unittest-style test files for OpenFHE-NumPy.",
        epilog="Examples:\n"
        "  %(prog)s                    # Run all tests\n"
        "  %(prog)s matrix_ops         # Run tests in matrix_ops folder\n"
        "  %(prog)s --details -x       # Run with details, exit on first failure",
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
        help="Enable verbose output with test summaries, timing, CPU and RSS usage",
    )

    return parser


# --- Formatting helpers -------------------------------------------------------
def _fmt_bytes(n_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    v = float(n_bytes)
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.1f} {units[i]}"


def _fmt_cpu(cpu_time: float, wall_time: float) -> str:
    """Format CPU time with effective core multiplier."""
    if wall_time > 0:
        multiplier = cpu_time / wall_time
        return f"{cpu_time:.2f}s ({multiplier:.1f}x cores)"
    return f"{cpu_time:.2f}s"


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
        return sorted(target_path.rglob(pattern))
    if target_path.is_file():
        return [target_path]

    # Try as project-relative path
    alt_path = PROJECT_ROOT / target
    if alt_path.is_dir():
        return sorted(alt_path.rglob(pattern))
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

    def walk(suite_obj: unittest.TestSuite) -> None:
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
def _get_children_cpu() -> Optional[Tuple[float, float]]:
    """Return (utime, stime) for all children, or None if unavailable."""
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime, r.ru_stime


def _parse_stderr(raw: str) -> Tuple[Optional[float], str]:
    """
    Extract __RSS__ marker from stderr and return (rss_bytes, clean_stderr).

    The __RSS__: line is always stripped from output — it's internal telemetry.
    """
    rss_bytes = None
    clean_lines = []
    for line in (raw or "").splitlines():
        if line.startswith("__RSS__:"):
            try:
                rss_bytes = float(line.split(":", 1)[1])
            except ValueError:
                pass
        else:
            clean_lines.append(line)
    return rss_bytes, "\n".join(clean_lines)


def _run_command(
    cmd: List[str], timeout: int, env: Dict[str, str]
) -> Tuple[int, float, float, Optional[float]]:
    """
    Execute a command with timeout and return results.

    Returns:
        Tuple of (exit_code, wall_seconds, cpu_seconds, rss_bytes).
        Exit codes:
            0 = pass
            1 = fail
            2 = timeout
            3 = killed by signal (e.g. OOM)
        rss_bytes:
            RSS reported by child via __RSS__: marker, or None if unavailable.
    """
    before_cpu = _get_children_cpu()
    start_time = time.perf_counter()

    def _cpu_delta() -> float:
        if before_cpu is None:
            return 0.0
        after = _get_children_cpu()
        if after is None:
            return 0.0
        return (after[0] - before_cpu[0]) + (after[1] - before_cpu[1])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration = time.perf_counter() - start_time
        cpu_time = _cpu_delta()

        rss_bytes, clean_stderr = _parse_stderr(proc.stderr)

        if proc.returncode < 0:
            # Negative return code means killed by a signal (e.g. -9 = SIGKILL/OOM)
            print(f"Process killed by signal {-proc.returncode} (likely OOM)")
            if clean_stderr:
                sys.stderr.write(clean_stderr + "\n")
            return EXIT_KILLED, duration, cpu_time, rss_bytes

        # Only print output on failure — passing tests stay silent
        if proc.returncode != 0:
            if proc.stdout:
                sys.stdout.write(proc.stdout)
            if clean_stderr:
                sys.stderr.write(clean_stderr + "\n")

        return proc.returncode, duration, cpu_time, rss_bytes

    except subprocess.TimeoutExpired as e:
        duration = time.perf_counter() - start_time
        print(f"Command timed out after {timeout}s")
        if e.stdout:
            sys.stdout.write(e.stdout)
        if e.stderr:
            rss_bytes, clean_stderr = _parse_stderr(e.stderr)
            if clean_stderr:
                sys.stderr.write(clean_stderr + "\n")
        else:
            rss_bytes = None
        return EXIT_TIMEOUT, duration, _cpu_delta(), rss_bytes

    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"Command failed with error: {e}")
        return EXIT_FAIL, duration, _cpu_delta(), None


def run_test_file(
    pyfile: Path,
    *,
    timeout: int,
    current: int,
    total: int,
    details: bool,
    exit_first: bool,
) -> Tuple[List[Tuple[str, int, float]], Dict[str, Dict[str, Any]], List[str]]:
    if details:
        print(f"\n\n=== Running {pyfile.name} ({current}/{total}) ===")
    else:
        print(f"\n\n... Running {pyfile.name} ({current}/{total}) ...")
    module_name = module_from_path(pyfile)

    # Find all tests in this module
    try:
        test_ids = get_test_from_module(module_name)
    except Exception as e:
        print(f"[discover] Failed to discover tests in '{module_name}': {e}")
        label = f"{pyfile.name}:<discover>"
        return [(label, EXIT_FAIL, 0.0)], {}, [label]

    if not test_ids:
        print(f"No tests found in file (module={module_name})")
        return [], {}, []

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
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["DETAILS"] = "1" if details else "0"

    results: List[Tuple[str, int, float]] = []
    class_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "pass": 0,
            "fail": 0,
            "timeout": 0,
            "killed": 0,
            "time": 0.0,
            "cpu_time": 0.0,
            "max_rss": 0.0,
        }
    )
    failed_labels: List[str] = []

    if details:
        print("\n...testing...\n")

    for idx, test_id in enumerate(test_ids, 1):
        _, cls, meth = split_test_id(test_id)
        label = f"{pyfile.name}:{cls}.{meth}"

        if details:
            print(f"  [{idx}/{len(test_ids)}] {label}")

        cmd = [sys.executable, "-m", "unittest", "-q"]
        if exit_first:
            cmd.append("--failfast")
        cmd.append(test_id)

        code, duration, cpu_time, rss_bytes = _run_command(cmd, timeout, env)
        results.append((label, code, duration))

        class_stats[cls]["time"] += duration
        class_stats[cls]["cpu_time"] += cpu_time
        if rss_bytes is not None:
            class_stats[cls]["max_rss"] = max(class_stats[cls]["max_rss"], rss_bytes)

        if code == EXIT_PASS:
            class_stats[cls]["pass"] += 1
        elif code == EXIT_TIMEOUT:
            class_stats[cls]["timeout"] += 1
            failed_labels.append(label)
        elif code == EXIT_KILLED:
            class_stats[cls]["killed"] += 1
            failed_labels.append(label)
        else:
            class_stats[cls]["fail"] += 1
            failed_labels.append(label)

        if details:
            if code == EXIT_KILLED:
                status = "KILLED"
            elif code == EXIT_TIMEOUT:
                status = "TIMEOUT"
            elif code == EXIT_PASS:
                status = "PASS"
            else:
                status = "FAIL"

            rss_str = _fmt_bytes(rss_bytes) if rss_bytes is not None else "n/a"
            print(
                f"        {status:>7} | "
                f"wall={duration:>8.2f}s | "
                f"cpu={_fmt_cpu(cpu_time, duration):>24} | "
                f"rss={rss_str:>10}"
            )

        if exit_first and code != EXIT_PASS:
            break

    return results, class_stats, failed_labels


# --- Main Function ------------------------------------------------------------
def main() -> None:
    """Main entry point for the test runner."""
    args = build_parser().parse_args()

    # Discover all test files to run
    all_tests: List[Path] = []
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

    if args.details:
        slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK")
        cpu_count = slurm_cpus if slurm_cpus else str(os.cpu_count() or 1)
        cpu_source = "SLURM" if slurm_cpus else "(os)"
        print()
        print("  ---- System Information ------------------------------------")
        print(f"    CPUs available:    {cpu_count:<6} ({cpu_source})")
        print(f"    OMP_NUM_THREADS:   {os.environ.get('OMP_NUM_THREADS', '1 (default)')}")
        try:
            mem_total = mem_avail = None
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1]) * 1024
                    if mem_total and mem_avail:
                        break
            if mem_total and mem_avail:
                used = mem_total - mem_avail
                pct = used / mem_total * 100
                print(f"    RAM total:         {_fmt_bytes(mem_total)}")
                print(
                    f"    RAM available:     {_fmt_bytes(mem_avail)}  "
                    f"(used {_fmt_bytes(used)}, {pct:.0f}%)"
                )
        except Exception:
            pass
        print("  ------------------------------------------------------------")
        print()

    print(f"Found {n_modules} test file(s)")

    if args.details:
        for test_file in all_tests:
            print(f"  {test_file.relative_to(CASES_DIR)}")

    # Run all tests
    passed_count = 0
    failed_count = 0
    timeout_count = 0
    killed_count = 0
    test_time = 0.0
    total_cpu_time = 0.0
    max_reported_rss = 0.0
    all_failed: List[str] = []

    for i, test_file in enumerate(all_tests, 1):
        file_results, class_stats, failed_labels = run_test_file(
            test_file,
            timeout=args.timeout,
            current=i,
            total=n_modules,
            details=args.details,
            exit_first=args.exitfirst,
        )

        all_failed.extend(failed_labels)

        if not class_stats and file_results:
            failed_count += 1
        else:
            for cls in class_stats:
                s = class_stats[cls]
                passed_count += s["pass"]
                failed_count += s["fail"]
                timeout_count += s["timeout"]
                killed_count += s["killed"]
                test_time += s["time"]
                total_cpu_time += s["cpu_time"]
                max_reported_rss = max(max_reported_rss, s["max_rss"])

        if args.exitfirst and (failed_count > 0 or timeout_count > 0 or killed_count > 0):
            break

    total_tests = passed_count + failed_count + timeout_count + killed_count

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total modules:       {n_modules}")
    print(f"  Total tests:         {total_tests}")
    print(f"  Passed:              {passed_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Timeouts:            {timeout_count}")
    print(f"  Killed (OOM/signal): {killed_count}")

    if all_failed:
        print("\n  Unsuccessful tests:")
        for label in all_failed:
            print(f"    ✗ {label}")

    print(f"\n  Total wall time:     {test_time:.2f}s")
    print(f"  Total CPU time:      {_fmt_cpu(total_cpu_time, test_time)}")

    if max_reported_rss > 0:
        print(f"  Max RSS reported:    {_fmt_bytes(max_reported_rss)}")

    if resource is not None:
        try:
            # ru_maxrss units: Linux = KiB, macOS = bytes
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak_bytes = float(peak) if sys.platform == "darwin" else float(peak) * 1024.0
            print(f"  Peak RSS (children): {_fmt_bytes(peak_bytes)}")
        except Exception as e:
            print(f"  Peak RSS (children): error reading ({e})")
    else:
        print("  Peak RSS (children): unavailable (no 'resource' module)")

    exit_code = 1 if (failed_count > 0 or timeout_count > 0 or killed_count > 0) else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
