"""
Master E2E Test Suite Runner for Sub-Quadratic Audio Transformer Engine.

Provides unified CLI invocation, programmatic pytest orchestration, rich visual
terminal progress tables, detailed tier-by-tier telemetry, and machine-readable
JSON report export with deterministic exit codes.

Usage:
    python tests/e2e/runner.py [--tier {1,2,3,4,all}] [--json-report <path>]
                               [--feature <F1..F14>] [--fail-fast] [--verbose]
                               [--collect-only]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Deterministic Exit Codes
EXIT_SUCCESS = 0
EXIT_TEST_FAILURES = 1
EXIT_INTERRUPTED = 2
EXIT_INTERNAL_ERROR = 3
EXIT_USAGE_ERROR = 4
EXIT_NO_TESTS = 5

TIER_FILE_MAP: Dict[str, str] = {
    "1": "tests/e2e/test_tier1_features.py",
    "2": "tests/e2e/test_tier2_boundaries.py",
    "3": "tests/e2e/test_tier3_pairwise.py",
    "4": "tests/e2e/test_tier4_workloads.py",
}


class JsonReportCollector:
    """Pytest plugin collecting per-test execution telemetry for JSON export and terminal tables."""

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self.end_time: float = 0.0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Invoked on setup, call, and teardown stages of each test."""
        # Only record the main call stage or a failed setup stage
        if report.when == "call" or (report.when == "setup" and report.failed):
            tier = 0
            if "test_tier1" in report.nodeid:
                tier = 1
            elif "test_tier2" in report.nodeid:
                tier = 2
            elif "test_tier3" in report.nodeid:
                tier = 3
            elif "test_tier4" in report.nodeid:
                tier = 4

            error_msg = None
            if report.failed:
                error_msg = str(report.longrepr)

            self.results.append({
                "nodeid": report.nodeid,
                "tier": tier,
                "name": report.location[2] if report.location else report.nodeid,
                "outcome": report.outcome,
                "duration_ms": round(report.duration * 1000.0, 2),
                "error": error_msg,
            })

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Invoked when pytest session finishes execution."""
        self.end_time = time.time()

    def generate_report(self, tier_requested: str) -> Dict[str, Any]:
        """Aggregates test results into structured dictionary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["outcome"] == "passed")
        failed = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")
        duration = round(self.end_time - self.start_time, 3)

        tier_breakdown: Dict[str, Dict[str, Any]] = {}
        for t in [1, 2, 3, 4]:
            t_results = [r for r in self.results if r["tier"] == t]
            if t_results:
                tier_breakdown[f"tier{t}"] = {
                    "total": len(t_results),
                    "passed": sum(1 for r in t_results if r["outcome"] == "passed"),
                    "failed": sum(1 for r in t_results if r["outcome"] == "failed"),
                    "skipped": sum(1 for r in t_results if r["outcome"] == "skipped"),
                    "duration_sec": round(sum(r["duration_ms"] for r in t_results) / 1000.0, 3)
                }

        pass_rate = round((passed / total * 100.0) if total > 0 else 0.0, 2)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tier_requested": tier_requested,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "duration_sec": duration,
                "pass_rate_pct": pass_rate,
                "tier_breakdown": tier_breakdown,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "cwd": os.getcwd(),
            },
            "results": self.results,
        }


def print_banner(tier: str, feature: Optional[str] = None) -> None:
    """Prints formatted execution banner to stdout."""
    print("\n" + "=" * 78)
    print("  SUB-QUADRATIC AUDIO TRANSFORMER ENGINE — E2E TEST RUNNER")
    print("=" * 78)
    print(f"  Target Tier:  {tier.upper():<10} | Start Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    if feature:
        print(f"  Filter:       Feature {feature}")
    print(f"  Platform:     {sys.platform} | Python {sys.version.split()[0]}")
    print("=" * 78 + "\n")


def print_summary_table(report: Dict[str, Any]) -> None:
    """Prints a clean, formatted ASCII table of test execution results."""
    summary = report["summary"]
    print("\n" + "=" * 78)
    print("  E2E TEST EXECUTION SUMMARY")
    print("=" * 78)
    print(f"  {'Tier / Category':<24} | {'Total':<7} | {'Passed':<7} | {'Failed':<7} | {'Skipped':<7} | {'Time (s)':<9}")
    print("  " + "-" * 74)

    tier_names = {
        "tier1": "Tier 1: Features",
        "tier2": "Tier 2: Boundaries",
        "tier3": "Tier 3: Pairwise",
        "tier4": "Tier 4: Workloads",
    }

    for tier_key, stats in summary.get("tier_breakdown", {}).items():
        display_name = tier_names.get(tier_key, tier_key.upper())
        print(
            f"  {display_name:<24} | {stats['total']:<7} | {stats['passed']:<7} | "
            f"{stats['failed']:<7} | {stats['skipped']:<7} | {stats['duration_sec']:<9.2f}"
        )

    print("  " + "-" * 74)
    print(
        f"  {'OVERALL TOTAL':<24} | {summary['total']:<7} | {summary['passed']:<7} | "
        f"{summary['failed']:<7} | {summary['skipped']:<7} | {summary['duration_sec']:<9.2f}"
    )
    print("=" * 78)

    if summary["failed"] == 0 and summary["total"] > 0:
        status_str = "SUCCESS: ALL TESTS PASSED"
    elif summary["total"] == 0:
        status_str = "NO TESTS EXECUTED"
    else:
        status_str = f"FAILURE: {summary['failed']} TEST(S) FAILED"

    print(f"  Result: {status_str} (Pass Rate: {summary['pass_rate_pct']}%)")
    print("=" * 78 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for test runner."""
    parser = argparse.ArgumentParser(
        description="Master E2E Test Suite Runner for Sub-Quadratic Audio Transformer Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/e2e/runner.py --tier 1
  python tests/e2e/runner.py --tier all --json-report build/e2e_report.json
  python tests/e2e/runner.py --tier 2 --feature F02 --fail-fast
  python tests/e2e/runner.py --collect-only
        """
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Target test tier to execute (default: 'all')"
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Destination path to export machine-readable JSON telemetry report"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Filter tests by feature key (e.g. F1, F02, F12)"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Halt test execution immediately upon first test failure"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose test output (prints individual test nodeids)"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Discover and list collected tests without executing them"
    )

    args = parser.parse_args(argv)
    print_banner(args.tier, args.feature)

    # Determine target test files based on tier
    if args.tier == "all":
        test_files = list(TIER_FILE_MAP.values())
    else:
        test_files = [TIER_FILE_MAP[args.tier]]

    # Check if target test files exist
    existing_files = [f for f in test_files if Path(f).exists()]

    pytest_args = ["-v" if args.verbose else "-q"]
    if args.fail_fast:
        pytest_args.append("-x")
    if args.collect_only:
        pytest_args.append("--collect-only")
    if args.feature:
        pytest_args.extend(["-k", args.feature])

    if existing_files:
        pytest_args.extend(existing_files)
    else:
        # Fallback to tests/e2e directory discovery if specific tier files are not yet created
        pytest_args.append("tests/e2e")

    collector = JsonReportCollector()
    try:
        pytest_exit_code = pytest.main(pytest_args, plugins=[collector])
    except KeyboardInterrupt:
        print("\n[Runner] Execution interrupted by user.")
        return EXIT_INTERRUPTED
    except Exception as e:
        print(f"\n[Runner] Internal execution error: {e}")
        return EXIT_INTERNAL_ERROR

    report_data = collector.generate_report(args.tier)
    print_summary_table(report_data)

    if args.json_report:
        try:
            report_path = Path(args.json_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"[Runner] Successfully saved JSON telemetry report to: {report_path.resolve()}")
        except Exception as e:
            print(f"[Runner] Failed writing JSON report to '{args.json_report}': {e}")

    # Map pytest exit code to deterministic runner exit code
    if pytest_exit_code == pytest.ExitCode.OK:
        return EXIT_SUCCESS
    elif pytest_exit_code == pytest.ExitCode.TESTS_FAILED:
        return EXIT_TEST_FAILURES
    elif pytest_exit_code == pytest.ExitCode.INTERRUPTED:
        return EXIT_INTERRUPTED
    elif pytest_exit_code == pytest.ExitCode.NO_TESTS_COLLECTED:
        return EXIT_NO_TESTS
    elif pytest_exit_code == pytest.ExitCode.USAGE_ERROR:
        return EXIT_USAGE_ERROR
    else:
        return EXIT_TEST_FAILURES


if __name__ == "__main__":
    sys.exit(main())
