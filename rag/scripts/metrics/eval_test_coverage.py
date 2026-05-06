"""
Test Coverage Measurement Script.

Runs the pytest suite with coverage enabled and parses the output into a
clean summary table. Reports line coverage per module and overall.

WHY THIS MATTERS FOR A CV:
  "190+ automated tests with X% line coverage" is a strong engineering
  discipline signal. Coverage % quantifies test thoroughness and is
  easily verifiable.

Usage:
    python -m scripts.metrics.eval_test_coverage
    python -m scripts.metrics.eval_test_coverage --html
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def _run_coverage(html: bool = False) -> str:
    """Run pytest with coverage and return the output."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=rag_framework",
        "--cov-report=term-missing",
        "--tb=no",
        "-q",
    ]
    if html:
        cmd.append("--cov-report=html:coverage_html")

    result = subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    return result.stdout + "\n" + result.stderr


def _parse_coverage_output(output: str):
    """Parse pytest-cov output into structured data."""
    lines = output.split("\n")

    # Find coverage table
    coverage_lines = []
    in_coverage = False
    total_line = None

    for line in lines:
        # Coverage table starts with a line containing "Name" and "Stmts"
        if "Name" in line and "Stmts" in line and "Cover" in line:
            in_coverage = True
            continue
        if in_coverage:
            if line.startswith("TOTAL"):
                total_line = line
                in_coverage = False
                continue
            if line.startswith("---") or not line.strip():
                continue
            if line.strip():
                coverage_lines.append(line)

    # Parse individual lines
    modules = []
    for line in coverage_lines:
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            try:
                stmts = int(parts[1])
                miss = int(parts[2])
                cover_str = parts[3].rstrip("%")
                cover = int(cover_str)
                modules.append((name, stmts, miss, cover))
            except (ValueError, IndexError):
                continue

    # Parse total
    total = None
    if total_line:
        parts = total_line.split()
        if len(parts) >= 4:
            try:
                total = (int(parts[1]), int(parts[2]), int(parts[3].rstrip("%")))
            except (ValueError, IndexError):
                pass

    # Parse test count from pytest output
    test_count = 0
    for line in lines:
        match = re.search(r"(\d+)\s+passed", line)
        if match:
            test_count = int(match.group(1))
            break

    return modules, total, test_count


def _print_results(modules, total, test_count, output: str):
    # ── Test count ──
    print("=" * 60)
    print("TEST COVERAGE SUMMARY")
    print("=" * 60)

    if test_count:
        print(f"\n  Tests passed: {test_count}")

    if not modules and not total:
        print("\n  Could not parse coverage output.")
        print("  Raw output snippet:")
        print("  " + output[:500])
        return

    # ── Per-module table ──
    print(f"\n{'Module':<50} {'Stmts':>6} {'Miss':>6} {'Cover':>6}")
    print("-" * 70)

    for name, stmts, miss, cover in modules:
        # Shorten module paths for readability
        short_name = name.replace("rag_framework/", "").replace("rag_framework\\", "")
        print(f"{short_name:<50} {stmts:>6} {miss:>6} {cover:>5}%")

    if total:
        print("-" * 70)
        print(f"{'TOTAL':<50} {total[0]:>6} {total[1]:>6} {total[2]:>5}%")

    # ── Markdown ──
    print("\n\n### Markdown (copy-paste for report)")
    print("```")
    print(f"| Module | Stmts | Miss | Cover |")
    print(f"|--------|-------|------|-------|")

    # Group by top-level package for conciseness
    packages = {}
    for name, stmts, miss, cover in modules:
        parts = name.replace("\\", "/").split("/")
        if len(parts) >= 2:
            pkg = parts[1] if parts[0] == "rag_framework" else parts[0]
        else:
            pkg = parts[0]
        if pkg not in packages:
            packages[pkg] = {"stmts": 0, "miss": 0}
        packages[pkg]["stmts"] += stmts
        packages[pkg]["miss"] += miss

    for pkg, data in sorted(packages.items()):
        cover_pct = (
            (data["stmts"] - data["miss"]) / data["stmts"] * 100
            if data["stmts"] > 0
            else 0
        )
        print(
            f"| {pkg} | {data['stmts']} | {data['miss']} | {cover_pct:.0f}% |"
        )

    if total:
        print(f"| **TOTAL** | **{total[0]}** | **{total[1]}** | **{total[2]}%** |")
    if test_count:
        print(f"\nTests passed: **{test_count}**")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Measure test coverage for the RAG framework."
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also generate HTML coverage report in coverage_html/",
    )
    args = parser.parse_args()

    print("Running pytest with coverage...")
    output = _run_coverage(html=args.html)
    modules, total, test_count = _parse_coverage_output(output)
    _print_results(modules, total, test_count, output)

    if args.html:
        print(f"\nHTML coverage report saved to: {_PROJECT_ROOT}/coverage_html/")


if __name__ == "__main__":
    main()
