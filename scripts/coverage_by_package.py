"""Coverage of each ``app`` package, read from the ``.coverage`` file the pytest step of the gate writes.

The table is a report only. ``--fail-under PACKAGE=PERCENT`` turns one package into a gate (exit code 1 below
it); scripts/check.ps1 and scripts/check.sh hold ``app/idm`` to its target (hardening plan §19.6).

    python scripts/coverage_by_package.py --fail-under app/idm=90

A package is a directory of ``app`` (``app/idm``); ``app/services`` is split by subpackage, and the modules
directly in ``app`` and in ``app/services`` are one group each. The percentages count branches, as the gate's
total does.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

from coverage import Coverage

REPO_ROOT = Path(__file__).resolve().parents[1]


def package_of(relative: str) -> str:
    """The report group of a measured file (``app/idm/fidelity.py`` -> ``app/idm``)."""

    parts = relative.split("/")
    depth = 3 if len(parts) > 3 and parts[1] == "services" else 2  # noqa: PLR2004 - app/services/<package>/...
    if len(parts) > depth:
        return "/".join(parts[:depth])
    return "/".join(parts[:-1]) + " (modules)"


def package_coverage(data_file: Path) -> dict[str, tuple[int, float]]:
    """File count and covered percent of every package, from the coverage data of the last test run."""

    coverage = Coverage(data_file=str(data_file))
    coverage.load()
    groups: dict[str, list[str]] = defaultdict(list)
    for measured in coverage.get_data().measured_files():
        path = Path(measured).resolve()
        if path.is_relative_to(REPO_ROOT / "app"):
            groups[package_of(path.relative_to(REPO_ROOT).as_posix())].append(measured)
    return {package: (len(files), coverage.report(include=files, file=io.StringIO()))
            for package, files in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coverage of each app package, from the .coverage file.")
    parser.add_argument("--data-file", type=Path, default=REPO_ROOT / ".coverage")
    parser.add_argument("--fail-under", action="append", default=[], metavar="PACKAGE=PERCENT",
                        help="fail when PACKAGE covers less than PERCENT (repeatable)")
    args = parser.parse_args(argv)
    minimums = {package: float(percent) for package, _, percent in
                (target.partition("=") for target in args.fail_under)}
    results = package_coverage(args.data_file)
    print("Coverage by package (report only, except the packages marked with a target):")
    print(f"{'package':<34} {'files':>5} {'cover':>7}  target")
    failures = []
    for package, (files, percent) in results.items():
        target = minimums.get(package)
        mark = "" if target is None else f">= {target:g}%" + ("" if percent >= target else "  FAILED")
        print(f"{package:<34} {files:>5} {percent:>6.2f}%  {mark}")
        if target is not None and percent < target:
            failures.append(package)
    failures += [package for package in minimums if package not in results]
    if failures:
        print(f"Package coverage below target (or not measured): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
