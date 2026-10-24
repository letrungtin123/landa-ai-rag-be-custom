"""Type checking of the whole service: no error in the strict paths, a shrink-only error count per legacy file.

mypy (the pyproject.toml settings, strict for every module) checks the strict paths given on the command line and
every other module of ``app``: the legacy lesson-author pipeline (app/services/lesson_author, app/workflows and the
legacy modules of app/). A strict file must have no error. A legacy file must have exactly the number of errors
recorded in scripts/mypy_baseline.json, which replaced ``ignore_errors = true`` for those modules (refactor wave 3b):

* more errors than recorded fail (new type errors in legacy code);
* fewer fail too, until the record is lowered with ``--update``, so the record only shrinks and a fix is kept.

    python scripts/mypy_baseline.py <strict paths>             # the gate (scripts/check.sh, scripts/check.ps1)
    python scripts/mypy_baseline.py --update <strict paths>    # lower the record after fixing legacy errors

``--update`` never raises a count or adds a file; it only writes the first record when there is none.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from mypy import api as mypy_api

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "scripts" / "mypy_baseline.json"
ERROR_LINE = re.compile(r"^(?P<path>[^:\n]+\.py):\d+:(?:\d+:)? error: ", re.MULTILINE)
MYPY_FATAL = 2


def python_files(paths: list[str]) -> list[str]:
    """The ``.py`` files of the given files and directories, as repository paths."""

    found: set[str] = set()
    for item in paths:
        path = REPO_ROOT / item
        for file in path.rglob("*.py") if path.is_dir() else [path]:
            found.add(file.relative_to(REPO_ROOT).as_posix())
    return sorted(found)


def error_counts(report: str) -> Counter[str]:
    """Errors per file of a mypy report (notes do not count)."""

    return Counter(match.group("path").replace("\\", "/") for match in ERROR_LINE.finditer(report))


def problems(counts: Counter[str], strict: set[str], legacy: set[str], recorded: dict[str, int]) -> list[str]:
    """Why the gate fails: errors in strict files, unrecorded or changed counts in legacy files."""

    found = [f"{name}: {count} type error(s) in strict code" for name, count in sorted(counts.items())
             if name in strict]
    found += [f"{name}: {count} type error(s) outside the checked files" for name, count in sorted(counts.items())
              if name not in strict and name not in legacy]
    for name in sorted((set(counts) & legacy) | set(recorded)):
        actual, expected = counts.get(name, 0), recorded.get(name, 0)
        if actual > expected:
            found.append(f"{name}: {actual} type errors, {expected} recorded: fix the new ones")
        elif actual < expected:
            found.append(f"{name}: {actual} type errors, {expected} recorded: lower the record "
                         "(python scripts/mypy_baseline.py --update <strict paths>)")
    return found


def updated(counts: Counter[str], legacy: set[str], recorded: dict[str, int] | None) -> dict[str, int]:
    """The new record: the legacy counts now, refused when one would grow or appear."""

    record = {name: count for name, count in sorted(counts.items()) if name in legacy and count}
    if recorded is not None:
        grown = [name for name, count in record.items() if count > recorded.get(name, 0)]
        if grown:
            raise SystemExit(f"--update only lowers the record; more errors than recorded in: {', '.join(grown)}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mypy: strict paths clean, legacy errors per file only shrink.")
    parser.add_argument("--update", action="store_true", help="lower the record to the current counts")
    parser.add_argument("strict_paths", nargs="+", help="the strict paths of scripts/check.sh")
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    strict = set(python_files(args.strict_paths))
    legacy = set(python_files(["app"])) - strict
    stdout, stderr, status = mypy_api.run([*sorted(strict), *sorted(legacy)])
    if status >= MYPY_FATAL:
        sys.stderr.write(stdout + stderr)
        return MYPY_FATAL
    counts = error_counts(stdout)
    recorded: dict[str, int] | None = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    if args.update:
        record = updated(counts, legacy, recorded)
        BASELINE.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"mypy baseline: {sum(record.values())} errors in {len(record)} legacy files recorded")
        return 0
    found = problems(counts, strict, legacy, recorded or {})
    if found:
        bad = {name.split(":", 1)[0] for name in found}
        sys.stdout.write("".join(line + "\n" for line in stdout.splitlines()
                                 if line.replace("\\", "/").split(":", 1)[0] in bad))
        print("\n".join(["mypy baseline check failed:", *found]))
        return 1
    print(f"mypy: {len(strict)} strict files without errors; {sum(counts.values())} errors in {len(counts)} legacy "
          f"files, as recorded in {BASELINE.relative_to(REPO_ROOT).as_posix()} ({len(legacy)} legacy files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
