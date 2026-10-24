"""scripts/mypy_baseline.py: the strict paths stay clean and the legacy type errors only shrink, file by file."""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "mypy_baseline.py"
# The record when it replaced ``ignore_errors = true`` (refactor wave 3b): 404 errors in these 31 legacy files.
# A hand edit cannot raise the total or add a file; scripts/mypy_baseline.py --update only lowers entries.
RECORDED_TOTAL = 404
RECORDED_FILES = 31


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mypy_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script()


def strict_paths() -> list[str]:
    match = re.search(r'^strict_paths="([^"]*)"', (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8"),
                      flags=re.MULTILINE)
    assert match is not None
    return match.group(1).split()


class MypyBaselineTests(unittest.TestCase):
    def test_error_lines_are_counted_per_file_on_every_platform(self) -> None:
        report = "\n".join([
            "app\\workflows\\contracts.py:373: error: Item None has no attribute get  [union-attr]",
            "app/workflows/contracts.py:374:5: error: Item None has no attribute get  [union-attr]",
            "app/workflows/contracts.py:375: note: See https://mypy.readthedocs.io",
            "app/source_map.py:12: error: Need type annotation  [var-annotated]",
            "Found 3 errors in 2 files (checked 257 source files)",
        ])
        self.assertEqual(baseline.error_counts(report),
                         Counter({"app/workflows/contracts.py": 2, "app/source_map.py": 1}))

    def test_strict_errors_new_legacy_errors_and_stale_records_fail(self) -> None:
        strict, legacy = {"app/idm/qa.py"}, {"app/a.py", "app/b.py", "app/c.py"}
        recorded = {"app/a.py": 3, "app/b.py": 2, "app/gone.py": 1}
        counts = Counter({"app/idm/qa.py": 1, "app/a.py": 4, "app/b.py": 1, "app/c.py": 1, "other.py": 1})
        self.assertEqual(baseline.problems(counts, strict, legacy, recorded), [
            "app/idm/qa.py: 1 type error(s) in strict code",
            "other.py: 1 type error(s) outside the checked files",
            "app/a.py: 4 type errors, 3 recorded: fix the new ones",
            "app/b.py: 1 type errors, 2 recorded: lower the record "
            "(python scripts/mypy_baseline.py --update <strict paths>)",
            "app/c.py: 1 type errors, 0 recorded: fix the new ones",
            "app/gone.py: 0 type errors, 1 recorded: lower the record "
            "(python scripts/mypy_baseline.py --update <strict paths>)",
        ])
        self.assertEqual(baseline.problems(Counter({"app/a.py": 3, "app/b.py": 2}), strict, legacy,
                                           {"app/a.py": 3, "app/b.py": 2}), [])

    def test_update_only_lowers_the_record(self) -> None:
        legacy = {"app/a.py", "app/b.py"}
        self.assertEqual(baseline.updated(Counter({"app/a.py": 2, "app/b.py": 0}), legacy, {"app/a.py": 3}),
                         {"app/a.py": 2})
        with self.assertRaises(SystemExit):
            baseline.updated(Counter({"app/a.py": 4}), legacy, {"app/a.py": 3})
        with self.assertRaises(SystemExit):
            baseline.updated(Counter({"app/b.py": 1}), legacy, {"app/a.py": 3})
        self.assertEqual(baseline.updated(Counter({"app/a.py": 4}), legacy, None), {"app/a.py": 4})

    def test_the_record_covers_legacy_files_only_and_never_grows(self) -> None:
        record = json.loads(baseline.BASELINE.read_text(encoding="utf-8"))
        strict = set(baseline.python_files(strict_paths()))
        legacy = set(baseline.python_files(["app"])) - strict
        self.assertTrue(record)
        self.assertLessEqual(set(record), legacy)
        self.assertTrue(all(type(count) is int and count > 0 for count in record.values()))
        self.assertLessEqual(sum(record.values()), RECORDED_TOTAL)
        self.assertLessEqual(len(record), RECORDED_FILES)
        # The legacy pipeline only: app/services/lesson_author, app/workflows and modules directly in app/.
        self.assertTrue(all(name.startswith(("app/services/lesson_author/", "app/workflows/")) or name.count("/") == 1
                            for name in record))


if __name__ == "__main__":
    unittest.main()
