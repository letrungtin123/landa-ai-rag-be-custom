"""scripts/check.ps1 (Windows) and scripts/check.sh (Linux, CI) run the same gate.

CI runs check.sh only, so a path or threshold added to one script and not the other would leave
code unchecked on one side without any failure.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PS1 = (REPO_ROOT / "scripts" / "check.ps1").read_text(encoding="utf-8")
SH = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
CI = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def ps1_list(name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\)", PS1, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"${name} not found in check.ps1")
    return re.findall(r'"([^"]+)"', match.group(1))


def sh_list(name: str) -> list[str]:
    match = re.search(rf'^{name}="([^"]*)"', SH, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"{name} not found in check.sh")
    return match.group(1).split()


class GateScriptParityTests(unittest.TestCase):
    def test_both_scripts_check_the_same_paths(self) -> None:
        for ps1, sh in (("StrictPaths", "strict_paths"), ("MypyStrictPaths", "mypy_strict_paths"),
                        ("StrictTests", "strict_tests")):
            with self.subTest(list=ps1):
                self.assertEqual(ps1_list(ps1), sh_list(sh))
                self.assertTrue(all((REPO_ROOT / path).exists() for path in ps1_list(ps1)), ps1)
        # Every IDM test file except the foundation one is added on top of the lists by both scripts.
        self.assertIn('-Filter "test_idm_*.py"', PS1)
        self.assertIn("tests/test_idm_*.py", SH)
        self.assertEqual(PS1.count('"test_idm_foundation.py"') + SH.count('"test_idm_foundation.py"'), 2)

    def test_both_scripts_run_the_same_steps_and_thresholds(self) -> None:
        for step in ("-m ruff check", "--select F", "-m mypy", "-m pytest -q --cov=app --cov-branch",
                     "-m pip_audit -r requirements.lock --disable-pip"):
            with self.subTest(step=step):
                self.assertIn(step, PS1)
                self.assertIn(step, SH)
        thresholds = {re.findall(r"--cov-fail-under=(\d+)", text)[0] for text in (PS1, SH)}
        self.assertEqual(len(thresholds), 1)

    def test_ci_runs_the_linux_gate_without_services_or_required_secrets(self) -> None:
        quality = CI.split("\n  image:", 1)[0]
        self.assertIn("run: sh scripts/check.sh", quality)
        self.assertNotIn("services:", quality)
        # The backend checkout for the one Node cross-language test is optional.
        self.assertIn("HAS_BACKEND_CHECKOUT", quality)
        self.assertEqual(re.findall(r"cov-fail-under=\d+", quality), [])  # the threshold lives in the scripts


if __name__ == "__main__":
    unittest.main()
