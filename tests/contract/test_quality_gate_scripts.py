"""scripts/check.ps1 (Windows) and scripts/check.sh (Linux, CI) run the same gate.

CI runs check.sh only, so a path or threshold added to one script and not the other would leave
code unchecked on one side without any failure.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        for ps1, sh in (("StrictPaths", "strict_paths"), ("StrictTests", "strict_tests")):
            with self.subTest(list=ps1):
                self.assertEqual(ps1_list(ps1), sh_list(sh))
                self.assertTrue(all((REPO_ROOT / path).exists() for path in ps1_list(ps1)), ps1)
        # Every test package except the legacy pipeline's (deleted with it) is held to the strict lint profile.
        packages = {f"tests/unit/{path.name}" for path in (REPO_ROOT / "tests" / "unit").iterdir()
                    if path.is_dir() and not path.name.startswith(("_", "."))}
        self.assertEqual(packages - set(ps1_list("StrictTests")), {"tests/unit/legacy"})

    def test_both_scripts_run_the_same_steps_and_thresholds(self) -> None:
        for step in ("-m ruff check", "--select F", "-m mypy", "-m pytest -q --cov=app --cov-branch",
                     "scripts/coverage_by_package.py --fail-under app/idm=90",
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
