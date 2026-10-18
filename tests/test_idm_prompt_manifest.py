"""The IDM prompt templates are versioned by content (``prompt_templates_hash``).

``tests/fixtures/prompt_templates_manifest.json`` records the hash and one digest per template
source. A prompt change fails this test until the manifest is updated in the same change, so the
reviewer sees it explicitly:

    LANDA_UPDATE_PROMPT_MANIFEST=1 python -m pytest tests/test_idm_prompt_manifest.py

``IDM_PROMPT_POLICY_VERSION`` ("idm-prompt-1") is a contract value shared with Node and stored in
course designs: never change it here, only together with the backend contract.
"""

from __future__ import annotations

import ast
import json
import os
import re
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from app import main
from app.api import deps as api_deps
from app.idm import fidelity, prompt_manifest
from app.idm.policy import IDM_PROMPT_POLICY_VERSION
from app.idm.prompt_manifest import PROMPT_TEMPLATE_SOURCES, prompt_template_digests, prompt_templates_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "prompt_templates_manifest.json"
UPDATE_ENV = "LANDA_UPDATE_PROMPT_MANIFEST"
# Functions named like prompt builders that build no prompt text.
NOT_PROMPT_TEXT = {
    "runtime:estimate_prompt_tokens",  # token estimate of a prompt
    "signals:states_rule",  # classifies source text
}
PROMPT_NAME = re.compile(r"(prompt|_rules?$|_suffix$|rule_lines?$|_hint$|^rule_line$|_cue_line$)")


def manifest() -> dict[str, Any]:
    return {
        "prompt_policy_version": IDM_PROMPT_POLICY_VERSION,
        "prompt_templates_hash": prompt_templates_hash(),
        "sources": prompt_template_digests(),
    }


def prompt_named_functions() -> set[str]:
    found: set[str] = set()
    for path in sorted((REPO_ROOT / "app" / "idm").glob("*.py")):
        if path.stem == "prompt_manifest":  # this module: it hashes the prompts, it builds none
            continue
        stack: list[tuple[ast.AST, str]] = [(ast.parse(path.read_text(encoding="utf-8")), "")]
        while stack:
            node, prefix = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and PROMPT_NAME.search(child.name):
                    found.add(f"{path.stem}:{prefix}{child.name}")
                elif isinstance(child, ast.ClassDef):
                    stack.append((child, prefix + child.name + "."))
    return found


class PromptManifestTests(unittest.TestCase):
    maxDiff = None

    def test_templates_match_the_reviewed_manifest(self) -> None:
        current = manifest()
        if os.environ.get(UPDATE_ENV) == "1":
            MANIFEST.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        recorded: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
        before: dict[str, str] = recorded["sources"]
        after: dict[str, str] = current["sources"]
        changed = sorted(entry for entry in {*before, *after} if before.get(entry) != after.get(entry))
        self.assertEqual(changed, [], f"Prompt templates changed: {changed}. Review the change, then run "
                                      f"{UPDATE_ENV}=1 python -m pytest tests/test_idm_prompt_manifest.py")
        self.assertEqual(current, recorded)

    def test_every_prompt_builder_of_the_package_is_covered(self) -> None:
        modules = {entry.removeprefix("app.idm.") for entry in PROMPT_TEMPLATE_SOURCES if ":" not in entry}
        covered = {entry.removeprefix("app.idm.") for entry in PROMPT_TEMPLATE_SOURCES if ":" in entry}
        uncovered = sorted(name for name in prompt_named_functions() - NOT_PROMPT_TEXT - covered
                           if name.split(":", 1)[0] not in modules)
        self.assertEqual(uncovered, [])

    def test_a_rule_constant_a_builder_reads_changes_the_hash(self) -> None:
        entry = "app.idm.fidelity:w2_treatment_rules"
        before = prompt_template_digests()[entry]
        prompt_template_digests.cache_clear()
        try:
            with patch.object(fidelity, "_TREATMENTS", fidelity._TREATMENTS + " (changed)"):
                self.assertNotEqual(prompt_template_digests()[entry], before)
        finally:
            prompt_template_digests.cache_clear()
        self.assertEqual(prompt_template_digests()[entry], before)

    def test_the_hash_is_stable_and_never_the_contract_version(self) -> None:
        self.assertRegex(prompt_templates_hash(), r"^[0-9a-f]{64}$")
        source = prompt_manifest.entry_source("app.idm.prompts")
        self.assertEqual(source, prompt_manifest.entry_source("app.idm.prompts"))
        self.assertNotIn("\r", source)
        self.assertEqual(IDM_PROMPT_POLICY_VERSION, "idm-prompt-1")


class MetaExposureTests(unittest.IsolatedAsyncioTestCase):
    async def test_meta_reports_the_hash_beside_the_unchanged_contract_versions(self) -> None:
        main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
                body = (await client.get("/v1/meta")).json()
        finally:
            main.app.dependency_overrides.pop(api_deps.require_internal_token, None)
        self.assertEqual(body["prompt_templates_hash"], prompt_templates_hash())
        self.assertEqual(body["contracts"]["idm_prompt_policy_version"], "idm-prompt-1")


if __name__ == "__main__":
    unittest.main()
