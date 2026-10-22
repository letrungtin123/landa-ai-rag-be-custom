"""The IDM prompt templates are versioned by content (``prompt_templates_hash``).

``tests/fixtures/prompt_templates_manifest.json`` records the hash and one digest per template
source. A prompt change fails this test until the manifest is updated in the same change, so the
reviewer sees it explicitly:

    LANDA_UPDATE_PROMPT_MANIFEST=1 python -m pytest tests/golden/test_idm_prompt_manifest.py

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

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "prompt_templates_manifest.json"
UPDATE_ENV = "LANDA_UPDATE_PROMPT_MANIFEST"
# Functions named like prompt builders that build no prompt text.
NOT_PROMPT_TEXT = {
    "runtime:estimate_prompt_tokens",  # token estimate of a prompt
    "signals:states_rule",  # classifies source text
}
# Reviewed values of prompt_templates_hash, oldest first; the last one is the manifest's. Append an entry with the
# reason whenever the manifest is re-recorded.
HASH_HISTORY: tuple[tuple[str, str], ...] = (
    ("dbf793b05060b7c86ab73066ccfe54ff1ed5ee88b4c4e2026419083c03f94ca0",
     "bd9d3d4: the manifest is introduced"),
    # Owner-approved split of app/idm/prompts.py by stage (refactor step 2.5) into prompts_common, prompts_course,
    # prompts_module, prompts_unit and prompts_judge (app.idm.prompts re-exports them). Lines moved verbatim; the
    # rendered prompts are byte-identical (tests/golden/test_idm_prompt_outputs.py, recorded before the split). Only the
    # hashed sources moved, so the hash changed while the prompt content did not.
    ("7954ecdcecab92fd9b419b6db3ef383e8ca76fb4839ef934281a8dd62a1acb61",
     "prompts.py split by stage: same prompt content, sources in five modules"),
    # Owner decisions of 2026-10-20 (P2, P3): a new-fact repair line quotes the new numbers and names of the writer's
    # answer (IDM_W5_NEW_FACT, IDM_W5_FAQ_NEW_FACT rule text, new_fact_hint), and the second targeted repair of a leak,
    # a new fact or a contested key adds its SECOND ATTEMPT line (SECOND_REPAIR_TEXT, second_repair_line).
    ("687191d16626ff02485edc88d1d5718f09295c6edb6e223b3be88a20ad40ded5",
     "new-fact repair lines quote their tokens; second targeted repair lines"),
    # Organization context (ORG_CONTEXT spec §4): the ORGANIZATION_CONTEXT block (app.idm.org_context) gets an optional
    # slot in W1-reduce, W2, W4, the module stage, the W5 writer and the W6 judge, empty without a context (the rendered
    # prompts of runs without one are byte-identical: tests/golden/test_idm_prompt_outputs.py); PROJECT_CONTEXT leaves
    # the key out (author_brief.NON_PLAN_CONTEXT_KEYS); two writer lines moved to module constants unchanged.
    ("cec80c040c1329b89ce09cdca144e880d9c40d12942c506a47935ad558bcf924",
     "organization context block slots; no change without a context"),
    # The organization profile distillation prompts (app.idm.prompts_org_distill, POST /v1/lesson-author/org-context/
    # distill): a new source; no existing prompt changed.
    ("4b6f4d57cc2359dd1a4e7e12ab4db56b22508cc4c32d5035317139cc28393dc0",
     "organization profile distillation prompts"),
    # QC feedback 2026-10-21, A7 (contract K3): the organization context never names the learners. The design, writer
    # and judge views and their rule lines leave out the learner groups; the distillation asks for none (org-distill-2).
    # Prompts of runs without an organization context are unchanged.
    ("0bfd97c4c36e765ce584cf95af57e4078f041793321ab6106484ecb1700e0f44",
     "QC A7: no learner groups in the organization context views or the distillation"),
    # QC feedback 2026-10-21, A6 (contract K2): no Nice to Know. The preamble, W1-reduce, W2 (decision flow,
    # classification list, term-definition rule, treatment line), W4 (duration fit) and the purpose background rules
    # name Remove only; the server stores a nice_to_know row as remove (blueprint.NICE_TO_KNOW_AS_REMOVE_CODE).
    ("ce47669c06590cc7aecb997df88169187ca5beedc4ef7ac28f3e6bef92e46846",
     "QC A6: no Nice to Know classification in the design prompts"),
    # QC feedback 2026-10-21, A5 (contract K2): no Hold. The preamble asks for suggested criteria instead of a Hold and
    # an SME question; W2 never holds and lists every Must Do on a course row (blocked_must_do_ids always []); the
    # module stage never holds a practice (step 2 moved to _criteria_rule: suggested criteria, labelled); the W5
    # worksheet rule (moved to _WORKSHEET_RULE) heads suggested criteria as such.
    ("ee2b60eb108325cdf0849ec5320fedac0b88711c162338f688fd22db5a46ef11",
     "QC A5: no Hold; suggested criteria for a practice the source gives no criteria"),
    # QC feedback 2026-10-21, A1/A2 (contract K1): QUIZ TEXT. The W5 writer gets the question and explanation shape
    # (line feeds, "- " list lines, principle line + one "A - Đúng/Sai vì …" line per option, caps, no empty praise),
    # COMPACT_UNIT and the judge's Q4 ask for the same shape, and the repair rules of the seven new codes are added
    # (``app.idm.quiz_text:quiz_hint`` joins the manifest).
    ("f8cfb8c52da83f4d612e4b01de984f266557d244aaec76be36d02bb7d7b01f08",
     "QC A1/A2: quiz question and explanation laid out in lines, with checks and repairs"),
    # QC feedback 2026-10-21, A3: the W5 callout rule (moved to ``_CALLOUT_RULE``) says a callout adds a point its slot
    # does not already make and that the server removes one that only repeats the paragraph or list above it.
    ("086a2ff4697c763b144cfeff5f7bd8a23719e464a82a28b113f5101cb7945205",
     "QC A3: a callout never repeats its slot's paragraphs, lists or tables"),
    # QC feedback 2026-10-21, A4: the W3/W4 media_brief rule (now ``prompts_module._MEDIA_BRIEF_RULE``) asks for a
    # context_description a designer can draw from: at least 40 words, the layout, one numbered element per content
    # point with its on-image text and numbers, arrows only for steps or a cause and its effect.
    ("c011de35a0dbd792e58649faf319caae9a1e67985e2a37806e47aa6395020b83",
     "QC A4: a media brief describes what to draw"),
    # Owner decision 2026-10-22: no question for an SME anywhere. W1 map (step 6, COMPACT_W1_MAP) and W1 reduce (step
    # 1) ask for none, the W1-reduce conflict schema's sme_question is optional, the module notes no longer ask for
    # "SME flags", and the FAQ rules and material (``faq_checks.faq_prompt_material``) carry no open SME question.
    ("9040b3a024bd9e3583798225f49fb9c69ad6476ff37685bcaaf10ebe810bd370",
     "No question for an SME in any prompt"),
    # Owner decision 2026-10-22 (plain words): the AI-drafted mark the module prompt asks for is addressed to the author
    # ("[AI soạn — tác giả kiểm tra lại]" / "[AI-drafted — author to check]") and the FAQ open-question repair rule
    # speaks of a point the source has not settled, no longer of the SME.
    ("89b5aac5a6e2df0983c90b70a65c66cb74f4441db9a9a364ae690be8e64b8b89",
     "No SME in the AI-drafted mark and repair rules"),
    # Owner decision 2026-10-22 (run dfe78482): QUIZ TEXT says stem items are never labelled with the option letters
    # A-H (number or name them); the server numbers them when they still are (``quiz_text.number_item_letters``).
    ("b4e5b81747f49d3e074bf7ddec1323aa7264c2cfed1718f9e97a0ec1676303a1",
     "Stem items numbered, never lettered"),
)
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
                                      f"{UPDATE_ENV}=1 python -m pytest tests/golden/test_idm_prompt_manifest.py")
        self.assertEqual(current, recorded)

    def test_hash_history_ends_with_the_reviewed_manifest(self) -> None:
        recorded: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(HASH_HISTORY[-1][0], recorded["prompt_templates_hash"])
        self.assertEqual(len({value for value, _ in HASH_HISTORY}), len(HASH_HISTORY))

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
