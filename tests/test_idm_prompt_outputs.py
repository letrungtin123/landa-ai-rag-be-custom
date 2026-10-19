"""Every IDM prompt builder renders byte-identical text on the golden fixtures (the safety net of the prompts split).

``tests/fixtures/idm_prompt_outputs.json`` holds, per rendering, the SHA-256 and length of the exact UTF-8 text:

* every prompt the golden flows send to the provider: the course design (W1 map per section, W1 reduce, W2, W4),
  the module design of every chapter, and units through ``/unit`` (writer + judge, the writer's repair prompts
  after malformed answers, a writer failure);
* every public builder of ``app.idm.prompts`` called directly with golden inputs (both locales).

It was recorded before ``app/idm/prompts.py`` was split by stage, so a split that moves code without changing any
character keeps this test green. Re-record only after a deliberate prompt change, together with the reviewed
prompt manifest (``tests/test_idm_prompt_manifest.py``):

    LANDA_UPDATE_IDM_PROMPT_OUTPUTS=1 python -m pytest tests/test_idm_prompt_outputs.py
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from app import main
from app.api import deps as api_deps
from app.core.config import settings
from app.idm import prompts
from app.idm.course_design import run_idm_course_design
from app.idm.html_rules import DENSITY_CODE, HtmlRuleViolation
from app.idm.module_design import run_idm_module_design
from tests import idm_golden as g
from tests import idm_golden_module as gm
from tests.idm_test_support import ALLOWANCE, golden_design, make_runtime
from tests.test_idm_storyboard import (
    JUDGE,
    WRITER,
    FakeGenerate,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
    unavailable,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "idm_prompt_outputs.json"
UPDATE_ENV = "LANDA_UPDATE_IDM_PROMPT_OUTPUTS"


def fingerprint(text: str) -> dict[str, Any]:
    data = text.encode("utf-8")
    return {"sha256": hashlib.sha256(data).hexdigest(), "chars": len(text)}


def provider_prompts(calls: list[dict[str, Any]], schema_key: str) -> list[dict[str, Any]]:
    # Order-free: the W1 map sections run concurrently.
    return sorted(({"schema": str(call[schema_key]), **fingerprint(call["prompt"])} for call in calls),
                  key=lambda item: (item["schema"], item["sha256"]))


def course_design_prompts() -> list[dict[str, Any]]:
    from app.idm.contracts import IdmCourseSkeletonRequestV1  # noqa: PLC0415  # test-local import

    provider = g.golden_provider()
    request = IdmCourseSkeletonRequestV1.model_validate({
        "pipeline_version": "idm-1", "project_context": g.project_context(),
        "source_facts": [fact.model_dump() for fact in g.source_facts()],
        "token_allowance": {"input_tokens": ALLOWANCE[0], "output_tokens": ALLOWANCE[1]},
        "remaining_budget_ms": 500_000,
    })
    asyncio.run(run_idm_course_design(request, source_snapshot_hash=g.SNAPSHOT_HASH, runtime=make_runtime(provider),
                                      parallelism=4, max_sections=16))
    return provider_prompts(provider.calls, "schema")


def module_prompts() -> dict[str, list[dict[str, Any]]]:
    design, skeleton = golden_design()
    out: dict[str, list[dict[str, Any]]] = {}
    for chapter_index, module in enumerate(design.modules):
        context, plan, facts = gm.module_context(design, skeleton, chapter_index, g.source_facts())
        answer = copy.deepcopy(gm.MODULE_RESPONSES[module.module_key])
        provider = g.FakeIdmProvider({"IdmW3W4ModuleResponseV1": [answer]})
        asyncio.run(run_idm_module_design(context=context, skeleton=skeleton, plan=plan, facts=facts,
                                          runtime=make_runtime(provider)))
        out[module.module_key] = provider_prompts(provider.calls, "schema")
    return out


async def post_unit(body: dict[str, Any], provider: FakeGenerate) -> None:
    with patch("app.services.provider.generate_content", provider), \
            patch.object(settings, "idm_judge_mode", "observe"):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://prompts") as client:
            reply = await client.post("/v1/lesson-author/orchestration-v2/unit", json=body)
    if reply.status_code != 200:
        raise AssertionError(reply.text)


def unit_prompts() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    severity = severity_body()
    good = severity_writer(severity)
    cases: dict[str, tuple[dict[str, Any], FakeGenerate]] = {
        "escalate": (escalate_body(), FakeGenerate(**{WRITER: [escalate_writer(escalate_body())], JUDGE: [judge()]})),
        "severity": (severity, FakeGenerate(**{WRITER: [good], JUDGE: [judge()]})),
        "writer_unavailable": (severity, FakeGenerate(**{WRITER: [unavailable()] * 4})),
        # Malformed answers drive the repair prompts.
        "missing_slot": (severity, FakeGenerate(**{WRITER: [{"components": {"c1": good["components"]["c1"]}}] * 4,
                                                   JUDGE: [judge()] * 2})),
        "semantic_null": (severity, FakeGenerate(**{
            WRITER: [{"components": {**good["components"],
                                     "c0": {**good["components"]["c0"], "semantic_content": None}}}] * 4,
            JUDGE: [judge()] * 2})),
    }
    main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
    try:
        for name, (body, provider) in cases.items():
            asyncio.run(post_unit(copy.deepcopy(body), provider))
            out[name] = [{"schema": call["name"], **fingerprint(call["prompt"])} for call in provider.calls]
    finally:
        main.app.dependency_overrides.pop(api_deps.require_internal_token, None)
    return out


def direct_builders() -> dict[str, dict[str, Any]]:
    facts = [(fact.fact_key, fact.fact_text, ["T", "S"] if index % 2 else None)
             for index, fact in enumerate(g.source_facts()[:6])]
    issues = [{"code": "IDM_SAMPLE_CODE", "path": "lessons[0].units[1]"}, {"code": "IDM_OTHER"}]
    errors: list[dict[str, Any]] = [{"type": "missing", "loc": ("lessons", 0, "units")},
                                    {"type": "string_too_long", "loc": ("title",)}]
    density = HtmlRuleViolation(code=DENSITY_CODE, location="semantic_content.sections[0]",
                                metrics=(("words", 260, 200), ("characters", 1900, 1500)))
    other = HtmlRuleViolation(code="HTML_SEMANTIC_INVALID", location="semantic_content")
    rendered: dict[str, str] = {}
    for locale in ("vi", "en"):
        rendered[f"locale_name/{locale}"] = prompts.locale_name(locale)
        rendered[f"preamble/{locale}"] = prompts.preamble(locale)
        for unit_path in ("chapter_1.lesson_1.unit_1", "chapter_1.lesson_2.unit_1", "chapter_3.lesson_1.unit_2"):
            rendered[f"example_opening/{locale}/{unit_path}"] = prompts.example_opening(unit_path, locale)
    rendered["fact_lines"] = prompts.fact_lines(facts)
    rendered["learner_fact_lines"] = prompts.learner_fact_lines(facts)
    rendered["repair_suffix"] = prompts.repair_suffix(issues)
    for name in ("COMPACT_W1_MAP", "COMPACT_W1_REDUCE", "COMPACT_W2", "COMPACT_W4", "COMPACT_MODULE", "COMPACT_UNIT"):
        hint = getattr(prompts, name)
        rendered[f"truncation_suffix/{name}"] = prompts.truncation_suffix(hint)
        rendered[f"answer_repair/{name}"] = prompts.answer_repair("IDM_STAGE_OUTPUT_INVALID", errors, hint)
        rendered[f"answer_repair/truncated/{name}"] = prompts.answer_repair(prompts.TRUNCATED_RESPONSE_CODE, [], hint)
    for words, chars, minimum in ((None, None, None), (180, 1200, 300), (90, 600, None)):
        for explain in (False, True):
            rendered[f"html_slot_rules/{words}/{chars}/{minimum}/{explain}"] = prompts.html_slot_rules(
                max_words=words, max_chars=chars, min_chars=minimum, explain=explain)
    for only_density in (False, True):
        for explain in (False, True):
            rendered[f"html_violation_line/density/{only_density}/{explain}"] = prompts.html_violation_line(
                "c0", density, only_density=only_density, explain=explain)
    rendered["html_violation_line/other"] = prompts.html_violation_line("c1", other)
    rendered["rule_line/known"] = prompts.rule_line("c1", "HTML_SEMANTIC_REQUIRED", "c1.semantic_content")
    rendered["rule_line/unknown"] = prompts.rule_line("c1", "IDM_UNKNOWN_CODE", "c1")
    rendered["unit_repair_suffix"] = prompts.unit_repair_suffix(issues, ["rule one", "rule two", "rule one"])
    rendered["unit_repair_suffix/no_rules"] = prompts.unit_repair_suffix(issues, [])
    rendered["UNIT_RULE_TEXT"] = json.dumps(prompts.UNIT_RULE_TEXT, ensure_ascii=False, sort_keys=True)
    return {name: fingerprint(text) for name, text in rendered.items()}


def render_all() -> dict[str, Any]:
    return {
        "course_design": course_design_prompts(),
        "modules": module_prompts(),
        "units": unit_prompts(),
        "builders": direct_builders(),
    }


class IdmPromptOutputTests(unittest.TestCase):
    def test_prompt_builders_render_the_recorded_text(self) -> None:
        current = render_all()
        if os.environ.get(UPDATE_ENV) == "1":
            FIXTURE.write_text(json.dumps(current, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for section in ("course_design", "modules", "units", "builders"):
            with self.subTest(section=section):
                self.assertEqual(current[section], recorded[section])

    def test_every_flow_reaches_the_provider(self) -> None:
        recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(recorded["course_design"]), 4)
        self.assertTrue(all(recorded["modules"].values()))
        self.assertTrue(any(item["schema"].startswith("StagedMultiRepair")
                            for calls in recorded["units"].values() for item in calls))
        self.assertTrue(any(item["schema"].startswith(JUDGE) for calls in recorded["units"].values() for item in calls))


if __name__ == "__main__":
    unittest.main()
