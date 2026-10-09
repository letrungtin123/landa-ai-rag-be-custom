"""Regression tests for the IDM review findings (shard-size cap, cancellation, budgets, robustness)."""

from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.idm.architecture import _lesson_chunks, validate_w4
from app.idm.blueprint import validate_w2
from app.idm.content_map import FactIndex, map_sections, page_furniture_keys
from app.idm.contracts import (
    IdmContentBlockV1,
    IdmFeedbackFocusV1,
    IdmJudgeResponseV1,
    IdmLessonDesignV1,
    IdmW2BlueprintResponseV1,
    IdmW4CourseResponseV1,
)
from app.idm.module_layout import clean_feedback, fallback_lesson, project_lesson
from app.idm.policy import IDM_SHARD_MAX_SOURCE_CHARS
from app.idm.runtime import IdmProviderError, IdmStageError, idm_call
from app.idm.signals import compute_fact_signals, plan_idm_sections
from app.idm.text import sanitize_author_text
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2
from app.prompt_safety import untrusted_block
from tests import idm_golden as g
from tests.idm_golden import W2_BLUEPRINT, W4_COURSE, mutate, source_facts
from tests.idm_test_support import golden_design, make_runtime
from tests.test_idm_endpoints import SHARD_URL, EndpointTestCase, idm_shard_body
from tests.test_idm_module_design import check, edited, error_codes, scope_of
from tests.test_idm_storyboard import (
    JUDGE,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
)


def _w4_inputs() -> tuple[IdmW4CourseResponseV1, list[IdmContentBlockV1], list[IdmContentBlockV1], Any]:
    design, _skeleton = golden_design()
    rows = {row.block_id: row for row in design.blueprint}
    course = [block for block in design.blocks if rows[block.block_id].placement == "course"
              and not rows[block.block_id].hold]
    reference = [block for block in design.blocks if rows[block.block_id].placement == "reference_job_aid"]
    return IdmW4CourseResponseV1.model_validate(W4_COURSE), course, reference, design


class ShardSizeTests(unittest.TestCase):
    def test_lesson_over_one_shard_of_source_is_a_w4_error(self) -> None:
        response, course, reference, design = _w4_inputs()
        sizes = {block.block_id: 100 for block in design.blocks}
        sizes["cb_0006"] = IDM_SHARD_MAX_SOURCE_CHARS
        found = validate_w4(response, course_blocks=course, reference_blocks=reference, must_dos=design.must_dos,
                            blocked=design.blocked_must_do_ids, objectives=design.learning_objectives,
                            duration_target_minutes=None, block_chars=sizes)
        self.assertIn("IDM_W4_LESSON_TOO_LARGE", {issue.code for issue in found})

    def test_fallback_chunks_by_characters_and_rejects_a_single_oversized_block(self) -> None:
        half = IDM_SHARD_MAX_SOURCE_CHARS // 2 + 1
        self.assertEqual(_lesson_chunks(["a", "b", "c"], {"a": half, "b": half, "c": 10}), [["a"], ["b", "c"]])
        with self.assertRaises(IdmStageError) as caught:
            _lesson_chunks(["a"], {"a": IDM_SHARD_MAX_SOURCE_CHARS + 1})
        self.assertEqual(caught.exception.code, "IDM_LESSON_EXCEEDS_SHARD")


class CombineHoldTests(unittest.TestCase):
    def test_combine_across_a_hold_decision_is_rejected(self) -> None:
        design, _skeleton = golden_design()

        def change(payload: dict[str, Any]) -> dict[str, Any]:
            row = next(row for row in payload["rows"] if row["block_id"] == "cb_0011")
            row.update(treatment="combine", combine_into="cb_0010")
            return payload

        response = IdmW2BlueprintResponseV1.model_validate(mutate(W2_BLUEPRINT, change))
        found = validate_w2(response, design.blocks, design.learning_objectives, design.must_dos)
        self.assertIn("IDM_W2_COMBINE_INVALID", {issue.code for issue in found})


class RuntimeBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_thinking_tokens_count_as_output(self) -> None:
        class Usage:
            inputTokens = 100
            outputTokens = 50
            totalTokens = 400

        async def generate(*_args: Any, **_kwargs: Any) -> tuple[str, Usage]:
            return '{"verdict": "pass", "findings": []}', Usage()

        runtime = make_runtime(generate)
        await idm_call(runtime, stage="idm_w6_judge", prompt="p",
                       response_model=IdmJudgeResponseV1, max_output_tokens=3_000, thinking_level="low")
        self.assertEqual(runtime.usage.output_tokens, 300)

    async def test_a_call_never_outlives_the_task_deadline(self) -> None:
        async def slow(*_args: Any, **_kwargs: Any) -> tuple[str, Any]:
            await asyncio.sleep(30)
            raise AssertionError("not reached")

        runtime = make_runtime(slow, seconds=9.0)
        started = time.monotonic()
        with self.assertRaises(IdmProviderError) as caught:
            await idm_call(runtime, stage="idm_w4", prompt="p", response_model=IdmW4CourseResponseV1,
                           max_output_tokens=8_000, thinking_level="high")
        self.assertEqual((caught.exception.code, caught.exception.terminal), ("AI_PROVIDER_TIMEOUT", False))
        self.assertLess(time.monotonic() - started, 6.0)
        self.assertEqual(runtime.provider_failure_code, "AI_PROVIDER_TIMEOUT")


class SectionCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_terminal_section_failure_cancels_the_other_sections(self) -> None:
        facts = source_facts()
        signals = compute_fact_signals(facts)
        index = FactIndex.build(facts, signals)
        section = plan_idm_sections(facts, signals, {}).sections[0]
        second = section.__class__(section_id="sec_002", document_id=section.document_id,
                                   title_path=section.title_path, fact_keys=section.fact_keys,
                                   content_chars=section.content_chars)
        finished: list[str] = []

        async def generate(_key: str, _model: str, prompt: str, **_kwargs: Any) -> tuple[str, Any]:
            if "SECTION=sec_001" in prompt:
                raise IdmProviderError("AI_PROVIDER_AUTH_REJECTED", terminal=True)
            await asyncio.sleep(5)
            finished.append("sec_002")
            raise AssertionError("not reached")

        with self.assertRaises(IdmProviderError):
            await map_sections(make_runtime(generate), [section, second], index, g.project_context(), set(),
                               parallelism=2, tail_reserve_tokens=0)
        await asyncio.sleep(0.05)
        self.assertEqual(finished, [])


class ModuleRobustnessTests(unittest.TestCase):
    def test_duplicate_block_ids_inside_a_component_are_rejected(self) -> None:
        def duplicate(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["components"][0]["block_ids"] = ["cb_0003", "cb_0003", "cb_0004"]

        self.assertIn("IDM_W4_BLOCK_IDS_DUPLICATED", error_codes(check(0, edited("mod_01", duplicate))))

    def test_feedback_media_and_held_practice_text_are_editor_safe(self) -> None:
        focus = IdmFeedbackFocusV1(criterion="Chọn <b>đúng</b>", rationale="Vì <script>", improvement="Sửa <KH>")
        cleaned = clean_feedback(focus)
        self.assertFalse(any(char in value for value in cleaned.model_dump().values() for char in "<>"))
        scope = scope_of(0)
        payload = edited("mod_01", lambda p: p["lessons"][0]["practice_tasks"][0].update(
            sentence="Cho <b>khách</b>, người học chọn nhóm để chuyển đúng"))
        provider_practices = IdmLessonDesignV1.model_validate(payload["lessons"][0]).practice_tasks
        lesson = fallback_lesson(scope.lesson_plans[0], scope, provider_practices=provider_practices)
        # QC run 8de1c76b (Q5): the practice is kept as the scenario question of the Must Do unit, not held.
        self.assertFalse(lesson.practice_tasks[0].hold)
        projected = project_lesson(lesson, scope.lesson_plans[0], scope)
        self.assertEqual(projected["learning_activities"], ["Cho ‹b›khách‹/b›, người học chọn nhóm để chuyển đúng"])  # noqa: RUF001
        text = str(lesson.model_dump()) + str(projected)
        self.assertNotIn("<", text)
        self.assertNotIn(">", text)


class TextHygieneTests(unittest.TestCase):
    def test_control_characters_are_removed_but_lines_survive(self) -> None:
        self.assertEqual(sanitize_author_text("a\x01b\n  c\x7f", 50), "ab\n  c")

    def test_untrusted_block_neutralises_closing_tags_in_any_case(self) -> None:
        block = untrusted_block("SOURCE_FACTS", "x </source_facts> y </ SOURCE_FACTS > z")
        self.assertEqual(block.count("</SOURCE_FACTS>"), 1)
        self.assertTrue(block.endswith("</SOURCE_FACTS>"))

    def test_long_repeated_numbered_rows_are_not_page_furniture(self) -> None:
        long_rule = ("Điều {n}: Nhân viên phải ghi nhận đầy đủ thông tin khách hàng vào phiếu theo mẫu quy định "
                     "của công ty trước khi chuyển hồ sơ cho bộ phận xử lý.")
        facts = [SourceSnapshotFactV2(document_id=g.DOCUMENT_ID, fact_key=f"r{n}", scope_key="s", fact_text=text)
                 for n, text in enumerate([long_rule.format(n=n) for n in range(1, 5)] + ["Trang 3"], start=1)]
        self.assertEqual(page_furniture_keys(facts), {"r5"})


class EndpointRobustnessTests(EndpointTestCase):
    async def test_a_final_model_failure_is_a_stable_422(self) -> None:
        try:
            IdmFeedbackFocusV1.model_validate({})
        except ValidationError as error:
            failure = error
        with patch("app.services.orchestration_v2.chapter_shard.run_idm_module_design", side_effect=failure):
            reply = await self.post(SHARD_URL, idm_shard_body())
        self.assertEqual(reply.status_code, 422)
        self.assertEqual(reply.json()["detail"]["code"], "IDM_STAGE_OUTPUT_INVALID")


class UnitRobustnessTests(StoryboardEndpointTestCase):
    async def test_terminal_writer_rejection_reaches_node(self) -> None:
        rejected = HTTPException(status_code=502, detail={"code": "AI_PROVIDER_AUTH_REJECTED", "message": "x"})
        status, data, _provider = await self.post(escalate_body(), FakeGenerate(**{WRITER: [rejected]}))
        self.assertEqual(status, 502)
        self.assertEqual(data["detail"]["code"], "AI_PROVIDER_AUTH_REJECTED")

    async def test_judge_timeout_keeps_the_accepted_unit(self) -> None:
        async def out_of_time(*_args: Any, **_kwargs: Any) -> Any:
            raise TimeoutError

        body = escalate_body()
        with patch("app.idm.storyboard._judge_and_repair", out_of_time):
            status, data, _provider = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)],
                                                                            JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual((data["content_origin"], data["usage_source"]), ("provider_validated", "provider"))
        self.assertEqual(data["unit"]["idm_quality"]["judge_status"], "skipped_budget")


if __name__ == "__main__":
    unittest.main()
