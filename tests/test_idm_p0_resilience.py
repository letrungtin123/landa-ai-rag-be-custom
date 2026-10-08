"""P0 fixes after the QC of course 234653 (2026-10-08): cut answers, bounded answers, 429 handling.

* D1: W1-reduce/W2/W4 answers were cut at ``max_output_tokens`` (thinking counts against it) and
  the repair re-sent the same prompt; caps, thinking levels and a shorter repair fix it.
* D3: one over-long string rejected a whole answer.
* D6: an exhausted key or a rate limit silently produced a deterministic course.

The provider is always a local fake; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

from app import main
from app.idm import runtime as idm_runtime
from app.idm.coerce import (
    CLAMPED_NUMBER_CODE,
    TRIMMED_LIST_CODE,
    TRIMMED_LONG_STRING_CODE,
    TRIMMED_STRING_CODE,
    coerce_provider_answer,
)
from app.idm.contracts import (
    IdmCourseSkeletonRequestV1,
    IdmLessonDesignV1,
    IdmW1ReduceResponseV1,
    IdmW1SectionResponseV1,
    IdmW4CourseResponseV1,
)
from app.idm.course_design import run_idm_course_design
from app.idm.policy import (
    IDM_W1_REDUCE_MAX_OUTPUT_TOKENS,
    IDM_W2_MAX_OUTPUT_TOKENS,
    IDM_W4_COURSE_MAX_OUTPUT_TOKENS,
    THINKING_W1_REDUCE,
)
from app.idm.prompts import COMPACT_W1_REDUCE, answer_repair, w1_reduce_prompt
from app.idm.runtime import (
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    IdmTokenAllowance,
    idm_call,
    idm_generate,
    idm_tail_reserve,
    minimum_admitted_output,
    repair_thinking,
)
from app.infra.provider_limits import classify_provider_limit
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorChapterShardV2Request
from app.services import provider as provider_service
from app.services.orchestration_v2 import idm as v2_idm
from app.services.orchestration_v2.common import ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES
from tests import idm_golden as g
from tests.idm_test_support import ALLOWANCE, assert_node_trace, make_runtime
from tests.test_idm_endpoints import (
    SHARD_URL,
    SKELETON_URL,
    EndpointTestCase,
    idm_shard_body,
    idm_skeleton_body,
    legacy_shard_body,
    legacy_skeleton_body,
)
from tests.test_idm_storyboard import WRITER, FakeGenerate, StoryboardEndpointTestCase, severity_body, severity_writer

CUT = '{"merges": [], "conflicts": [], "target_audience": {"description": "Nhân viên'


class Usage:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int | None = None) -> None:
        self.inputTokens = input_tokens
        self.outputTokens = output_tokens
        self.totalTokens = input_tokens + output_tokens if total_tokens is None else total_tokens


class Answer(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    value: str


def runtime(generate: Any, *, seconds: float = 100.0, wait_ms: int = 0) -> IdmRuntime:
    return IdmRuntime(generate=generate, api_key="secret-key", model="m", locale="vi", deadline=seconds,
                      clock=lambda: 0.0, rate_limit_max_wait_ms=wait_ms)


class TelemetryProvider:
    """Wraps a ``FakeIdmProvider``; the first call of ``cut_schema`` is cut at the output limit."""

    def __init__(self, inner: g.FakeIdmProvider, cut_schema: str) -> None:
        self.inner, self.cut_schema, self.cut_done = inner, cut_schema, False
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, Any]:
        name = options["response_schema"].model_json_schema().get("title")
        self.calls.append({"schema": name, "prompt": prompt, **{k: v for k, v in options.items()
                                                                 if k != "on_provider_telemetry"}})
        if name == self.cut_schema and not self.cut_done:
            self.cut_done = True
            options["on_provider_telemetry"]({"provider_finish_reason": "FinishReason.MAX_TOKENS",
                                              "provider_input_tokens": 900, "provider_output_tokens": 300,
                                              "provider_total_tokens": 900 + options["max_output_tokens"]})
            return CUT, Usage(900, 300, 900 + options["max_output_tokens"])
        return await self.inner(api_key, model, prompt, **options)


def skeleton_request() -> IdmCourseSkeletonRequestV1:
    return IdmCourseSkeletonRequestV1.model_validate({
        "pipeline_version": "idm-1", "project_context": g.project_context(),
        "source_facts": [fact.model_dump() for fact in g.source_facts()],
        "token_allowance": {"input_tokens": ALLOWANCE[0], "output_tokens": ALLOWANCE[1]},
        "remaining_budget_ms": 500_000,
    })


async def design(provider: Any) -> dict[str, Any]:
    return await run_idm_course_design(skeleton_request(), source_snapshot_hash=g.SNAPSHOT_HASH,
                                       runtime=make_runtime(provider), parallelism=4, max_sections=16)


class TruncationTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_that_spent_the_whole_cap_is_truncated_without_a_finish_reason(self) -> None:
        async def provider(*_args: Any, **options: Any) -> tuple[str, Usage]:
            return '{"value": "cut', Usage(100, 40, 100 + options["max_output_tokens"])

        item = runtime(provider)
        with self.assertRaises(IdmResponseInvalidError) as caught:
            await idm_call(item, stage="idm_w2", prompt="p", response_model=Answer, max_output_tokens=2_000,
                           thinking_level="medium")
        self.assertEqual(caught.exception.code, "IDM_RESPONSE_TRUNCATED")
        self.assertEqual(item.trace[0]["failure_code"], "IDM_RESPONSE_TRUNCATED")
        assert_node_trace(self, item.trace)

    async def test_custom_parser_rejection_of_a_cut_answer_is_truncated_and_keeps_its_errors(self) -> None:
        async def provider(*_args: Any, **options: Any) -> tuple[str, Usage]:
            options["on_provider_telemetry"]({"provider_finish_reason": "MAX_TOKENS"})
            return "{", Usage(10, 5)

        def parse(_text: str) -> Any:
            raise IdmResponseInvalidError("IDM_W5_INSTANCE_INVALID", [{"type": "x", "loc": ["c0"]}])

        with self.assertRaises(IdmResponseInvalidError) as caught:
            await idm_generate(runtime(provider), stage="idm_w5_writer", prompt="p", response_schema=Answer,
                               parse=parse, max_output_tokens=16_000, thinking_level="medium")
        self.assertEqual((caught.exception.code, caught.exception.errors),
                         ("IDM_RESPONSE_TRUNCATED", [{"type": "x", "loc": ["c0"]}]))

    async def test_a_complete_answer_at_the_cap_is_accepted(self) -> None:
        async def provider(*_args: Any, **options: Any) -> tuple[str, Usage]:
            return '{"value": "ok"}', Usage(10, options["max_output_tokens"])

        result = await idm_call(runtime(provider), stage="idm_w2", prompt="p", response_model=Answer,
                                max_output_tokens=500, thinking_level="medium")
        self.assertEqual(result.value, "ok")

    def test_repair_after_truncation_asks_for_less_with_less_thinking(self) -> None:
        suffix = answer_repair("IDM_RESPONSE_TRUNCATED", [{"type": "json_invalid", "loc": []}], COMPACT_W1_REDUCE)
        self.assertIn("cut off at the output limit", suffix)
        self.assertIn(COMPACT_W1_REDUCE, suffix)
        self.assertNotIn("json_invalid", suffix)
        self.assertEqual(repair_thinking("IDM_RESPONSE_TRUNCATED", "medium"), "low")
        self.assertEqual(repair_thinking("IDM_RESPONSE_SCHEMA_INVALID", "medium"), "medium")
        schema_suffix = answer_repair("IDM_RESPONSE_SCHEMA_INVALID", [{"type": "string_too_long", "loc": ["a", 1]}],
                                      COMPACT_W1_REDUCE)
        self.assertIn('"path":"a.1"', schema_suffix)

    async def test_cut_w1_reduce_is_repaired_shorter_instead_of_falling_back(self) -> None:
        provider = TelemetryProvider(g.golden_provider(), "IdmW1ReduceResponseV1")
        result = await design(provider)
        reduce_calls = [call for call in provider.calls if call["schema"] == "IdmW1ReduceResponseV1"]
        self.assertEqual(len(reduce_calls), 2)
        self.assertEqual([call["thinking_level"] for call in reduce_calls], [THINKING_W1_REDUCE, "low"])
        self.assertIn("cut off at the output limit", reduce_calls[1]["prompt"])
        self.assertIn(COMPACT_W1_REDUCE, reduce_calls[1]["prompt"])
        self.assertEqual(result["idm"]["stage_origins"]["w1_reduce"], "provider")
        self.assertEqual(result["content_origin"], "provider_validated")

    def test_later_stages_reserve_only_their_minimum_share_of_the_allowance(self) -> None:
        tail = idm_tail_reserve(IDM_W1_REDUCE_MAX_OUTPUT_TOKENS, IDM_W2_MAX_OUTPUT_TOKENS,
                                IDM_W4_COURSE_MAX_OUTPUT_TOKENS)
        self.assertEqual(tail, (IDM_W1_REDUCE_MAX_OUTPUT_TOKENS + IDM_W2_MAX_OUTPUT_TOKENS
                                + IDM_W4_COURSE_MAX_OUTPUT_TOKENS) // 2)
        self.assertEqual(minimum_admitted_output(1), 1)
        # Four parallel W1-map calls (12k each) plus the tail fit the 131,072-token task allowance.
        self.assertLessEqual(4 * 12_000 + tail, ALLOWANCE[1])

    def test_w1_reduce_prompt_asks_only_for_serving_links(self) -> None:
        prompt = w1_reduce_prompt("vi", project_context={}, catalog=[])
        self.assertIn("relation direct (needed to perform it) or", prompt)
        self.assertNotIn("relation direct | supporting | context | unrelated | unknown", prompt)


class CoercionTests(unittest.TestCase):
    def section_answer(self, **block: Any) -> dict[str, Any]:
        base = {"local_id": "b1", "name": "Khi nào phải chuyển khiếu nại cho quản lý?", "summary": "Tóm tắt.",
                "intent": "know", "support_role": None, "content_kind": "concept", "fact_keys": ["k1"],
                "issues": [], "gaps": [], "sme_questions": []}
        return {"blocks": [{**base, **block}], "noise": []}

    def test_slightly_long_text_and_extra_sme_questions_are_trimmed_and_counted(self) -> None:
        value = self.section_answer(summary="x" * 320, sme_questions=[f"Câu hỏi SME số {n} cho chuyên gia?"
                                                                      for n in range(7)])
        adjusted, codes = coerce_provider_answer(IdmW1SectionResponseV1, value)
        block = IdmW1SectionResponseV1.model_validate(adjusted).blocks[0]
        self.assertEqual((len(block.summary), len(block.sme_questions)), (300, 5))
        self.assertEqual(dict(codes), {TRIMMED_STRING_CODE: 1, TRIMMED_LIST_CODE: 1})

    def test_identifiers_and_structure_stay_strict(self) -> None:
        value = self.section_answer(local_id="b" + "1" * 300, fact_keys=["k" * 300])
        value["blocks"] = value["blocks"] * 41
        adjusted, codes = coerce_provider_answer(IdmW1SectionResponseV1, value)
        self.assertEqual(adjusted, value)
        self.assertEqual(dict(codes), {})

    def test_far_too_long_text_is_cut_at_a_boundary_and_counted_apart(self) -> None:
        # QC course 364564 (N1): a far longer string used to stay as it was and reject the whole answer.
        sentence = "Người học điền từng ô của phiếu theo đúng thứ tự các bước đã học. "
        words = "Người học ghi rõ mục tiêu, người phụ trách và thời hạn của từng việc " * 30
        value = self.section_answer(summary=sentence * 30, name=words)
        adjusted, codes = coerce_provider_answer(IdmW1SectionResponseV1, value)
        block = IdmW1SectionResponseV1.model_validate(adjusted).blocks[0]
        self.assertEqual(dict(codes), {TRIMMED_LONG_STRING_CODE: 2})
        self.assertLessEqual(len(block.summary), 300)
        self.assertTrue(block.summary.endswith("các bước đã học."))  # the last whole sentence
        self.assertGreater(len(block.summary), 150)
        self.assertLessEqual(len(block.name), 180)
        self.assertTrue(block.name.endswith("…"))  # no sentence end: the last whole word
        head = block.name[:-1]
        self.assertTrue(words.startswith(head))
        self.assertIn(words[len(head)], " ,")  # cut between two words

    def test_estimates_are_clamped_and_lesson_objective_strings_trimmed(self) -> None:
        lesson = {"lesson_key": "lsn_001", "kind": "learning", "title": "Phân loại khiếu nại",
                  "primary_must_do_id": "md_1", "secondary_must_do_ids": [], "block_ids": ["cb_0001"],
                  "est_screens": 1, "est_minutes": 75, "ordering_rationale": "Trước tiên."}
        course = {"course_title": "Khoá học", "course_summary": "Tóm tắt khoá học đủ hai mươi ký tự.",
                  "assessment_strategy": "Luyện tập cuối mỗi mục.", "prerequisites": [],
                  "modules": [{"module_key": "mod_01", "title": "Mô-đun", "performance_goal": "Mục tiêu hiệu suất.",
                               "lo_ids": ["lo_1"], "lessons": [lesson]}]}
        adjusted, codes = coerce_provider_answer(IdmW4CourseResponseV1, course)
        plan = IdmW4CourseResponseV1.model_validate(adjusted).modules[0].lessons[0]
        self.assertEqual((plan.est_screens, plan.est_minutes), (2, 60))
        self.assertEqual(codes[CLAMPED_NUMBER_CODE], 2)
        objectives = {"learning_objectives": [f"Mục tiêu {n}" for n in range(9)]}
        adjusted_lesson, _ = coerce_provider_answer(IdmLessonDesignV1, objectives)
        self.assertEqual(len(adjusted_lesson["learning_objectives"]), 8)
        reduce = {"learning_objectives": [{"lo_id": f"lo_{n}"} for n in range(1, 10)]}
        self.assertEqual(coerce_provider_answer(IdmW1ReduceResponseV1, reduce)[0], reduce)

    def test_idm_call_accepts_a_301_character_reason_and_records_the_adjustment(self) -> None:
        answer = json.loads(json.dumps(g.W1_REDUCE))
        answer["merges"] = [{"keep_block_id": "cb_0001", "merged_block_ids": ["cb_0002"], "reason": "r" * 301}]

        async def provider(*_args: Any, **_options: Any) -> tuple[str, Usage]:
            return json.dumps(answer), Usage(10, 10)

        item = runtime(provider)
        result = asyncio.run(idm_call(item, stage="idm_w1_reduce", prompt="p", response_model=IdmW1ReduceResponseV1,
                                      max_output_tokens=24_000, thinking_level="medium"))
        self.assertEqual(len(result.merges[0].reason), 300)
        self.assertEqual(item.adjustments[TRIMMED_STRING_CODE], 1)


class ProviderStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_key_stops_the_course_design_instead_of_a_fallback_course(self) -> None:
        quota = IdmProviderError("AI_PROVIDER_QUOTA_EXHAUSTED", terminal=True, http_status=503)
        provider = g.golden_provider({"IdmW1ReduceResponseV1": [quota]})
        with self.assertRaises(IdmProviderError) as caught:
            await design(provider)
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("AI_PROVIDER_QUOTA_EXHAUSTED", 503))

    async def test_rate_limit_wait_is_bounded_by_the_setting_and_the_deadline(self) -> None:
        seen: list[int | None] = []

        async def provider(*_args: Any, **options: Any) -> tuple[str, Usage]:
            seen.append(options["rate_limit_max_wait_ms"])
            return '{"value": "ok"}', Usage(1, 1)

        for seconds, wait_ms, expected in ((200.0, 60_000, 60_000), (40.0, 60_000, 27_000), (9.0, 60_000, 0),
                                           (200.0, 0, None)):
            await idm_call(runtime(provider, seconds=seconds, wait_ms=wait_ms), stage="s", prompt="p",
                           response_model=Answer, max_output_tokens=100, thinking_level="low")
            self.assertEqual(seen[-1], expected)

    async def test_deadline_spent_waiting_on_a_rate_limit_is_rate_limited_not_a_timeout(self) -> None:
        async def provider(*_args: Any, **options: Any) -> tuple[str, Usage]:
            options["on_provider_telemetry"]({"event": "provider_rate_limited_retry", "retry_after_ms": 30_000})
            await asyncio.sleep(10)
            return "{}", Usage(1, 1)

        loop = asyncio.get_running_loop()
        item = IdmRuntime(generate=provider, api_key="k", model="m", locale="vi",  # type: ignore[arg-type]
                          deadline=loop.time() + 8.5, clock=loop.time)
        # 8.5 s left minus 8.4 s headroom: the outer bound fires after ~0.1 s, inside the rate-limit wait.
        with (patch.object(idm_runtime, "IDM_CALL_HEADROOM_SECONDS", 8.4),
              self.assertRaises(IdmProviderError) as caught):
            await idm_call(item, stage="s", prompt="p", response_model=Answer, max_output_tokens=10,
                           thinking_level="low")
        self.assertEqual((caught.exception.code, caught.exception.terminal, caught.exception.http_status),
                         ("AI_PROVIDER_RATE_LIMITED", True, 503))
        self.assertIsNone(item.provider_failure_code)

    def test_runtime_takes_the_rate_limit_wait_from_settings(self) -> None:
        request = RagLessonAuthorChapterShardV2Request.model_validate(idm_shard_body())
        with patch.object(main.settings, "provider_rate_limit_max_wait_ms", 45_000):
            item = v2_idm._idm_runtime(request, budget_ms=60_000, allowance=None)
        self.assertEqual(item.rate_limit_max_wait_ms, 45_000)
        self.assertEqual(type(main.settings).model_fields["provider_rate_limit_max_wait_ms"].default, 60_000)


class ProviderStopEndpointTests(EndpointTestCase):
    def provider_failure(self, code: str) -> AsyncMock:
        return AsyncMock(side_effect=HTTPException(503, {"code": code, "message": "private"}))

    async def test_idm_course_skeleton_and_chapter_shard_report_quota_and_rate_limit(self) -> None:
        for url, body in ((SKELETON_URL, idm_skeleton_body()), (SHARD_URL, idm_shard_body())):
            for code in ("AI_PROVIDER_QUOTA_EXHAUSTED", "AI_PROVIDER_RATE_LIMITED"):
                with self.subTest(url=url, code=code):
                    reply = await self.post(url, body, self.provider_failure(code))
                    self.assertEqual((reply.status_code, reply.json()["detail"]["code"]), (503, code))

    async def test_legacy_v2_planning_still_falls_back_on_a_rate_limit(self) -> None:
        for url, body in ((SKELETON_URL, legacy_skeleton_body()), (SHARD_URL, legacy_shard_body())):
            with self.subTest(url=url):
                reply = await self.post(url, body, self.provider_failure("AI_PROVIDER_RATE_LIMITED"))
                self.assertEqual(reply.status_code, 200)
                data = reply.json()
                self.assertEqual((data["content_origin"], data["usage_source"]),
                                 ("structured_fallback", "reserved_upper_bound"))


class UnitProviderStopTests(StoryboardEndpointTestCase):
    async def test_quota_and_rate_limit_reach_node_instead_of_a_fallback_unit(self) -> None:
        for code in ("AI_PROVIDER_QUOTA_EXHAUSTED", "AI_PROVIDER_RATE_LIMITED"):
            with self.subTest(code=code):
                failure = HTTPException(503, {"code": code, "message": "private"})
                status, data, _provider = await self.post(severity_body(), FakeGenerate(**{WRITER: [failure]}))
                self.assertEqual((status, data["detail"]["code"]), (503, code))

    async def test_fallback_only_attempt_needs_no_provider(self) -> None:
        body = {**severity_body(), "fallback_only": True}
        status, data, provider = await self.post(body, FakeGenerate())
        self.assertEqual((status, data["usage_source"], provider.names), (200, "deterministic_fallback", []))

    async def test_cut_writer_answer_is_retried_shorter_with_low_thinking(self) -> None:
        body = severity_body()
        answers: list[Callable[[dict[str, Any]], str]] = [
            lambda options: (options["on_provider_telemetry"]({"provider_finish_reason": "MAX_TOKENS"})
                             or '{"components": {"c0": {'),
            lambda _options: json.dumps(severity_writer(body), ensure_ascii=False),
        ]
        calls: list[dict[str, Any]] = []

        async def generate(_key: str, _model: str, prompt: str, **options: Any) -> tuple[str, Any]:
            name = options["response_schema"].__name__
            if not name.startswith(WRITER):
                return json.dumps({"verdict": "pass", "findings": []}), AiUsage(totalTokens=2)
            calls.append({"prompt": prompt, **options})
            return answers.pop(0)(options), AiUsage(inputTokens=100, outputTokens=50, totalTokens=150)

        with patch("app.services.provider.generate_content", generate), \
                patch.object(main.settings, "idm_judge_mode", "off"):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
                reply = await client.post("/v1/lesson-author/orchestration-v2/unit", json=body)
        self.assertEqual(reply.status_code, 200)
        self.assertEqual([call["thinking_level"] for call in calls], ["medium", "low"])
        self.assertIn("cut off at the output limit", calls[1]["prompt"])
        self.assertEqual(reply.json()["content_origin"], "provider_validated")


class ProviderLimitClassificationTests(unittest.TestCase):
    def error(self, message: str, details: Any = None) -> Exception:
        error = Exception(message)
        error.details = details  # type: ignore[attr-defined]
        return error

    def test_billing_and_daily_quota_are_exhausted_even_with_a_retry_hint(self) -> None:
        for message in ("429 RESOURCE_EXHAUSTED. Your prepayment credits are depleted. Please go to AI Studio.",
                        "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly spending cap.",
                        "429 Billing is disabled for this project."):
            with self.subTest(message=message):
                limit = classify_provider_limit(self.error(message), 5.0)
                self.assertEqual((limit.kind, limit.code, limit.retry_after_seconds),
                                 ("quota_exhausted", "AI_PROVIDER_QUOTA_EXHAUSTED", None))
        daily = {"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}
        self.assertEqual(classify_provider_limit(self.error("429 quota", daily), 30.0).kind, "quota_exhausted")

    def test_per_minute_limits_are_rate_limits_with_their_hint(self) -> None:
        minute = {"error": {"details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}}
        message = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing "
                   "details. Please retry in 41.5s.")
        limit = classify_provider_limit(self.error(message, minute), None)
        self.assertEqual((limit.kind, limit.code, limit.retry_after_seconds),
                         ("rate_limited", "AI_PROVIDER_RATE_LIMITED", 41.5))
        self.assertEqual(classify_provider_limit(self.error("Please retry in 500ms"), None).retry_after_seconds, 0.5)
        self.assertEqual(classify_provider_limit(self.error("429 RESOURCE_EXHAUSTED"), 2.0).kind, "rate_limited")
        self.assertEqual(classify_provider_limit(self.error("429 Too Many Requests"), None).kind, "rate_limited")

    def test_bare_resource_exhausted_keeps_its_legacy_meaning(self) -> None:
        self.assertEqual(classify_provider_limit(self.error("429 RESOURCE_EXHAUSTED"), None).kind,
                         "quota_exhausted")


class RateLimited(Exception):
    status_code = 429

    def __init__(self, message: str, delay: str | None = "1s") -> None:
        super().__init__(message)
        retry = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}] if delay else []
        self.details = {"error": {"message": message, "details": retry}}


class ProviderRetryLoopTests(unittest.TestCase):
    def call(self, answers: list[Any], **options: Any) -> tuple[Any, list[float], list[dict[str, Any]]]:
        pending = list(answers)
        sleeps: list[float] = []
        events: list[dict[str, Any]] = []

        def run() -> Any:
            answer = pending.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with patch("app.services.provider.asyncio.sleep", sleep):
            try:
                result: Any = asyncio.run(provider_service.call_provider_with_timeout(
                    run, "m", on_provider_diagnostic=events.append, **options))
            except HTTPException as error:
                result = error
        return result, sleeps, events

    def test_idm_waits_out_per_minute_limits_without_using_transient_attempts(self) -> None:
        result, sleeps, events = self.call([RateLimited("PRIVATE per minute")] * 3 + ["ok"],
                                           rate_limit_max_wait_ms=60_000)
        self.assertEqual((result, sleeps), ("ok", [1.0, 1.0, 1.0]))
        retries = [event for event in events if event["event"] == "provider_rate_limited_retry"]
        self.assertEqual([event["provider_limit_category"] for event in retries], ["rate_limited"] * 3)
        self.assertNotIn("PRIVATE", json.dumps(events))

    def test_idm_rate_limit_that_outlives_the_bound_is_reported_as_rate_limited(self) -> None:
        with self.assertLogs("app.main", "WARNING") as logs:
            result, sleeps, _ = self.call([RateLimited("PRIVATE", "25s")] * 5, rate_limit_max_wait_ms=60_000)
        self.assertIsInstance(result, HTTPException)
        self.assertEqual((result.status_code, result.detail["code"]), (503, "AI_PROVIDER_RATE_LIMITED"))
        self.assertEqual(sleeps, [25.0, 25.0])
        self.assertNotIn("PRIVATE", "\n".join(logs.output))
        self.assertIn("category=rate_limited retry_after_s=25.0", "\n".join(logs.output))

    def test_legacy_calls_keep_one_short_retry_and_report_long_limits_at_once(self) -> None:
        result, sleeps, _ = self.call([RateLimited("x", "0.5s"), "ok"])
        self.assertEqual((result, sleeps), ("ok", [0.5]))
        result, sleeps, _ = self.call([RateLimited("x", "30s")])
        self.assertEqual((result.detail["code"], sleeps), ("AI_PROVIDER_RATE_LIMITED", []))

    def test_exhausted_key_is_never_retried(self) -> None:
        result, sleeps, events = self.call([RateLimited("Your prepayment credits are depleted", "0.5s")],
                                           rate_limit_max_wait_ms=60_000)
        self.assertEqual((result.detail["code"], sleeps), ("AI_PROVIDER_QUOTA_EXHAUSTED", []))
        self.assertEqual(events[-1]["provider_limit_category"], "quota_exhausted")

    def test_legacy_code_sets_treat_a_rate_limit_like_the_quota_code_they_knew(self) -> None:
        error = HTTPException(503, {"code": "AI_PROVIDER_RATE_LIMITED"})
        self.assertTrue(provider_service.is_non_retryable_provider_error(error))
        self.assertIn("AI_PROVIDER_RATE_LIMITED", ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES)


class TokenAllowanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_golden_design_fits_the_task_allowance_with_full_caps(self) -> None:
        provider = g.golden_provider()
        item = make_runtime(provider, allowance=ALLOWANCE)
        await run_idm_course_design(skeleton_request(), source_snapshot_hash=g.SNAPSHOT_HASH, runtime=item,
                                    parallelism=4, max_sections=16)
        self.assertEqual([call["max_output_tokens"] for call in provider.calls], [12_000, 24_000, 32_000, 16_000])
        self.assertIsInstance(item.token_allowance, IdmTokenAllowance)


if __name__ == "__main__":
    unittest.main()
