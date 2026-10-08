"""IDM dispatch in the three orchestration-v2 endpoints (spec §11.2-§11.4, §15.1).

``app.main.generate_content`` is replaced by local fakes; nothing reaches the network.
"""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from app import main
from app.api import deps as api_deps
from app.core.errors import AppError
from app.idm.contracts import IdmCourseDesignV1, IdmShardDesignV1
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmStageError
from app.lesson_author_orchestration_v2 import ChapterBlueprintShardV2, CourseSkeletonV2
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorChapterShardV2Request, RagLessonAuthorCourseSkeletonV2Request
from app.services.orchestration_v2 import chapter_shard as chapter_shard_service
from app.services.orchestration_v2 import course_skeleton as course_skeleton_service
from app.services.orchestration_v2 import idm as v2_idm
from tests import idm_golden as g
from tests import idm_golden_module as gm
from tests.idm_test_support import golden_design, module_inputs, module_response
from tests.test_lesson_author_orchestration_v2_endpoints import (
    SOURCE_HASH,
    common,
    lesson_wire,
    model_designed_authority,
    skeleton_wire,
)

SKELETON_URL = "/v1/lesson-author/orchestration-v2/course-skeleton"
SHARD_URL = "/v1/lesson-author/orchestration-v2/chapter-shard"
MODULE = "IdmW3W4ModuleResponseV1"


def provider_error(status: int) -> Exception:
    return type("ProviderError", (Exception,), {"status_code": status})("PRIVATE_PROVIDER_BODY")


def idm_skeleton_body(facts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    facts = facts or [fact.model_dump(mode="json") for fact in g.source_facts()]
    scopes = sorted({fact["scope_key"] for fact in facts})
    return {
        **common(), "source_snapshot_hash": g.SNAPSHOT_HASH, "source_authority": model_designed_authority(),
        "scope_catalog": [{"scope_key": scope, "title": scope, "source_ref": None, "fact_count": 1,
                           "content_chars": 5} for scope in scopes],
        "idm": {"pipeline_version": "idm-1", "project_context": g.project_context(), "source_facts": facts,
                "token_allowance": {"input_tokens": 1_800_000, "output_tokens": 131_072},
                "remaining_budget_ms": 500_000},
    }


def idm_shard_body(chapter_index: int = 1) -> dict[str, Any]:
    context, plan, facts = module_inputs(chapter_index)
    _, skeleton = golden_design()
    return {**common(), "skeleton": skeleton.model_dump(mode="json"), "shard_plan": plan.model_dump(mode="json"),
            "source_facts": [fact.model_dump(mode="json") for fact in facts],
            "idm_module_context": context.model_dump(mode="json")}


def legacy_skeleton_body() -> dict[str, Any]:
    return {**common(), "source_snapshot_hash": SOURCE_HASH, "source_authority": model_designed_authority(),
            "scope_catalog": [{"scope_key": "scope-1", "title": "Mục 1", "source_ref": None, "fact_count": 1,
                               "content_chars": 5}], "max_attempts": 2}


def legacy_shard_body() -> dict[str, Any]:
    return {**common(), "skeleton": skeleton_wire(),
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1"], "source_fact_count": 1, "source_content_chars": 5},
            "source_facts": [{"document_id": "document-1", "fact_key": "fact-1", "scope_key": "scope-1",
                              "fact_text": "Alpha", "source_ref": None, "source_page": 1, "source_chunk": 0,
                              "locator": {}}], "max_attempts": 2}


class EndpointTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None

    def tearDown(self) -> None:
        main.app.dependency_overrides.pop(api_deps.require_internal_token, None)

    async def post(self, url: str, body: dict[str, Any], provider: Any = None) -> httpx.Response:
        fake = provider or AsyncMock(side_effect=AssertionError("no call"))
        with patch("app.services.provider.generate_content", fake):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
                return await client.post(url, json=body)


class CourseSkeletonDispatchTests(EndpointTestCase):
    async def test_idm_request_returns_skeleton_and_design(self) -> None:
        provider = g.golden_provider()
        reply = await self.post(SKELETON_URL, idm_skeleton_body(), provider)
        self.assertEqual(reply.status_code, 200)
        data = reply.json()
        CourseSkeletonV2.model_validate(data["skeleton"])
        design = IdmCourseDesignV1.model_validate(data["idm"])
        self.assertEqual(design.design_hash, golden_design()[0].design_hash)
        self.assertEqual((data["content_origin"], data["usage_source"]), ("provider_validated", "provider"))
        self.assertEqual([call["schema"] for call in provider.calls],
                         ["IdmW1SectionResponseV1", "IdmW1ReduceResponseV1", "IdmW2BlueprintResponseV1",
                          "IdmW4CourseResponseV1"])
        self.assertTrue(all(call["json_mode"] for call in provider.calls))
        self.assertEqual({call["thinking_level"] for call in provider.calls[1:]}, {"medium"})

    # Regression (fixed): IdmCapacityError subclasses ValueError, not IdmError, so the course-skeleton wrapper
    # (``except IdmError``) lets it escape and the request ends as HTTP 500 INTERNAL_ERROR instead of
    # the spec's 422 IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY (Appendix C, §5.5).
    # Evidence: app/idm/signals.py ``class IdmCapacityError(ValueError)``; app/main.py
    # lesson_author_orchestration_v2_course_skeleton only catches IdmError.
    async def test_source_over_capacity_is_422(self) -> None:
        facts = [g.source_facts()[0].model_copy(update={"fact_key": f"big-{n}", "fact_text": "x" * 32_000})
                 .model_dump(mode="json") for n in range(13)]
        reply = await self.post(SKELETON_URL, idm_skeleton_body(facts))
        self.assertEqual(reply.status_code, 422)
        self.assertEqual(reply.json()["detail"]["code"], "IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY")

    async def test_terminal_provider_rejection_is_502(self) -> None:
        with self.assertLogs("app.main", "WARNING") as logs:
            reply = await self.post(SKELETON_URL, idm_skeleton_body(), AsyncMock(side_effect=provider_error(400)))
        self.assertEqual(reply.status_code, 502)
        self.assertEqual(reply.json()["detail"]["code"], "AI_PROVIDER_REQUEST_REJECTED")
        self.assertNotIn("PRIVATE_PROVIDER_BODY", "\n".join(logs.output) + reply.text)

    async def test_transient_provider_failure_returns_a_reviewable_design(self) -> None:
        failure = HTTPException(status_code=503, detail={"code": "AI_PROVIDER_UNAVAILABLE", "message": "private"})
        reply = await self.post(SKELETON_URL, idm_skeleton_body(), AsyncMock(side_effect=failure))
        self.assertEqual(reply.status_code, 200)
        data = reply.json()
        self.assertEqual((data["content_origin"], data["quality_state"], data["usage_source"]),
                         ("structured_fallback", "review_required", "reserved_upper_bound"))
        IdmCourseDesignV1.model_validate(data["idm"])

    async def test_locale_mismatch_is_422(self) -> None:
        body = idm_skeleton_body()
        body["idm"]["project_context"]["locale"] = "en"
        reply = await self.post(SKELETON_URL, body)
        self.assertEqual((reply.status_code, reply.json()["detail"]["code"]),
                         (422, "ORCHESTRATION_V2_CONTRACT_INVALID"))


class ChapterShardDispatchTests(EndpointTestCase):
    async def test_idm_request_returns_shard_with_design(self) -> None:
        provider = g.FakeIdmProvider({MODULE: [module_response("mod_02")]})
        reply = await self.post(SHARD_URL, idm_shard_body(), provider)
        self.assertEqual(reply.status_code, 200)
        data = reply.json()
        shard = data["shard"]
        IdmShardDesignV1.model_validate(shard["idm_design"])
        ChapterBlueprintShardV2.model_validate({k: v for k, v in shard.items() if k != "idm_design"})
        self.assertEqual((data["content_origin"], shard["idm_design"]["stage_origin"]),
                         ("provider_validated", "provider"))
        self.assertEqual(provider.calls[0]["max_output_tokens"], 48_000)

    async def test_invalid_shard_context_is_422(self) -> None:
        body = idm_shard_body()
        body["idm_module_context"]["project_context"]["locale"] = "en"
        reply = await self.post(SHARD_URL, body)
        self.assertEqual((reply.status_code, reply.json()["detail"]["code"]),
                         (422, "ORCHESTRATION_V2_CONTRACT_INVALID"))
        body = idm_shard_body()
        body["source_facts"][0]["scope_key"] = "scope3_99"
        self.assertEqual((await self.post(SHARD_URL, body)).status_code, 422)
        body = idm_shard_body()
        body["shard_plan"]["source_scope_ids"].reverse()
        reply = await self.post(SHARD_URL, body)
        self.assertEqual((reply.status_code, reply.json()["detail"]["code"]), (422, "IDM_MODULE_CONTEXT_INVALID"))

    async def test_terminal_provider_rejection_is_502(self) -> None:
        reply = await self.post(SHARD_URL, idm_shard_body(), AsyncMock(side_effect=provider_error(403)))
        self.assertEqual((reply.status_code, reply.json()["detail"]["code"]), (502, "AI_PROVIDER_AUTH_REJECTED"))


class LegacyDispatchTests(unittest.IsolatedAsyncioTestCase):
    """Requests without IDM fields must produce exactly the legacy response."""

    async def compare(self, endpoint: Any, legacy: Any, request: Any, answers: list[Any]) -> dict[str, Any]:
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=copy.deepcopy(answers))), \
                patch("app.services.orchestration_v2.course_skeleton.run_idm_course_design", AsyncMock()) as course, \
                patch("app.services.orchestration_v2.chapter_shard.run_idm_module_design", AsyncMock()) as module:
            dispatched = await endpoint(request)
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=copy.deepcopy(answers))):
            direct = await legacy(request=request)
        course.assert_not_awaited()
        module.assert_not_awaited()
        self.assertEqual(dispatched, direct)
        return dispatched

    async def test_course_skeleton_without_idm(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate(legacy_skeleton_body())
        self.assertIsNone(request.idm)
        draft = {k: v for k, v in skeleton_wire().items()
                 if k not in {"contract_version", "source_snapshot_hash", "locale"}}
        usage = AiUsage(outputTokens=20)
        for answers in ([(json.dumps(draft), usage)], [("{}", usage), ("{}", usage)]):
            result = await self.compare(course_skeleton_service.lesson_author_orchestration_v2_course_skeleton,
                                        course_skeleton_service._lesson_author_orchestration_v2_course_skeleton,
                request, answers)
            self.assertNotIn("idm", result)

    async def test_chapter_shard_without_idm_context(self) -> None:
        request = RagLessonAuthorChapterShardV2Request.model_validate(legacy_shard_body())
        self.assertIsNone(request.idm_module_context)
        usage = AiUsage(outputTokens=20)
        for answers in ([(json.dumps({"lessons": [lesson_wire(["scope-1"])]}), usage)],
                        [("{}", usage), ("{}", usage)]):
            result = await self.compare(chapter_shard_service.lesson_author_orchestration_v2_chapter_shard,
                                        chapter_shard_service._lesson_author_orchestration_v2_chapter_shard, request,
                answers)
            self.assertNotIn("idm_design", result["shard"])


class TransportMappingTests(unittest.IsolatedAsyncioTestCase):
    async def mapped(self, error: Exception) -> IdmProviderError:
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=error)), \
                self.assertRaises(IdmProviderError) as caught:
            await v2_idm._idm_generate("key", "model", "prompt", json_mode=True)
        return caught.exception

    async def test_success_passes_options_through(self) -> None:
        generate = AsyncMock(return_value=("{}", AiUsage(outputTokens=1)))
        with patch("app.services.provider.generate_content", generate):
            self.assertEqual((await v2_idm._idm_generate("key", "model", "prompt", thinking_level="low"))[0], "{}")
        generate.assert_awaited_once_with("key", "model", "prompt", thinking_level="low")

    async def test_service_errors_map_to_idm_provider_errors(self) -> None:
        cases = [
            (HTTPException(504, {"code": "AI_PROVIDER_TIMEOUT"}), "AI_PROVIDER_TIMEOUT", False, 504),
            # QC course 234653 (D6): an exhausted key or a persistent rate limit must reach Node (503), not
            # become deterministic content.
            (HTTPException(503, {"code": "AI_PROVIDER_QUOTA_EXHAUSTED"}), "AI_PROVIDER_QUOTA_EXHAUSTED", True, 503),
            (HTTPException(503, {"code": "AI_PROVIDER_RATE_LIMITED"}), "AI_PROVIDER_RATE_LIMITED", True, 503),
            (HTTPException(502, {"code": "AI_PROVIDER_REQUEST_REJECTED"}), "AI_PROVIDER_REQUEST_REJECTED", True, 502),
            (HTTPException(503, "plain detail"), "AI_PROVIDER_UNAVAILABLE", False, 503),
            (AppError(code="SERVICE_BUSY", http_status=503, safe_message="busy"), "SERVICE_BUSY", False, 503),
        ]
        for error, code, terminal, status in cases:
            with self.subTest(code=code):
                mapped = await self.mapped(error)
                self.assertEqual((mapped.code, mapped.terminal, mapped.http_status), (code, terminal, status))

    async def test_sdk_errors_map_by_http_status_without_leaking_messages(self) -> None:
        cases = {400: ("AI_PROVIDER_REQUEST_REJECTED", True), 401: ("AI_PROVIDER_AUTH_REJECTED", True),
                 403: ("AI_PROVIDER_AUTH_REJECTED", True), 500: ("AI_PROVIDER_UNAVAILABLE", False)}
        for status, (code, terminal) in cases.items():
            with self.subTest(status=status), self.assertLogs("app.main", "WARNING") as logs:
                mapped = await self.mapped(provider_error(status))
            self.assertEqual((mapped.code, mapped.terminal, mapped.http_status), (code, terminal, 502))
            self.assertNotIn("PRIVATE_PROVIDER_BODY", "\n".join(logs.output))
        with self.assertLogs("app.main", "WARNING"):
            self.assertFalse((await self.mapped(RuntimeError("PRIVATE"))).terminal)

    def test_http_error_and_runtime_helpers(self) -> None:
        provider = v2_idm._idm_http_error(IdmProviderError("AI_PROVIDER_TIMEOUT", terminal=False, http_status=504))
        stage = v2_idm._idm_http_error(IdmStageError("IDM_W5_BRIEF_CONTRACT_MISMATCH"))
        budget = v2_idm._idm_http_error(IdmBudgetError("IDM_DEADLINE_EXCEEDED"))
        self.assertEqual((provider.status_code, provider.detail["code"]), (504, "AI_PROVIDER_TIMEOUT"))
        self.assertEqual((stage.status_code, stage.detail["code"]), (422, "IDM_W5_BRIEF_CONTRACT_MISMATCH"))
        self.assertEqual((budget.status_code, budget.detail["code"]), (422, "IDM_DEADLINE_EXCEEDED"))
        request = RagLessonAuthorChapterShardV2Request.model_validate(idm_shard_body())
        context = request.idm_module_context
        assert context is not None
        runtime = v2_idm._idm_runtime(request, budget_ms=60_000, allowance=context.token_allowance)
        self.assertEqual((runtime.locale, runtime.model, runtime.correlation_id), ("vi", "test-model",
                                                                                   request.correlation_id))
        self.assertEqual((runtime.token_allowance.input_tokens, runtime.token_allowance.output_tokens),
                         # type: ignore[union-attr]
                         (400_000, 131_072))
        self.assertTrue(55 < runtime.remaining_seconds() <= 60)
        self.assertEqual(runtime.provider_call_timeout_ms, main.settings.idm_provider_call_timeout_ms)
        self.assertIsNone(v2_idm._idm_runtime(request, budget_ms=1_000, allowance=None).token_allowance)
        self.assertEqual(gm.ALLOWED_TYPES, list(context.allowed_component_types))


if __name__ == "__main__":
    unittest.main()
