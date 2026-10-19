"""Broad exception handlers report what they turn into a fallback (wave 1: observe, never narrow).

Control flow is unchanged: each case checks the response or result is what it was, and that the
handler logged one ``broad_exception_handled`` line (stage, outcome, bounded type, sanitised
frames, never the message) and counted it. Providers are local fakes; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from app import main
from app.api import deps as api_deps
from app.core import metrics
from app.core.errors import ProviderError, ServiceError
from app.core.handled_exceptions import HANDLED_EXCEPTION_EVENT, count_handled_exception, report_handled_exception
from app.core.logging import exception_fields, exception_type_label
from app.idm.runtime import IdmProviderError
from app.idm.storyboard import run_idm_unit
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.semantic_review import run_bounded_semantic_review
from app.services.orchestration_v2 import idm as v2_idm
from app.services.orchestration_v2 import unit as unit_service
from tests.support.idm_test_support import make_runtime
from tests.support.log_support import log_text, logged_events
from tests.unit.idm.test_idm_endpoints import SKELETON_URL, legacy_skeleton_body
from tests.unit.idm.test_idm_storyboard import WRITER, FakeGenerate, severity_body, severity_writer

COUNTER = "ai_rag_handled_exceptions_total"


def counted(stage: str, exception_type: str, outcome: str) -> float:
    value = metrics.REGISTRY.get_sample_value(COUNTER, {"stage": stage, "exception_type": exception_type,
                                                        "outcome": outcome})
    return 0.0 if value is None else value


def raise_private(error_type: type[Exception]) -> Any:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise error_type("PRIVATE_PROVIDER_TEXT")

    return fail


class ReportTests(unittest.TestCase):
    def test_report_logs_type_stage_outcome_and_frames_but_never_the_message(self) -> None:
        logger = logging.getLogger("tests.handled")
        before = counted("idm_w5_bind", "KeyError", "IDM_W5_INSTANCE_INVALID")
        with self.assertLogs(logger, "WARNING") as logs:
            try:
                raise_private(KeyError)()
            except KeyError as error:
                report_handled_exception(logger, error, stage="idm_w5_bind", outcome="IDM_W5_INSTANCE_INVALID",
                                         family="lesson_author_idm", unit_path="c1.l1.u1")
        [event] = logged_events(logs.records, HANDLED_EXCEPTION_EVENT)
        self.assertEqual((logs.records[0].levelno, logs.records[0].getMessage()),
                         (logging.WARNING, "lesson_author_idm broad_exception_handled"))
        self.assertEqual((event["stage"], event["outcome"], event["exception_type"], event["unit_path"]),
                         ("idm_w5_bind", "IDM_W5_INSTANCE_INVALID", "KeyError", "c1.l1.u1"))
        self.assertEqual(event["exception_frames"][-1].rsplit(":", 1)[0],
                         "tests.unit.core.test_handled_exceptions:fail")
        self.assertNotIn("PRIVATE", log_text(logs.records))
        self.assertEqual(counted("idm_w5_bind", "KeyError", "IDM_W5_INSTANCE_INVALID"), before + 1)

    def test_labels_stay_bounded(self) -> None:
        dynamic = type("ProviderError", (Exception,), {})("PRIVATE")
        before = counted("other", "OtherException", "other")
        count_handled_exception(dynamic, stage="Free Text", outcome="not a code!")
        self.assertEqual(counted("other", "OtherException", "other"), before + 1)


    def test_service_and_provider_errors_keep_the_old_http_exception_label(self) -> None:
        # They were fastapi.HTTPException (OtherException) until wave 2b; the label holds until the review.
        for error in (ServiceError(status_code=502, detail="PRIVATE"),
                      ProviderError("AI_PROVIDER_UNAVAILABLE", "PRIVATE")):
            with self.subTest(error=type(error).__name__):
                self.assertEqual(exception_type_label(error), "OtherException")
                try:
                    try:
                        raise error
                    except ServiceError as cause:
                        raise IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False) from cause
                except IdmProviderError as wrapped:
                    fields = exception_fields(wrapped)
                self.assertEqual((fields["exception_type"], fields["exception_cause_type"]),
                                 ("IdmProviderError", "OtherException"))
                before = counted("idm", "OtherException", "AI_PROVIDER_UNAVAILABLE")
                count_handled_exception(error, stage="idm", outcome="AI_PROVIDER_UNAVAILABLE")
                self.assertEqual(counted("idm", "OtherException", "AI_PROVIDER_UNAVAILABLE"), before + 1)


class OrchestrationFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None

    def tearDown(self) -> None:
        main.app.dependency_overrides.pop(api_deps.require_internal_token, None)

    async def test_unexpected_skeleton_error_still_falls_back_and_is_reported(self) -> None:
        before = counted("course_skeleton", "TypeError", "AI_PROVIDER_UNAVAILABLE")
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=raise_private(TypeError))), \
                self.assertLogs("app.main", "WARNING") as logs:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
                reply = await client.post(SKELETON_URL, json=legacy_skeleton_body())
        self.assertEqual(reply.status_code, 200)
        self.assertEqual((reply.json()["content_origin"], reply.json()["quality_state"]),
                         ("structured_fallback", "review_required"))
        [event] = logged_events(logs.records, HANDLED_EXCEPTION_EVENT)
        self.assertEqual((event["stage"], event["outcome"], event["exception_type"], event["provider_attempt"]),
                         ("course_skeleton", "AI_PROVIDER_UNAVAILABLE", "TypeError", 1))
        self.assertTrue(event["exception_frames"])
        self.assertNotIn("PRIVATE", log_text(logs.records) + reply.text)
        self.assertEqual(counted("course_skeleton", "TypeError", "AI_PROVIDER_UNAVAILABLE"), before + 1)

    async def test_idm_transport_reports_the_stage_of_the_failed_call(self) -> None:
        before = counted("idm_w2", "AttributeError", "AI_PROVIDER_UNAVAILABLE")
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=raise_private(AttributeError))), \
                patch.object(v2_idm, "current_provider_stage", return_value="idm_w2"), \
                self.assertLogs("app.main", "WARNING") as logs, self.assertRaises(IdmProviderError) as caught:
            await v2_idm._idm_generate("key", "model", "prompt")
        self.assertEqual((caught.exception.code, caught.exception.terminal), ("AI_PROVIDER_UNAVAILABLE", False))
        [event] = logged_events(logs.records, HANDLED_EXCEPTION_EVENT)
        self.assertEqual((event["stage"], event["exception_type"]), ("idm_w2", "AttributeError"))
        self.assertEqual(counted("idm_w2", "AttributeError", "AI_PROVIDER_UNAVAILABLE"), before + 1)


class StoryboardRecoverableErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_binder_key_error_is_reported_before_it_becomes_an_invalid_answer(self) -> None:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = unit_service._idm_unit_deps(request)
        seen: list[tuple[str, str, str]] = []

        def observe(stage: str, code: str, error: BaseException) -> None:
            seen.append((stage, code, type(error).__name__))
            assert deps.on_recoverable_error is not None
            deps.on_recoverable_error(stage, code, error)  # the service observer logs and counts it

        broken = dataclasses.replace(deps, judge_mode="off", bind_instance_payload=raise_private(KeyError),
                                     on_recoverable_error=observe)
        provider = FakeGenerate(**{WRITER: [severity_writer(body)] * 2})
        before = counted("idm_w5_bind", "KeyError", "IDM_W5_INSTANCE_INVALID")
        with self.assertLogs("app.main", "WARNING") as logs:
            result = await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(provider, allowance=None),
                                        deps=broken, fallback_only=False)
        # Unchanged: the answers are invalid and the unit settles on its deterministic content.
        self.assertEqual(result["content_origin"], "structured_fallback")
        self.assertEqual(seen, [("idm_w5_bind", "IDM_W5_INSTANCE_INVALID", "KeyError")] * len(provider.calls))
        events = logged_events(logs.records, HANDLED_EXCEPTION_EVENT)
        self.assertEqual({(event["stage"], event["unit_path"]) for event in events},
                         {("idm_w5_bind", request.unit_contract.unit_path)})
        self.assertEqual(counted("idm_w5_bind", "KeyError", "IDM_W5_INSTANCE_INVALID"), before + len(provider.calls))
        self.assertNotIn("PRIVATE", log_text(logs.records) + json.dumps(result, ensure_ascii=False))


class SemanticReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_reviewer_is_reported_and_still_needs_review(self) -> None:
        before = counted("semantic_review", "RuntimeError", "SEMANTIC_REVIEW_UNAVAILABLE")
        with self.assertLogs("app.semantic_review", "WARNING") as logs:
            outcome = await run_bounded_semantic_review(
                {"components": [{"type": "html"}]}, reviewer=AsyncMock(side_effect=raise_private(RuntimeError)),
                repairer=AsyncMock(), deterministic_validate=lambda unit: None, allow_repair=True)
        self.assertEqual((outcome.quality_state, outcome.failure_code),
                         ("review_required", "SEMANTIC_REVIEW_UNAVAILABLE"))
        [event] = logged_events(logs.records, HANDLED_EXCEPTION_EVENT)
        self.assertEqual((event["stage"], event["exception_type"]), ("semantic_review", "RuntimeError"))
        self.assertEqual(counted("semantic_review", "RuntimeError", "SEMANTIC_REVIEW_UNAVAILABLE"), before + 1)


if __name__ == "__main__":
    unittest.main()
