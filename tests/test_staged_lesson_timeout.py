from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.core.config import settings
from app.main import (
    AiUsage,
    STAGED_LESSON_WORKFLOW_TIMEOUT_MAX_MS,
    StagedLessonWorkflowDeadline,
    RagLessonAuthorRequest,
    call_provider_with_timeout,
    generate_staged_lesson_author_proposal,
    is_non_retryable_provider_error,
    staged_lesson_content_output_tokens,
)


def staged_request(*, max_output_tokens: int = 30_000) -> RagLessonAuthorRequest:
    return RagLessonAuthorRequest(
        tenant_id="11111111-1111-4111-8111-111111111111",
        kb_id="22222222-2222-4222-8222-222222222222",
        conversation_id="33333333-3333-4333-8333-333333333333",
        target="lesson_author",
        model="gemini-3.5-flash",
        max_output_tokens=max_output_tokens,
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
        system_prompt="test",
        user_message="Soạn chi tiết chương đã chọn.",
        output_schema_hint="schema",
        api_key="test-key",
    )


def staged_content() -> str:
    html = (
        "<h3>Nhận diện mối nguy</h3><p>Người học quan sát công việc, thiết bị và môi trường "
        "trước khi thao tác để nhận diện điều kiện có thể gây mất an toàn. Việc rà soát cần được "
        "thực hiện có hệ thống, ghi nhận rõ mối nguy và dùng kết quả làm cơ sở cho bước đánh giá "
        "rủi ro tiếp theo. Khi phát hiện dấu hiệu bất thường, người học cần mô tả vị trí, điều kiện "
        "và đối tượng bị ảnh hưởng để các biện pháp kiểm soát có thể được lựa chọn phù hợp. Việc trao "
        "đổi với người phụ trách trước khi tiếp tục công việc giúp tránh bỏ sót những thay đổi tại hiện trường.</p>"
    )
    return json.dumps({
        "title": "Nhận diện mối nguy",
        "source_fact_ids": ["p10-f1"],
        "components": [{
            "type": "html",
            "title": "Nhận diện mối nguy",
            "html": html,
            "source_fact_ids": ["p10-f1"],
            "covered_source_fact_ids": ["p10-f1"],
        }],
    }, ensure_ascii=False)


class StagedLessonTimeoutTests(unittest.TestCase):
    def _run_staged_request(self, stage_two_result: object) -> tuple[object, AsyncMock]:
        manifest = {
            "facts": [{
                "fact_id": "p10-f1",
                "source_page": 10,
                "text": "Nhận diện mối nguy: quan sát công việc, thiết bị và môi trường trước khi thao tác.",
            }],
        }
        usage = AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)
        provider = AsyncMock(side_effect=[("{", usage), ("{", usage), stage_two_result])
        self._last_provider = provider
        with patch("app.main.generate_content", new=provider):
            result = asyncio.run(generate_staged_lesson_author_proposal(
                staged_request(),
                "Nguồn liên quan",
                "",
                "- [p10-f1] trang/slide 10: Nhận diện mối nguy",
                [{
                    "source_page": 10,
                    "content": manifest["facts"][0]["text"],
                    "document_name": "source.pdf",
                    "source_section": "",
                    "document_id": "doc-1",
                    "score": 0,
                }],
                manifest,
            ))
        return result, provider

    def test_stage_two_uses_dedicated_timeout_while_skeleton_keeps_general_timeout(self) -> None:
        result, provider = self._run_staged_request((staged_content(), AiUsage()))

        self.assertIsInstance(result, tuple)
        self.assertEqual(provider.await_count, 3)
        self.assertNotIn("request_timeout_ms", provider.await_args_list[0].kwargs)
        self.assertNotIn("request_timeout_ms", provider.await_args_list[1].kwargs)
        self.assertEqual(
            provider.await_args_list[2].kwargs["request_timeout_ms"],
            settings.staged_lesson_content_provider_timeout_ms,
        )
        self.assertNotEqual(
            provider.await_args_list[2].kwargs["request_timeout_ms"],
            settings.provider_request_timeout_ms,
        )

    def test_thirty_thousand_token_stage_two_budget_retains_dedicated_timeout(self) -> None:
        self.assertEqual(
            staged_lesson_content_output_tokens(
                request_max_output_tokens=30_000,
                max_facts_per_unit=25,
            ),
            30_000,
        )
        self.assertEqual(settings.staged_lesson_content_provider_timeout_ms, 180_000)
        self.assertNotEqual(settings.staged_lesson_content_provider_timeout_ms, 60_000)

    def test_thirty_thousand_token_stage_two_call_does_not_inherit_general_timeout(self) -> None:
        fact_ids = [f"p10-f{index}" for index in range(1, 26)]
        skeleton = json.dumps({
            "chapters": [{
                "title": "Nhận diện mối nguy",
                "lessons": [{
                    "title": "Phân tích mối nguy",
                    "units": [{
                        "title": "Thực hành nhận diện",
                        "source_fact_ids": fact_ids,
                        "component_plan": [{
                            "type": "html",
                            "rationale": "Giải thích các dữ kiện nguồn.",
                            "purpose": "explain",
                            "source_fact_ids": fact_ids,
                            "content_requirements": ["Bao quát các dữ kiện bắt buộc."],
                        }],
                    }],
                }],
            }],
        }, ensure_ascii=False)
        manifest = {"facts": [
            {
                "fact_id": fact_id,
                "source_page": 10,
                "text": f"Dữ kiện nguồn {index} cho nhận diện mối nguy.",
            }
            for index, fact_id in enumerate(fact_ids, start=1)
        ]}
        timeout_error = HTTPException(
            status_code=504,
            detail={"code": "AI_PROVIDER_TIMEOUT", "message": "safe"},
        )
        provider = AsyncMock(side_effect=[
            (skeleton, AiUsage()),
            timeout_error,
        ])

        with patch("app.main.generate_content", new=provider):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(generate_staged_lesson_author_proposal(
                    staged_request(max_output_tokens=30_000),
                    "Nguồn liên quan",
                    "",
                    "",
                    source_rows=[],
                    source_coverage_manifest=manifest,
                ))

        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_TIMEOUT")
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(provider.await_args_list[1].kwargs["max_output_tokens"], 30_000)
        self.assertEqual(
            provider.await_args_list[1].kwargs["request_timeout_ms"],
            settings.staged_lesson_content_provider_timeout_ms,
        )
        self.assertNotEqual(provider.await_args_list[1].kwargs["request_timeout_ms"], 60_000)

    def test_stage_two_timeout_is_non_retryable_and_fails_closed(self) -> None:
        timeout_error = HTTPException(
            status_code=504,
            detail={"code": "AI_PROVIDER_TIMEOUT", "message": "safe"},
        )
        with self.assertRaises(HTTPException) as raised:
            self._run_staged_request(timeout_error)

        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_TIMEOUT")
        self.assertTrue(is_non_retryable_provider_error(raised.exception))
        # The timeout exits the Stage-2 call immediately: no recovery prompt
        # or identical provider call is sent after the timed-out batch.
        self.assertEqual(self._last_provider.await_count, 3)

    def test_workflow_budget_bounds_multiple_stage_two_batches_before_node_envelope(self) -> None:
        with patch("app.main.perf_counter", side_effect=[100.0, 100.0, 580.1]):
            deadline = StagedLessonWorkflowDeadline()
            first_timeout_ms, first_remaining_ms = deadline.stage_two_provider_timeout_ms()
            self.assertEqual(first_timeout_ms, 180_000)
            self.assertEqual(first_remaining_ms, 480_000)
            with self.assertRaises(HTTPException) as raised:
                deadline.stage_two_provider_timeout_ms()

        self.assertEqual(raised.exception.detail["code"], "AI_STAGED_LESSON_WORKFLOW_TIMEOUT")
        self.assertTrue(is_non_retryable_provider_error(raised.exception))
        self.assertEqual(STAGED_LESSON_WORKFLOW_TIMEOUT_MAX_MS, 480_000)
        self.assertGreaterEqual(600_000 - STAGED_LESSON_WORKFLOW_TIMEOUT_MAX_MS, 120_000)

    def test_blueprint_timeout_remains_separate_and_unchanged(self) -> None:
        self.assertEqual(settings.blueprint_provider_request_timeout_ms, 300_000)
        self.assertNotEqual(
            settings.blueprint_provider_request_timeout_ms,
            settings.staged_lesson_content_provider_timeout_ms,
        )


class ProviderTimeoutCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_timeout_does_not_prove_sync_provider_thread_was_cancelled(self) -> None:
        started = threading.Event()
        completed = threading.Event()

        def slow_sync_provider() -> None:
            started.set()
            time.sleep(0.08)
            completed.set()

        with self.assertRaises(HTTPException) as raised:
            await call_provider_with_timeout(
                slow_sync_provider,
                "gemini-test",
                request_timeout_ms=20,
            )

        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_TIMEOUT")
        self.assertTrue(started.wait(timeout=1))
        # The coroutine timed out, but the to_thread work still completed.
        # This is not a claim about a real Google HTTP cancellation.
        self.assertTrue(completed.wait(timeout=1))


if __name__ == "__main__":
    unittest.main()
