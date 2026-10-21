"""``POST /v1/lesson-author/org-context/distill`` (ORG_CONTEXT spec §4): reader, sampling, map-reduce, errors, route.

The database is a fake that answers only the tenant and knowledge base it holds; the provider is a local fake behind
``app.services.provider.generate_content`` (the real IDM transport, limiter mapping and usage ledger run). Nothing
reaches the network or a database.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

import httpx

from app import main
from app.api import deps as api_deps
from app.core.config import settings
from app.core.errors import AppError, ServiceError
from app.schemas.org_context import RagOrgContextDistillRequest
from app.services.org_context import distill as distill_service
from tests import idm_contract_bridge as bridge
from tests.support.log_support import log_text, logged_events
from tests.support.org_context_support import DISTILLED, FakeContextPool, FakeProfileProvider, profile

TENANT = "00000000-0000-4000-8000-0000000000a1"
KB = "00000000-0000-4000-8000-0000000000b1"
OTHER_TENANT = "00000000-0000-4000-8000-0000000000a2"
DOC_A = "00000000-0000-4000-8000-00000000d00a"
DOC_B = "00000000-0000-4000-8000-00000000d00b"
DOC_UNREQUESTED = "00000000-0000-4000-8000-00000000d00c"
DOC_LEARNING = "00000000-0000-4000-8000-00000000d00d"
URL = "/v1/lesson-author/org-context/distill"


def body(**changes: Any) -> dict[str, Any]:
    return {"tenant_id": TENANT, "kb_id": KB, "document_ids": [DOC_B, DOC_A], "locale": "vi",
            "max_source_chars": 1_500_000, "model": "fake-model", "api_key": "fake-key",
            "correlation_id": "00000000-0000-4000-8000-0000000000c1", **changes}


DOCUMENTS = {
    DOC_A: ["Hải Đăng Logistics vận hành kho lạnh.", "Điều phối viên ca đêm phân tuyến xe."],
    DOC_B: ["Phiếu HD-07 là biên bản bàn giao hàng lạnh."],
    DOC_UNREQUESTED: ["Tài liệu khác của cùng doanh nghiệp."],
    DOC_LEARNING: [],
}


class DistillServiceTests(unittest.IsolatedAsyncioTestCase):
    async def run_distill(self, provider: FakeProfileProvider, pool: FakeContextPool | None = None,
                          **changes: Any) -> Any:
        request = RagOrgContextDistillRequest.model_validate(body(**changes))
        with patch("app.services.provider.generate_content", provider):
            return await distill_service.org_context_distill(
                request, pool or FakeContextPool(DOCUMENTS, tenant=TENANT, kb=KB))

    async def test_one_batch_reads_exactly_the_requested_documents_in_order(self) -> None:
        provider = FakeProfileProvider()
        pool = FakeContextPool(DOCUMENTS, tenant=TENANT, kb=KB)
        result = await self.run_distill(provider, pool, document_ids=[DOC_B, DOC_A, DOC_LEARNING])
        self.assertEqual(result.profile.model_dump(mode="json"), DISTILLED)
        self.assertEqual(result.source_document_ids, [DOC_B, DOC_A])
        self.assertEqual(result.source_chars, sum(len(text) for text in [*DOCUMENTS[DOC_A], *DOCUMENTS[DOC_B]]))
        self.assertFalse(result.truncated)
        self.assertEqual((result.usage, result.usage_complete, result.provider_calls),
                         # Billed output counts the thinking tokens (total - input), as on the IDM routes.
                         ({"inputTokens": 1_000, "outputTokens": 300, "embeddingTokens": 0, "totalTokens": 1_300},
                          True, 1))
        prompt = provider.calls[0]["prompt"]
        self.assertLess(prompt.index(DOCUMENTS[DOC_B][0]), prompt.index(DOCUMENTS[DOC_A][0]))
        self.assertIn("[DOCUMENT 1]\n" + DOCUMENTS[DOC_B][0], prompt)
        self.assertIn("Vietnamese (with full diacritics)", prompt)
        self.assertEqual((provider.calls[0]["model"], provider.calls[0]["json_mode"]), ("fake-model", True))
        # Tenant and knowledge base filter every statement; an unrequested document is never read.
        self.assertTrue(all(args[0] == TENANT and args[1] == KB for _sql, args in pool.calls))
        self.assertTrue(all(".tenant_id=$1::uuid AND" in sql and ".kb_id=$2::uuid AND" in sql for sql, _ in pool.calls))
        self.assertNotIn(DOC_UNREQUESTED, json.dumps([args for _sql, args in pool.calls], default=str))
        self.assertNotIn(DOCUMENTS[DOC_UNREQUESTED][0], prompt)

    async def test_several_batches_are_mapped_then_reduced(self) -> None:
        merged = profile(products_services=["Gói gộp"])
        provider = FakeProfileProvider(maps=[profile(products_services=["Gói A"]), profile(products_services=[
            "Gói B"])], reduces=[merged])
        with patch.object(distill_service, "ORG_DISTILL_BATCH_MAX_CHARS", 60):
            result = await self.run_distill(provider)
        kinds = [call["kind"] for call in provider.calls]
        self.assertEqual((kinds.count("map"), kinds.count("reduce")), (3, 1))
        self.assertEqual(result.profile.products_services, ["Gói gộp"])
        self.assertEqual(result.provider_calls, 4)
        reduce_prompt = provider.calls[-1]["prompt"]
        self.assertIn('"products_services":["Gói A"]', reduce_prompt)
        self.assertIn('"products_services":["Gói B"]', reduce_prompt)

    async def test_a_source_over_the_limit_is_sampled_evenly_and_reported_truncated(self) -> None:
        many = {DOC_A: [f"Đoạn A{index:02d} " + "a" * 990 for index in range(60)],
                DOC_B: [f"Đoạn B{index:02d} " + "b" * 990 for index in range(60)]}
        pool = FakeContextPool(many, tenant=TENANT, kb=KB)
        provider = FakeProfileProvider()
        result = await self.run_distill(provider, pool, max_source_chars=20_000, document_ids=[DOC_A, DOC_B])
        self.assertTrue(result.truncated)
        self.assertLessEqual(result.source_chars, 20_000)
        self.assertEqual(result.source_document_ids, [DOC_A, DOC_B])
        prompt = provider.calls[0]["prompt"]
        self.assertIn("excerpts taken evenly", prompt)
        self.assertIn("Đoạn A0", prompt)
        self.assertIn("Đoạn B5", prompt)
        # Only the sampled chunks' texts are read.
        read = [number for sql, args in pool.calls if "c.chunk_no=ANY" in sql for number in args[4]]
        self.assertEqual(len(read) * 999, result.source_chars)

    async def test_the_answer_is_brought_into_the_profile_bounds(self) -> None:
        oversized = profile(products_services=[f"Sản phẩm {index}: " + "x" * 180 for index in range(20)],
                            key_processes=[f"Quy trình {index}: " + "y" * 280 for index in range(15)],
                            organization_summary="s" * 800, culture_tone="t" * 600)
        # The documents number every product and process, so the grounding step keeps what the bounds keep.
        numbered = {**DOCUMENTS, DOC_A: [", ".join(f"sản phẩm {index}" for index in range(20))]}
        result = await self.run_distill(FakeProfileProvider(maps=[oversized]),
                                        FakeContextPool(numbered, tenant=TENANT, kb=KB))
        fitted = result.profile
        self.assertTrue(fitted.products_services and fitted.key_processes)
        self.assertLessEqual(len(fitted.products_services), 15)
        self.assertEqual(fitted.products_services, oversized["products_services"][:len(fitted.products_services)])
        total = (len(fitted.organization_summary) + len(fitted.culture_tone) + sum(map(len, fitted.products_services))
                 + sum(map(len, fitted.key_processes)) + sum(map(len, fitted.design_conventions))
                 + sum(len(group.name) + len(group.description) for group in fitted.learner_groups)
                 + sum(len(entry.term) + len(entry.meaning) for entry in fitted.glossary))
        self.assertLessEqual(total, 6_000)

    async def test_details_the_source_does_not_state_are_dropped_and_counted(self) -> None:
        # QC run 2026-10-21: the documents said "IRC", the draft said "xin IRC/ERC".
        documents = {DOC_A: ["Dự án điện mặt trời rộng 3400 ha, đấu nối lưới 110kV.",
                             "Đường dây 22kV cấp điện cho khu vận hành."],
                     DOC_B: ["Nhà đầu tư xin IRC (Giấy chứng nhận đăng ký đầu tư) trước khi khởi công."]}
        kept = {"products_services": ["Điện mặt trời: 3.400 ha nối lưới 110kV/22kV."], "learner_groups": [],
                "culture_tone": "An toàn là trên hết.", "design_conventions": []}
        drafted = profile(
            **kept, organization_summary="Công ty vận hành dự án điện mặt trời 3.400 ha. Đạt chứng chỉ I-REC năm 2020.",
            glossary=[{"term": "IRC", "meaning": "Giấy chứng nhận đăng ký đầu tư."},
                      {"term": "Trạm nâng áp", "meaning": "Trạm biến áp 110kV, công suất 50MVA."}],
            key_processes=["Xin IRC trước khi khởi công.", "Xin IRC/ERC cho dự án.", "Bảo trì trạm mỗi 6 tháng."])
        with self.assertLogs("app.idm", "INFO") as logs:
            result = await self.run_distill(FakeProfileProvider(maps=[drafted]),
                                            FakeContextPool(documents, tenant=TENANT, kb=KB))
        self.assertEqual(result.profile.model_dump(mode="json"), profile(
            **kept, organization_summary="Công ty vận hành dự án điện mặt trời 3.400 ha.",
            glossary=[{"term": "IRC", "meaning": "Giấy chứng nhận đăng ký đầu tư."}],
            key_processes=["Xin IRC trước khi khởi công."]))
        completed = logged_events(logs.records, "org_context_distill_completed")
        self.assertEqual(completed[0]["ungrounded_removed"],
                         {"glossary": 1, "key_processes": 2, "organization_summary": 1})
        # Counts only: nothing of the profile reaches the log.
        self.assertTrue(all(text not in log_text(logs.records) for text in ("I-REC", "50MVA", "IRC/ERC")))

    async def test_no_learned_text_is_409(self) -> None:
        cases = {"no learned index": {"document_ids": [DOC_LEARNING]},
                 "other tenant": {"tenant_id": OTHER_TENANT},
                 "other knowledge base": {"kb_id": "00000000-0000-4000-8000-0000000000b2"}}
        for name, changes in cases.items():
            with self.subTest(case=name):
                provider = FakeProfileProvider()
                with self.assertRaises(AppError) as caught:
                    await self.run_distill(provider, **changes)
                self.assertEqual((caught.exception.code, caught.exception.http_status),
                                 ("ORG_CONTEXT_NO_LEARNED_SOURCE", 409))
                self.assertEqual(provider.calls, [])

    async def test_provider_failures_map_to_their_status(self) -> None:
        unavailable = ServiceError(status_code=503, detail={"code": "AI_PROVIDER_UNAVAILABLE", "message": "x"})
        cases: list[tuple[Any, int, str]] = [
            (unavailable, 503, "AI_PROVIDER_UNAVAILABLE"),
            (AppError("SERVICE_BUSY", 503, "busy"), 503, "SERVICE_BUSY"),
            (ServiceError(status_code=503, detail={"code": "AI_PROVIDER_QUOTA_EXHAUSTED", "message": "x"}), 503,
             "AI_PROVIDER_QUOTA_EXHAUSTED"),
            (RuntimeError("transport"), 502, "AI_PROVIDER_UNAVAILABLE"),
        ]
        for failure, status, code in cases:
            with self.subTest(code=code, status=status):
                with self.assertRaises(ServiceError) as caught:
                    await self.run_distill(FakeProfileProvider(maps=[failure]))
                self.assertEqual((caught.exception.status_code, caught.exception.detail["code"]), (status, code))

    async def test_two_invalid_answers_are_502_and_no_time_is_504(self) -> None:
        with self.assertRaises(AppError) as caught:
            await self.run_distill(FakeProfileProvider(maps=["{}", "{}"]))
        self.assertEqual((caught.exception.code, caught.exception.http_status),
                         ("ORG_CONTEXT_DISTILL_OUTPUT_INVALID", 502))
        with patch.object(settings, "lesson_author_deadline_ms", 5_000), self.assertRaises(AppError) as late:
            await self.run_distill(FakeProfileProvider())
        self.assertEqual((late.exception.code, late.exception.http_status), ("REQUEST_DEADLINE_EXCEEDED", 504))

    async def test_an_invalid_answer_is_repaired_once(self) -> None:
        provider = FakeProfileProvider(maps=["not json", profile()])
        result = await self.run_distill(provider)
        self.assertEqual(result.provider_calls, 2)
        self.assertIn("REPAIR_REQUIREMENTS", provider.calls[1]["prompt"])
        self.assertEqual(result.profile.model_dump(mode="json"), DISTILLED)


class DistillRouteTests(unittest.IsolatedAsyncioTestCase):
    async def post(self, payload: dict[str, Any], *, provider: FakeProfileProvider | None = None,
                   authorized: bool = True) -> httpx.Response:
        pool = FakeContextPool(DOCUMENTS, tenant=TENANT, kb=KB)
        main.app.dependency_overrides[api_deps.get_db] = lambda: pool
        if authorized:
            main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
        try:
            with patch("app.services.provider.generate_content", provider or FakeProfileProvider()):
                transport = httpx.ASGITransport(app=main.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                    return await client.post(URL, json=payload)
        finally:
            main.app.dependency_overrides.pop(api_deps.get_db, None)
            main.app.dependency_overrides.pop(api_deps.require_internal_token, None)

    async def test_the_route_answers_the_published_shape(self) -> None:
        reply = await self.post(body())
        self.assertEqual(reply.status_code, 200)
        data = reply.json()
        self.assertEqual(list(data), ["profile", "source_document_ids", "source_chars", "truncated", "usage",
                                      "usage_complete", "provider_calls"])
        self.assertEqual(data["profile"], DISTILLED)
        self.assertEqual(data["source_document_ids"], [DOC_B, DOC_A])

    async def test_requests_are_validated(self) -> None:
        cases = {
            "repeated document": body(document_ids=[DOC_A, DOC_A.upper()]),
            "no document": body(document_ids=[]),
            "too many documents": body(document_ids=[f"00000000-0000-4000-8000-{index:012d}" for index in range(51)]),
            "limit too small": body(max_source_chars=999),
            "limit too large": body(max_source_chars=2_000_001),
            "limit as text": body(max_source_chars="20000"),
            "locale": body(locale="fr"),
            "extra field": body(target="lesson_author"),
            "tenant": body(tenant_id="tenant-1"),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                reply = await self.post(payload)
                self.assertEqual((reply.status_code, reply.json()["detail"]["code"]),
                                 (422, "REQUEST_VALIDATION_FAILED"))

    async def test_the_route_needs_the_internal_token(self) -> None:
        reply = await self.post(body(), authorized=False)
        self.assertEqual(reply.status_code, 401)

    async def test_an_upper_case_document_id_is_read_in_its_canonical_form(self) -> None:
        reply = await self.post(body(document_ids=[DOC_A.upper()]))
        self.assertEqual((reply.status_code, reply.json()["source_document_ids"]), (200, [DOC_A]))


if __name__ == "__main__":
    unittest.main()


class DistillBridgeTests(unittest.TestCase):
    def test_the_bridge_runs_the_real_route_on_the_given_chunks(self) -> None:
        options = {"documents": {DOC_A: DOCUMENTS[DOC_A], DOC_B: DOCUMENTS[DOC_B]},
                   "maps": [profile(culture_tone="Ngắn gọn.")]}
        result = asyncio.run(bridge.run("org_context_distill", body(), options))
        self.assertEqual((result["status"], result["calls"]), (200, ["map"]))
        self.assertEqual(result["response"]["profile"]["culture_tone"], "Ngắn gọn.")
        self.assertEqual(result["response"]["source_document_ids"], [DOC_B, DOC_A])
        empty = asyncio.run(bridge.run("org_context_distill", body(), {"documents": {}}))
        self.assertEqual((empty["status"], empty["response"]["detail"]["code"]),
                         (409, "ORG_CONTEXT_NO_LEARNED_SOURCE"))


class GoneBetweenReadsPool(FakeContextPool):
    """A document learned again after its sizes were read: its old chunks are gone when the texts are read."""

    def __init__(self, gone: str) -> None:
        super().__init__(DOCUMENTS, tenant=TENANT, kb=KB)
        self.gone = gone

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        rows = await super().fetch(sql, *args)
        return [] if "c.chunk_no=ANY" in sql and args[2] == self.gone else rows


class ReaderEdgeTests(unittest.IsolatedAsyncioTestCase):
    async def distill(self, pool: FakeContextPool, provider: FakeProfileProvider) -> Any:
        request = RagOrgContextDistillRequest.model_validate(body())
        with patch("app.services.provider.generate_content", provider):
            return await distill_service.org_context_distill(request, pool)

    async def test_chunk_sizes_are_read_in_pages(self) -> None:
        pool = FakeContextPool({DOC_A: [f"đoạn {index}" for index in range(7)], DOC_B: ["một đoạn"]}, tenant=TENANT,
                               kb=KB)
        provider = FakeProfileProvider()
        with patch.object(distill_service, "_SIZE_PAGE", 3):
            result = await self.distill(pool, provider)
        pages = [args[4] for sql, args in pool.calls if "char_length" in sql and args[2] == DOC_A]
        self.assertEqual(pages, [-1, 2, 5])
        self.assertTrue(all(f"đoạn {index}" in provider.calls[0]["prompt"] for index in range(7)))
        self.assertEqual(result.source_document_ids, [DOC_B, DOC_A])

    async def test_a_document_gone_between_the_reads_is_left_out(self) -> None:
        result = await self.distill(GoneBetweenReadsPool(DOC_A), FakeProfileProvider())
        self.assertEqual(result.source_document_ids, [DOC_B])
        self.assertEqual(result.source_chars, len(DOCUMENTS[DOC_B][0]))
