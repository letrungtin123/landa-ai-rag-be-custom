import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.main import (
    AiUsage,
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
    lesson_author_orchestration_v2_chapter_shard,
    lesson_author_orchestration_v2_course_skeleton,
    lesson_author_orchestration_v2_source_snapshot,
    lesson_author_orchestration_v2_unit,
)
from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    ChapterShardProviderWireV2,
    CourseSkeletonProviderWireV2,
)


SOURCE_HASH = "a" * 64
TENANT_ID = "00000000-0000-4000-8000-000000000001"
KB_ID = "00000000-0000-4000-8000-000000000002"
CONVERSATION_ID = "00000000-0000-4000-8000-000000000003"
CORRELATION_ID = "00000000-0000-4000-8000-000000000004"


def model_designed_authority() -> dict:
    payload = {
        "mode": "model_designed", "source": "none", "complete": True,
        "confidence": 0.0, "reason_codes": [], "chapters": [],
    }
    return {**payload, "structure_hash": canonical_hash(payload)}


def common() -> dict:
    return {
        "tenant_id": TENANT_ID,
        "kb_id": KB_ID,
        "conversation_id": CONVERSATION_ID,
        "target": "lesson_author",
        "model": "test-model",
        "max_output_tokens": 8192,
        "embedding_model": "test-embedding",
        "embedding_dimensions": 768,
        "system_prompt": "server-owned",
        "user_message": "Tạo nội dung bài học",
        "history": [],
        "source_documents": [],
        "locale": "vi",
        "correlation_id": CORRELATION_ID,
        "api_key": "private-test-key",
        "contract_version": 2,
    }


def skeleton_wire() -> dict:
    return {
        "contract_version": 2,
        "source_snapshot_hash": SOURCE_HASH,
        "locale": "vi",
        "title": "Khóa học",
        "summary": "Tóm tắt",
        "target_audience": "Quản lý",
        "prerequisites": [],
        "learning_outcomes": ["Áp dụng"],
        "assessment_strategy": "Đánh giá",
        "assumptions": [],
        "chapters": [{
            "chapter_key": "chapter-1", "order": 0, "title": "Chương 1", "objective": "Mục tiêu",
            "learning_outcomes": ["Hoàn thành mục tiêu"],
            "source_scope_ids": ["scope-1"],
        }],
    }


def lesson_wire(scopes: list[str]) -> dict:
    return {
        "title": "Bài 1", "objective": "Hiểu nội dung", "learning_objectives": ["Áp dụng"],
        "learning_activities": ["Đọc và thực hành"], "assessment": "Bài kiểm tra",
        "units": [{"title": "Nội dung", "purpose": "Giải thích nội dung nguồn",
                   "learning_objective_refs": ["lo_1"], "source_scope_ids": scopes,
                   "component_plan": [{"type": "html", "title": "Giải thích", "rationale": "Nội dung chính",
                                       "author_review": {"purpose": "Giải thích nội dung", "example_scenario": None,
                                                         "visual_asset": None, "user_behavior_navigation": None},
                                       "source_scope_ids": scopes}], "media_brief": None}],
    }


def unit_contract_wire() -> dict:
    base = {
        "contract_version": 2, "source_snapshot_hash": SOURCE_HASH, "assembly_hash": "b" * 64,
        "chapter_key": "chapter-1", "unit_path": "chapter_1.lesson_1.unit_1",
        "chapter_title": "Chương 1", "lesson_title": "Bài 1",
        "lesson_learning_objectives": ["Áp dụng"], "unit_title": "Nội dung",
        "unit_purpose": "Giải thích", "unit_learning_objective_refs": ["lo_1"],
        "unit_source_scope_ids": ["scope-1"], "unit_source_fact_ids": ["fact-1"],
        "component_plan": [{"component_plan_id": "cp2_" + "c" * 32, "type": "html",
            "title": "Giải thích", "rationale": "Nội dung chính", "purpose": "explain",
            "source_fact_ids": ["fact-1"], "supporting_evidence_fact_ids": [],
            "learning_objective_refs": ["lo_1"], "source_scope_ids": ["scope-1"],
            "content_requirements": [], "learning_block_ids": [], "required_artifacts": []}],
        "source_facts": [{"document_id": "00000000-0000-4000-8000-000000000005",
            "fact_key": "fact-1", "scope_key": "scope-1", "fact_text": "Alpha",
            "source_ref": None, "source_page": 1, "source_chunk": 0, "locator": {}}],
    }
    return {**base, "contract_hash": canonical_hash(base)}


class LessonAuthorOrchestrationV2EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_unit_fallback_only_preserves_every_supported_component_without_provider_call(self) -> None:
        component_types = ["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]
        for component_type in component_types:
            with self.subTest(component_type=component_type):
                contract = unit_contract_wire()
                contract["unit_source_fact_ids"] = ["fact-1", "fact-2", "fact-3"]
                contract["source_facts"] = [{
                    "document_id": "00000000-0000-4000-8000-000000000005",
                    "fact_key": f"fact-{index}", "scope_key": "scope-1",
                    "fact_text": text, "source_ref": "Mục nguồn", "source_page": index,
                    "source_chunk": index - 1, "locator": {},
                } for index, text in enumerate([
                    "Người học đọc quy trình đầy đủ, xác định mục tiêu và kiểm tra từng điều kiện an toàn trước khi bắt đầu.",
                    "Tiếp theo người học đối chiếu ví dụ, thực hành theo đúng thứ tự và ghi lại kết quả quan sát được.",
                    "Cuối cùng người học tự đánh giá kết quả, giải thích lựa chọn và rà soát lại nội dung theo tài liệu nguồn.",
                ], start=1)]
                plan = contract["component_plan"][0]
                plan["source_fact_ids"] = ["fact-1", "fact-2", "fact-3"]
                if component_type != "html":
                    contract["component_plan"].append({
                        **plan, "component_plan_id": "cp2_" + f"{len(component_type):032x}",
                        "type": component_type, "title": f"Nội dung {component_type}",
                        "purpose": ({"problem": "assess", "la_faq": "clarify",
                                     "la_sortable": "sequence", "la_crossword": "terminology",
                                     "la_diagram": "relationship"})[component_type],
                    })
                contract["contract_hash"] = canonical_hash({
                    key: value for key, value in contract.items() if key != "contract_hash"
                })
                request = RagLessonAuthorUnitV2Request.model_validate({
                    **common(),
                    "source_documents": [{
                        "document_id": "00000000-0000-4000-8000-000000000005",
                        "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
                    }],
                    "unit_contract": contract, "max_attempts": 1,
                    "remaining_workflow_budget_ms": 60_000, "fallback_only": True,
                })
                with patch("app.main.generate_staged_lesson_author_proposal", AsyncMock()) as provider:
                    response = await lesson_author_orchestration_v2_unit(request)
                provider.assert_not_called()
                self.assertEqual(response["usage_source"], "deterministic_fallback")
                self.assertTrue(response["usage_complete"])
                expected_types = ["html"] if component_type == "html" else ["html", component_type]
                self.assertEqual([item["type"] for item in response["unit"]["components"]], expected_types)
                for item, expected_plan in zip(response["unit"]["components"], contract["component_plan"]):
                    self.assertEqual(item["component_plan_id"], expected_plan["component_plan_id"])
                    self.assertEqual(item["source_fact_ids"], expected_plan["source_fact_ids"])

    async def test_unit_provider_failure_returns_valid_upper_bound_fallback(self) -> None:
        request_data = common()
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorUnitV2Request.model_validate({
            **request_data, "unit_contract": unit_contract_wire(), "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
        })
        with patch("app.main.generate_staged_lesson_author_proposal",
                   AsyncMock(side_effect=ValueError("private provider payload"))) as provider:
            response = await lesson_author_orchestration_v2_unit(request)
        provider.assert_awaited_once()
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["unit"]["components"][0]["type"], "html")

    async def test_unit_incomplete_provider_usage_never_crosses_the_service_boundary(self) -> None:
        request_data = common()
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorUnitV2Request.model_validate({
            **request_data, "unit_contract": unit_contract_wire(), "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
        })
        component = {"type": "html", "title": "Giải thích", "data": "<p>Alpha</p>",
            "component_plan_id": "cp2_" + "c" * 32, "source_fact_ids": ["fact-1"],
            "covered_source_fact_ids": ["fact-1"], "supporting_evidence_fact_ids": []}
        staged = AsyncMock(return_value=({"unit": {"title": "Nội dung",
            "source_fact_ids": ["fact-1"], "components": [component]},
            "provider_usage_complete": False}, AiUsage(inputTokens=10, outputTokens=5)))
        with patch("app.main.generate_staged_lesson_author_proposal", staged):
            response = await lesson_author_orchestration_v2_unit(request)
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertTrue(response["unit"]["source_locked_fallback"])

    async def test_unit_endpoint_invokes_one_bounded_staged_generation_without_database_work(self) -> None:
        request_data = common()
        request_data["source_documents"] = [{"document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned"}]
        request = RagLessonAuthorUnitV2Request.model_validate({**request_data,
            "unit_contract": unit_contract_wire(), "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000})
        component = {"type": "html", "title": "Giải thích", "data": "<p>Alpha</p>",
            "component_plan_id": "cp2_" + "c" * 32, "source_fact_ids": ["fact-1"],
            "covered_source_fact_ids": ["fact-1"], "supporting_evidence_fact_ids": []}
        staged = AsyncMock(return_value=({"unit": {"title": "Nội dung",
            "source_fact_ids": ["fact-1"], "components": [component]},
            "provider_usage_complete": True}, AiUsage(inputTokens=10, outputTokens=5)))
        with patch("app.main.generate_staged_lesson_author_proposal", staged):
            response = await lesson_author_orchestration_v2_unit(request)
        staged.assert_awaited_once()
        args, kwargs = staged.await_args
        self.assertEqual(kwargs["checkpoint_unit_index"], 0)
        self.assertEqual(kwargs["source_coverage_manifest"]["facts"][0]["fact_id"], "fact-1")
        self.assertEqual(response["unit_path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(response["usage_source"], "provider")

    async def test_source_snapshot_uses_no_generation_provider_call(self) -> None:
        request_data = common()
        request_data["source_snapshot_hash"] = SOURCE_HASH
        request_data.pop("api_key")
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID,
            "name": "Nguồn.pdf",
            "type": "pdf",
            "status": "learned",
        }]
        request_data["page_max_facts"] = 1
        request = RagLessonAuthorSourceSnapshotV2Request.model_validate(request_data)
        document_id = request.source_documents[0].document_id
        index_row = {"index_id": "00000000-0000-4000-8000-000000000006",
                     "document_id": document_id, "content_sha256": "content", "chunk_count": 2,
                     "embedding_model": "test-embedding", "embedding_dimensions": 768}
        chunks = [{"content": "Alpha.", "source_page": 1, "source_section": "Mục 1", "metadata": {},
                   "chunk_no": 0, "index_id": index_row["index_id"], "document_id": document_id,
                   "document_name": "Nguồn.pdf"},
                  {"content": "Beta.", "source_page": 2, "source_section": "Mục 2", "metadata": {},
                   "chunk_no": 1, "index_id": index_row["index_id"], "document_id": document_id,
                   "document_name": "Nguồn.pdf"}]

        class Pool:
            async def fetch(self, sql: str, *args: object) -> list[dict[str, object]]:
                if "FROM rag_document_indexes" in sql:
                    return [index_row]
                if "metadata->'source_structure'" in sql:
                    return []
                after_chunk = int(args[4])
                return [row for row in chunks if int(row["chunk_no"]) > after_chunk]

        with patch("app.main.generate_content", AsyncMock()) as provider, \
                patch("app.main.embed_texts", AsyncMock()) as embedder:
            response = await lesson_author_orchestration_v2_source_snapshot(request, pool=Pool())
            next_request = RagLessonAuthorSourceSnapshotV2Request.model_validate({
                **request_data, "cursor": response["next_cursor"],
                "expected_source_revision": response["source_revision"],
            })
            final = await lesson_author_orchestration_v2_source_snapshot(next_request, pool=Pool())
        provider.assert_not_called()
        embedder.assert_not_called()
        self.assertEqual(response["contract_version"], 2)
        self.assertEqual(response["source_snapshot_hash"], SOURCE_HASH)
        # Missing parser evidence is not proof that the document has no
        # structure. Current-parser NONE evidence is model-designed; an absent
        # structure row fails closed for review.
        self.assertEqual(response["source_authority"]["mode"], "needs_review")
        self.assertEqual(len(response["facts"]), 1)
        self.assertTrue(response["has_more"])
        self.assertEqual(final["facts"][0]["fact_key"], "d1-c2-f1")
        self.assertNotEqual(response["facts"][0]["scope_key"], final["facts"][0]["scope_key"])
        self.assertFalse(final["has_more"])
        changed_request = RagLessonAuthorSourceSnapshotV2Request.model_validate({
            **request_data, "expected_source_revision": "f" * 64,
        })
        with self.assertRaises(HTTPException) as changed:
            await lesson_author_orchestration_v2_source_snapshot(changed_request, pool=Pool())
        self.assertEqual(changed.exception.status_code, 409)
        self.assertEqual(changed.exception.detail["code"], "SOURCE_REVISION_CHANGED")

    async def test_course_skeleton_binds_exact_scope_catalog(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate({
            **common(), "source_snapshot_hash": SOURCE_HASH,
            "source_authority": model_designed_authority(),
            "scope_catalog": [{"scope_key": "scope-1", "title": "Mục 1", "source_ref": None,
                               "fact_count": 1, "content_chars": 5}], "max_attempts": 2,
        })
        draft = {key: value for key, value in skeleton_wire().items()
                 if key not in {"contract_version", "source_snapshot_hash", "locale"}}
        with patch("app.main.generate_content", AsyncMock(return_value=(json.dumps(draft), AiUsage(outputTokens=20)))) as provider:
            response = await lesson_author_orchestration_v2_course_skeleton(request)
        provider.assert_awaited_once()
        self.assertIs(provider.await_args.kwargs["response_schema"], CourseSkeletonProviderWireV2)
        self.assertEqual(response["skeleton"]["source_snapshot_hash"], SOURCE_HASH)
        self.assertEqual(response["skeleton"]["chapters"][0]["source_scope_ids"], ["scope-1"])

    async def test_course_skeleton_returns_safe_terminal_code_for_provider_400(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate({
            **common(), "source_snapshot_hash": SOURCE_HASH,
            "source_authority": model_designed_authority(),
            "scope_catalog": [{"scope_key": "scope-1", "title": "Mục 1", "source_ref": None,
                               "fact_count": 1, "content_chars": 5}], "max_attempts": 1,
        })
        provider_error = type("ProviderError", (Exception,), {"status_code": 400})("PRIVATE_PROVIDER_BODY")
        with patch("app.main.generate_content", AsyncMock(side_effect=provider_error)), \
                self.assertLogs("app.main", "WARNING") as logs, self.assertRaises(HTTPException) as failure:
            await lesson_author_orchestration_v2_course_skeleton(request)
        self.assertEqual(failure.exception.status_code, 502)
        self.assertEqual(failure.exception.detail["code"], "AI_PROVIDER_REQUEST_REJECTED")
        self.assertNotIn("PRIVATE_PROVIDER_BODY", "\n".join(logs.output))

    async def test_course_skeleton_uses_source_complete_fallback_after_malformed_outputs(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate({
            **common(), "source_snapshot_hash": SOURCE_HASH,
            "source_authority": model_designed_authority(),
            "scope_catalog": [
                {"scope_key": "scope-1", "title": "Mục 1", "source_ref": None,
                 "fact_count": 1, "content_chars": 5},
                {"scope_key": "scope-2", "title": "Mục 2", "source_ref": None,
                 "fact_count": 1, "content_chars": 5},
            ], "max_attempts": 2,
        })
        with patch("app.main.generate_content", AsyncMock(return_value=("{}", AiUsage(outputTokens=3)))) as provider:
            response = await lesson_author_orchestration_v2_course_skeleton(request)
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(response["usage"]["outputTokens"], 6)
        self.assertEqual(
            [scope for chapter in response["skeleton"]["chapters"] for scope in chapter["source_scope_ids"]],
            ["scope-1", "scope-2"],
        )

    async def test_chapter_shard_freezes_server_owned_identity(self) -> None:
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            **common(), "skeleton": skeleton_wire(),
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1"], "source_fact_count": 1, "source_content_chars": 5},
            "source_facts": [{"document_id": "document-1", "fact_key": "fact-1", "scope_key": "scope-1",
                              "fact_text": "Alpha", "source_ref": None, "source_page": 1,
                              "source_chunk": 0, "locator": {}}],
            "max_attempts": 2,
        })
        with patch("app.main.generate_content", AsyncMock(return_value=(
            json.dumps({"lessons": [lesson_wire(["scope-1"])]}), AiUsage(outputTokens=20),
        ))) as provider:
            response = await lesson_author_orchestration_v2_chapter_shard(request)
        provider.assert_awaited_once()
        self.assertIs(provider.await_args.kwargs["response_schema"], ChapterShardProviderWireV2)
        self.assertEqual(response["shard"]["title"], "Chương 1")
        self.assertEqual(response["shard"]["source_scope_ids"], ["scope-1"])

    async def test_chapter_shard_falls_back_without_terminal_failure_after_invalid_provider_output(self) -> None:
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            **common(), "skeleton": skeleton_wire(),
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1"], "source_fact_count": 1, "source_content_chars": 5},
            "source_facts": [{"document_id": "document-1", "fact_key": "fact-1", "scope_key": "scope-1",
                              "fact_text": "Alpha", "source_ref": None, "source_page": 1,
                              "source_chunk": 0, "locator": {}}],
            "max_attempts": 1,
        })
        with patch("app.main.generate_content", AsyncMock(return_value=(
            json.dumps({"lessons": [{"invalid": True}]}), AiUsage(outputTokens=20),
        ))) as provider:
            response = await lesson_author_orchestration_v2_chapter_shard(request)
        provider.assert_awaited_once()
        self.assertEqual(response["shard"]["source_scope_ids"], ["scope-1"])
        self.assertEqual(response["shard"]["lessons"][0]["units"][0]["source_scope_ids"], ["scope-1"])

    async def test_unit_adapts_complete_source_rows_before_staged_generation(self) -> None:
        unit_contract = unit_contract_wire()
        unit_contract["source_facts"][0]["source_ref"] = "Phần mở đầu"
        unit_contract["contract_hash"] = canonical_hash({
            key: value for key, value in unit_contract.items() if key != "contract_hash"
        })
        request_data = common()
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorUnitV2Request.model_validate({
            **request_data,
            "remaining_workflow_budget_ms": 60_000,
            "unit_contract": unit_contract,
        })
        generated = {"unit": {"title": "Nội dung", "components": []}, "provider_usage_complete": True}
        with patch("app.main.generate_staged_lesson_author_proposal",
                   AsyncMock(return_value=(generated, AiUsage(outputTokens=20)))) as provider:
            await lesson_author_orchestration_v2_unit(request)
        rows = provider.await_args.kwargs["source_rows"]
        self.assertEqual(rows[0]["document_name"], "Nguồn.pdf")
        self.assertEqual(rows[0]["source_section"], "Phần mở đầu")
        self.assertEqual(rows[0]["score"], 1.0)
        self.assertEqual(rows[0]["metadata"]["fact_key"], "fact-1")


if __name__ == "__main__":
    unittest.main()
