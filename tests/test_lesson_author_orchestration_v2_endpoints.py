import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.main import (
    ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS,
    STAGED_COMPONENT_PAYLOAD_FIELDS,
    validate_staged_unit_content,
    lesson_author_orchestration_v2_chapter_shard,
    lesson_author_orchestration_v2_course_skeleton,
    lesson_author_orchestration_v2_source_snapshot,
    lesson_author_orchestration_v2_unit,
)
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import (
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
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


def relationship_unit_contract_wire(*, include_html: bool) -> dict:
    base = unit_contract_wire()
    base["unit_content_policy_version"] = "unit-content-v4-alignment-1"
    fact_ids = ["fact-1", "fact-2", "fact-3"]
    base["unit_source_fact_ids"] = fact_ids
    base["source_facts"] = [{
        "document_id": "00000000-0000-4000-8000-000000000005",
        "fact_key": fact_id,
        "scope_key": "scope-1",
        "fact_text": text,
        "source_ref": "Bảng nguồn nội bộ",
        "source_page": 4,
        "source_chunk": index,
        "locator": {},
    } for index, (fact_id, text) in enumerate(zip(fact_ids, (
        "Row 1: Chu trình | Công cụ BiC",
        "Row 2: PLAN | Change Mindset + BQS/BMQ",
        "Row 3: DO | SPD (Standard Product Dev)",
    )))]
    diagram = {
        **base["component_plan"][0],
        "component_plan_id": "cp2_" + "d" * 32,
        "type": "la_diagram",
        "title": "Quan hệ chu trình và công cụ",
        "purpose": "relationship",
        "source_fact_ids": [] if include_html else fact_ids,
        "supporting_evidence_fact_ids": fact_ids if include_html else [],
        "required_artifacts": [],
    }
    if include_html:
        html = {
            **base["component_plan"][0],
            "source_fact_ids": fact_ids,
            "required_artifacts": [{"type": "table", "minimum_items": 2}],
        }
        base["component_plan"] = [html, diagram]
    else:
        base["component_plan"] = [diagram]
    base["contract_hash"] = canonical_hash({
        key: value for key, value in base.items() if key != "contract_hash"
    })
    return base


class LessonAuthorOrchestrationV2EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_component_unit_reaches_provider_with_supporting_evidence(self) -> None:
        contract = relationship_unit_contract_wire(include_html=True)
        request_payload = {
            **common(),
            "source_documents": [{
                "document_id": "00000000-0000-4000-8000-000000000005",
                "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
            }],
            "unit_contract": contract,
            "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
        }
        fallback = await lesson_author_orchestration_v2_unit(
            RagLessonAuthorUnitV2Request.model_validate({**request_payload, "fallback_only": True}),
        )
        slots = {}
        for index, component in enumerate(fallback["unit"]["components"]):
            fields = STAGED_COMPONENT_PAYLOAD_FIELDS[component["type"]] | {
                "title", "selection_rationale", "covered_source_fact_ids",
            }
            slots[f"c{index}"] = {key: deepcopy(value) for key, value in component.items() if key in fields}
        slots["c0"]["html"] = (
            "<h2>Giải thích mối quan hệ</h2>"
            "<p>Bảng dữ liệu nguồn ghép từng giai đoạn trong chu trình với đúng công cụ BiC tương ứng. "
            "Ở hàng thứ nhất, giai đoạn PLAN đi cùng Change Mindset và BQS/BMQ. Ở hàng thứ hai, "
            "giai đoạn DO đi cùng SPD, tức Standard Product Dev.</p>"
            "<table><thead><tr><th>Chu trình</th><th>Công cụ BiC</th></tr></thead><tbody>"
            "<tr><td>PLAN</td><td>Change Mindset + BQS/BMQ</td></tr>"
            "<tr><td>DO</td><td>SPD (Standard Product Dev)</td></tr></tbody></table>"
            "<h3>Cách đọc</h3><p>Đọc theo từng hàng để giữ nguyên cặp quan hệ mà tài liệu nguồn đã nêu, "
            "sau đó đối chiếu với sơ đồ ở phần tiếp theo.</p>"
        )
        async def generate(*_args, **kwargs):
            telemetry = kwargs.get("on_provider_telemetry")
            if telemetry:
                telemetry({
                    "event": "provider_response_received",
                    "provider_attempt": 1,
                    "provider_http_status": 200,
                    "provider_finish_reason": "FinishReason.STOP",
                    "usage_source": "provider",
                    "provider_input_tokens": 20,
                    "provider_output_tokens": 30,
                    "provider_total_tokens": 50,
                })
            return json.dumps({"components": slots}), AiUsage(
                inputTokens=20, outputTokens=30, totalTokens=50,
            )

        provider = AsyncMock(side_effect=generate)

        with patch("app.services.provider.generate_content", provider):
            response = await lesson_author_orchestration_v2_unit(
                RagLessonAuthorUnitV2Request.model_validate(request_payload),
            )

        self.assertGreaterEqual(provider.await_count, 1)
        self.assertEqual(response["content_origin"], "provider_validated")
        self.assertEqual([item["type"] for item in response["unit"]["components"]], ["html", "la_diagram"])

    async def test_relationship_fallback_preserves_table_html_and_exact_diagram_pairs(self) -> None:
        contract = relationship_unit_contract_wire(include_html=True)
        request = RagLessonAuthorUnitV2Request.model_validate({
            **common(),
            "source_documents": [{
                "document_id": "00000000-0000-4000-8000-000000000005",
                "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
            }],
            "unit_contract": contract,
            "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
            "fallback_only": True,
        })

        response = await lesson_author_orchestration_v2_unit(request)

        html, diagram = response["unit"]["components"]
        self.assertEqual([html["type"], diagram["type"]], ["html", "la_diagram"])
        self.assertIn("<table>", html["html"])
        self.assertNotIn("Nguồn.pdf", html["html"])
        self.assertNotIn("Bảng nguồn nội bộ", html["html"])
        labels = [node["label"] for node in diagram["nodes"]]
        pairs = {(labels[edge["source"]], labels[edge["target"]]) for edge in diagram["edges"]}
        self.assertEqual(pairs, {
            ("PLAN", "Change Mindset + BQS/BMQ"),
            ("DO", "SPD (Standard Product Dev)"),
        })
        self.assertEqual(response["quality_state"], "review_required")

    async def test_interaction_led_relationship_unit_does_not_require_html(self) -> None:
        contract = relationship_unit_contract_wire(include_html=False)
        request = RagLessonAuthorUnitV2Request.model_validate({
            **common(),
            "source_documents": [{
                "document_id": "00000000-0000-4000-8000-000000000005",
                "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
            }],
            "unit_contract": contract,
            "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
            "fallback_only": True,
        })

        response = await lesson_author_orchestration_v2_unit(request)

        self.assertEqual([item["type"] for item in response["unit"]["components"]], ["la_diagram"])
        self.assertEqual(response["unit"]["components"][0]["source_fact_ids"], contract["unit_source_fact_ids"])
        self.assertEqual(response["quality_state"], "review_required")

    async def test_diagram_validator_rejects_a_missing_locked_source_relation(self) -> None:
        contract = relationship_unit_contract_wire(include_html=False)
        request = RagLessonAuthorUnitV2Request.model_validate({
            **common(),
            "source_documents": [{
                "document_id": "00000000-0000-4000-8000-000000000005",
                "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
            }],
            "unit_contract": contract,
            "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
            "fallback_only": True,
        })
        response = await lesson_author_orchestration_v2_unit(request)
        unit = deepcopy(response["unit"])
        first_edge = unit["components"][0]["edges"][0]
        unit["components"][0]["edges"] = [
            first_edge,
            {"source": 1, "target": 2, "label": "Liên quan"},
            {"source": 1, "target": 3, "label": "Sai quan hệ"},
        ]
        expected = {
            "component_types": ["la_diagram"],
            "component_plan": contract["component_plan"],
            "source_fact_ids": contract["unit_source_fact_ids"],
            "supporting_evidence_fact_ids": [],
            "diagram_relationships_by_plan_id": {
                contract["component_plan"][0]["component_plan_id"]: [
                    ["PLAN", "Change Mindset + BQS/BMQ", "Công cụ BiC"],
                    ["DO", "SPD (Standard Product Dev)", "Công cụ BiC"],
                ],
            },
        }

        finding = validate_staged_unit_content(unit, expected, strict_payload=True)

        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding.code, "DIAGRAM_SOURCE_RELATION_MISSING")

    async def test_unit_fallback_only_preserves_every_supported_component_without_provider_call(self) -> None:
        component_types = ["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]
        for component_type in component_types:
            with self.subTest(component_type=component_type):
                contract = unit_contract_wire()
                contract["unit_source_fact_ids"] = ["fact-1", "fact-2", "fact-3"]
                source_texts = ([
                    "Câu hỏi: Bước đầu tiên khi đánh giá rủi ro là gì? A. Nhận diện mối nguy. B. Bỏ qua nguồn phát sinh. C. Chờ xảy ra sự cố.",
                    "Đáp án: A",
                    "Giải thích: Nhận diện mối nguy là dữ kiện đầu tiên của quy trình đánh giá rủi ro.",
                ] if component_type == "problem" else [
                    "Mối nguy cơ khí: Nguồn chuyển động có thể gây va đập, cuốn hoặc kẹp người lao động.",
                    "Mối nguy điện: Nguồn điện không được kiểm soát có thể gây điện giật hoặc hồ quang.",
                    "Mối nguy hóa chất: Phơi nhiễm cần được kiểm soát theo đặc tính của hóa chất.",
                ] if component_type == "la_crossword" else [
                    "Nếu phát hiện điều kiện không an toàn, người lao động phải dừng công việc và báo cho người phụ trách.",
                    "Khi biện pháp kiểm soát chưa được xác nhận hiệu quả, công việc chỉ được tiếp tục sau khi hoàn tất đánh giá lại.",
                    "Lưu ý: Không được bỏ qua bước xác nhận vì công việc đã từng được thực hiện trước đó.",
                ] if component_type == "la_faq" else [
                    "Bước 1: Nhận diện mối nguy và ghi rõ nguồn phát sinh trước khi bắt đầu công việc.",
                    "Bước 2: Đánh giá khả năng xảy ra và mức hậu quả theo tiêu chí đã quy định.",
                    "Bước 3: Chọn biện pháp kiểm soát phù hợp, phân công thực hiện và xác nhận kết quả.",
                ])
                contract["source_facts"] = [{
                    "document_id": "00000000-0000-4000-8000-000000000005",
                    "fact_key": f"fact-{index}", "scope_key": "scope-1",
                    "fact_text": text, "source_ref": "Mục nguồn", "source_page": index,
                    "source_chunk": index - 1, "locator": {},
                } for index, text in enumerate(source_texts, start=1)]
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
                self.assertEqual(response["content_origin"], "structured_fallback")
                self.assertEqual(response["quality_state"], "review_required")
                self.assertEqual(response["attempt_trace"][0]["outcome"], "fallback")
                self.assertEqual(response["attempt_trace"][0]["usage_source"], "unknown")
                expected_types = ["html"] if component_type == "html" else ["html", component_type]
                self.assertEqual([item["type"] for item in response["unit"]["components"]], expected_types)
                for item, expected_plan in zip(response["unit"]["components"], contract["component_plan"]):
                    self.assertEqual(item["component_plan_id"], expected_plan["component_plan_id"])
                    self.assertEqual(item["source_fact_ids"], expected_plan["source_fact_ids"])

    async def test_unreviewed_visual_evidence_preserves_fallback_draft_as_review_required(self) -> None:
        contract = unit_contract_wire()
        contract["source_facts"][0]["locator"] = {
            "source_revision": "d" * 64,
            "visual_regions": [{
                "region_kind": "embedded_image",
                "asset_revision": "e" * 64,
                "locator": {"page": 1, "bbox_normalized": [0.1, 0.2, 0.8, 0.9]},
                "observation": {"status": "unreviewed", "facts": []},
                "inference": {"status": "not_performed", "claims": []},
            }],
        }
        contract["contract_hash"] = canonical_hash({
            key: value for key, value in contract.items() if key != "contract_hash"
        })
        request = RagLessonAuthorUnitV2Request.model_validate({
            **common(),
            "source_documents": [{
                "document_id": "00000000-0000-4000-8000-000000000005",
                "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
            }],
            "unit_contract": contract,
            "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
            "fallback_only": True,
        })

        response = await lesson_author_orchestration_v2_unit(request)

        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["unit"]["title"], contract["unit_title"])

    async def test_unit_first_attempt_failure_preserves_the_durable_dispatch_reservation(self) -> None:
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
        # Node persists the dispatch fence before this HTTP call. Even if the
        # Python provider adapter fails before Gemini, the first response must
        # keep the reservation conservative; the durable fallback-only replay
        # is the provider-free completion path.
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["unit"]["components"][0]["type"], "html")
        self.assertEqual(response["attempt_trace"][-1]["outcome"], "fallback")

    async def test_unit_post_dispatch_failure_preserves_upper_bound_accounting(self) -> None:
        request_data = common()
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorUnitV2Request.model_validate({
            **request_data, "unit_contract": unit_contract_wire(), "max_attempts": 1,
            "remaining_workflow_budget_ms": 60_000,
        })

        async def staged(*_args, **kwargs):
            kwargs["on_provider_dispatch"]()
            raise ValueError("private provider payload")

        with patch("app.main.generate_staged_lesson_author_proposal", staged):
            response = await lesson_author_orchestration_v2_unit(request)
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["quality_state"], "review_required")

    async def test_unit_deadline_keeps_response_headroom_for_reviewable_fallback(self) -> None:
        request_data = common()
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorUnitV2Request.model_validate({
            **request_data, "unit_contract": unit_contract_wire(), "max_attempts": 1,
            "remaining_workflow_budget_ms": ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS + 1,
        })

        async def stalled(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            raise AssertionError("the inner generation deadline must cancel this call")

        with patch("app.main.generate_staged_lesson_author_proposal", stalled):
            response = await lesson_author_orchestration_v2_unit(request)

        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["attempt_trace"][-1]["failure_code"], "AI_STAGED_LESSON_WORKFLOW_TIMEOUT")

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
        self.assertEqual(
            kwargs["remaining_workflow_budget_ms"],
            request.remaining_workflow_budget_ms - ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS,
        )
        self.assertEqual(kwargs["source_coverage_manifest"]["facts"][0]["fact_id"], "fact-1")
        self.assertEqual(response["unit_path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(response["usage_source"], "provider")
        self.assertEqual(response["content_origin"], "provider_validated")

    async def test_unit_repair_mode_records_a_successful_independent_semantic_review(self) -> None:
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
            "provider_usage_complete": True},
            AiUsage(inputTokens=10, outputTokens=5, totalTokens=15)))

        async def semantic_provider(*_args, **kwargs):
            telemetry = kwargs["on_provider_telemetry"]
            telemetry({"event": "provider_http_attempt_started", "provider_attempt": 1})
            telemetry({"event": "provider_http_attempt_succeeded", "provider_attempt": 1,
                       "provider_http_status": 200})
            telemetry({"event": "provider_response_received", "provider_attempt": 1,
                       "provider_http_status": 200, "usage_source": "provider",
                       "provider_input_tokens": 2, "provider_output_tokens": 1,
                       "provider_total_tokens": 3})
            return (json.dumps({
                "contract_version": "semantic-review-v1", "verdict": "pass",
                "reviewed_component_indices": [0], "findings": [],
            }), AiUsage(inputTokens=2, outputTokens=1, totalTokens=3))

        with patch("app.main.settings.semantic_review_mode", "repair"), \
                patch("app.main.generate_staged_lesson_author_proposal", staged), \
                patch("app.services.provider.generate_content", semantic_provider):
            response = await lesson_author_orchestration_v2_unit(request)

        self.assertEqual(response["semantic_review"]["status"], "passed")
        self.assertEqual(response["semantic_review"]["quality_state"], "validated")
        self.assertEqual(response["quality_state"], "validated")
        self.assertEqual(response["usage"], {
            "inputTokens": 12, "outputTokens": 6, "embeddingTokens": 0, "totalTokens": 18,
        })
        self.assertTrue(response["usage_complete"])
        self.assertTrue(any(event["invocation_kind"] == "evaluator"
                            for event in response["attempt_trace"]))

    async def test_unit_semantic_provider_failure_keeps_visible_draft_and_reserved_accounting(self) -> None:
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
            "provider_usage_complete": True},
            AiUsage(inputTokens=10, outputTokens=5, totalTokens=15)))

        async def failed_semantic_provider(*_args, **kwargs):
            kwargs["on_provider_telemetry"]({
                "event": "provider_http_attempt_started", "provider_attempt": 1,
            })
            raise RuntimeError("private provider payload")

        with patch("app.main.settings.semantic_review_mode", "repair"), \
                patch("app.main.generate_staged_lesson_author_proposal", staged), \
                patch("app.services.provider.generate_content", failed_semantic_provider):
            response = await lesson_author_orchestration_v2_unit(request)

        self.assertEqual(response["unit"]["components"], [component])
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["semantic_review"]["status"], "unavailable")
        self.assertEqual(response["semantic_review"]["failure_code"], "SEMANTIC_REVIEW_UNAVAILABLE")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertNotIn("private provider payload", json.dumps(response))

    async def test_unit_provider_usage_can_preserve_valid_siblings_with_component_fallback(self) -> None:
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
            "provider_usage_complete": True, "fallback_component_indices": [0]},
            AiUsage(inputTokens=10, outputTokens=5)))
        with patch("app.main.generate_staged_lesson_author_proposal", staged):
            response = await lesson_author_orchestration_v2_unit(request)
        self.assertEqual(response["usage_source"], "provider")
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "validated")

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
        chunks = [{"content": "Alpha.", "source_page": 1, "source_section": "Mục 1",
                   "metadata": {"content_kinds": ["text", "table"], "table_count": 1},
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

        with patch("app.services.provider.generate_content", AsyncMock()) as provider, \
                patch("app.services.provider.embed_texts", AsyncMock()) as embedder:
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
        self.assertEqual(response["source_revision"], canonical_hash([{
            "document_id": document_id,
            "index_id": index_row["index_id"],
            "content_sha256": "content",
            "chunk_count": 2,
            "embedding_model": "test-embedding",
            "embedding_dimensions": 768,
            "source_evidence_status": "legacy_review_required",
        }]))
        # Missing parser evidence is not proof that the document has no
        # structure. Current-parser NONE evidence is model-designed; an absent
        # structure row fails closed for review.
        self.assertEqual(response["source_authority"]["mode"], "needs_review")
        self.assertIn(
            "STRUCTURED_EVIDENCE_REVISION_MISSING",
            response["source_authority"]["reason_codes"],
        )
        self.assertEqual(len(response["facts"]), 1)
        # Chunk-level table metadata must not leak into a prose-only density
        # lane; otherwise the unit contract requires an unreconstructable table.
        self.assertEqual(response["facts"][0]["locator"]["content_kinds"], ["text"])
        self.assertEqual(response["facts"][0]["locator"]["table_count"], 0)
        self.assertEqual(
            response["facts"][0]["locator"]["source_evidence_status"],
            "legacy_review_required",
        )
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

    async def test_source_snapshot_materializes_explicit_evidence_revision_after_reindex(self) -> None:
        request_data = common()
        request_data["source_snapshot_hash"] = SOURCE_HASH
        request_data.pop("api_key")
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorSourceSnapshotV2Request.model_validate(request_data)
        document_id = request.source_documents[0].document_id
        index_row = {
            "index_id": "00000000-0000-4000-8000-000000000006",
            "document_id": document_id,
            "content_sha256": "content",
            "chunk_count": 1,
            "embedding_model": "test-embedding",
            "embedding_dimensions": 768,
        }
        evidence_revision = "e" * 64
        chunk = {
            "content": "Alpha.", "source_page": 1, "source_section": "Mục 1",
            "metadata": {"source_evidence_revision": evidence_revision},
            "chunk_no": 0, "index_id": index_row["index_id"],
            "document_id": document_id, "document_name": "Nguồn.pdf",
        }

        class Pool:
            async def fetch(self, sql: str, *args: object) -> list[dict[str, object]]:
                if "FROM rag_document_indexes" in sql:
                    return [index_row]
                if "metadata->'source_structure'" in sql:
                    return [{
                        "document_id": document_id,
                        "source_structure": {
                            "structure_source": "none", "confidence": 0.0,
                            "parser_version": "source-structure-v1", "nodes": [],
                        },
                        "source_evidence_revision": evidence_revision,
                    }]
                return [chunk]

        response = await lesson_author_orchestration_v2_source_snapshot(request, pool=Pool())

        self.assertEqual(response["source_revision"], canonical_hash([{
            "document_id": document_id,
            "index_id": index_row["index_id"],
            "content_sha256": "content",
            "chunk_count": 1,
            "embedding_model": "test-embedding",
            "embedding_dimensions": 768,
            "source_evidence_status": "ready",
            "source_evidence_revision": evidence_revision,
        }]))
        self.assertEqual(
            response["facts"][0]["locator"]["source_evidence_revision"],
            evidence_revision,
        )
        self.assertEqual(response["facts"][0]["locator"]["source_evidence_status"], "ready")
        self.assertNotIn(
            "STRUCTURED_EVIDENCE_REVISION_MISSING",
            response["source_authority"]["reason_codes"],
        )

    async def test_source_snapshot_rejects_a_mixed_evidence_revision_index(self) -> None:
        request_data = common()
        request_data["source_snapshot_hash"] = SOURCE_HASH
        request_data.pop("api_key")
        request_data["source_documents"] = [{
            "document_id": "00000000-0000-4000-8000-000000000005",
            "kb_id": KB_ID, "name": "Nguồn.pdf", "type": "pdf", "status": "learned",
        }]
        request = RagLessonAuthorSourceSnapshotV2Request.model_validate(request_data)
        document_id = request.source_documents[0].document_id
        index_row = {
            "index_id": "00000000-0000-4000-8000-000000000006",
            "document_id": document_id, "content_sha256": "content", "chunk_count": 2,
            "embedding_model": "test-embedding", "embedding_dimensions": 768,
        }

        class Pool:
            async def fetch(self, sql: str, *_args: object) -> list[dict[str, object]]:
                if "FROM rag_document_indexes" in sql:
                    return [index_row]
                if "metadata->'source_structure'" in sql:
                    return [{
                        "document_id": document_id, "source_structure": None,
                        "source_evidence_revision": "e" * 64,
                        "actual_chunk_count": 2,
                        "evidence_revision_chunk_count": 1,
                        "evidence_revision_distinct_count": 1,
                    }]
                raise AssertionError("chunk fetch must not run for a mixed evidence revision")

        with self.assertRaises(HTTPException) as rejected:
            await lesson_author_orchestration_v2_source_snapshot(request, pool=Pool())
        self.assertEqual(rejected.exception.status_code, 409)
        self.assertEqual(
            rejected.exception.detail["code"],
            "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
        )

    async def test_course_skeleton_binds_exact_scope_catalog(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate({
            **common(), "source_snapshot_hash": SOURCE_HASH,
            "source_authority": model_designed_authority(),
            "scope_catalog": [{"scope_key": "scope-1", "title": "Mục 1", "source_ref": None,
                               "fact_count": 1, "content_chars": 5}], "max_attempts": 2,
        })
        draft = {key: value for key, value in skeleton_wire().items()
                 if key not in {"contract_version", "source_snapshot_hash", "locale"}}
        with patch("app.services.provider.generate_content", AsyncMock(return_value=(json.dumps(draft), AiUsage(outputTokens=20)))) as provider:
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
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=provider_error)), \
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
        with patch("app.services.provider.generate_content", AsyncMock(return_value=("{}", AiUsage(outputTokens=3)))) as provider:
            response = await lesson_author_orchestration_v2_course_skeleton(request)
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(response["usage"]["outputTokens"], 6)
        self.assertEqual(
            [scope for chapter in response["skeleton"]["chapters"] for scope in chapter["source_scope_ids"]],
            ["scope-1", "scope-2"],
        )
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["usage_source"], "provider")

    async def test_course_skeleton_provider_timeout_returns_reviewable_source_fallback(self) -> None:
        request = RagLessonAuthorCourseSkeletonV2Request.model_validate({
            **common(), "source_snapshot_hash": SOURCE_HASH,
            "source_authority": model_designed_authority(),
            "scope_catalog": [{"scope_key": "scope-1", "title": "Mục 1", "source_ref": None,
                               "fact_count": 1, "content_chars": 5}], "max_attempts": 1,
        })
        provider_failure = HTTPException(status_code=504, detail={
            "code": "AI_PROVIDER_TIMEOUT", "message": "private",
        })
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=provider_failure)):
            response = await lesson_author_orchestration_v2_course_skeleton(request)
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")

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
        with patch("app.services.provider.generate_content", AsyncMock(return_value=(
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
        with patch("app.services.provider.generate_content", AsyncMock(return_value=(
            json.dumps({"lessons": [{"invalid": True}]}), AiUsage(outputTokens=20),
        ))) as provider:
            response = await lesson_author_orchestration_v2_chapter_shard(request)
        provider.assert_awaited_once()
        self.assertEqual(response["shard"]["source_scope_ids"], ["scope-1"])
        self.assertEqual(response["shard"]["lessons"][0]["units"][0]["source_scope_ids"], ["scope-1"])

    async def test_chapter_shard_salvages_valid_provider_unit_before_scope_fallback(self) -> None:
        skeleton = skeleton_wire()
        skeleton["chapters"][0]["source_scope_ids"] = ["scope-1", "scope-2"]
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            **common(), "skeleton": skeleton,
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1", "scope-2"], "source_fact_count": 2,
                           "source_content_chars": 20},
            "source_facts": [
                {"document_id": "document-1", "fact_key": f"fact-{index}", "scope_key": scope_id,
                 "fact_text": f"Nội dung nguồn đủ dài cho {scope_id}.", "source_ref": None,
                 "source_page": 1, "source_chunk": index - 1, "locator": {}}
                for index, scope_id in enumerate(("scope-1", "scope-2"), start=1)
            ],
            "max_attempts": 1,
        })
        provider_lesson = lesson_wire(["scope-1"])
        provider_lesson["title"] = "Bài provider được giữ"
        invalid_unit = json.loads(json.dumps(provider_lesson["units"][0]))
        invalid_unit["source_scope_ids"] = ["scope-2"]
        invalid_unit["component_plan"][0]["source_scope_ids"] = ["scope-2"]
        invalid_unit.pop("purpose")
        provider_lesson["units"].append(invalid_unit)
        with patch("app.services.provider.generate_content", AsyncMock(return_value=(
            json.dumps({"lessons": [provider_lesson]}, ensure_ascii=False), AiUsage(outputTokens=20),
        ))) as provider:
            response = await lesson_author_orchestration_v2_chapter_shard(request)

        provider.assert_awaited_once()
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["shard"]["lessons"][0]["title"], "Bài provider được giữ")
        self.assertEqual(
            [scope_id for item in response["shard"]["lessons"]
             for unit in item["units"] for scope_id in unit["source_scope_ids"]],
            ["scope-1", "scope-2"],
        )

    async def test_chapter_shard_keeps_best_partial_salvage_across_repair_attempts(self) -> None:
        skeleton = skeleton_wire()
        skeleton["chapters"][0]["source_scope_ids"] = ["scope-1", "scope-2"]
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            **common(), "skeleton": skeleton,
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1", "scope-2"], "source_fact_count": 2,
                           "source_content_chars": 20},
            "source_facts": [
                {"document_id": "document-1", "fact_key": f"fact-{index}", "scope_key": scope_id,
                 "fact_text": f"Nội dung nguồn đủ dài cho {scope_id}.", "source_ref": None,
                 "source_page": 1, "source_chunk": index - 1, "locator": {}}
                for index, scope_id in enumerate(("scope-1", "scope-2"), start=1)
            ],
            "max_attempts": 2,
        })
        provider_lesson = lesson_wire(["scope-1"])
        provider_lesson["title"] = "Bài provider tốt nhất"
        invalid_unit = json.loads(json.dumps(provider_lesson["units"][0]))
        invalid_unit["source_scope_ids"] = ["scope-2"]
        invalid_unit["component_plan"][0]["source_scope_ids"] = ["scope-2"]
        invalid_unit.pop("purpose")
        provider_lesson["units"].append(invalid_unit)
        responses = [
            (json.dumps({"lessons": [provider_lesson]}, ensure_ascii=False), AiUsage(outputTokens=20)),
            (json.dumps({"lessons": [{"invalid": True}]}), AiUsage(outputTokens=10)),
        ]
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=responses)) as provider:
            response = await lesson_author_orchestration_v2_chapter_shard(request)

        self.assertEqual(provider.await_count, 2)
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")
        self.assertEqual(response["shard"]["lessons"][0]["title"], "Bài provider tốt nhất")
        self.assertEqual(
            [scope_id for item in response["shard"]["lessons"]
             for unit in item["units"] for scope_id in unit["source_scope_ids"]],
            ["scope-1", "scope-2"],
        )

    async def test_chapter_shard_provider_unavailable_returns_bounded_reviewable_fallback(self) -> None:
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            **common(), "skeleton": skeleton_wire(),
            "shard_plan": {"chapter_key": "chapter-1", "order": 0, "shard_index": 0, "shard_count": 1,
                           "source_scope_ids": ["scope-1"], "source_fact_count": 1, "source_content_chars": 5},
            "source_facts": [{"document_id": "document-1", "fact_key": "fact-1", "scope_key": "scope-1",
                              "fact_text": "Alpha", "source_ref": None, "source_page": 1,
                              "source_chunk": 0, "locator": {}}],
            "max_attempts": 1,
        })
        provider_failure = HTTPException(status_code=503, detail={
            "code": "AI_PROVIDER_UNAVAILABLE", "message": "private",
        })
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=provider_failure)):
            response = await lesson_author_orchestration_v2_chapter_shard(request)
        self.assertEqual(response["usage_source"], "reserved_upper_bound")
        self.assertFalse(response["usage_complete"])
        self.assertEqual(response["content_origin"], "structured_fallback")
        self.assertEqual(response["quality_state"], "review_required")

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
