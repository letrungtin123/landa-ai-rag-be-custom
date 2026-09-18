from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from google import genai
from google.genai import models, types
from fastapi import HTTPException

from app.lesson_author_blueprint import (
    LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA,
    LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL,
    LessonAuthorBlueprintResponse,
    LessonAuthorBlueprintValidationError,
    parse_and_validate_lesson_author_blueprint,
)
from app.main import (
    AiUsage,
    LessonAuthorBlueprintGenerationError,
    LessonAuthorProposalValidationError,
    RagLessonAuthorBlueprintRequest,
    RagLessonAuthorRequest,
    build_lesson_author_blueprint_prompt,
    build_lesson_author_proposal_response_schema,
    build_source_locked_staged_skeleton,
    build_staged_component_plan,
    consolidate_staged_thin_units,
    extract_lesson_author_unit_batches,
    build_lesson_author_skeleton_response_schema,
    build_lesson_author_unit_response_schema,
    build_source_locked_html_unit,
    drop_invalid_lesson_author_source_refs,
    embed_text_batch,
    enforce_lesson_author_source_structure,
    generate_content,
    generate_staged_lesson_author_proposal,
    generate_validated_lesson_author_blueprint,
    normalize_lesson_author_proposal_tree,
    prepare_source_locked_expected,
    should_stage_lesson_author_proposal,
    staged_unit_source_material,
    validate_staged_unit_content,
    validate_staged_skeleton_source_facts,
    validate_lesson_author_proposal_shape,
)


def valid_blueprint() -> dict[str, object]:
    return {
        "title": "Nhập môn an toàn thông tin",
        "summary": "Khóa học giúp nhân viên nhận biết và xử lý các rủi ro an toàn thông tin cơ bản.",
        "target_audience": "Nhân viên mới và nhân viên không chuyên kỹ thuật.",
        "prerequisites": [],
        "learning_outcomes": [
            "Nhận biết các rủi ro phổ biến.",
            "Áp dụng quy trình báo cáo phù hợp.",
            "Thực hành xử lý một tình huống giả định.",
        ],
        "assessment_strategy": "Đánh giá bằng tình huống thực hành và bài kiểm tra cuối khóa.",
        "assumptions": ["Cần xác nhận thời lượng học chính thức."],
        "chapters": [
            {
                "title": "Nền tảng",
                "objective": "Người học nhận biết được các rủi ro cơ bản.",
                "duration_minutes": 45,
                "lessons": [
                    {
                        "title": "Rủi ro thường gặp",
                        "objective": "Người học phân biệt được các rủi ro thường gặp.",
                        "duration_minutes": 20,
                        "learning_activities": ["Phân tích tình huống ngắn."],
                        "assessment": "Trả lời câu hỏi tình huống.",
                    }
                ],
            }
        ],
    }


def blueprint_request(system_prompt: str = "legacy") -> RagLessonAuthorBlueprintRequest:
    return RagLessonAuthorBlueprintRequest(
        tenant_id="11111111-1111-4111-8111-111111111111",
        kb_id="22222222-2222-4222-8222-222222222222",
        conversation_id="33333333-3333-4333-8333-333333333333",
        target="lesson_author",
        model="gemini-3.5-flash",
        max_output_tokens=4096,
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
        system_prompt=system_prompt,
        user_message="Tạo một khóa học an toàn thông tin.",
        blueprint_schema_hint="Server-enforced Blueprint schema.",
        locale="vi",
        api_key="test-key",
    )


def proposal_request(**overrides: object) -> RagLessonAuthorRequest:
    values: dict[str, object] = {
        "tenant_id": "11111111-1111-4111-8111-111111111111",
        "kb_id": "22222222-2222-4222-8222-222222222222",
        "conversation_id": "33333333-3333-4333-8333-333333333333",
        "target": "lesson_author",
        "model": "gemini-3.5-flash",
        "max_output_tokens": 4096,
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        "system_prompt": "test",
        "user_message": "Soạn chi tiết Chương 1.",
        "output_schema_hint": "schema",
        "api_key": "test-key",
    }
    values.update(overrides)
    return RagLessonAuthorRequest(**values)


def valid_proposal() -> dict[str, object]:
    return {
        "summary": "Đề xuất nội dung bài học bám sát tài liệu nguồn.",
        "chapters": [
            {
                "title": "Chương nguồn",
                "lessons": [
                    {
                        "title": "Bài học nguồn",
                        "units": [
                            {
                                "title": "Nội dung chính",
                                "components": [
                                    {"type": "html", "title": "Tóm tắt", "html": "<p>Phân tích an toàn công việc giúp nhận diện mối nguy, đánh giá rủi ro và lựa chọn biện pháp kiểm soát phù hợp trước khi bắt đầu công việc. Người học cần quan sát từng bước, xác định điều kiện nguy hiểm, xem xét mức độ rủi ro và đề xuất biện pháp kiểm soát theo thứ tự ưu tiên để làm việc an toàn.</p>"},
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }


class LessonAuthorBlueprintContractTests(unittest.TestCase):
    def test_staged_skeleton_schema_requires_complete_structure(self) -> None:
        schema = build_lesson_author_skeleton_response_schema()
        self.assertEqual(schema.required, ["chapters"])
        chapter_schema = schema.properties["chapters"].items
        self.assertEqual(chapter_schema.required, ["title", "lessons"])
        lesson_schema = chapter_schema.properties["lessons"].items
        self.assertEqual(lesson_schema.required, ["title", "units"])
        unit_schema = lesson_schema.properties["units"].items
        self.assertEqual(unit_schema.required, ["title", "component_plan", "source_fact_ids"])
        self.assertEqual(unit_schema.properties["component_plan"].items.required, ["type", "rationale", "source_fact_ids"])

    def test_proposal_schema_requires_a_complete_lesson_tree(self) -> None:
        schema = build_lesson_author_proposal_response_schema()
        chapter_schema = schema.properties["chapters"].items
        self.assertEqual(chapter_schema.required, ["title", "lessons"])
        lesson_schema = chapter_schema.properties["lessons"].items
        self.assertEqual(lesson_schema.required, ["title", "units"])
        unit_schema = lesson_schema.properties["units"].items
        self.assertEqual(unit_schema.required, ["title", "components"])
        self.assertEqual(unit_schema.properties["components"].items.required, ["type", "source_fact_ids"])

        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_staged_unit_schema_is_bounded_and_serializable(self) -> None:
        schema = build_lesson_author_unit_response_schema()
        self.assertEqual(schema.required, ["title", "components", "source_fact_ids"])
        self.assertEqual(schema.properties["components"].items.required, ["type", "source_fact_ids"])

        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_explicit_staged_mode_cannot_fall_back_to_single_generation(self) -> None:
        request = proposal_request(generation_mode="staged", max_output_tokens=512)
        self.assertTrue(should_stage_lesson_author_proposal(request, "short context"))

        single_request = proposal_request(generation_mode="single", operation="create", target_type="chapter")
        self.assertFalse(should_stage_lesson_author_proposal(single_request, "large context"))

    def test_staged_skeleton_must_assign_every_manifest_fact(self) -> None:
        batches = [[{
            "source_fact_ids": ["p1-f1", "p2-f1"],
        }]]
        manifest = {
            "facts": [
                {"fact_id": "p1-f1", "source_page": 1, "text": "A"},
                {"fact_id": "p2-f1", "source_page": 2, "text": "B"},
            ],
        }
        validate_staged_skeleton_source_facts(batches, manifest)

        with self.assertRaises(LessonAuthorProposalValidationError):
            validate_staged_skeleton_source_facts(
                [[{"source_fact_ids": ["p1-f1"]}]],
                manifest,
            )

    def test_source_locked_skeleton_is_bounded_and_covers_every_fact_once(self) -> None:
        manifest = {
            "facts": [
                {
                    "fact_id": f"p{page}-f1",
                    "source_page": page,
                    "text": f"Chủ đề trang {page}: nội dung nguồn bắt buộc.",
                }
                for page in range(10, 18)
            ],
        }
        request = proposal_request(
            user_message=(
                "Soạn chi tiết Chương 3: Thực hành Nhận diện mối nguy, "
                "Đánh giá Rủi ro, Lựa chọn và sử dụng PPE phù hợp"
            ),
            locale="vi",
        )

        skeleton = build_source_locked_staged_skeleton(request, manifest)
        chapter = skeleton["chapters"][0]
        units = chapter["lessons"][0]["units"]
        assigned = [fact_id for unit in units for fact_id in unit["source_fact_ids"]]

        self.assertEqual(
            chapter["title"],
            "Thực hành Nhận diện mối nguy, Đánh giá Rủi ro, Lựa chọn và sử dụng PPE phù hợp",
        )
        self.assertLessEqual(len(units), 4)
        self.assertEqual(assigned, [fact["fact_id"] for fact in manifest["facts"]])
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_staged_generation_recovers_from_two_malformed_skeletons(self) -> None:
        manifest = {
            "facts": [{
                "fact_id": "p10-f1",
                "source_page": 10,
                "text": "Nhận diện mối nguy: quan sát công việc, thiết bị và môi trường trước khi thao tác.",
            }],
        }
        html = (
            "<h3>Nhận diện mối nguy</h3><p>Người học quan sát công việc, thiết bị và môi trường "
            "trước khi thao tác để nhận diện điều kiện có thể gây mất an toàn. Việc rà soát cần được "
            "thực hiện có hệ thống, ghi nhận rõ mối nguy và dùng kết quả làm cơ sở cho bước đánh giá "
            "rủi ro tiếp theo.</p>"
        )
        content = json.dumps({
            "title": "Nhận diện mối nguy",
            "source_fact_ids": ["p10-f1"],
            "components": [{
                "type": "html",
                "title": "Nhận diện mối nguy",
                "html": html,
                "source_fact_ids": ["p10-f1"],
            }],
        }, ensure_ascii=False)
        usage = AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)

        with patch(
            "app.main.generate_content",
            new=AsyncMock(side_effect=[("{", usage), ("{", usage), (content, usage)]),
        ) as generate:
            proposal, combined_usage = asyncio.run(generate_staged_lesson_author_proposal(
                proposal_request(user_message="Soạn chi tiết Chương 3: Nhận diện mối nguy", locale="vi"),
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

        self.assertEqual(generate.await_count, 3)
        self.assertEqual(proposal["chapters"][0]["title"], "Nhận diện mối nguy")
        self.assertEqual(
            proposal["chapters"][0]["lessons"][0]["units"][0]["source_fact_ids"],
            ["p10-f1"],
        )
        self.assertEqual(combined_usage.totalTokens, 6)

    def test_docx_blueprint_chapter_survives_malformed_skeleton_and_content(self) -> None:
        request = proposal_request(
            user_message="Soạn chi tiết Chương 1: Tiêu chuẩn và Chuẩn bị",
            locale="vi",
        )
        request.target_scope_instruction = (
            "HARD SCOPE LIMIT: Generate detailed content for Blueprint Chapter 1 only: Tiêu chuẩn và Chuẩn bị.\n"
            "Chapter source references: src-002, src-006\n"
            "Blueprint lessons: 1. Tiêu chuẩn yêu cầu đối với khu vực văn phòng (Mô tả; 10 minutes; sources src-002) | "
            "2. Chuẩn bị dụng cụ, thiết bị và hóa chất (Mô tả; 20 minutes; sources src-006)"
        )
        manifest = {
            "facts": [
                {"fact_id": "src-002-f1", "source_ref": "src-002", "text": "The office area is neat and orderly."},
                {"fact_id": "src-002-f2", "source_ref": "src-002", "text": "All waste has been collected and removed."},
                {"fact_id": "src-006-f1", "source_ref": "src-006", "text": "Prepare glass cleaner, all-purpose cleaner, and telephone disinfectant."},
                {"fact_id": "src-006-f2", "source_ref": "src-006", "text": "Prepare clean wiping cloths, a window squeegee kit, and rubber gloves."},
            ],
        }
        usage = AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)

        with patch(
            "app.main.generate_content",
            new=AsyncMock(side_effect=[("{", usage)] * 6),
        ) as generate:
            proposal, combined_usage = asyncio.run(generate_staged_lesson_author_proposal(
                request,
                "Full document context that must not be required by the fallback.",
                "",
                "",
                source_rows=[],
                source_coverage_manifest=manifest,
            ))

        units = proposal["chapters"][0]["lessons"][0]["units"]
        self.assertEqual(generate.await_count, 4)
        self.assertEqual(combined_usage.totalTokens, 8)
        self.assertEqual(len(units), 1)
        self.assertIn("và", units[0]["title"])
        self.assertTrue(all(unit["source_locked_fallback"] for unit in units))
        self.assertEqual(
            {fact_id for unit in units for fact_id in unit["source_fact_ids"]},
            {fact["fact_id"] for fact in manifest["facts"]},
        )

    def test_component_plan_is_evidence_gated_and_not_html_only(self) -> None:
        manifest = {
            "facts": [
                {"fact_id": "p3-f1", "source_page": 3, "text": "Câu hỏi kiểm tra: Định nghĩa an toàn lao động là gì?"},
                {"fact_id": "p3-f2", "source_page": 3, "text": "Đáp án: bảo vệ người lao động."},
                {"fact_id": "p5-f1", "source_page": 5, "text": "1. Chính sách. 2. Đào tạo. 3. Phân tích rủi ro. 4. Kiểm soát. 5. Cải tiến."},
            ],
        }
        theory_plan = build_staged_component_plan(
            {"source_fact_ids": ["p3-f1", "p3-f2"], "components": [{"type": "html"}]},
            manifest,
        )
        process_plan = build_staged_component_plan(
            {
                "title": "Mô hình triển khai ATSKNN",
                "source_fact_ids": ["p5-f1"],
                "components": [{"type": "html"}],
            },
            manifest,
        )
        self.assertEqual([item["type"] for item in theory_plan], ["html", "problem"])
        self.assertEqual([item["type"] for item in process_plan], ["html", "la_diagram", "la_sortable"])
        self.assertTrue(all(item["source_fact_ids"] for item in process_plan))

    def test_component_plan_does_not_turn_taxonomy_into_ordering_activity(self) -> None:
        taxonomy_plan = build_staged_component_plan(
            {
                "source_fact_ids": ["p6-f1"],
                "components": [{"type": "html"}],
            },
            {
                "facts": [{
                    "fact_id": "p6-f1",
                    "source_page": 6,
                    "text": (
                        "1. Rủi ro cơ khí: máy móc quay. "
                        "2. Rủi ro điện: dây hở. "
                        "3. Rủi ro hóa chất: tiếp xúc không bảo hộ."
                    ),
                }],
            },
        )
        self.assertEqual([item["type"] for item in taxonomy_plan], ["html"])

    def test_component_plan_does_not_treat_generic_compliance_as_quiz(self) -> None:
        compliance_plan = build_staged_component_plan(
            {
                "source_fact_ids": ["p7-f1", "p7-f2"],
                "components": [{"type": "html"}],
            },
            {
                "facts": [
                    {"fact_id": "p7-f1", "source_page": 7, "text": "Người lao động thực hiện đúng hướng dẫn an toàn."},
                    {"fact_id": "p7-f2", "source_page": 7, "text": "Thiết bị phải được kiểm tra trước khi vận hành."},
                ],
            },
        )
        self.assertEqual([item["type"] for item in compliance_plan], ["html"])

    def test_staged_chapter_adds_grounded_recall_checks_when_all_units_are_html(self) -> None:
        manifest = {
            "facts": [
                {
                    "fact_id": "p1-f1",
                    "text": (
                        "Office cleaning requires the work area to remain orderly, all waste to be removed promptly, "
                        "and each completed task to be checked against the assigned cleanliness standard before handover. "
                        "The documented completion record is reviewed by the supervisor before the service is closed."
                    ),
                },
                {
                    "fact_id": "p1-f2",
                    "text": (
                        "Cleaning staff must prepare the correct supplies, protect nearby equipment, record completed "
                        "tasks, and report any condition that prevents the area from being cleaned safely. The assigned "
                        "cleaning procedure and local safety controls remain in force throughout the task."
                    ),
                },
            ],
        }
        skeleton = {
            "chapters": [{
                "title": "Office preparation",
                "lessons": [{
                    "title": "Core practices",
                    "units": [
                        {"title": "Area standard", "source_fact_ids": ["p1-f1"]},
                        {"title": "Preparation and reporting", "source_fact_ids": ["p1-f2"]},
                    ],
                }],
            }],
        }

        batches = extract_lesson_author_unit_batches(skeleton, manifest, "en")
        planned_types = [
            component_type
            for batch in batches
            for unit in batch
            for component_type in unit["component_types"]
        ]

        self.assertEqual(planned_types.count("problem"), 2)
        self.assertTrue(all("html" in unit["component_types"] for batch in batches for unit in batch))

    def test_consolidate_staged_thin_units_preserves_adjacent_fact_coverage(self) -> None:
        manifest = {
            "facts": [
                {"fact_id": "p2-f1", "text": "The office area is neat and orderly, and completed work is checked before handover."},
                {"fact_id": "p2-f2", "text": "All waste is collected, removed, and recorded according to the local cleaning procedure."},
            ],
        }
        skeleton = {
            "chapters": [{
                "title": "Office standards",
                "lessons": [{
                    "title": "Required standards",
                    "units": [
                        {"title": "Area condition", "source_fact_ids": ["p2-f1"]},
                        {"title": "Waste handling", "source_fact_ids": ["p2-f2"]},
                    ],
                }],
            }],
        }

        consolidated = consolidate_staged_thin_units(skeleton, manifest, "en")
        units = consolidated["chapters"][0]["lessons"][0]["units"]

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["source_fact_ids"], ["p2-f1", "p2-f2"])
        self.assertIn(" and ", units[0]["title"])

    def test_staged_unit_rejects_component_plan_drift_and_unmapped_facts(self) -> None:
        expected = {
            "component_types": ["html", "problem"],
            "source_fact_ids": ["p3-f1", "p3-f2"],
        }
        unit = {
            "title": "Mục kiểm tra",
            "source_fact_ids": ["p3-f1", "p3-f2"],
            "components": [{
                "type": "html",
                "source_fact_ids": ["p3-f1", "p3-f2"],
                "html": "<p>Phân tích an toàn lao động giúp bảo vệ người lao động và giảm rủi ro trong công việc hằng ngày. Người học cần nhận biết điều kiện nguy hiểm, hiểu mục tiêu phòng ngừa và thực hiện đúng hướng dẫn an toàn trước khi bắt đầu công việc.</p>",
            }],
        }
        reason = validate_staged_unit_content(unit, expected)
        self.assertIn("formats", reason or "")

        unit["components"].append({
            "type": "problem",
            "source_fact_ids": ["p3-f1", "p9-f1"],
            "question": "Ý chính là gì?",
            "choices": [
                {"text": "Bảo vệ người lao động", "correct": True},
                {"text": "Không liên quan", "correct": False},
            ],
        })
        reason = validate_staged_unit_content(unit, expected)
        self.assertIn("outside", reason or "")

    def test_staged_unit_context_is_scoped_to_assigned_pages(self) -> None:
        coverage, source_context = staged_unit_source_material(
            {"source_fact_ids": ["p2-f1"]},
            [
                {
                    "source_page": 1,
                    "content": "Không thuộc Mục này.",
                    "document_name": "source.pdf",
                    "source_section": "",
                    "document_id": "doc-1",
                    "score": 0,
                },
                {
                    "source_page": "2",
                    "content": "Nội dung nguồn của Mục này.",
                    "document_name": "source.pdf",
                    "source_section": "",
                    "document_id": "doc-1",
                    "score": 0,
                },
            ],
            {"facts": [{"fact_id": "p2-f1", "source_page": 2, "text": "Fact bắt buộc."}]},
        )
        self.assertIn("p2-f1", coverage)
        self.assertIn("Nội dung nguồn của Mục này", source_context)
        self.assertNotIn("Không thuộc Mục này", source_context)

    def test_staged_unit_context_uses_only_assigned_inferred_heading_facts(self) -> None:
        coverage, source_context = staged_unit_source_material(
            {"source_fact_ids": ["src-002-f1", "src-002-f2"]},
            [{
                "document_id": "doc-office",
                "chunk_no": 0,
                "source_page": None,
                "content": "Content from every chapter must not leak into the unit.",
                "document_name": "office.docx",
                "source_section": "",
                "score": 0,
            }],
            {
                "facts": [
                    {"fact_id": "src-002-f1", "source_ref": "src-002", "text": "The office area is neat and orderly."},
                    {"fact_id": "src-002-f2", "source_ref": "src-002", "text": "All waste has been collected and removed."},
                    {"fact_id": "src-007-f1", "source_ref": "src-007", "text": "Perform cleaning."},
                ],
            },
        )

        self.assertIn("mục nguồn src-002", coverage)
        self.assertIn("The office area is neat", source_context)
        self.assertNotIn("Perform cleaning", source_context)
        self.assertNotIn("every chapter", source_context)

    def test_source_locked_html_fallback_preserves_assigned_raw_facts(self) -> None:
        expected = {
            "unit_title": "Khái niệm an toàn lao động",
            "component_types": ["html"],
            "component_plan": [{"type": "html", "source_fact_ids": ["p1-f1", "p1-f2"]}],
            "source_fact_ids": ["p1-f1", "p1-f2"],
        }
        manifest = {
            "facts": [
                {
                    "fact_id": "p1-f1",
                    "source_page": 1,
                    "text": (
                        "An toàn lao động là hệ thống các biện pháp nhằm phòng ngừa tai nạn, "
                        "bệnh nghề nghiệp và bảo vệ sức khỏe người lao động trong quá trình làm việc."
                    ),
                },
                {
                    "fact_id": "p1-f2",
                    "source_page": 1,
                    "text": "Người lao động cần nhận diện nguy cơ và tuân thủ hướng dẫn kiểm soát phù hợp.",
                },
            ],
        }
        unit = build_source_locked_html_unit(expected, manifest)
        self.assertIsNotNone(unit)
        self.assertEqual(unit["source_fact_ids"], ["p1-f1", "p1-f2"])
        self.assertIn("An toàn lao động là hệ thống", unit["components"][0]["html"])
        self.assertIsNone(validate_staged_unit_content(unit, expected))

    def test_source_locked_fallback_rebuilds_grounded_short_text_problem(self) -> None:
        unit = build_source_locked_html_unit(
            {
                "unit_title": "Bài kiểm tra",
                "component_types": ["html", "problem"],
                "source_fact_ids": ["p1-f1"],
            },
            {
                "facts": [{
                    "fact_id": "p1-f1",
                    "text": (
                        "Nội dung nguồn mô tả rõ các yêu cầu kiểm tra khu vực làm việc trước khi bắt đầu, "
                        "cách ghi nhận mối nguy, biện pháp kiểm soát cần áp dụng, và trách nhiệm báo cáo "
                        "ngay khi phát hiện điều kiện không an toàn trong quá trình thực hiện công việc."
                    ),
                }],
            },
        )
        self.assertIsNotNone(unit)
        self.assertEqual([component["type"] for component in unit["components"]], ["html", "problem"])
        self.assertEqual(unit["components"][1]["problem_type"], "short_text")
        self.assertIn("Nội dung nguồn mô tả rõ", unit["components"][1]["answer"])
        self.assertIsNone(validate_staged_unit_content(unit, {
            "component_types": ["html", "problem"],
            "source_fact_ids": ["p1-f1"],
        }))

    def test_source_locked_html_fallback_rejects_thin_raw_fact_without_padding(self) -> None:
        expected = {
            "unit_title": "Nhận diện mối nguy cơ khí",
            "component_types": ["html"],
            "source_fact_ids": ["p6-f1"],
        }
        raw_text = (
            "1. Rủi ro cơ khí: máy móc quay, chuyển động không có tấm chắn, "
            "làm việc với vật sắc nhọn. Phân loại rủi ro tại nơi làm việc"
        )
        unit = build_source_locked_html_unit(
            expected,
            {"facts": [{"fact_id": "p6-f1", "text": raw_text}]},
        )
        self.assertIsNone(unit)

    def test_source_locked_fallback_rebuilds_ordered_interactive_formats(self) -> None:
        expected = {
            "unit_title": "Mô hình triển khai",
            "component_types": ["html", "la_diagram", "la_sortable"],
            "component_plan": [
                {"type": "html", "source_fact_ids": ["p5-f1"]},
                {"type": "la_diagram", "source_fact_ids": ["p5-f1"]},
                {"type": "la_sortable", "source_fact_ids": ["p5-f1"]},
            ],
            "source_fact_ids": ["p5-f1"],
        }
        manifest = {
            "facts": [{
                "fact_id": "p5-f1",
                "source_page": 5,
                "text": (
                    "1. Chính sách rõ ràng: Cam kết từ lãnh đạo. "
                    "2. Đào tạo và nhận thức: Hướng dẫn an toàn trước khi làm việc. "
                    "3. Phân tích rủi ro: HIRA và JSA. "
                    "4. Biện pháp kiểm soát: PPE, thiết bị và quy trình. "
                    "5. Theo dõi, báo cáo và cải tiến liên tục."
                ),
            }],
        }
        unit = build_source_locked_html_unit(expected, manifest)
        self.assertIsNotNone(unit)
        self.assertEqual(
            [component["type"] for component in unit["components"]],
            ["html", "la_diagram", "la_sortable"],
        )
        self.assertEqual(len(unit["components"][1]["nodes"]), 5)
        self.assertEqual(len(unit["components"][1]["edges"]), 4)
        self.assertEqual(len(unit["components"][2]["items"]), 5)
        self.assertIsNone(validate_staged_unit_content(unit, expected))

    def test_source_locked_fallback_extracts_bullet_steps_for_ordered_formats(self) -> None:
        expected = {
            "unit_title": "Cleaning workflow",
            "component_types": ["html", "la_diagram", "la_sortable"],
            "source_fact_ids": ["p6-f1"],
        }
        unit = build_source_locked_html_unit(
            expected,
            {
                "facts": [{
                    "fact_id": "p6-f1",
                    "text": (
                        "- Prepare the approved cleaning supplies and protect nearby equipment before work begins.\n"
                        "- Clean each assigned surface using the specified method and replace materials when contaminated.\n"
                        "- Inspect the completed area, remove all waste, record completion, and report unresolved issues."
                    ),
                }],
            },
        )

        self.assertIsNotNone(unit)
        self.assertEqual(len(unit["components"][1]["nodes"]), 3)
        self.assertEqual(unit["components"][2]["items"][0], "Prepare the approved cleaning supplies and protect nearby equipment before work begins.")
        self.assertIsNone(validate_staged_unit_content(unit, expected))

    def test_source_locked_recovery_drops_only_unreconstructable_formats(self) -> None:
        expected, dropped = prepare_source_locked_expected(
            {
                "unit_title": "Tiêu chuẩn yêu cầu",
                "component_types": ["html", "la_faq", "la_crossword"],
                "component_plan": [
                    {"type": "html", "source_fact_ids": ["src-002-f1"]},
                    {"type": "la_faq", "source_fact_ids": ["src-002-f1"]},
                    {"type": "la_crossword", "source_fact_ids": ["src-002-f1"]},
                ],
                "source_fact_ids": ["src-002-f1"],
            },
            {"facts": [{
                "fact_id": "src-002-f1",
                "source_ref": "src-002",
                "text": (
                    "The office area is neat and orderly, with all waste collected and removed. Cleaning staff "
                    "inspect the completed area, protect nearby equipment, record the completion status, and report "
                    "any condition that prevents the required standard from being achieved safely."
                ),
            }]},
        )

        self.assertEqual(expected["component_types"], ["html"])
        self.assertEqual(dropped, ["la_faq", "la_crossword"])
        unit = build_source_locked_html_unit(expected, {"facts": [{
            "fact_id": "src-002-f1",
            "source_ref": "src-002",
            "text": (
                "The office area is neat and orderly, with all waste collected and removed. Cleaning staff "
                "inspect the completed area, protect nearby equipment, record the completion status, and report "
                "any condition that prevents the required standard from being achieved safely."
            ),
        }]})
        self.assertIsNotNone(unit)
        self.assertIsNone(validate_staged_unit_content(unit, expected))

    def test_source_locked_skeleton_preserves_blueprint_lesson_titles_for_docx(self) -> None:
        request = proposal_request(
            user_message="Soạn chi tiết Chương 1: Tiêu chuẩn và Chuẩn bị",
            locale="vi",
        )
        request.target_scope_instruction = (
            "HARD SCOPE LIMIT: Generate detailed content for Blueprint Chapter 1 only: Tiêu chuẩn và Chuẩn bị.\n"
            "Chapter source references: src-002, src-006\n"
            "Blueprint lessons: 1. Tiêu chuẩn yêu cầu đối với khu vực văn phòng (Mô tả; 10 minutes; sources src-002) | "
            "2. Chuẩn bị dụng cụ, thiết bị và hóa chất (Mô tả; 20 minutes; sources src-006)"
        )
        skeleton = build_source_locked_staged_skeleton(request, {
            "facts": [
                {"fact_id": "src-002-f1", "source_ref": "src-002", "text": "The office area is neat and orderly."},
                {"fact_id": "src-006-f1", "source_ref": "src-006", "text": "Prepare glass cleaner and clean wiping cloths."},
            ],
        })

        titles = [
            unit["title"]
            for lesson in skeleton["chapters"][0]["lessons"]
            for unit in lesson["units"]
        ]
        self.assertEqual(titles, [
            "Tiêu chuẩn yêu cầu đối với khu vực văn phòng",
            "Chuẩn bị dụng cụ, thiết bị và hóa chất",
        ])


    def test_schema_requires_core_blueprint_fields(self) -> None:
        self.assertEqual(
            LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA.required,
            [
                "title",
                "summary",
                "target_audience",
                "prerequisites",
                "learning_outcomes",
                "assessment_strategy",
                "assumptions",
                "chapters",
            ],
        )
        chapters = LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA.properties["chapters"]
        self.assertIsNotNone(chapters.items)

    def test_schema_avoids_unsupported_gemini_developer_api_fields(self) -> None:
        unsupported_fields = (
            "min_items",
            "example",
            "property_ordering",
            "pattern",
            "minimum",
            "default",
            "any_of",
            "max_length",
            "title",
            "min_length",
            "min_properties",
            "max_items",
            "maximum",
            "nullable",
            "max_properties",
        )

        def assert_compatible(schema: object) -> None:
            for field in unsupported_fields:
                self.assertIsNone(getattr(schema, field, None), field)
            item_schema = getattr(schema, "items", None)
            if item_schema is not None:
                assert_compatible(item_schema)
            for child in (getattr(schema, "properties", None) or {}).values():
                assert_compatible(child)

        assert_compatible(LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA)

    def test_schema_serializes_with_the_installed_gemini_developer_api_sdk(self) -> None:
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_pydantic_response_model_serializes_with_the_installed_gemini_developer_api_sdk(self) -> None:
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_generate_content_uses_sdk_parsed_structured_response(self) -> None:
        parsed = LessonAuthorBlueprintResponse.model_validate(valid_blueprint())
        response = SimpleNamespace(
            text="not-json",
            parsed=parsed,
            usage_metadata=None,
            candidates=[],
            prompt_feedback=None,
        )
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch("app.main.genai.Client", return_value=client):
            text, _usage = asyncio.run(
                generate_content(
                    "test-key",
                    "gemini-3.5-flash",
                    "SERVER MODE: COURSE_BLUEPRINT.",
                    max_output_tokens=4096,
                    json_mode=True,
                    response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL,
                )
            )

        self.assertEqual(json.loads(text)["title"], "Nhập môn an toàn thông tin")

    def test_generate_content_passes_thinking_config_for_structured_generation(self) -> None:
        response = SimpleNamespace(
            text=json.dumps(valid_blueprint(), ensure_ascii=False),
            parsed=None,
            usage_metadata=None,
            candidates=[],
            prompt_feedback=None,
        )
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch("app.main.genai.Client", return_value=client):
            asyncio.run(
                generate_content(
                    "test-key",
                    "gemini-3.5-flash",
                    "SERVER MODE: COURSE_BLUEPRINT.",
                    max_output_tokens=8192,
                    json_mode=True,
                    response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL,
                    thinking_config=types.ThinkingConfig(include_thoughts=False),
                )
            )

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertFalse(config["thinking_config"].include_thoughts)

    def test_valid_blueprint_is_normalized(self) -> None:
        candidate = valid_blueprint()
        candidate["chapters"][0]["source_refs"] = ["src-001"]
        candidate["chapters"][0]["lessons"][0]["source_refs"] = ["src-001"]
        parsed = parse_and_validate_lesson_author_blueprint(json.dumps(candidate))
        self.assertEqual(parsed["title"], "Nhập môn an toàn thông tin")
        self.assertEqual(len(parsed["chapters"]), 1)
        self.assertEqual(parsed["chapters"][0]["source_refs"], ["src-001"])

    def test_blueprint_accepts_twelve_distinct_lessons_in_one_inferred_chapter(self) -> None:
        candidate = valid_blueprint()
        lesson = candidate["chapters"][0]["lessons"][0]
        candidate["chapters"][0]["lessons"] = [
            {
                **lesson,
                "title": f"Quy trình làm sạch {index}",
                "objective": f"Người học thực hiện đúng quy trình làm sạch {index}.",
                "source_refs": [f"src-{index:03d}"],
            }
            for index in range(1, 13)
        ]

        parsed = parse_and_validate_lesson_author_blueprint(json.dumps(candidate, ensure_ascii=False))

        self.assertEqual(len(parsed["chapters"][0]["lessons"]), 12)
        self.assertEqual(parsed["chapters"][0]["lessons"][11]["source_refs"], ["src-012"])

    def test_blueprint_rejects_more_than_twelve_lessons_in_one_chapter(self) -> None:
        candidate = valid_blueprint()
        lesson = candidate["chapters"][0]["lessons"][0]
        candidate["chapters"][0]["lessons"] = [
            {**lesson, "title": f"Quy trình {index}"}
            for index in range(1, 14)
        ]

        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            parse_and_validate_lesson_author_blueprint(json.dumps(candidate, ensure_ascii=False))

        self.assertIn("1 to 12 items", str(raised.exception))

    def test_provider_shape_drift_is_normalized_without_inventing_content(self) -> None:
        candidate = valid_blueprint()
        candidate["chapters"][0]["duration_minutes"] = "45"
        candidate["chapters"][0]["source_refs"] = "src-001"
        candidate["chapters"][0]["lessons"][0]["duration_minutes"] = "20"
        candidate["chapters"][0]["lessons"][0]["source_refs"] = "src-001"
        parsed = parse_and_validate_lesson_author_blueprint(
            "json\n" + json.dumps(candidate),
        )
        self.assertEqual(parsed["chapters"][0]["duration_minutes"], 45)
        self.assertEqual(parsed["chapters"][0]["source_refs"], ["src-001"])
        self.assertEqual(parsed["chapters"][0]["lessons"][0]["duration_minutes"], 20)

    def test_malformed_provider_json_is_repaired_only_when_structure_is_complete(self) -> None:
        candidate = valid_blueprint()
        candidate["summary"] = "Dòng một\nDòng hai"
        raw = json.dumps(candidate, ensure_ascii=False).replace("\\n", "\n")
        raw = raw.replace("\"assumptions\": [\"Cần xác nhận thời lượng học chính thức.\"]", "\"assumptions\": [\"Cần xác nhận thời lượng học chính thức.\",]")
        parsed = parse_and_validate_lesson_author_blueprint(raw)
        self.assertEqual(parsed["summary"], "Dòng một\nDòng hai")

        with self.assertRaises(LessonAuthorBlueprintValidationError):
            parse_and_validate_lesson_author_blueprint(raw[:-1])

    def test_last_complete_object_is_used_when_provider_adds_an_extra_object(self) -> None:
        candidate = valid_blueprint()
        raw = '{"partial": true}\n' + json.dumps(candidate, ensure_ascii=False) + "\nĐã hoàn tất."
        parsed = parse_and_validate_lesson_author_blueprint(raw)
        self.assertEqual(parsed["title"], "Nhập môn an toàn thông tin")

    def test_invalid_json_is_rejected(self) -> None:
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            parse_and_validate_lesson_author_blueprint('{"title":')
        self.assertEqual(raised.exception.code, "BLUEPRINT_INVALID_JSON")

    def test_legacy_template_format_is_not_in_blueprint_prompt(self) -> None:
        stored_prompt = "Return chapters, lessons, units, components and an old proposal schema."
        prompt = build_lesson_author_blueprint_prompt(blueprint_request(stored_prompt), "Tài liệu nguồn")
        self.assertIn("SERVER MODE: COURSE_BLUEPRINT.", prompt)
        self.assertIn(stored_prompt, prompt)
        self.assertIn("<STORED_SYSTEM_PROMPT>", prompt)
        self.assertIn("<SOURCE_MATERIAL>", prompt)

    def test_blueprint_prompt_explains_toc_and_fallback_rules(self) -> None:
        prompt = build_lesson_author_blueprint_prompt(
            blueprint_request(),
            "Tài liệu nguồn",
            "Cấu trúc nguồn (toc):\n[src-001] 1. Nền tảng",
        )
        self.assertIn("<SOURCE_OUTLINE>", prompt)
        self.assertIn("mục lục/tiêu đề nguồn", prompt)
        self.assertIn("Không tự tạo mã nguồn", prompt)
        self.assertIn("Không đưa hậu tố phạm vi nguồn", prompt)
        self.assertIn("1 to 12 chapters and 1 to 12 lessons", prompt)
        self.assertIn("authorized whole-course operation", prompt)

    def test_toc_structure_canonicalizes_chapter_name_and_source_ref(self) -> None:
        candidate = valid_blueprint()
        canonical = enforce_lesson_author_source_structure(
            candidate,
            structure_source="toc",
            authoritative_source_nodes=[{
                "source_ref": "src-001",
                "title": "1. Tên chương nguyên văn (từ Slide 1 đến slide 5)",
                "level": 1,
            }],
        )

        self.assertEqual(
            canonical["chapters"][0]["title"],
            "1. Tên chương nguyên văn",
        )
        self.assertEqual(canonical["chapters"][0]["source_refs"], ["src-001"])

    def test_blueprint_validation_removes_source_ranges_from_chapter_and_lesson_titles(self) -> None:
        candidate = valid_blueprint()
        candidate["chapters"][0]["title"] = "Nền tảng (từ slide 1 đến slide 5)"
        candidate["chapters"][0]["lessons"][0]["title"] = "Rủi ro thường gặp (from page 2 to page 4)"

        parsed = parse_and_validate_lesson_author_blueprint(json.dumps(candidate, ensure_ascii=False))

        self.assertEqual(parsed["chapters"][0]["title"], "Nền tảng")
        self.assertEqual(parsed["chapters"][0]["lessons"][0]["title"], "Rủi ro thường gặp")

    def test_proposal_normalization_removes_source_ranges_from_all_structural_titles(self) -> None:
        proposal = {
            "summary": "Đề xuất.",
            "chapters": [{
                "title": "Chương nguồn (từ slide 1 đến slide 5)",
                "lessons": [{
                    "title": "Bài nguồn (from page 2 to page 4)",
                    "units": [{
                        "title": "Mục nguồn (trang 3 đến trang 4)",
                        "components": [{"type": "html", "title": "Nội dung", "html": "<p>Nội dung.</p>"}],
                    }],
                }],
            }],
        }

        normalized = normalize_lesson_author_proposal_tree(proposal)

        chapter = normalized["chapters"][0]
        lesson = chapter["lessons"][0]
        unit = lesson["units"][0]
        self.assertEqual(chapter["title"], "Chương nguồn")
        self.assertEqual(lesson["title"], "Bài nguồn")
        self.assertEqual(unit["title"], "Mục nguồn")

    def test_toc_structure_rejects_missing_or_extra_chapters(self) -> None:
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            enforce_lesson_author_source_structure(
                valid_blueprint(),
                structure_source="toc",
                authoritative_source_nodes=[
                    {"source_ref": "src-001", "title": "Chương một", "level": 1},
                    {"source_ref": "src-002", "title": "Chương hai", "level": 1},
                ],
            )
        self.assertEqual(raised.exception.code, "BLUEPRINT_SOURCE_STRUCTURE_MISMATCH")

    def test_unknown_toc_lesson_refs_are_removed_without_inventing_evidence(self) -> None:
        candidate = valid_blueprint()
        candidate["chapters"][0]["source_refs"] = ["src-009"]
        candidate["chapters"][0]["lessons"][0]["source_refs"] = ["src-001", "src-010"]
        sanitized, dropped = drop_invalid_lesson_author_source_refs(candidate, {"src-001"})

        self.assertEqual(dropped, ["src-009", "src-010"])
        self.assertEqual(sanitized["chapters"][0]["source_refs"], [])
        self.assertEqual(sanitized["chapters"][0]["lessons"][0]["source_refs"], ["src-001"])

    def test_detailed_proposal_requires_a_complete_content_tree(self) -> None:
        validate_lesson_author_proposal_shape(valid_proposal())

        incomplete = valid_proposal()
        incomplete["chapters"][0]["lessons"][0]["units"][0]["components"] = []
        with self.assertRaises(LessonAuthorProposalValidationError):
            validate_lesson_author_proposal_shape(incomplete)

    def test_detailed_proposal_rejects_thin_html_content(self) -> None:
        incomplete = valid_proposal()
        incomplete["chapters"][0]["lessons"][0]["units"][0]["components"][0]["html"] = "<p>Ngắn.</p>"
        with self.assertRaises(LessonAuthorProposalValidationError):
            validate_lesson_author_proposal_shape(incomplete)

    def test_detailed_proposal_rejects_problem_with_one_choice(self) -> None:
        incomplete = valid_proposal()
        incomplete["chapters"][0]["lessons"][0]["units"][0]["components"] = [{
            "type": "multiple_choice",
            "question": "Hành động đầu tiên là gì?",
            "choices": [{"text": "Giữ bình tĩnh", "correct": True}],
        }]
        with self.assertRaisesRegex(LessonAuthorProposalValidationError, "at least 2"):
            validate_lesson_author_proposal_shape(incomplete)

        valid = incomplete.copy()
        valid["chapters"] = [dict(incomplete["chapters"][0])]
        valid["chapters"][0]["lessons"] = [dict(incomplete["chapters"][0]["lessons"][0])]
        valid["chapters"][0]["lessons"][0]["units"] = [dict(incomplete["chapters"][0]["lessons"][0]["units"][0])]
        valid["chapters"][0]["lessons"][0]["units"][0]["components"] = [{
            "type": "multiple_choice",
            "question": "Hành động đầu tiên là gì?",
            "choices": [
                {"text": "Giữ bình tĩnh", "correct": True},
                {"text": "Chạy ngay", "correct": False},
            ],
        }]
        self.assertIsNone(validate_staged_unit_content(valid["chapters"][0]["lessons"][0]["units"][0]))

    def test_detailed_proposal_accepts_nested_html_component_content(self) -> None:
        nested = valid_proposal()
        nested["chapters"][0]["lessons"][0]["units"][0]["components"] = [{
            "type": "html",
            "title": "Tóm tắt",
            "content": {
                "html": "<p>Phân tích an toàn công việc giúp nhận diện mối nguy, đánh giá rủi ro và lựa chọn biện pháp kiểm soát phù hợp trước khi bắt đầu công việc. Người học cần quan sát từng bước, xác định điều kiện nguy hiểm, xem xét mức độ rủi ro và đề xuất biện pháp kiểm soát theo thứ tự ưu tiên để làm việc an toàn.</p>",
            },
        }]
        validate_lesson_author_proposal_shape(nested)

    def test_detailed_proposal_rejects_incomplete_sortable_component(self) -> None:
        incomplete = valid_proposal()
        incomplete["chapters"][0]["lessons"][0]["units"][0]["components"] = [{
            "type": "la_sortable",
            "title": "Sắp xếp quy trình",
            "items": ["Bước một", "Bước hai"],
        }]
        with self.assertRaisesRegex(LessonAuthorProposalValidationError, "at least 3"):
            validate_lesson_author_proposal_shape(incomplete)

    def test_detailed_proposal_accepts_nested_sortable_component_content(self) -> None:
        nested = valid_proposal()
        nested["chapters"][0]["lessons"][0]["units"][0]["components"] = [{
            "type": "la_sortable",
            "title": "Sắp xếp quy trình",
            "content": {
                "items": [
                    {"text": "Bước một"},
                    {"text": "Bước hai"},
                    {"text": "Bước ba"},
                ],
            },
        }]
        validate_lesson_author_proposal_shape(nested)


class LessonAuthorBlueprintRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_provider_quota_exhaustion_is_returned_as_non_retryable_service_error(self) -> None:
        with patch(
            "app.main.asyncio.to_thread",
            side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
        ):
            with self.assertRaises(HTTPException) as raised:
                await embed_text_batch(
                    "test-key",
                    "gemini-embedding-001",
                    ["source text"],
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_QUOTA_EXHAUSTED")

    async def test_provider_unavailable_is_returned_as_retryable_service_error(self) -> None:
        provider_error = RuntimeError("503 UNAVAILABLE")
        provider_error.status_code = 503
        with patch(
            "app.main.asyncio.to_thread",
            side_effect=provider_error,
        ):
            with self.assertRaises(HTTPException) as raised:
                await generate_content(
                    "test-key",
                    "gemini-3.5-flash",
                    "Return JSON.",
                    max_output_tokens=1024,
                    json_mode=True,
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_UNAVAILABLE")

    async def test_provider_quota_exhaustion_is_returned_as_non_retryable_service_error(self) -> None:
        with patch(
            "app.main.asyncio.to_thread",
            side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
        ):
            with self.assertRaises(HTTPException) as raised:
                await generate_content(
                    "test-key",
                    "gemini-3.5-flash",
                    "Return JSON.",
                    max_output_tokens=1024,
                    json_mode=True,
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_QUOTA_EXHAUSTED")

    async def test_invalid_first_candidate_retries_once_and_counts_both_attempts(self) -> None:
        first_usage = AiUsage(inputTokens=12, outputTokens=3, totalTokens=15)
        second_usage = AiUsage(inputTokens=14, outputTokens=7, totalTokens=21)
        with patch(
            "app.main.generate_content",
            new=AsyncMock(side_effect=[("{", first_usage), (json.dumps(valid_blueprint()), second_usage)]),
        ) as generate:
            blueprint, usage = await generate_validated_lesson_author_blueprint(
                blueprint_request(),
                "SERVER MODE: COURSE_BLUEPRINT.",
            )

        self.assertEqual(generate.await_count, 2)
        self.assertEqual(blueprint["title"], "Nhập môn an toàn thông tin")
        self.assertEqual(usage.inputTokens, 26)
        self.assertEqual(usage.outputTokens, 10)
        self.assertEqual(usage.totalTokens, 36)
        retry_prompt = generate.await_args_list[1].args[2]
        self.assertIn("<SERVER_VALIDATION_FEEDBACK>", retry_prompt)
        self.assertIn("Blueprint response is not valid JSON.", retry_prompt)

    async def test_retry_receives_structural_feedback_and_accepts_twelve_lessons(self) -> None:
        invalid_candidate = valid_blueprint()
        lesson = invalid_candidate["chapters"][0]["lessons"][0]
        invalid_candidate["chapters"][0]["lessons"] = [
            {**lesson, "title": f"Quy trình {index}"}
            for index in range(1, 14)
        ]
        repaired_candidate = valid_blueprint()
        repaired_candidate["chapters"][0]["lessons"] = [
            {**lesson, "title": f"Quy trình {index}"}
            for index in range(1, 13)
        ]
        usage = AiUsage(inputTokens=10, outputTokens=10, totalTokens=20)

        with patch(
            "app.main.generate_content",
            new=AsyncMock(side_effect=[
                (json.dumps(invalid_candidate, ensure_ascii=False), usage),
                (json.dumps(repaired_candidate, ensure_ascii=False), usage),
            ]),
        ) as generate:
            blueprint, _ = await generate_validated_lesson_author_blueprint(
                blueprint_request(),
                "SERVER MODE: COURSE_BLUEPRINT.",
            )

        self.assertEqual(len(blueprint["chapters"][0]["lessons"]), 12)
        retry_prompt = generate.await_args_list[1].args[2]
        self.assertIn("chapters[0].lessons must contain 1 to 12 items.", retry_prompt)

    async def test_persistent_invalid_candidates_raise_safe_error_with_usage(self) -> None:
        attempt_usage = AiUsage(inputTokens=10, outputTokens=2, totalTokens=12)
        with patch(
            "app.main.generate_content",
            new=AsyncMock(side_effect=[("{", attempt_usage), ("{", attempt_usage)]),
        ) as generate:
            with self.assertRaises(LessonAuthorBlueprintGenerationError) as raised:
                await generate_validated_lesson_author_blueprint(
                    blueprint_request(),
                    "SERVER MODE: COURSE_BLUEPRINT.",
                )

        self.assertEqual(generate.await_count, 2)
        self.assertEqual(raised.exception.code, "BLUEPRINT_INVALID_JSON")
        self.assertEqual(raised.exception.usage.totalTokens, 24)


if __name__ == "__main__":
    unittest.main()
