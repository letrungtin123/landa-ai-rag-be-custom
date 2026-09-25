from __future__ import annotations

import unittest

from app.main import (
    ExtractedSection,
    RagLessonAuthorRequest,
    build_retrieval_diagnostics,
    build_index_diagnostics,
    build_source_coverage_manifest,
    build_retrieval_query_texts,
    build_target_source_scopes,
    extract_source_coverage_facts,
    extract_lesson_author_unit_batches,
    format_source_coverage_manifest,
    LessonAuthorProposalValidationError,
    parse_source_range,
    should_stage_lesson_author_proposal,
    source_coverage_metrics,
    target_source_scope_is_incomplete,
    validate_lesson_author_source_coverage,
)
from app.source_structure import analyze_source_structure, chunk_structure_metadata, structure_outline


class SourceStructureTests(unittest.TestCase):
    def test_index_diagnostics_report_extracted_and_indexed_volume_without_raw_content(self) -> None:
        sections = [
            ExtractedSection(text="Nội dung trang một", page=1),
            ExtractedSection(text="Nội dung trang hai", page=2),
        ]
        chunks = [
            {"content": "Nội dung trang một", "token_count": 5},
            {"content": "Nội dung trang hai", "token_count": 5},
        ]

        diagnostics = build_index_diagnostics(sections, chunks, raw_bytes=128, file_name="source.pdf")

        self.assertEqual(diagnostics["extracted_page_count"], 2)
        self.assertEqual(diagnostics["extracted_chars"], len("Nội dung trang mộtNội dung trang hai"))
        self.assertEqual(diagnostics["indexed_chars"], len("Nội dung trang mộtNội dung trang hai"))
        self.assertEqual(diagnostics["raw_bytes"], 128)
        self.assertIn("PDF_TEXT_LAYER_ONLY; OCR_OR_EMBEDDED_IMAGE_TEXT_IS_NOT_EXTRACTED", diagnostics["warnings"])

    def test_source_range_is_parsed_from_vietnamese_toc_title(self) -> None:
        self.assertEqual(parse_source_range("Chương 6 (từ slide 30 đến slide 32 )"), (30, 32))
        self.assertEqual(parse_source_range("Chapter 2 (from page 4 to page 9)"), (4, 9))
        self.assertIsNone(parse_source_range("Chương không có phạm vi nguồn"))

    def test_lesson_author_scope_uses_selected_toc_chapter_range(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn chi tiết Chương 6: Quy định an toàn lao động và bảo vệ môi trường trong sản xuất",
            outline_context="Chương 6: Quy định an toàn lao động và bảo vệ môi trường trong sản xuất",
            target_scope_instruction="Phạm vi mục tiêu là Chương 6.",
            output_schema_hint="{}",
            api_key="test-key",
        )
        context = {
            "structure_source": "toc",
            "authoritative_source_nodes": [
                {
                    "document_id": "44444444-4444-4444-8444-444444444444",
                    "document_name": "raw.pdf",
                    "source_ref": "src-006",
                    "title": "Quy định an toàn lao động và bảo vệ môi trường trong sản xuất (từ slide 30 đến slide 32 )",
                },
            ],
        }

        scopes = build_target_source_scopes(request, context)

        self.assertEqual(scopes[0]["source_ref"], "src-006")
        self.assertEqual((scopes[0]["start_page"], scopes[0]["end_page"]), (30, 32))

    def test_lesson_author_scope_uses_explicit_chapter_number_over_full_outline_context(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn chi tiết Chương 1: An toàn Lao động và Sức khỏe Nghề nghiệp",
            outline_context="Chương 1 ... Chương 2 ... Chương 3 ... Chương 4 ... Chương 5 ... Chương 6 ...",
            target_scope_instruction="SERVER-RESOLVED OPERATION: UPDATE_CONTENT. Exact target is Chương 1.",
            output_schema_hint="{}",
            api_key="test-key",
        )
        nodes = [
            {
                "document_id": "44444444-4444-4444-8444-444444444444",
                "document_name": "raw.pdf",
                "source_ref": f"src-{index:03d}",
                "title": title,
                "level": 1,
                "logical_page": start_page,
            }
            for index, (title, start_page) in enumerate([
                ("An toàn Lao động và Sức khỏe Nghề nghiệp (từ Slide 1 đến slide 5)", 1),
                ("Nhận diện mối nguy và đánh giá rủi ro (từ Slide 6 đến slide 10)", 6),
                ("Thực hành PPE (từ slide 11 đến slide 25)", 11),
                ("Bảo vệ môi trường (từ slide 26 đến slide 27)", 26),
                ("Ứng phó khẩn cấp (từ slide 28 đến slide 29)", 28),
                ("Quy định an toàn (từ slide 30 đến slide 32)", 30),
            ], start=1)
        ]

        scopes = build_target_source_scopes(
            request,
            {"structure_source": "toc", "authoritative_source_nodes": nodes},
        )

        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0]["source_ref"], "src-001")
        self.assertEqual((scopes[0]["start_page"], scopes[0]["end_page"]), (1, 5))

    def test_toc_assigns_source_ref_to_body_chunks_by_page_range(self) -> None:
        structure = {
            "structure_source": "toc",
            "confidence": 0.95,
            "nodes": [
                {"source_ref": "src-001", "title": "Chương một", "level": 1, "logical_page": 1},
                {"source_ref": "src-002", "title": "Chương hai", "level": 1, "logical_page": 6},
            ],
        }

        first_chapter = chunk_structure_metadata(
            structure,
            page=5,
            text="Nội dung cuối chương một.",
            include_outline=False,
        )
        second_chapter = chunk_structure_metadata(
            structure,
            page=6,
            text="Nội dung đầu chương hai.",
            include_outline=False,
        )

        self.assertEqual(first_chapter["source_ref"], "src-001")
        self.assertEqual(second_chapter["source_ref"], "src-002")

    def test_source_coverage_manifest_preserves_bullets_and_numbered_steps(self) -> None:
        manifest = build_source_coverage_manifest([
            {"source_page": 3, "chunk_no": 1, "content": "• Định nghĩa một\n• Định nghĩa hai"},
            {"source_page": 4, "chunk_no": 2, "content": "1. Bước một\n2. Bước hai"},
        ])

        self.assertEqual(manifest["pages"], [3, 4])
        self.assertEqual([fact["fact_id"] for fact in manifest["facts"]], ["p3-f1", "p3-f2", "p4-f1", "p4-f2"])
        rendered = format_source_coverage_manifest(manifest)
        self.assertIn("[p3-f1]", rendered)
        self.assertIn("[p4-f2]", rendered)

    def test_36_chunk_371_fact_regression_keeps_full_canonical_manifest(self) -> None:
        rows = []
        fact_number = 1
        for page in range(1, 37):
            page_fact_count = 11 if page <= 11 else 10
            lines = []
            for _ in range(page_fact_count):
                lines.append(f"• HSE source rule {fact_number}: a distinct source-grounded requirement.")
                fact_number += 1
            rows.append({
                "document_id": "doc-hse",
                "source_page": page,
                "chunk_no": page - 1,
                "content": "\n".join(lines),
            })

        manifest = build_source_coverage_manifest(rows, canonical_max_chars=1_000_000)

        self.assertEqual(fact_number - 1, 371)
        self.assertEqual(manifest["total_fact_count"], 371)
        self.assertEqual(manifest["represented_fact_count"], 371)
        self.assertTrue(manifest["fact_scope_complete"])
        self.assertFalse(manifest["truncated"])
        self.assertEqual(len({fact["fact_id"] for fact in manifest["facts"]}), 371)

        incomplete = build_source_coverage_manifest(rows, canonical_max_chars=1_000)
        self.assertEqual(incomplete["total_fact_count"], 371)
        self.assertLess(incomplete["represented_fact_count"], 371)
        self.assertFalse(incomplete["fact_scope_complete"])
        self.assertEqual(incomplete["incomplete_reason"], "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED")

    def test_source_coverage_manifest_skips_branded_cover_page(self) -> None:
        manifest = build_source_coverage_manifest([
            {"source_page": 1, "chunk_no": 0, "content": "HSE training www.example.com"},
            {"source_page": 3, "chunk_no": 1, "content": "• Định nghĩa an toàn lao động"},
        ])

        self.assertEqual(manifest["pages"], [1, 3])
        self.assertEqual([fact["fact_id"] for fact in manifest["facts"]], ["p3-f1"])

    def test_paginated_manifest_resolves_toc_source_ref_from_matching_page_range(self) -> None:
        manifest = build_source_coverage_manifest(
            [{
                "document_id": "doc-hse",
                "source_page": 5,
                "chunk_no": 3,
                "content": "Nội dung chính xác thuộc phạm vi Chương một.",
            }],
            structure_nodes=[{
                "document_id": "doc-hse",
                "source_ref": "src-001",
                "title": "Chương một (từ trang 4 đến trang 6)",
            }],
            target_source_refs={"src-001"},
            target_scopes=[{
                "document_id": "doc-hse",
                "source_ref": "src-001",
                "start_page": 4,
                "end_page": 6,
            }],
        )

        self.assertFalse(manifest["scope_unresolved"])
        self.assertEqual(manifest["resolved_source_refs"], ["src-001"])

    def test_paginated_manifest_resolves_heading_ref_from_canonical_page_metadata(self) -> None:
        manifest = build_source_coverage_manifest(
            [{
                "document_id": "doc-hse",
                "source_page": 2,
                "chunk_no": 1,
                "content": "Nội dung thuộc tiêu đề đã được parser định vị ở trang hai.",
            }],
            structure_nodes=[{
                "document_id": "doc-hse",
                "source_ref": "src-010",
                "title": "Chuẩn bị dụng cụ và hóa chất",
                "page": 2,
                "structure_source": "heading_inferred",
            }],
            target_source_refs={"src-010"},
        )

        self.assertFalse(manifest["scope_unresolved"])
        self.assertEqual(manifest["resolved_source_refs"], ["src-010"])

    def test_paginated_manifest_does_not_resolve_heading_without_matching_page_chunk(self) -> None:
        manifest = build_source_coverage_manifest(
            [{
                "document_id": "doc-hse",
                "source_page": 3,
                "chunk_no": 1,
                "content": "Nội dung trang khác.",
            }],
            structure_nodes=[{
                "document_id": "doc-hse",
                "source_ref": "src-010",
                "title": "Chuẩn bị dụng cụ và hóa chất",
                "page": 2,
                "structure_source": "heading_inferred",
            }],
            target_source_refs={"src-010"},
        )

        self.assertTrue(manifest["scope_unresolved"])
        self.assertEqual(manifest["resolved_source_refs"], [])

    def test_unpaginated_manifest_scopes_inferred_docx_to_blueprint_refs(self) -> None:
        manifest = build_source_coverage_manifest(
            [{
                "document_id": "doc-office",
                "document_name": "office.docx",
                "source_page": None,
                "chunk_no": 0,
                "metadata": '{"source_ref":"src-001"}',
                "content": (
                    "OFFICE AREA CLEANING INSTRUCTIONS\n"
                    "I. REQUIRED STANDARDS\n"
                    "The office area is neat and orderly.\n"
                    "1. Prepare tools, equipment, and chemicals | - Glass cleaner\n"
                    "- Clean wiping cloths\n"
                    "2. PERFORM CLEANING | 2. PERFORM CLEANING\n"
                    "2.1. Request permission to clean | Ask before cleaning."
                ),
            }],
            structure_nodes=[
                {"document_id": "doc-office", "source_ref": "src-001", "title": "OFFICE AREA CLEANING INSTRUCTIONS", "order": 0},
                {"document_id": "doc-office", "source_ref": "src-002", "title": "REQUIRED STANDARDS", "order": 1},
                {"document_id": "doc-office", "source_ref": "src-006", "title": "Prepare tools, equipment, and chemicals | - Glass cleaner", "order": 2},
                {"document_id": "doc-office", "source_ref": "src-007", "title": "PERFORM CLEANING | 2. PERFORM CLEANING", "order": 3},
                {"document_id": "doc-office", "source_ref": "src-008", "title": "Request permission to clean | Ask before cleaning.", "order": 4, "parent_source_ref": "src-007"},
            ],
            target_source_refs={"src-002", "src-006"},
        )

        fact_text = " ".join(fact["text"] for fact in manifest["facts"])
        self.assertFalse(manifest["scope_unresolved"])
        self.assertEqual(manifest["pages"], [])
        self.assertEqual(manifest["chunks"], [0])
        self.assertTrue(all(fact["source_ref"] in {"src-002", "src-006"} for fact in manifest["facts"]))
        self.assertIn("The office area is neat and orderly", fact_text)
        self.assertIn("Clean wiping cloths", fact_text)
        self.assertNotIn("Request permission", fact_text)
        self.assertIn("mục nguồn src-002", format_source_coverage_manifest(manifest))

    def test_unpaginated_manifest_reports_missing_blueprint_ref(self) -> None:
        manifest = build_source_coverage_manifest(
            [{
                "document_id": "doc-office",
                "source_page": None,
                "chunk_no": 0,
                "content": "I. REQUIRED STANDARDS\nThe office is clean.",
            }],
            structure_nodes=[{
                "document_id": "doc-office",
                "source_ref": "src-002",
                "title": "REQUIRED STANDARDS",
                "order": 1,
            }],
            target_source_refs={"src-002", "src-999"},
        )

        self.assertTrue(manifest["scope_unresolved"])
        self.assertEqual(manifest["resolved_source_refs"], ["src-002"])

    def test_source_fact_split_does_not_truncate_long_docx_lines(self) -> None:
        source = "A" * 1000
        facts = extract_source_coverage_facts(source)

        self.assertGreater(len(facts), 1)
        self.assertTrue(all(len(fact) <= 420 for fact in facts))
        self.assertEqual("".join(facts), source)

    def test_source_fact_extraction_rejoins_wrapped_pdf_sentences(self) -> None:
        facts = extract_source_coverage_facts(
            "An toàn lao động: Trạng thái làm việc không xảy ra tai\n"
            "nạn hoặc tổn thương.\n"
            "Sức khỏe nghề nghiệp: Phòng ngừa và kiểm soát\n"
            "các yếu tố có thể ảnh hưởng đến người lao động."
        )

        self.assertEqual(facts, [
            "An toàn lao động: Trạng thái làm việc không xảy ra tai nạn hoặc tổn thương.",
            "Sức khỏe nghề nghiệp: Phòng ngừa và kiểm soát các yếu tố có thể ảnh hưởng đến người lao động.",
        ])

    def test_source_coverage_requires_every_manifest_fact(self) -> None:
        manifest = build_source_coverage_manifest([
            {"source_page": 3, "chunk_no": 1, "content": "• Định nghĩa một\n• Định nghĩa hai"},
        ])
        proposal = {
            "chapters": [{
                "lessons": [{
                    "units": [{"source_fact_ids": ["p3-f1"]}],
                }],
            }],
        }

        metrics = source_coverage_metrics(proposal, manifest)
        self.assertEqual(metrics["covered_count"], 1)
        self.assertEqual(metrics["status"], "incomplete")
        with self.assertRaises(LessonAuthorProposalValidationError):
            validate_lesson_author_source_coverage(proposal, manifest)

    def test_blank_page_gap_does_not_fail_hard_locked_scope(self) -> None:
        structure_context = {
            "target_source_scope_hard_locked": True,
            "target_source_scope_missing_pages": [23],
            "target_source_scope_truncated": False,
        }
        rows = [
            {"source_page": page, "chunk_no": index, "content": f"Nội dung trang {page}"}
            for index, page in enumerate([22, 24])
        ]
        sources = [{"source_page": row["source_page"]} for row in rows]

        self.assertFalse(target_source_scope_is_incomplete(structure_context, rows, sources, None))

    def test_blank_page_gap_does_not_mark_retrieval_incomplete(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn nội dung Chương 3",
            target_scope_instruction="Chương 3",
            output_schema_hint="{}",
            api_key="test-key",
        )
        structure_context = {
            "target_source_scope_hard_locked": True,
            "target_source_scope_missing_pages": [23],
        }
        rows = [{"content": "source", "score": 1.0, "method": "source_scope"}]
        sources = [{"content": "source"}]

        diagnostics = build_retrieval_diagnostics(request, rows, sources, structure_context)

        self.assertIsNone(diagnostics["reason"])

    def test_hard_locked_scope_still_rejects_empty_or_truncated_context(self) -> None:
        base_context = {"target_source_scope_hard_locked": True}

        self.assertTrue(target_source_scope_is_incomplete(base_context, [], [], None))
        self.assertTrue(target_source_scope_is_incomplete(
            {**base_context, "target_source_scope_truncated": True},
            [{"content": "source"}],
            [{"content": "source"}],
            None,
        ))
        self.assertTrue(target_source_scope_is_incomplete(
            base_context,
            [{"content": "source"}],
            [],
            None,
        ))

    def test_authoring_embedding_queries_include_outline_scope(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn nội dung chương 6",
            outline_context="Chương 6: Bảo vệ môi trường",
            target_scope_instruction="Chỉ lấy dữ liệu trong chương 6.",
            output_schema_hint="{}",
            api_key="test-key",
        )

        queries = build_retrieval_query_texts(request)

        self.assertEqual(queries[0], "Soạn nội dung chương 6")
        self.assertIn("Chương 6: Bảo vệ môi trường", queries)
        self.assertIn("Chỉ lấy dữ liệu trong chương 6.", queries)

    def test_large_explicit_chapter_scope_is_staged_and_batched(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=4096,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn chi tiết Chương 1",
            outline_context="Chương 1: Nền tảng",
            target_scope_instruction="Chỉ lấy dữ liệu trong chương 1.",
            output_schema_hint="{}",
            api_key="test-key",
        )
        skeleton = {
            "chapters": [{
                "title": "Nền tảng",
                "lessons": [{
                    "title": "Bài chính",
                    "units": [
                        {"title": f"Mục {index}", "components": [{"type": "html"}]}
                        for index in range(1, 7)
                    ],
                }],
            }],
        }

        self.assertTrue(should_stage_lesson_author_proposal(request, "x" * 16000))
        batches = extract_lesson_author_unit_batches(skeleton)
        self.assertEqual([len(batch) for batch in batches], [1, 1, 1, 1, 1, 1])

    def test_medium_chapter_scope_uses_staged_generation_before_json_overflow(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=8192,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn chi tiết Chương 3: Thực hành nhận diện mối nguy",
            outline_context="Chương 3: Thực hành nhận diện mối nguy",
            target_scope_instruction="Chỉ lấy dữ liệu trong chương 3.",
            output_schema_hint="{}",
            api_key="test-key",
        )

        self.assertTrue(should_stage_lesson_author_proposal(request, "x" * 6559))

    def test_blueprint_target_scope_also_enables_staged_generation(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=8192,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Soạn chi tiết Chương 3",
            target_scope_instruction="Chương 3: Thực hành nhận diện mối nguy.",
            output_schema_hint="{}",
            api_key="test-key",
        )

        self.assertTrue(should_stage_lesson_author_proposal(request, "x" * 6559))

    def test_large_new_chapter_request_without_selected_scope_is_staged(self) -> None:
        request = RagLessonAuthorRequest(
            tenant_id="11111111-1111-4111-8111-111111111111",
            kb_id="22222222-2222-4222-8222-222222222222",
            conversation_id="33333333-3333-4333-8333-333333333333",
            target="lesson_author",
            model="gemini-3.5-flash",
            max_output_tokens=8192,
            embedding_model="gemini-embedding-001",
            system_prompt="system",
            user_message="Ok hãy tạo chương tiếp theo cho khóa học này",
            output_schema_hint="{}",
            api_key="test-key",
        )

        self.assertTrue(should_stage_lesson_author_proposal(request, "small context"))

    def test_source_outline_hides_trailing_slide_range_from_authoring_prompt(self) -> None:
        structure = analyze_source_structure(
            [ExtractedSection(text="1. An toàn lao động (từ slide 30 đến slide 32)", page=1)]
        )

        self.assertEqual(structure["nodes"][0]["title"], "An toàn lao động (từ slide 30 đến slide 32)")
        outline = structure_outline(structure)
        self.assertIn("An toàn lao động", outline)
        self.assertNotIn("từ slide 30 đến slide 32", outline)

    def test_table_of_contents_is_used_as_authoritative_outline(self) -> None:
        structure = analyze_source_structure(
            [
                ExtractedSection(
                    text="""
                    MỤC LỤC
                    1. Nền tảng học tập........ 3
                    1.1. Mục tiêu khóa học........ 4
                    2. Thực hành áp dụng........ 9
                    """,
                    page=1,
                ),
                ExtractedSection(
                    text="1. Nền tảng học tập\nNội dung chi tiết.",
                    page=3,
                ),
            ]
        )

        self.assertEqual(structure["structure_source"], "toc")
        self.assertEqual(structure["confidence"], 0.95)
        self.assertEqual([node["title"] for node in structure["nodes"]], [
            "Nền tảng học tập",
            "Mục tiêu khóa học",
            "Thực hành áp dụng",
        ])
        self.assertEqual(structure["nodes"][1]["level"], 2)
        self.assertEqual(structure["nodes"][1]["parent_source_ref"], "src-001")
        self.assertIn("[src-001]", structure_outline(structure))

    def test_numbered_headings_are_inferred_without_toc(self) -> None:
        structure = analyze_source_structure(
            [
                ExtractedSection(
                    text="""
                    1. Nhận diện vấn đề
                    Phần nội dung thứ nhất.
                    2. Xử lý tình huống
                    Các bước thực hiện.
                    """,
                    page=2,
                )
            ]
        )

        self.assertEqual(structure["structure_source"], "heading_inferred")
        self.assertEqual(len(structure["nodes"]), 2)
        self.assertIn("TOC_NOT_FOUND_HEADING_INFERRED", structure["warnings"])

    def test_split_toc_title_and_page_lines_are_supported(self) -> None:
        structure = analyze_source_structure(
            [
                ExtractedSection(
                    text="""
                    TABLE OF CONTENTS
                    Chapter I: Nền tảng
                    1
                    Mục tiêu khóa học
                    3
                    Nội dung thực hành
                    8
                    """,
                    page=2,
                )
            ]
        )

        self.assertEqual(structure["structure_source"], "toc")
        self.assertEqual([node["title"] for node in structure["nodes"]], [
            "Nền tảng",
            "Mục tiêu khóa học",
            "Nội dung thực hành",
        ])
        self.assertEqual([node["logical_page"] for node in structure["nodes"]], [1, 3, 8])
        self.assertEqual(structure["nodes"][1]["parent_source_ref"], "src-001")

    def test_multiline_numbered_program_contents_is_authoritative(self) -> None:
        structure = analyze_source_structure(
            [
                ExtractedSection(
                    text="""
                    1.
                    An toàn Lao động và Sức khỏe Nghề nghiệp
                    (từ Slide 1 đến slide 5)
                    2.
                    Nhận diện mối nguy và đánh giá rủi ro (từ
                    Slide 6 đến slide 10)
                    3.
                    Thực hành Nhận diện mối nguy, Đánh giá
                    Rủi ro, Lựa chọn và sử dụng PPE phù hợp
                    (từ slide 11 đến slide 25)
                    4.
                    Bảo vệ môi trường trong sản xuất (từ slide
                    26 đến slide 27)
                    5.
                    Quy Trình Ứng Phó Khẩn Cấp Và Bảo Vệ Môi
                    Trường (từ slide 28 đến slide 29 )
                    6.
                    Quy định an toàn lao động và bảo vệ môi
                    trường trong sản xuất (từ slide 30 đến slide
                    32 )
                    Nội dung Chương trình
                    """,
                    page=2,
                ),
                ExtractedSection(text="1. Nội dung chi tiết.", page=3),
            ]
        )

        self.assertEqual(structure["structure_source"], "toc")
        self.assertEqual(len(structure["nodes"]), 6)
        self.assertEqual([node["number_label"] for node in structure["nodes"]], ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(
            [node["title"] for node in structure["nodes"]],
            [
                "An toàn Lao động và Sức khỏe Nghề nghiệp (từ Slide 1 đến slide 5)",
                "Nhận diện mối nguy và đánh giá rủi ro (từ Slide 6 đến slide 10)",
                "Thực hành Nhận diện mối nguy, Đánh giá Rủi ro, Lựa chọn và sử dụng PPE phù hợp (từ slide 11 đến slide 25)",
                "Bảo vệ môi trường trong sản xuất (từ slide 26 đến slide 27)",
                "Quy Trình Ứng Phó Khẩn Cấp Và Bảo Vệ Môi Trường (từ slide 28 đến slide 29 )",
                "Quy định an toàn lao động và bảo vệ môi trường trong sản xuất (từ slide 30 đến slide 32 )",
            ],
        )
        self.assertEqual([node["logical_page"] for node in structure["nodes"]], [1, 6, 11, 26, 28, 30])
        self.assertNotIn("TOC_NOT_FOUND_HEADING_INFERRED", structure["warnings"])

    def test_roman_headings_are_supported(self) -> None:
        structure = analyze_source_structure(
            [ExtractedSection(text="I. Cơ sở\nNội dung.\nII. Ứng dụng\nThực hành.", page=1)]
        )

        self.assertEqual([node["number_label"] for node in structure["nodes"]], ["I", "II"])
        self.assertEqual([node["title"] for node in structure["nodes"]], ["Cơ sở", "Ứng dụng"])

    def test_missing_headings_use_explicit_low_confidence_fallback(self) -> None:
        structure = analyze_source_structure(
            [ExtractedSection(text="Đây là nội dung liên tục không có tiêu đề rõ ràng.", page=1)]
        )

        self.assertEqual(structure["structure_source"], "semantic_inferred")
        self.assertEqual(structure["confidence"], 0.35)
        self.assertIn("NO_EXPLICIT_HEADINGS", structure["warnings"])


if __name__ == "__main__":
    unittest.main()
