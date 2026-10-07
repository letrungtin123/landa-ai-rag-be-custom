from __future__ import annotations

import unittest

from app.instructional_quality import (
    build_source_grounded_single_choice,
    build_source_locked_single_choice,
    clean_source_facts,
    faq_answer_is_complete,
    normalize_visible_text,
    ordered_source_steps,
    render_source_locked_html,
    source_relationship_pairs,
    source_clarification_signals,
    source_term_definitions,
)


class InstructionalQualityTests(unittest.TestCase):
    def test_faq_signals_match_the_learner_facing_answer_contract(self) -> None:
        valid_answers = [
            "Nếu phát hiện điều kiện không an toàn, người lao động phải dừng công việc và báo người phụ trách.",
            "Khi biện pháp kiểm soát chưa có hiệu lực, công việc chỉ được tiếp tục sau khi đánh giá lại.",
        ]
        signals = source_clarification_signals([
            "Lưu ý: Dừng lại.",
            "nếu thiếu dữ liệu, người thực hiện phải hỏi người phụ trách trước khi tiếp tục công việc.",
            *valid_answers,
        ])

        self.assertEqual(signals, valid_answers)
        self.assertFalse(faq_answer_is_complete("Lưu ý: Dừng lại."))
        self.assertFalse(faq_answer_is_complete(
            "nếu thiếu dữ liệu, người thực hiện phải hỏi người phụ trách trước khi tiếp tục công việc."
        ))
        self.assertTrue(all(faq_answer_is_complete(answer) for answer in signals))

    def test_visible_wire_newline_tokens_never_reach_learner_text(self) -> None:
        self.assertEqual(
            normalize_visible_text(r"Mối nguy\nBiện pháp /n Kiểm soát\r\nXác nhận"),
            "Mối nguy Biện pháp Kiểm soát Xác nhận",
        )

    def test_source_cleaning_removes_presentation_and_contact_debris(self) -> None:
        facts = clean_source_facts([
            "THANK YOU",
            "www.l-a.com.vn",
            "training@example.com",
            "vvvvvvvvvvvvvvvv",
            "Kiểm tra mối nguy trước khi bắt đầu công việc.",
        ])

        self.assertEqual(facts, ["Kiểm tra mối nguy trước khi bắt đầu công việc."])

    def test_table_rows_render_once_as_semantic_table(self) -> None:
        value = render_source_locked_html(
            "Ma trận rủi ro",
            [
                "Ma trận dùng để xác định mức ưu tiên kiểm soát.",
                "Row 1: Khả năng | Hậu quả | Mức rủi ro",
                "Row 2: Có thể xảy ra | Nghiêm trọng | Cao",
                "Row 3: Hiếm khi | Nhẹ | Thấp",
            ],
            locale="vi",
            required_artifacts=[{"type": "table", "minimum_items": 2}],
        )

        self.assertEqual(value.count("<table>"), 1)
        self.assertIn("<th>Khả năng</th>", value)
        self.assertIn("<td>Cao</td>", value)
        self.assertNotIn("Row 1", value)

    def test_checklist_and_callouts_survive_source_locked_rendering(self) -> None:
        value = render_source_locked_html(
            "Kiểm tra trước công việc",
            [
                "☐ Xác nhận khu vực làm việc đã được cô lập.",
                "☑ Ghi nhận người chịu trách nhiệm kiểm tra.",
                "Cảnh báo: Không tiếp tục nếu biện pháp kiểm soát chưa hiệu lực.",
                "Yêu cầu: Lưu kết quả kiểm tra theo quy trình đã phê duyệt.",
                "Ngoại lệ: Chỉ người có thẩm quyền mới được phê duyệt thay đổi.",
            ],
            locale="vi",
            required_artifacts=[
                {"type": "checklist", "minimum_items": 2},
                {"type": "warning", "minimum_items": 1},
                {"type": "requirement", "minimum_items": 1},
                {"type": "exception", "minimum_items": 1},
            ],
        )

        self.assertEqual(value.count("<li>"), 2)
        self.assertEqual(value.count("<blockquote>"), 3)
        self.assertNotIn("☐", value)
        self.assertNotIn("☑", value)

    def test_relationship_parser_preserves_explicit_table_pairs(self) -> None:
        pairs = source_relationship_pairs([
            "Row 1: Chu trình | Công cụ BiC",
            "Row 2: PLAN | Change Mindset + BQS/BMQ",
            "Row 3: DO | SPD (Standard Product Dev)",
        ])

        self.assertEqual(pairs, [
            ("PLAN", "Change Mindset + BQS/BMQ", "Công cụ BiC"),
            ("DO", "SPD (Standard Product Dev)", "Công cụ BiC"),
        ])

    def test_relationship_parser_preserves_arrow_direction_and_normalizes_wire_newlines(self) -> None:
        pairs = source_relationship_pairs([
            r"SME ↔ Expert ↔ Technology\n↔ AI/Data",
        ])

        self.assertEqual(pairs, [
            ("SME", "Expert", "↔"),
            ("Expert", "Technology", "↔"),
            ("Technology", "AI/Data", "↔"),
        ])
        self.assertNotIn(r"\n", " ".join(value for pair in pairs for value in pair if value))

    def test_relationship_parser_does_not_infer_edges_from_nearby_prose(self) -> None:
        self.assertEqual(source_relationship_pairs([
            "Vai trò lãnh đạo được mô tả trong nội dung này.",
            "Nhóm thực hiện báo cáo tiến độ theo quy định.",
        ]), [])

    def test_problem_fallback_requires_an_explicit_source_authored_check(self) -> None:
        problem = build_source_locked_single_choice(
            "Kiểm tra trước công việc",
            ["Người thực hiện phải kiểm tra điều kiện an toàn trước khi bắt đầu công việc."],
            locale="vi",
        )

        self.assertIsNone(problem)

    def test_grounded_problem_can_use_definitions_table_or_explicit_steps(self) -> None:
        candidates = [
            [
                "Mối nguy cơ khí: Nguồn chuyển động có thể gây va đập, cuốn hoặc kẹp người lao động.",
                "Mối nguy điện: Nguồn điện không được kiểm soát có thể gây điện giật hoặc hồ quang.",
                "Mối nguy hóa chất: Phơi nhiễm cần được kiểm soát theo đặc tính của hóa chất.",
            ],
            [
                "Row 1: Mối nguy | Biện pháp kiểm soát",
                "Row 2: Nhiệt | Lắp tấm chắn trước khi bắt đầu",
                "Row 3: Điện | Cô lập nguồn điện",
                "Row 4: Hóa chất | Dùng phương tiện bảo hộ phù hợp",
            ],
            [
                "Bước 1: Nhận diện mối nguy tại khu vực làm việc.",
                "Bước 2: Đánh giá khả năng và hậu quả của rủi ro.",
                "Bước 3: Lựa chọn biện pháp kiểm soát phù hợp.",
            ],
        ]
        for facts in candidates:
            with self.subTest(facts=facts[0]):
                problem = build_source_grounded_single_choice("Kiểm tra", facts, locale="vi")
                self.assertIsNotNone(problem)
                assert problem is not None
                self.assertEqual(sum(choice["correct"] for choice in problem["choices"]), 1)
                self.assertFalse(problem["choices"][0]["correct"])
                self.assertTrue(all("Bước 1:" not in choice["text"] for choice in problem["choices"]))

        self.assertIsNone(build_source_grounded_single_choice(
            "Nội dung chung",
            ["Người học cần hiểu nội dung và áp dụng phù hợp trong công việc."],
            locale="vi",
        ))

    def test_html_fallback_uses_semantic_hierarchy_without_source_citations(self) -> None:
        value = render_source_locked_html(
            "Kiểm soát rủi ro",
            ["Theo tài liệu nguồn, người lao động phải kiểm tra điều kiện trước khi bắt đầu công việc."],
            locale="vi",
        )
        self.assertTrue(value.startswith("<h2>Kiểm soát rủi ro</h2>"))
        self.assertNotIn("tài liệu nguồn", value.casefold())
        self.assertIn("Người lao động phải kiểm tra", value)

    def test_explicit_source_authored_check_is_recovered_without_invented_distractors(self) -> None:
        problem = build_source_locked_single_choice(
            "Kiểm tra trước công việc",
            [
                "Câu hỏi: Việc nào phải thực hiện trước khi bắt đầu công việc?",
                "A. Kiểm tra điều kiện an toàn tại nơi làm việc",
                "B. Bỏ qua mối nguy đã nhận diện",
                "C. Chờ đến khi xảy ra sự cố mới kiểm tra",
                "Đáp án: A",
                "Giải thích: Tài liệu yêu cầu kiểm tra điều kiện an toàn trước khi công việc bắt đầu.",
            ],
            locale="vi",
        )

        self.assertIsNotNone(problem)
        assert problem is not None
        self.assertEqual(problem["problem_type"], "multiple_choice")
        self.assertEqual(len(problem["choices"]), 3)
        self.assertEqual(sum(choice["correct"] for choice in problem["choices"]), 1)
        self.assertEqual(problem["choices"][0]["text"], "Kiểm tra điều kiện an toàn tại nơi làm việc")

    def test_sortable_requires_a_real_three_step_sequence(self) -> None:
        self.assertEqual(ordered_source_steps([
            "Khái niệm nhận diện mối nguy.",
            "Mức độ rủi ro phụ thuộc khả năng và hậu quả.",
            "Biện pháp kiểm soát cần phù hợp với mối nguy.",
        ], locale="vi"), [])
        self.assertEqual(len(ordered_source_steps([
            "Bước 1: Nhận diện mối nguy tại khu vực làm việc.",
            "Bước 2: Đánh giá khả năng và hậu quả của rủi ro.",
            "Bước 3: Lựa chọn biện pháp kiểm soát phù hợp.",
        ], locale="vi")), 3)

    def test_crossword_requires_explicit_term_definitions(self) -> None:
        self.assertEqual(source_term_definitions([
            "Bước 1: Nhận diện mối nguy tại khu vực làm việc.",
            "Nội dung chung không định nghĩa một thuật ngữ cụ thể.",
        ]), [])
        self.assertEqual(source_term_definitions([
            "Mối nguy cơ khí: Nguồn chuyển động có thể gây va đập, cuốn hoặc kẹp người lao động.",
        ]), [("MOINGUYCOKHI", "Nguồn chuyển động có thể gây va đập, cuốn hoặc kẹp người lao động.")])
        self.assertEqual(source_term_definitions([
            "5S: Phương pháp tổ chức nơi làm việc nhằm duy trì trật tự, sạch sẽ và kỷ luật.",
        ])[0][0], "5S")


if __name__ == "__main__":
    unittest.main()
