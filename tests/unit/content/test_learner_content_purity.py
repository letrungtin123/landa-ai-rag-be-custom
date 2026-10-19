import unittest

from app.learner_content_purity import (
    build_learner_content_purity_context,
    component_learner_text,
    is_provenance_only_source_line,
    learner_content_purity_finding,
    sanitize_source_fact_for_learner,
)


class LearnerContentPurityTests(unittest.TestCase):
    def test_rejects_source_attribution_locator_filename_and_internal_id(self) -> None:
        context = {
            "exact_identifiers": ["p5-f2", "src-004"],
            "source_filenames": ["Executive Training Playbook.pdf"],
        }
        self.assertEqual(
            learner_content_purity_finding("Theo tài liệu nguồn, cần đánh giá rủi ro.", context),
            "LEARNER_CONTENT_SOURCE_ATTRIBUTION",
        )
        self.assertEqual(
            learner_content_purity_finding("Nội dung này được nêu trong tài liệu nguồn.", context),
            "LEARNER_CONTENT_SOURCE_ATTRIBUTION",
        )
        self.assertEqual(
            learner_content_purity_finding("Tài liệu mô tả quy trình kiểm soát rủi ro.", context),
            "LEARNER_CONTENT_SOURCE_ATTRIBUTION",
        )
        self.assertEqual(
            learner_content_purity_finding("Xem nội dung tại trang 12.", context),
            "LEARNER_CONTENT_SOURCE_LOCATOR",
        )
        self.assertEqual(
            learner_content_purity_finding("Executive Training Playbook.pdf", context),
            "LEARNER_CONTENT_SOURCE_FILENAME",
        )
        self.assertEqual(
            learner_content_purity_finding("Nội dung p5-f2 cần được học.", context),
            "LEARNER_CONTENT_INTERNAL_IDENTIFIER",
        )

    def test_allows_legitimate_page_word_without_numeric_locator(self) -> None:
        self.assertIsNone(learner_content_purity_finding("Kiểm tra trang thiết bị trước khi vận hành."))

    def test_context_uses_only_unit_identifiers_and_source_filenames(self) -> None:
        context = build_learner_content_purity_context(
            {"source_fact_ids": ["p1-f1"], "supporting_evidence_fact_ids": []},
            {"facts": [
                {"fact_id": "p1-f1", "source_ref": "src-1", "text": "Approved fact"},
                {"fact_id": "p2-f1", "source_ref": "src-2", "text": "Other fact"},
            ]},
            [{"document_name": "Playbook.pdf"}],
        )
        self.assertEqual(context["exact_identifiers"], ["p1-f1", "src-1"])
        self.assertEqual(context["source_filenames"], ["Playbook.pdf"])

    def test_component_projection_excludes_provenance_fields(self) -> None:
        text = component_learner_text({
            "type": "html",
            "source_fact_ids": ["p1-f1"],
            "semantic_content": {"sections": [{"heading": "HIRA", "blocks": [
                {"kind": "paragraph", "text": "Đánh giá rủi ro trước khi làm việc."},
            ]}]},
        })
        self.assertIn("Đánh giá rủi ro", text)
        self.assertNotIn("p1-f1", text)

    def test_only_removes_provenance_furniture(self) -> None:
        self.assertTrue(is_provenance_only_source_line("Trang 12"))
        self.assertTrue(is_provenance_only_source_line("Nguồn: Playbook.pdf"))
        self.assertFalse(is_provenance_only_source_line("Kiểm tra trang thiết bị trước khi vận hành."))

    def test_source_fact_sanitizer_preserves_the_proposition_only(self) -> None:
        self.assertEqual(
            sanitize_source_fact_for_learner("Theo tài liệu nguồn, cần đánh giá rủi ro trước công việc."),
            "Cần đánh giá rủi ro trước công việc.",
        )
        self.assertEqual(
            sanitize_source_fact_for_learner("Giải thích: Tài liệu yêu cầu kiểm tra điều kiện an toàn."),
            "Giải thích: Kiểm tra điều kiện an toàn.",
        )
        self.assertEqual(
            sanitize_source_fact_for_learner("Nội dung này được nêu trong tài liệu nguồn."),
            "",
        )

    def test_source_fact_sanitizer_removes_joined_locators_without_losing_content(self) -> None:
        cases = {
            "Trang 27: Nhận diện mối nguy trước khi bắt đầu công việc.":
                "Nhận diện mối nguy trước khi bắt đầu công việc.",
            "Đánh giá mức rủi ro trước khi chọn biện pháp (slide 28, chunk 4).":
                "Đánh giá mức rủi ro trước khi chọn biện pháp.",
            "Page 9 - Confirm the control before work starts.":
                "Confirm the control before work starts.",
            "Xem nội dung tại trang 12.": "",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                cleaned = sanitize_source_fact_for_learner(source)
                self.assertEqual(cleaned, expected)
                self.assertIsNone(learner_content_purity_finding(cleaned))

    def test_source_fact_sanitizer_keeps_equipment_word_and_business_numbers(self) -> None:
        value = "Kiểm tra 12 trang thiết bị trước khi vận hành."
        self.assertEqual(sanitize_source_fact_for_learner(value), value)


if __name__ == "__main__":
    unittest.main()
