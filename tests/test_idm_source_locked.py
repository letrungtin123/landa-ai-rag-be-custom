"""IDM source-locked fallback content (QC course 234653, defects D8, D11, D12).

The legacy renderer dumped every short PDF line as ``<h3>`` (75 consecutive headings in one
course), made one-item ``<ol>`` lists of numbered headings, moved every table to the end under a
"Bảng dữ liệu" label and left wrapped lines split. The recovered question asked about the
crossword answer form ("KIEMTOANTHUCTRANG") with up to six options and a one-line explanation.
IDM units now use ``app.idm.source_locked``; the shared legacy builders are unchanged.
"""

from __future__ import annotations

import re
import unittest
from typing import Any
from unittest.mock import patch

from app import instructional_quality
from app.idm.source_locked import idm_source_grounded_single_choice, render_idm_source_locked_html
from app.instructional_quality import render_source_locked_html
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.lesson_author.staged import validation as staged_validation
from app.services.orchestration_v2 import source_locked as v2_source_locked
from app.services.orchestration_v2 import unit as unit_service
from tests.test_idm_storyboard import severity_body


def finding(component: dict[str, Any], plan: dict[str, Any] | None = None) -> str | None:
    result = staged_validation.staged_instructional_finding({"source_locked_fallback": True, **component}, 0, plan,
                                                            None, None)
    return None if result is None else result.code


class IdmHtmlRendererTests(unittest.TestCase):
    def render(self, *lines: str, title: str = "Bối cảnh mới", required: list[dict[str, Any]] | None = None) -> str:
        return render_idm_source_locked_html(title, list(lines), locale="vi", required_artifacts=required or [])

    def test_labels_never_become_a_run_of_headings(self) -> None:
        html = self.render("TRỤ CỘT 01", "Business Excellence", "TRỤ CỘT 02", "ERA5.0 Connection",
                           "Mọi hoạt động quản trị phải được tích hợp thành một cấu trúc vững chãi.")
        self.assertNotIn("</h3><h3>", html)
        self.assertEqual(html.count("<h3>"), 1)
        self.assertIn("<ul><li>TRỤ CỘT 01</li><li>Business Excellence</li><li>TRỤ CỘT 02</li></ul>", html)
        self.assertIn("<h3>ERA5.0 Connection</h3><p>Mọi hoạt động", html)
        two = self.render("CHƯƠNG 01", "WHY CHANGE? • BỐI CẢNH MỚI", "Môi trường kinh tế đã thay đổi hoàn toàn.")
        self.assertIn("<h3>CHƯƠNG 01 - WHY CHANGE? • BỐI CẢNH MỚI</h3>", two)
        trailing = self.render("Một đoạn nội dung đầy đủ để mở đầu phần này.", "Ghi chú cuối")
        self.assertTrue(trailing.endswith("<p>Ghi chú cuối</p>"))

    def test_a_lone_numbered_heading_is_a_heading_and_real_steps_stay_a_list(self) -> None:
        html = self.render("1. Thế Giới Mới: Biến Động Là Trạng Thái Mặc Định",
                           "Môi trường kinh tế toàn cầu đã chấm dứt kỷ nguyên ổn định tương đối.")
        self.assertIn("<h3>1. Thế Giới Mới: Biến Động Là Trạng Thái Mặc Định</h3>", html)
        self.assertNotIn("<ol>", html)
        steps = self.render("Quy trình xử lý:", "1. Tiếp nhận khiếu nại từ khách hàng.",
                            "2. Phân loại mức độ theo bảng.", "3. Chuyển cho người phụ trách.")
        self.assertEqual(len(re.findall(r"<li>", steps)), 3)
        self.assertIn("<ol>", steps)
        required = self.render("1. Thế Giới Mới: Biến Động", "Nội dung mô tả bối cảnh mới của doanh nghiệp.",
                               required=[{"type": "ordered_list", "minimum_items": 1}])
        self.assertIn("<ol><li>", required)
        self.assertIsNone(finding({"type": "html", "html": required},
                                  {"required_artifacts": [{"type": "ordered_list", "minimum_items": 1}]}))

    def test_wrapped_lines_are_joined_and_markdown_markers_removed(self) -> None:
        html = self.render("Doanh nghiệp cần áp dụng các công cụ quản trị chất lượng như Lean, Six", "Sigma.",
                           "Mục tiêu là nâng tỷ suất lợi nhuận và năng lực cạnh tranh trong", "90 ngày.",
                           "Thoát khỏi cuộc đua giá rẻ (*Race to the bottom*) bằng giá trị thật.")
        self.assertIn("<p>Doanh nghiệp cần áp dụng các công cụ quản trị chất lượng như Lean, Six Sigma.</p>", html)
        self.assertIn("năng lực cạnh tranh trong 90 ngày.</p>", html)
        self.assertIn("(Race to the bottom)", html)
        self.assertNotIn("*", html)
        heading = self.render("Chuyển Dịch Tư Duy Sống Còn Của Lãnh Đạo Doanh Nghiệp", "Đây là nền tảng.")
        self.assertIn("<h3>Chuyển Dịch Tư Duy Sống Còn Của Lãnh Đạo Doanh Nghiệp</h3><p>Đây là nền tảng.</p>",
                      heading)

    def test_tables_stay_in_place_without_a_generic_label(self) -> None:
        html = self.render("BiC tích hợp chu trình PDCA thành bộ công cụ hoàn chỉnh.",
                           "Row 1: CHU TRÌNH | CÔNG CỤ | Ý NGHĨA",
                           "Row 2: PLAN | Change Mindset | Cài đặt tư duy chất lượng lãnh đạo.",
                           "Row 3: ADVANCED | Lean / Six Sigma | Tinh gọn dòng chảy, loại bỏ các lãng phí",
                           "(Muda) và kiểm soát độ biến thiên.",
                           "Bảng trên tóm tắt vai trò của từng cấu phần trong chu trình.")
        self.assertNotIn("Bảng dữ liệu", html)
        self.assertLess(html.index("<table>"), html.index("Bảng trên tóm tắt"))
        self.assertIn("<td>Tinh gọn dòng chảy, loại bỏ các lãng phí (Muda) và kiểm soát độ biến thiên.</td>", html)
        self.assertNotIn("<p>(Muda)", html)
        form = self.render("Row 1: 01 Hành Động Đo Được Trong 90 Ngày;..............................",
                           "Row 2: Người Phụ Trách Triển Khai (Owner);....................")
        self.assertIn("<p>01 Hành Động Đo Được Trong 90 Ngày;</p>", form)
        self.assertNotIn("Row ", form)
        self.assertIsNone(finding({"type": "html", "html": form}))

    def test_title_repeats_are_dropped_unless_nothing_else_remains(self) -> None:
        html = self.render("30 Ngày: SEE DIFFERENT", "Kiểm toán thực trạng hệ điều hành tư duy của doanh nghiệp.",
                           title="Ngày: SEE DIFFERENT")
        self.assertNotIn("30 Ngày", html)
        alone = self.render("30 Ngày: SEE DIFFERENT", title="Ngày: SEE DIFFERENT")
        self.assertIn("30 Ngày", alone)
        self.assertIsNone(finding({"type": "html", "html": alone}))

    def test_qc_card_page_passes_the_shared_validator_with_far_fewer_headings(self) -> None:
        lines = ["CHƯƠNG 03", "THE BEST-IN-CLASS HOUSE • KIẾN TRÚC NĂNG LỰC", "Ngôi Nhà Năng Lực Best-in-Class",
                 "Mọi hoạt động quản trị của doanh nghiệp phải được tích hợp thành một cấu trúc vững chãi.",
                 "CHIẾN LƯỢC", "BMQ / Canvas / Balanced", "R&D SÁNG TẠO", "PMP / NQC Chuẩn Hóa", "KIỂM SOÁT",
                 "Hệ Thống ROSS & DQI", "KHÁCH HÀNG", "Customer Delight & QA"]
        html = self.render(*lines, title="Kiến trúc Ngôi nhà Năng lực")
        legacy = render_source_locked_html("Kiến trúc Ngôi nhà Năng lực", lines, locale="vi")
        self.assertGreater(legacy.count("<h3>"), 5)
        self.assertLessEqual(html.count("<h3>"), 1)
        self.assertIsNone(finding({"type": "html", "html": html}))


class IdmSingleChoiceTests(unittest.TestCase):
    DEFINITIONS = (
        "Kiểm toán thực trạng: Audit lại hệ điều hành tư duy và nhận diện các điểm nghẽn nghiêm trọng nhất.",
        "Lựa chọn chiến lược: Thiết lập tham vọng mới và chọn 01 năng lực lõi cần vươn lên Best-in-Class.",
        "Thực thi & Chuẩn hóa: Triển khai thí điểm và đo lường kết quả hàng tuần bằng dữ liệu thực.",
        "Đo lường kết quả: Theo dõi chỉ số hàng tuần bằng dữ liệu thực để điều chỉnh kế hoạch kịp thời.",
        "Nhân rộng mô hình: Đóng gói thành quy trình chuẩn SOP rồi triển khai cho các phòng ban khác.",
    )

    def test_definition_question_names_the_source_term_and_explains_the_other_options(self) -> None:
        problem = idm_source_grounded_single_choice("Lộ trình 90 ngày", self.DEFINITIONS, locale="vi")
        assert problem is not None
        self.assertIn("Kiểm toán thực trạng", problem["question"])
        self.assertNotIn("KIEMTOAN", problem["question"] + problem["explanation"])
        self.assertEqual(len(problem["choices"]), 4)
        self.assertEqual(sum(choice["correct"] for choice in problem["choices"]), 1)
        self.assertFalse(problem["choices"][0]["correct"])
        self.assertIn("“Lựa chọn chiến lược”", problem["explanation"])
        self.assertIsNone(finding({"type": "problem", **problem}))

    def test_table_and_step_questions_show_at_most_four_options(self) -> None:
        rows = ["Row 1: Bậc | Tên gọi | Mô tả", *[f"Row {n + 1}: Bậc 0{n} | Tên gọi số {n} | Mô tả {n}"
                                                for n in range(1, 7)]]
        table = idm_source_grounded_single_choice("Nấc thang", rows, locale="vi")
        assert table is not None
        self.assertEqual(len(table["choices"]), 4)
        for n in (2, 3, 4):
            self.assertIn(f"Tên gọi số {n} ↔ Bậc 0{n}", table["explanation"])
        self.assertIsNone(finding({"type": "problem", **table}))
        steps = idm_source_grounded_single_choice(
            "Quy trình", [f"{n}. Bước xử lý khiếu nại thứ {n} theo quy định" for n in range(1, 7)], locale="en")
        assert steps is not None
        self.assertEqual(len(steps["choices"]), 4)
        self.assertIn("is a later step", steps["explanation"])

    def test_authored_questions_pass_through_and_prose_yields_none(self) -> None:
        authored = ["Câu hỏi 1: Khi nào phải chuyển khiếu nại cho quản lý ngay lập tức?",
                    "A. Khi khiếu nại liên quan an toàn", "B. Khi khách hàng gọi lần đầu", "C. Khi có thời gian",
                    "Đáp án: A", "Giải thích: Khiếu nại liên quan an toàn phải escalate ngay theo quy định."]
        problem = idm_source_grounded_single_choice("Escalate", authored, locale="vi")
        assert problem is not None
        self.assertEqual(problem["question"], "Khi nào phải chuyển khiếu nại cho quản lý ngay lập tức?")
        self.assertIsNone(idm_source_grounded_single_choice("Văn xuôi", ["Một câu văn bình thường."], locale="vi"))


class IdmWiringTests(unittest.TestCase):
    def test_idm_units_use_the_idm_builders_and_legacy_defaults_are_unchanged(self) -> None:
        request = RagLessonAuthorUnitV2Request.model_validate(severity_body())
        with patch("app.services.orchestration_v2.unit.render_idm_source_locked_html",
                   wraps=render_idm_source_locked_html) as renderer, \
                patch("app.services.orchestration_v2.unit.idm_source_grounded_single_choice",
                      wraps=idm_source_grounded_single_choice) as question:
            deps = unit_service._idm_unit_deps(request)
            legacy = v2_source_locked.build_orchestration_v2_source_locked_components(request.unit_contract, "vi")
        self.assertEqual((renderer.call_count, question.call_count), (1, 1))
        assert deps.source_locked_components is not None
        html_slot = deps.source_locked_components[0]
        legacy_slot = legacy[0]
        assert html_slot is not None and legacy_slot is not None
        plan = request.unit_contract.component_plan[0]
        facts = instructional_quality.clean_source_facts([fact.fact_text for fact in request.unit_contract.source_facts
                                         if fact.fact_key in plan.source_fact_ids], preserve_table_numeric=True)
        self.assertEqual(legacy_slot["html"], render_source_locked_html(
            legacy_slot["title"], facts, locale="vi", required_artifacts=plan.required_artifacts))
        self.assertEqual(html_slot["html"], render_idm_source_locked_html(
            html_slot["title"], facts, locale="vi", required_artifacts=plan.required_artifacts))
        self.assertEqual({key: value for key, value in html_slot.items() if key != "html"},
                         {key: value for key, value in legacy_slot.items() if key != "html"})


if __name__ == "__main__":
    unittest.main()
