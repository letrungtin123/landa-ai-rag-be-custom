"""Fixes from the independent review 6 of the review 5 fixes (345663c..90ee210), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import unittest

from app.idm.node_acceptance import html_quality_code
from app.idm.source_locked import _is_label, render_idm_source_locked_html


class DashLineTests(unittest.TestCase):
    """M1: a source line holding an en dash is measured whole; only a label the fallback led with a section name is
    measured without that name."""

    def test_rules_with_an_en_dash_stay_paragraphs(self) -> None:
        rules = ["Khách hàng VIP – được ưu tiên xử lý trong vòng 2 giờ làm việc",  # noqa: RUF001 - the en dash
                 "Khách hàng thường – được xử lý trong vòng 24 giờ kể từ khi tiếp nhận",  # noqa: RUF001
                 "Mọi khiếu nại đều được ghi vào hệ thống trước khi xử lý."]
        title = "CHƯƠNG 2 – QUY TRÌNH XỬ LÝ KHIẾU NẠI CỦA KHÁCH HÀNG DOANH NGHIỆP LỚN"  # noqa: RUF001 - the en dash
        self.assertEqual([_is_label(line) for line in [*rules, title]], [False, False, False, False])
        html = render_idm_source_locked_html("Xử lý khiếu nại", rules, locale="vi")
        self.assertEqual(html, "<h2>Xử lý khiếu nại</h2>" + "".join(f"<p>{rule}</p>" for rule in rules))
        capitals = render_idm_source_locked_html("Xử lý khiếu nại", [title, rules[2]], locale="vi")
        self.assertIn(f"<p>{title}</p>", capitals)


class PrefixTests(unittest.TestCase):
    """M2 and L7: only a label another section uses gets the section name; never a figure or a data line, and never
    a label the renderer joins with its section title."""

    def test_figures_and_data_lines_keep_their_words(self) -> None:
        html = render_idm_source_locked_html("Hồ sơ khách hàng", [
            "HỒ SƠ KHÁCH HÀNG (VD1: KẾ TOÁN TRƯỞNG)", "Mục tiêu", "Đóng sổ sách đúng hạn vào cuối mỗi tháng.",
            "HỒ SƠ KHÁCH HÀNG (VD2: GIÁM ĐỐC TÀI CHÍNH)", "Mục tiêu", "Phê duyệt khoản đầu tư có hiệu quả rõ ràng.",
            "Ngân sách tối đa 2 tỷ đồng", "Thời hạn hoàn vốn 18 tháng", "85%",
            "Tỷ lệ dự án đạt mục tiêu năm ngoái theo báo cáo nội bộ."], locale="vi")
        self.assertNotIn("Giám đốc tài chính – Ngân sách", html)  # noqa: RUF001 - the en dash
        self.assertNotIn("– 85%", html)  # noqa: RUF001 - the en dash
        self.assertIn("<p>85%</p>", html)
        self.assertIn("<p>Ngân sách tối đa 2 tỷ đồng</p>", html)
        self.assertIsNone(html_quality_code(html))

    def test_a_label_joined_with_its_section_title_is_not_led_again(self) -> None:
        steps = ["BƯỚC 1: CHUẨN BỊ", "Mục tiêu của bước này",
                 "Thu thập đủ hồ sơ của khách hàng trước buổi gặp đầu tiên.", "Công cụ cần dùng",
                 "Bảng kiểm hồ sơ và mẫu email xác nhận lịch hẹn.", "BƯỚC 2: THỰC HIỆN",
                 "Mục tiêu của bước này", "Trình bày giải pháp phù hợp với đúng nhu cầu đã ghi nhận.",
                 "Công cụ cần dùng", "Bộ slide giới thiệu và bảng so sánh phương án."]
        html = render_idm_source_locked_html("Quy trình", steps, locale="vi")
        self.assertIn("<h3>Bước 2: thực hiện - Mục tiêu của bước này</h3>", html)
        self.assertIn("<h3>Thực hiện – Công cụ cần dùng</h3>", html)  # noqa: RUF001 - after a sentence: led
        self.assertIsNone(html_quality_code(html))
        parts = render_idm_source_locked_html("Quy trình", [
            "PHẦN 1: TIẾP NHẬN", "Người phụ trách", "Nhân viên trực tổng đài ghi nhận thông tin của khách.",
            "PHẦN 2: XỬ LÝ", "Người phụ trách", "Trưởng nhóm phân loại và giao việc trong ngày."], locale="vi")
        self.assertIn("<h3>Phần 2: xử lý - Người phụ trách</h3>", parts)
        bare = render_idm_source_locked_html("Quy trình", [
            "BƯỚC 1", "Mục tiêu", "Thu thập đủ hồ sơ của khách hàng trước buổi gặp đầu tiên.", "BƯỚC 2", "Mục tiêu",
            "Trình bày giải pháp phù hợp với đúng nhu cầu đã ghi nhận."], locale="vi")
        self.assertIn("<h3>Bước 2 - Mục tiêu</h3>", bare)
        for html_text in (parts, bare):
            self.assertIsNone(html_quality_code(html_text))
