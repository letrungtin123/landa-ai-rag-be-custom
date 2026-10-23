"""Fixes from the independent review 7 of the review 6 fixes (90ee210..d74c42f), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

from app.idm.node_acceptance import html_quality_code
from app.idm.source_locked import render_idm_source_locked_html
from tests.unit.idm.test_idm_qc_run_2f8781d6 import many_facts_body
from tests.unit.idm.test_idm_storyboard import (
    StoryboardEndpointTestCase,
)

# Review 7 (M1): a figure ends each section, so the next section's title and its repeated label are no pair the
# renderer joins into one heading.
FIGURE_FACTS = ["HỒ SƠ KHÁCH HÀNG (VD1: KẾ TOÁN TRƯỞNG)", "Mục tiêu công việc",
                "Đóng sổ sách đúng hạn vào cuối mỗi tháng.", "85%", "HỒ SƠ KHÁCH HÀNG (VD2: GIÁM ĐỐC TÀI CHÍNH)",
                "Mục tiêu công việc", "Phê duyệt khoản đầu tư có hiệu quả rõ ràng.", "72%",
                "HỒ SƠ KHÁCH HÀNG (VD3: NHÂN VIÊN KHO)", "Mục tiêu công việc",
                "Xuất kho đúng hạn và đúng số lượng mỗi ngày."]


class FallbackDuplicateTests(StoryboardEndpointTestCase):
    """M1: a repeated label is left without its section name only when the renderer really joins it with the title;
    the fallback never repeats a paragraph and fallback_only answers as it did at 345663c."""

    def test_a_figure_before_the_title_keeps_the_name(self) -> None:
        html = render_idm_source_locked_html("Hồ sơ", FIGURE_FACTS, locale="vi")
        self.assertIsNone(html_quality_code(html))
        self.assertIn("Giám đốc tài chính – Mục tiêu công việc", html)  # noqa: RUF001 - the en dash
        self.assertIn("Nhân viên kho – Mục tiêu công việc", html)  # noqa: RUF001 - the en dash
        # A step title the plan renders as a list keeps its name on the label (it was "<h3>Mục tiêu …</h3>" alone).
        steps = ["BƯỚC 1: CHUẨN BỊ", "Mục tiêu của bước này", "Thu thập đủ hồ sơ của khách hàng trước buổi gặp.",
                 "BƯỚC 2: THỰC HIỆN", "Mục tiêu của bước này", "Trình bày giải pháp phù hợp với nhu cầu."]
        listed = render_idm_source_locked_html("Quy trình", steps, locale="vi",
                                               required_artifacts=[{"type": "ordered_list"}])
        self.assertIn("Thực hiện – Mục tiêu của bước này", listed)  # noqa: RUF001 - the en dash
        self.assertIsNone(html_quality_code(listed))

    async def test_fallback_only_answers(self) -> None:
        body = {**many_facts_body(FIGURE_FACTS), "fallback_only": True}
        status, data, provider = await self.post(body)
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assertEqual((data["content_origin"], data["quality_state"]), ("structured_fallback", "review_required"))
