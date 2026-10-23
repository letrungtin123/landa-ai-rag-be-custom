"""Fixes from the independent review 7 of the review 6 fixes (90ee210..d74c42f), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

from app.core.errors import AppError, ServiceError
from app.idm.node_acceptance import html_quality_code
from app.idm.source_locked import render_idm_source_locked_html
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_qc_run_2f8781d6 import many_facts_body
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
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


def busy() -> AppError:
    return AppError("SERVICE_BUSY", 503, "The AI service is busy. Please retry shortly.")


class UndispatchedRepairTests(StoryboardEndpointTestCase):
    """M2: a soft-only repair this service's own limiter refuses (SERVICE_BUSY, never sent) leaves the provider usage
    of the calls that were sent and the unit validated; the soft finding stays an edit note."""

    async def test_a_busy_soft_repair_keeps_the_provider_lane(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [busy()], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual((data["content_origin"], data["quality_state"], data["usage_source"], data["usage_complete"]),
                         ("provider_validated", "validated", "provider", True))
        self.assertEqual(provider.names.count(JUDGE), 1)
        self.assertEqual(logged_events(logs.records, "idm_unit_quality")[-1]["failure_codes"], [])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 1 (Quiz): đáp án đúng", note)
        self.assertNotIn("Cần tác giả xem lại", note)

    async def test_a_sent_attempt_before_the_refusal_stays_unknown(self) -> None:
        # A timed-out attempt was sent, then its retry was refused: the first call's usage is unknown (review 6, M5).
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        timeout = ServiceError(status_code=504, detail={"code": "AI_PROVIDER_TIMEOUT", "message": "private"})
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [timeout, busy()], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual((data["usage_source"], data["usage_complete"], data["quality_state"]),
                         ("reserved_upper_bound", False, "review_required"))
