"""Fixes from the independent review 7 of the review 6 fixes (90ee210..d74c42f), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import json
import unittest

from app.core.errors import AppError, ServiceError
from app.idm.fallback_lines import restore_stops
from app.idm.node_acceptance import html_quality_code
from app.idm.slot_words import exercise_details
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.source_names import NameGuide
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_qc_run_2f8781d6 import many_facts_body
from tests.unit.idm.test_idm_review5_2f8781d6 import FIXED, REPEATED, writer_with
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
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


class DiscardedJudgeRepairTests(StoryboardEndpointTestCase):
    """L1: the codes of a judge repair the second judge does not keep are never reported as fixed."""

    async def test_no_fix_line_for_a_dropped_repair(self) -> None:
        body = severity_body()
        writer = writer_with(body, [REPEATED])
        fixed = writer_with(body, [REPEATED, FIXED])
        repair = {"components": {"c0": {key: value for key, value in fixed["components"]["c0"].items()
                                        if key != "covered_source_fact_ids"}}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [
            judge(("Q7_cognitive_load", "major", 0)), judge(("Q7_cognitive_load", "major", 0))]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200, data.get("detail"))
        self.assertNotIn(FIXED[:30], json.dumps(data["unit"]["components"][0], ensure_ascii=False))
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual((quality["codes"], quality["fixed_codes"]), ([], []))
        self.assertNotIn("Đã tự sửa", data["unit"]["idm_quality"]["author_note"])


class NamedExerciseTests(unittest.TestCase):
    """L2: "Thực hành: …" and "Thảo luận: …" open an exercise again."""

    def test_a_bare_exercise_header(self) -> None:
        self.assertEqual(exercise_details(["Thực hành: chia nhóm 4 người, thảo luận trong 15 phút về một khách hàng."]),
                         ["15 phút"])
        self.assertEqual(exercise_details(["Thảo luận: Các nhóm làm việc trong 20 phút, trình bày theo nhóm."]),
                         ["20 phút", "theo nhóm"])


class LowerCaseStopTests(unittest.TestCase):
    """L3: only the title-case dotted abbreviations of a name keep their dot ("Tp.", "Mr.", "Ths.")."""

    def test_lower_case_words_get_their_space(self) -> None:
        for glued, mended in (("thời gian phản hồi 200 ms.Sau đó", "thời gian phản hồi 200 ms. Sau đó"),
                              ("say no.The manager agreed", "say no. The manager agreed"),
                              ("vd.Khách hàng mua 2 sản phẩm", "vd. Khách hàng mua 2 sản phẩm")):
            self.assertEqual(restore_stops(glued), mended)
        for name in ("Tp.Hồ Chí Minh", "Mr.Smith", "Ths.Nguyễn Văn A", "St.Petersburg", "Node.Js là"):
            self.assertEqual(restore_stops(name), name)


class TranslatedToolNameTests(unittest.TestCase):
    """L5: an English tool name the source writes with its Vietnamese tool noun ("Ma trận SWOT") is the source's; a
    one-word name at the start of a clause is caught."""

    def test_names(self) -> None:
        source = NameGuide.build(["Ma trận SWOT giúp đánh giá điểm mạnh và điểm yếu.", "Bản đồ thấu cảm gồm 4 ô."])
        for text, found in (("Áp dụng SWOT Matrix cho sản phẩm mới", []), ("Phân tích SWOT Matrix", []),
                            ("Empathy Map giúp gì cho bạn?", ["Empathy Map"]),
                            ("Eisenhower Matrix: xếp ưu tiên công việc", ["Eisenhower Matrix"]),
                            ("Building Your Prioritization Matrix", [])):
            self.assertEqual(source.missing(text), found, text)
