"""Fixes from the independent review 6 of the review 5 fixes (345663c..90ee210), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter
from types import SimpleNamespace
from typing import Any

from app.core.errors import ServiceError
from app.idm.answer_near import ANSWER_NEAR_EXAMPLE_CODE
from app.idm.fallback_lines import restore_stops
from app.idm.fidelity_checks import NEW_FACT_CODE, SENTENCE_COPY_CODE, SENTENCE_LENGTH_CODE
from app.idm.node_acceptance import html_quality_code
from app.idm.qa import ANSWER_LEAK_CODE
from app.idm.source_locked import _is_label, render_idm_source_locked_html
from app.idm.source_names import NAME_NOT_IN_SOURCE_CODE, NameGuide
from app.idm.storyboard_context import _Draft
from app.idm.storyboard_repair import IdmUnitRepairSteps
from tests.support.log_support import logged_events
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
    unavailable,
)


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


class JudgeRepairSoftTests(StoryboardEndpointTestCase):
    """M3: a soft finding never drops a judge repair (judge mode "repair", the default)."""

    async def test_a_judge_repair_is_kept_when_the_quiz_turns_soft(self) -> None:
        body = severity_body()
        writer = writer_with(body, [REPEATED])
        fixed = writer_with(body, [REPEATED, FIXED])
        repair = {"components": {"c0": {key: value for key, value in fixed["components"]["c0"].items()
                                        if key != "covered_source_fact_ids"}}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair],
                                   JUDGE: [judge(("Q7_cognitive_load", "major", 0)), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(provider.names.count(JUDGE), 2)  # the second judge ran on the kept repair
        self.assertIn(FIXED[:30], json.dumps(data["unit"]["components"][0], ensure_ascii=False))
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual(quality["remaining_codes"], [ANSWER_NEAR_EXAMPLE_CODE])
        self.assertIn("Nên biên tập khối 2 (Quiz): đáp án đúng", data["unit"]["idm_quality"]["author_note"])


class _Guard(IdmUnitRepairSteps):
    """The real ``guard_repair`` over stubbed slot findings (review 6, ``h1_guard.py``)."""

    def __init__(self, findings: dict[str, dict[int, set[str]]]) -> None:
        self.findings = findings
        self.runtime = SimpleNamespace(adjustments=Counter(), correlation_id=None)  # type: ignore[assignment]
        self.contract = SimpleNamespace(unit_path="u")  # type: ignore[assignment]

    def slot_findings(self, draft: _Draft, *, advisory: bool = True) -> tuple[Any, Any, bool]:
        del advisory
        return {}, {index: frozenset(codes) for index, codes in self.findings[draft.unit["tag"]].items()}, True

    def is_worksheet(self, index: int) -> bool:
        del index
        return False


class SoftGuardTests(unittest.TestCase):
    """M4 (lead's rule): soft findings are ignored both ways; the other findings decide as before the round."""

    def decide(self, before_codes: dict[int, set[str]], after_codes: dict[int, set[str]], *, html: bool,
               question: bool) -> str:
        def draft(tag: str, changed_html: bool, changed_quiz: bool) -> _Draft:
            return _Draft({"tag": tag, "components": [{"html": "h1" if changed_html else "h0"},
                                                      {"q": "q1" if changed_quiz else "q0"}]}, [])

        before, after = draft("before", False, False), draft("after", html, question)
        kept = _Guard({"before": before_codes, "after": after_codes}).guard_repair(before, after, "w5")
        return "kept" if kept is after else "dropped" if kept is before else "slot"

    def test_soft_codes_count_neither_way(self) -> None:
        near, name = ANSWER_NEAR_EXAMPLE_CODE, NAME_NOT_IN_SOURCE_CODE
        cases = [  # (before, after, html changed, quiz changed, decision)
            ({1: {near}}, {1: {SENTENCE_LENGTH_CODE}}, False, True, "dropped"),  # S1: no credit for a soft fix
            ({0: {name}}, {0: {SENTENCE_COPY_CODE}}, True, False, "dropped"),  # S1b
            ({0: {SENTENCE_LENGTH_CODE}}, {0: {SENTENCE_LENGTH_CODE, name}}, True, False, "kept"),  # S2: no blame
            ({0: {NEW_FACT_CODE}}, {0: {NEW_FACT_CODE, name}}, True, False, "kept"),  # S2b
            ({0: {SENTENCE_COPY_CODE}}, {1: {ANSWER_LEAK_CODE}}, True, False, "dropped"),  # S3: a hard finding
            ({0: {SENTENCE_COPY_CODE}}, {1: {near}}, True, False, "kept"),  # S4 (H1)
            ({0: {SENTENCE_COPY_CODE}}, {1: {NEW_FACT_CODE}}, True, False, "dropped"),  # S5
        ]
        for before, after, html, question, decision in cases:
            self.assertEqual(self.decide(before, after, html=html, question=question), decision, (before, after))


def timeout() -> ServiceError:
    return ServiceError(status_code=504, detail={"code": "AI_PROVIDER_TIMEOUT", "message": "private"})


class SoftRepairUsageTests(StoryboardEndpointTestCase):
    """M5 (lead's rule): a dispatched provider call that failed is unknown usage, so the backend charges the reserved
    ceiling (usage_source reserved_upper_bound, usage_complete false, which Node only takes as a reviewable
    structured draft); the judge still runs and the AI draft stays."""

    async def test_a_failed_soft_repair_reports_unknown_usage(self) -> None:
        for failures in ([timeout(), timeout()], [unavailable(), unavailable()]):
            body = escalate_body()
            writer = escalate_writer(body)
            writer["components"]["c0"]["question"] = (
                "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm "
                "gì trước tiên?")
            provider = FakeGenerate(**{WRITER: [writer], REPAIR: failures, JUDGE: [judge()]})
            status, data, _ = await self.post(body, provider)
            with self.subTest(failure=failures[0].detail["code"]):
                self.assertEqual(status, 200, data.get("detail"))
                self.assertEqual((data["usage_source"], data["usage_complete"]), ("reserved_upper_bound", False))
                self.assertEqual((data["content_origin"], data["quality_state"]),
                                 ("structured_fallback", "review_required"))
                self.assertEqual(provider.names.count(JUDGE), 1)  # the judge still ran
                self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
                failed = [event for event in data["attempt_trace"] if event["outcome"] == "failed"]
                self.assertTrue(failed and all(event["provider_dispatched"] for event in failed))

    async def test_an_undispatched_soft_repair_reports_the_provider_usage(self) -> None:
        # A repair the time left does not allow is never sent: nothing unknown, the unit keeps the provider lane.
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        from app.idm import storyboard  # noqa: PLC0415 - patched for this case only

        original = storyboard.repair_thinking_for
        storyboard.repair_thinking_for = lambda *args, **kwargs: None  # type: ignore[assignment]
        try:
            status, data, _ = await self.post(body, provider)
        finally:
            storyboard.repair_thinking_for = original
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(provider.names.count(REPAIR), 0)
        self.assertEqual((data["usage_source"], data["usage_complete"]), ("provider", True))


class StopTests(unittest.TestCase):
    """L1: a glued full stop is mended between a lower-case letter and a capital with a lower-case letter, whatever
    the words' language; only a name's dot stays."""

    def test_sentences_are_mended_names_are_kept(self) -> None:
        for glued, mended in (("thị trường Việt Nam.Bạn cần", "thị trường Việt Nam. Bạn cần"),
                              ("sữa Vinamilk.Công ty mở rộng", "sữa Vinamilk. Công ty mở rộng"),
                              ("chi nhánh Hà Nội.Khách hàng", "chi nhánh Hà Nội. Khách hàng"),
                              ("chi phí tăng mạnh.Do đó, cần cắt giảm", "chi phí tăng mạnh. Do đó, cần cắt giảm"),
                              ("rõ ràng.Ta cần", "rõ ràng. Ta cần"),
                              ("the customer journey.It starts", "the customer journey. It starts"),
                              ("đối thủ.On the other hand", "đối thủ. On the other hand")):
            self.assertEqual(restore_stops(glued), mended)
        for name in ("Node.Js là", "System.Console", "TP.HCM", "Tp.Hồ Chí Minh", "Mr.Smith", "St.Petersburg",
                     "Ths.Nguyễn Văn A", "file README.Md", "v.v.Các bước", "trang www.abc.Com", "giá 3.5.Tiếp"):
            self.assertEqual(restore_stops(name), name)


class ToolNameTests(unittest.TestCase):
    """L2: a one-word English tool name inside a sentence is found again; a Title Case heading stays quiet."""

    def test_one_word_names_mid_sentence(self) -> None:
        source = NameGuide.build(["Hành trình trải nghiệm khách hàng gồm 5 giai đoạn.",
                                  "Điểm chạm là nơi khách hàng tương tác với doanh nghiệp."])
        for text, found in (("Hãy dùng Eisenhower Matrix để xếp ưu tiên.", ["Eisenhower Matrix"]),
                            ("Dùng Eisenhower Matrix để xếp ưu tiên các điểm chạm.", ["Eisenhower Matrix"]),
                            ("Bạn nên vẽ Empathy Map cho từng nhóm.", ["Empathy Map"]),
                            ("Use the Eisenhower Matrix to rank touchpoints.", ["Eisenhower Matrix"]),
                            ("Building Your Prioritization Matrix", []), ("Our Decision Framework", []),
                            ("A Practical Framework for Touchpoints", []), ("Mapping Each Stage on a Simple Map", [])):
            self.assertEqual(source.missing(text), found, text)
