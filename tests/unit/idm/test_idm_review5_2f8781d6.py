"""Fixes from the independent review 5 of the run 2f8781d6 changes (7f67137..345663c).

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from app.idm.fallback_lines import restore_stops
from app.idm.node_acceptance import html_quality_code
from app.idm.quiz_letters import expand_short_item_labels
from app.idm.source_fallback import idm_statement_faq
from app.idm.source_locked import render_idm_source_locked_html
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
    rehashed_brief,
    severity_body,
    severity_writer,
    unavailable,
)

# H1: a quiz whose key the html repair below happens to restate ("hẹn gọi lại cho khách").
KEY = "Ghi nhận phản hồi và hẹn gọi lại cho khách trong ngày"
QUIZ = {
    "question": "Giả sử một khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ, chưa có thiệt hại tài "
                "chính. Bạn nên làm gì trước tiên?",
    "choices": [{"text": "Chuyển hồ sơ sang bộ phận Pháp chế ngay lập tức", "correct": False},
                {"text": KEY, "correct": True},
                {"text": "Đề nghị hoàn tiền toàn bộ giá trị của đơn hàng", "correct": False}],
    "explanation": "Nguyên tắc: khiếu nại cấp 2 cần ghi nhận và thông báo trưởng nhóm.\nA - Sai vì không có yếu tố "
                   "pháp lý.\nB - Đúng vì khách phàn nàn lần thứ hai là cấp 2.\nC - Sai vì chưa có thiệt hại."}
REPEATED = "Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định ai xử lý."
FIXED = "Với khiếu nại cấp 2, nhân viên nên hẹn gọi lại cho khách và báo trưởng nhóm."


def writer_with(body: dict[str, Any], paragraphs: list[str]) -> dict[str, Any]:
    writer = severity_writer(body)
    blocks = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"]
    blocks[:] = [{"kind": "paragraph", "text": text, "items": [], "rows": []} for text in paragraphs] + [blocks[1]]
    writer["components"]["c1"].update(QUIZ)
    return writer


class SoftFindingRepairTests(StoryboardEndpointTestCase):
    """H1 and M6: a soft finding never makes a repair worse, never vetoes it, and a failed call that only repairs soft
    findings leaves the unit as it was."""

    async def test_an_html_repair_is_kept_when_the_unchanged_quiz_turns_soft(self) -> None:
        body = severity_body()
        writer = writer_with(body, [REPEATED, REPEATED])  # the same paragraph twice: HTML_DUPLICATE_BLOCK
        fixed = writer_with(body, [REPEATED, FIXED])
        repair = {"components": {"c0": {key: value for key, value in fixed["components"]["c0"].items()
                                        if key != "covered_source_fact_ids"}}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        # Old code (review 5): the guard dropped the whole repair and the html slot fell back.
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        self.assertEqual(logged_events(logs.records, "idm_unit_repair_guard"), [])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual(quality["fallback_slots"], [])
        html = json.dumps(data["unit"]["components"][0], ensure_ascii=False)
        self.assertIn("hẹn gọi lại cho khách và báo trưởng nhóm", html)
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertIn("Đã tự sửa: phần lý thuyết lặp lại cùng một đoạn", data["unit"]["idm_quality"]["author_note"])

    async def test_a_failed_soft_repair_leaves_the_unit_as_it_was(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable(), unavailable()], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        # Old code: structured_fallback / review_required / reserved_upper_bound, and the judge never ran.
        self.assertEqual((data["content_origin"], data["quality_state"], data["usage_source"]),
                         ("provider_validated", "validated", "provider"))
        self.assertEqual(provider.names.count(JUDGE), 1)
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual(quality["failure_codes"], [])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 1 (Quiz): đáp án đúng", note)
        self.assertNotIn("Sửa tự động không thành công", note)
        self.assertNotIn("Cần tác giả xem lại", note)
        self.assertNotIn("không chạy được", note)

    async def test_a_failed_hard_repair_still_marks_the_unit(self) -> None:
        # Unchanged for any other finding: the failure is the unit's (here a leak, repaired at targeted level).
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Có"
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable(), unavailable()], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(data["usage_source"], "reserved_upper_bound")


class ShortLabelTests(unittest.TestCase):
    """M1: only the stem's item labels and the options that are only such a name are written out."""

    def test_words_inside_sentences_stay(self) -> None:
        items = "\n".join(f"- Ý {number}: {text}" for number, text in enumerate(
            ("Gọi lại cho khách trong ngày.", "Gửi thư xin lỗi theo mẫu chuẩn.", "Chuyển ngay cho quản lý cấp cao.",
             "Hẹn khách mang máy đến cửa hàng."), 1))
        quiz = {"problem_type": "multiple_choice", "title": "Chú ý 3 điểm khi xử lý",
                "question": f"Giả sử khách báo máy sấy bốc khói. Bạn cân nhắc bốn ý (gợi ý 2 cách):\n{items}\n"
                            "Ý nào đúng?",
                "choices": [{"text": f"Ý {number}", "correct": number == 3} for number in (3, 1, 2, 4)],
                "hints": ["Đồng ý 1 lần là đủ."],
                "explanation": "Nguyên tắc: cần lưu ý 2 điều: an toàn trước, rồi mới tới bồi thường.\nA - Đúng vì Ý 3 "
                               "chuyển ngay.\nB - Sai vì Ý 1 chậm.\nC - Sai vì Ý 2 chưa xử lý.\nD - Sai vì Ý 4 chậm."}
        expanded, count = expand_short_item_labels(quiz)
        self.assertEqual(count, 8)
        self.assertEqual([choice["text"] for choice in expanded["choices"]],
                         ["Ý kiến 3", "Ý kiến 1", "Ý kiến 2", "Ý kiến 4"])
        self.assertIn("\n- Ý kiến 3: Chuyển ngay cho quản lý cấp cao.", expanded["question"])
        self.assertIn("(gợi ý 2 cách)", expanded["question"])
        self.assertIn("\nÝ nào đúng?", expanded["question"])
        for key in ("title", "explanation", "hints"):
            self.assertEqual(expanded[key], quiz[key], key)
        self.assertTrue(quiz["choices"][0]["text"] == "Ý 3")  # the input is never changed


SAME_LABEL = ["Lưu ý quan trọng: Khách hàng VIP luôn được ưu tiên xử lý trong ngày làm việc đầu tiên.",
              "Lưu ý quan trọng: Nhân viên không được hứa hoàn tiền trước khi phân loại xong khiếu nại."]


class StatementFaqTests(StoryboardEndpointTestCase):
    """M2: the last-resort FAQ asks by place, as before; never fewer items than before, never a 422 for it."""

    def test_two_statements_of_one_label_give_two_items(self) -> None:
        items = idm_statement_faq("Hỏi đáp: Xử lý khiếu nại", SAME_LABEL, locale="vi")
        self.assertEqual(items, [
            {"question": "Người học cần ghi nhớ gì về 'Xử lý khiếu nại'?", "answer": SAME_LABEL[0]},
            {"question": "Điểm thứ 2 cần lưu ý trong 'Xử lý khiếu nại' là gì?", "answer": SAME_LABEL[1]}])

    async def test_fallback_only_still_answers(self) -> None:
        body = escalate_body()
        contract = body["unit_contract"]
        for number, fact in enumerate(contract["source_facts"]):
            fact["fact_text"] = SAME_LABEL[number % 2] if number < 2 else f"BÀI TẬP {number}"
        contract["idm_unit_brief"]["lesson_context_facts"] = []
        body = rehashed_brief(body)
        status, data, _ = await self.post({**body, "fallback_only": True})
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(data["content_origin"], "structured_fallback")
        self.assertEqual([item["answer"] for item in data["unit"]["components"][1]["items"]], SAME_LABEL)


STEP_FACTS = ["BƯỚC 1: CHUẨN BỊ", "Mục tiêu của bước này", "Thu thập đủ hồ sơ của khách hàng trước buổi gặp đầu tiên.",
              "Công cụ cần dùng", "Bảng kiểm hồ sơ và mẫu email xác nhận lịch hẹn.", "BƯỚC 2: THỰC HIỆN",
              "Mục tiêu của bước này", "Trình bày giải pháp phù hợp với đúng nhu cầu đã ghi nhận.", "Công cụ cần dùng",
              "Bộ slide giới thiệu và bảng so sánh phương án."]


class FallbackLabelTests(unittest.TestCase):
    """M3, L2, L3, L5: the source-locked html keeps a step's words, never makes up a label, gives the labels of one
    section one form and leaves the dots of names alone."""

    def test_a_step_section_keeps_its_words_and_no_label_is_a_step(self) -> None:
        for required in ((), ({"type": "ordered_list"},)):
            html = render_idm_source_locked_html("Quy trình bán hàng", STEP_FACTS, locale="vi",
                                                 required_artifacts=required)
            self.assertIsNone(html_quality_code(html))
            self.assertEqual(html.count("Mục tiêu của bước này"), 2, required)
            self.assertEqual(html.count("Công cụ cần dùng"), 2, required)
            self.assertNotRegex(html, r"<li>[^<]*(?:Mục tiêu của bước này|Công cụ cần dùng)", required)
            self.assertIn("Thực hiện – Công cụ cần dùng", html)  # noqa: RUF001 - the en dash
        plain = render_idm_source_locked_html("Quy trình bán hàng", STEP_FACTS, locale="vi")
        self.assertNotIn("<ol>", plain)
        self.assertIn("Bước 2: thực hiện", plain)

    def test_a_cut_label_is_never_made_up(self) -> None:
        html = render_idm_source_locked_html("Hồ sơ", [
            "HỒ SƠ (VD1: KHO HÀNG)", "- Về Ý nghĩa / Xã hội (Social/Meaningful",
            "Job): * Được công nhận là nhân viên lành nghề trong phòng ban.",
            "Indicator): * Tỷ lệ đơn xuất kho đúng hạn mỗi ngày của cả ca."], locale="vi")
        self.assertNotIn("Meaningful Indicator", html)
        self.assertIn("<p>Tỷ lệ đơn xuất kho đúng hạn mỗi ngày của cả ca.</p>", html)

    def test_the_labels_of_a_section_have_one_form(self) -> None:
        social = "Về Ý nghĩa / Xã hội (Social/Meaningful Job)"
        html = render_idm_source_locked_html("Hồ sơ", [
            "1.3 HỒ SƠ (VD1: NGƯỜI GÁC CỔNG)", f"Row 2: {social}: * Được sếp tin tưởng giao việc lớn hơn.",
            "Về Cảm xúc (Emotional Job): * Thấy an tâm khi lịch của sếp được giữ gọn.",
            "1.3 HỒ SƠ (VD3: NGƯỜI SỬ DỤNG TRỰC TIẾP)", "Về Cảm xúc (Emotional Job)",
            "Muốn làm việc thoải mái với công cụ mới mỗi ngày.", social,
            "Được công nhận là nhân viên lành nghề trong phòng ban."], locale="vi")
        # The 14-word prefixed label is a heading as its 9-word sibling is (it was a paragraph).
        self.assertIn("Người sử dụng trực tiếp – Về Cảm xúc (Emotional Job)</h3>", html)  # noqa: RUF001
        self.assertIn(f"<h3>Người sử dụng trực tiếp – {social}</h3>", html)  # noqa: RUF001 - the en dash
        self.assertIsNone(html_quality_code(html))

    def test_a_name_keeps_its_dot(self) -> None:
        for text in ("Node.Js là nền tảng", "System.Console ghi lỗi", "Văn phòng TP.HCM mở cửa", "giá 3.5.Tiếp",
                     "xem v.v.Các bước", "trang www.abc.Com"):
            self.assertEqual(restore_stops(text), text)
        self.assertEqual(restore_stops("bộ phận nội bộ.Phân loại"), "bộ phận nội bộ. Phân loại")
        self.assertEqual(restore_stops("lõi.Bị các sales"), "lõi. Bị các sales")
        self.assertEqual(restore_stops("Lưu ý.Đây là bước đầu"), "Lưu ý. Đây là bước đầu")
