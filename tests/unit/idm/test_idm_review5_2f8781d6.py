"""Fixes from the independent review 5 of the run 2f8781d6 changes (7f67137..345663c).

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from app.idm.acronyms import explained
from app.idm.answer_near import ANSWER_NEAR_EXAMPLE_CODE, near_example
from app.idm.fallback_lines import restore_stops
from app.idm.node_acceptance import html_quality_code
from app.idm.quiz_letters import expand_short_item_labels
from app.idm.slot_words import classroom_words, exercise_details
from app.idm.sortable import without_order_labels
from app.idm.source_fallback import idm_statement_faq
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.source_names import NameGuide
from app.idm.visual_pages import evidence_needs_check, unread_image_pages
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_qc_run_5496155c import region
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
                "chính. Với khiếu nại cấp 2 này, bạn nên làm gì trước tiên?",
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
        # The repaired html now holds the key's words and the case names it: the unchanged quiz turns soft.
        self.assertEqual(quality["remaining_codes"], [ANSWER_NEAR_EXAMPLE_CODE])
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


class SortableNumberTests(unittest.TestCase):
    """M4: only a pure ordinal prefix of one form with numbers one by one goes; a number that is content stays."""

    @staticmethod
    def texts(items: list[str]) -> tuple[list[str], int]:
        component, count = without_order_labels({"question_text": "Sắp xếp", "items": [{"text": t} for t in items]})
        return [item["text"] for item in component["items"]], count

    def test_numbers_that_are_content_stay(self) -> None:
        for items in (["Mốc 7 ngày: gọi điện hỏi thăm khách hàng mới", "Mốc 30 ngày: gửi khảo sát mức độ hài lòng",
                       "Mốc 90 ngày: đề xuất gói gia hạn hợp đồng"],
                      ["Tier 1 agents log the ticket and greet the caller", "Tier 2 engineers reproduce the fault",
                       "Tier 3 vendors ship the firmware fix"],
                      ["Nhân viên tiếp nhận yêu cầu hoàn tiền của khách", "Cấp 2 phê duyệt khoản hoàn tiền",
                       "Kế toán chuyển tiền và báo lại cho khách"],
                      ["Phần 1 của buổi gặp: hỏi về mục tiêu kinh doanh",
                       "Phần 2 của buổi gặp: trình bày giải pháp phù hợp",
                       "Phần 3 của buổi gặp: thống nhất bước tiếp theo"],
                      ["Bước 1: xác định nhóm khách hàng mục tiêu", "Bước 3: phân tích hành trình mua hàng",
                       "Bước 4: thiết kế điểm chạm tạo khác biệt"]):  # 2 is missing: not one by one
            self.assertEqual(self.texts(items), (items, 0))

    def test_a_pure_ordinal_prefix_goes(self) -> None:
        self.assertEqual(self.texts(["Tầng 1: Cột mốc hành trình theo thời gian", "Tầng 2: Tâm thế và kỳ vọng khách",
                                     "Tầng 3: Điểm đau trọng yếu từng giai đoạn"]),
                         (["Cột mốc hành trình theo thời gian", "Tâm thế và kỳ vọng khách",
                           "Điểm đau trọng yếu từng giai đoạn"], 3))
        # Source steps numbered from 9 still give their order away.
        self.assertEqual(self.texts([f"Bước {n}: Thực hiện việc số {n} của quy trình" for n in (10, 9, 11)])[1], 3)


class ImagePageCoverTests(unittest.TestCase):
    """M5: only a certain section cover loses the image-page warning; chart labels or a plain title keep it."""

    @staticmethod
    def regions(*pages: tuple[int, float]) -> list[dict[str, Any]]:
        return [{"visual_regions": [region(page, share, asset=f"{page:02d}a" * 21)]} for page, share in pages]

    def test_only_a_section_cover_is_left_out(self) -> None:
        texts = {12: ["Row 1; MODULE 1: THẤU HIỂU HIỆN TRẠNG",  # the run's fact form of its cover page
                      "Góc nhìn Outside – In: Khách hàng thực sự đang trải qua điều gì?"],  # noqa: RUF001
                 45: ["Row 1: | BÀI TẬP VỀ NHÀ"],
                 30: ["OPEX", "CAPEX", "ROI"],  # a chart whose only text is its labels
                 31: ["BẢN ĐỒ HÀNH TRÌNH KHÁCH HÀNG"],  # a plain title over a journey-map picture
                 32: ["Module 2: chiến lược và chiến thuật"],  # not a title in capitals: not certain
                 33: ["MODULE 3: TRIỂN KHAI", "Lộ trình ba tháng", "Các mốc kiểm tra"]}  # two lines under it
        regions = self.regions((12, 0.95), (45, 0.95), (30, 0.92), (31, 0.88), (32, 0.9), (33, 0.9))
        self.assertEqual(unread_image_pages(regions, texts), [30, 31, 32, 33])
        self.assertTrue(evidence_needs_check(regions[2:4], (), texts))
        self.assertFalse(evidence_needs_check(regions[:2], (), texts))


def html_slot(*sections: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"version": 2, "sections": [
        {"heading": heading, "blocks": blocks} for heading, blocks in sections]}}


def block(kind: str, text: str | None = None, rows: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": [],
            "rows": [{"label": label, "value": value} for label, value in rows]}


def quiz(question: str, options: list[str], key: int = 0) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question, "explanation": "x",
            "choices": [{"text": text, "correct": index == key} for index, text in enumerate(options)]}


class RuleApplicationTests(unittest.TestCase):
    """M7: a new case that applies a taught rule is no give-away, whether the key copies or rewords the rule."""

    RULE = html_slot(("Khi nào chuyển khiếu nại lên cấp trên?", [
        block("paragraph", "Khiếu nại liên quan đến an toàn sản phẩm phải được chuyển ngay cho quản lý cấp cao xử lý "
                           "trong vòng hai giờ."),
        block("paragraph", "Khiếu nại về giao hàng chậm do nhân viên tuyến đầu tự xử lý và gọi lại cho khách trong "
                           "ngày.")]))
    DISTRACTORS = ("Hứa đổi máy mới cho khách vào cuối tuần này", "Gửi thư xin lỗi theo mẫu chuẩn của công ty",
                   "Hẹn khách mang máy đến cửa hàng để kiểm tra")
    CASE = "Giả sử bạn nhận cuộc gọi của khách nói máy sấy vừa mua bốc khói khi dùng lần đầu. Bạn làm gì trước?"

    def test_applying_a_rule_to_a_new_case_is_quiet(self) -> None:
        for key in ("Chuyển ngay cho quản lý cấp cao xử lý trong vòng hai giờ",
                    "Báo ngay cho quản lý cấp cao để xử lý trong hai giờ"):
            self.assertIsNone(near_example(quiz(self.CASE, [key, *self.DISTRACTORS]), [self.RULE]), key)
        english = html_slot(("When do you escalate?", [block("table", rows=(
            ("Safety complaint", "Escalate to a senior manager within two hours."),
            ("Late delivery", "The front-line agent resolves it and calls the customer back the same day.")))]))
        question = quiz("Suppose a customer says the new heater smoked on first use. What do you do first?",
                        ["Escalate to a senior manager within two hours",
                         "Promise a replacement heater by the end of the week",
                         "Send the standard apology email template",
                         "Ask the customer to bring the heater to the store"])
        self.assertIsNone(near_example(question, [english]))

    def test_a_case_that_names_the_condition_still_counts(self) -> None:
        named = quiz("Khiếu nại liên quan đến an toàn sản phẩm thì bạn làm gì trước?",
                     ["Chuyển ngay cho quản lý cấp cao xử lý trong vòng hai giờ", *self.DISTRACTORS])
        self.assertEqual(near_example(named, [self.RULE]), "option")


class ClassroomWordTests(unittest.TestCase):
    """M8: only a fact that opens as an exercise gives details, and its homework or group wording counts only after an
    exercise word."""

    def test_a_mention_is_no_exercise(self) -> None:
        details = exercise_details([
            "Sở hữu các tính năng tự động hóa \"1 chạm\", giúp họ cắt giảm được 1-2 tiếng làm việc bù đầu mỗi ngày để "
            "được về nhà đúng giờ.",
            "Trưởng nhóm thảo luận với khách hàng trong 30 phút đầu để chốt phạm vi dự án."])
        self.assertEqual(details, [])
        teaching = html_slot(("Người dùng muốn gì?", [block("paragraph", "Họ muốn về nhà đúng giờ. Trưởng nhóm cần 30 "
                                                                         "phút đầu để chốt phạm vi.")]))
        self.assertEqual(classroom_words([teaching], (), details), {})

    def test_group_wording_counts_only_in_an_exercise(self) -> None:
        details = exercise_details(["Bài tập về nhà theo nhóm: hoàn thiện bản đồ hành trình cho dự án của bạn.",
                                    "Thực hành Bộ lọc: các nhóm chỉ chọn ra đúng 2 sáng kiến trong 10 phút."])
        self.assertEqual(details, ["về nhà", "theo nhóm", "đúng 2", "10 phút"])
        ordinary = html_slot(("Phân khúc", [block("paragraph", "Phân khúc là gom khách hàng theo nhóm có đặc điểm "
                                                               "chung, để về nhà đúng giờ.")]))
        exercise = html_slot(("Đồ án", [block("paragraph", "Bài tập về nhà theo nhóm yêu cầu đội ngũ hoàn thiện đồ án; "
                                                          "chọn đúng 2 sáng kiến.")]))
        self.assertEqual(classroom_words([ordinary, exercise], (), details), {1: ["về nhà", "theo nhóm", "đúng 2"]})


class ToolNameTests(unittest.TestCase):
    """M9: a tool noun as a verb and the generic words of an English Title Case heading name no tool."""

    def test_the_reviewer_s_headings_are_quiet(self) -> None:
        vietnamese = NameGuide.build(["Hành trình trải nghiệm khách hàng gồm 5 giai đoạn: nhận biết, cân nhắc, mua.",
                                      "Điểm chạm là nơi khách hàng tương tác với doanh nghiệp."])
        for title in ("Sơ đồ hóa hành trình khách hàng", "Sơ đồ hoá điểm chạm của khách hàng",
                      "Bản đồ hoá các điểm chạm", "Ma trận hóa ưu tiên"):
            self.assertEqual(vietnamese.missing(title), [], title)
        english = NameGuide.build(["A customer journey has five stages: awareness, consideration and purchase.",
                                   "Prioritise the touchpoints that matter most to the customer."])
        for heading in ("A Practical Framework for Touchpoints", "Building Your Prioritization Matrix",
                        "Mapping Each Stage on a Simple Map", "Our Decision Framework", "Use the Framework"):
            self.assertEqual(english.missing(heading), [], heading)

    def test_a_name_the_source_does_not_use_is_still_found(self) -> None:
        source = NameGuide.build(["Bản đồThấu cảm (Empathy Map) giúp hiểu điều khách hàng nghĩ và cảm nhận."])
        self.assertEqual(source.missing("Khung ma trận thấu cảm và 3 tầng nhận thức"), ["ma trận thấu cảm"])
        self.assertEqual(source.missing("The Value Proposition Canvas"), ["Value Proposition Canvas"])
        self.assertEqual(source.missing("Building the Customer Journey Map"), ["Customer Journey Map"])


class DashedAbbreviationTests(unittest.TestCase):
    """L1: "<words> - ABBR" explains ABBR only when the words spell it."""

    def test_only_spelled_words_explain(self) -> None:
        for text, token in (("khoảnh khắc sự thật (Moments of Truth - MOT) quyết định", "MOT"),
                            ("Net Promoter Score - NPS đo lòng trung thành", "NPS")):
            self.assertTrue(explained(token, None, [text]), text)
        for text, token in (("Ba chỉ số cần theo dõi hằng tuần - NPS, CSAT và CES.", "NPS"),
                            ("Doanh thu quý này tăng mạnh - KPI vượt kế hoạch.", "KPI")):
            self.assertFalse(explained(token, None, [text]), text)
