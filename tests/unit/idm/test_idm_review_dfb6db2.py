"""Fixes from the independent review of db022e7..dfb6db2 (2026-10-22) and its replay of the paid runs r4/r5.

Every case is synthetic text with the shape the review found (no course text of the runs); every provider is a local
fake. Each test failed before its fix.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter
from typing import Any

from app.idm.answer_checks import copies_example_measure, stem_copies_answer, unit_teaching
from app.idm.hypothetical import SUPPOSITION_ADDED, as_supposition, suppose_semantic
from app.idm.prompts_unit import unit_writer_prompt
from app.idm.qa import (
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_KEPT_SLOTS,
    JudgeOutcome,
    SlotFinding,
    build_unit_author_note,
)
from app.idm.quiz_text import lay_out_quiz
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.text import evidence_index
from app.idm.worksheet import compact_worksheet, plan_worksheet_html
from app.idm.worksheet_input import is_example_section, is_input_section, missing_input, reused_entries
from tests.unit.idm.test_idm_qc_feedback import SUGGESTED
from tests.unit.idm.test_idm_qc_run_5496155c import facts_of
from tests.unit.idm.test_idm_storyboard import StoryboardEndpointTestCase, rehashed_brief, severity_body, worksheet_body


def flat(text: str) -> str:
    return " ".join(text.split())


def section(heading: str, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"heading": heading, "learning_block_ids": [],
            "blocks": [{"text": None, "items": [], "rows": [], **block} for block in blocks]}


def html(*sections: dict[str, Any]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": list(sections)}}


def quiz(question: str, choices: list[tuple[str, bool]], explanation: str = "") -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question,
            "choices": [{"text": text, "correct": correct} for text, correct in choices], "explanation": explanation}


PRACTICE_RULES = unit_writer_prompt(
    "vi", course_title="", audience="", lesson_title="", lesson_objective="", practice_sentences=[],
    previous_title=None, next_title=None, unit_brief={}, facts=[], context_facts=[])

REQUESTS = [
    "Yêu cầu 1: Xây cổng tra cứu hợp đồng trực tuyến có xác thực cho khách hàng.",
    "Yêu cầu 2: Tùy biến toàn bộ giao diện phần mềm theo màu riêng của từng phòng ban.",
    "Yêu cầu 3: Gửi thông báo tiến độ giao hàng tự động qua tin nhắn theo thời gian thực.",
    "Yêu cầu 4: Bố trí kỹ thuật viên trực tại chỗ cả ngày lẫn đêm cho mọi câu hỏi nhỏ.",
    "Yêu cầu 5: Thiết kế lại toàn bộ quy trình ký duyệt nội bộ của khách hàng.",
]
# r5 c3.l2.u1: the worksheet's input data, then a question on the same five requests.
WORKSHEET = html(
    section("Nhiệm vụ", {"kind": "task", "text": "Đánh giá 5 yêu cầu dưới đây và chọn 2 yêu cầu nên làm trước."}),
    section("Dữ liệu đầu vào gồm những yêu cầu nào?", {"kind": "bullets", "items": REQUESTS}),
    section("Mẫu đánh giá", {"kind": "table", "rows": [
        {"label": "Yêu cầu", "value": "Yêu cầu nào đang được đánh giá?"},
        {"label": "Mức tác động", "value": "Yêu cầu tạo ra lợi ích gì cho khách hàng?"}]}),
    section("Ví dụ đánh giá một yêu cầu khác", {"kind": "table", "rows": [
        {"label": "Yêu cầu", "value": "Tự động gửi báo cáo đối soát hằng tuần qua thư điện tử."},
        {"label": "Mức tác động", "value": "Giảm 4 giờ làm việc thủ công mỗi tuần cho đối tác."}]}),
    section("Tự kiểm tra", {"kind": "bullets", "items": ["Chọn đúng 2 yêu cầu.", "Nêu lý do cho mỗi yêu cầu."]}),
)
SAME_REQUESTS = quiz(
    "Giả sử nhóm bạn nhận năm yêu cầu sau:\n" + "\n".join(f"- {item}" for item in REQUESTS)
    + "\nHai yêu cầu nào nên đưa vào danh mục trước?",
    [("Yêu cầu 1 và Yêu cầu 3", True), ("Yêu cầu 2 và Yêu cầu 4", False), ("Yêu cầu 4 và Yêu cầu 5", False),
     ("Yêu cầu 2 và Yêu cầu 5", False)])


class InputDataIsNoTeachingTests(unittest.TestCase):
    """Item 1 (HIGH): the worksheet's input data was read as taught table rows, so a question practising on the same
    requests was a "leak" (r5 c3.l2.u1: two more paid repairs and a false "Cần xem")."""

    def test_a_question_on_the_worksheets_own_data_is_no_leak(self) -> None:
        laid = lay_out_quiz(SAME_REQUESTS)[0]
        teaching = unit_teaching([WORKSHEET], [])
        self.assertEqual(teaching.rows[:1], (("Yêu cầu", "Yêu cầu nào đang được đánh giá?"),))
        self.assertFalse(any(label.startswith("Yêu cầu 1") for label, _value in teaching.rows))
        self.assertFalse(stem_copies_answer(laid, teaching, laid))

    def test_a_taught_table_row_still_counts(self) -> None:
        taught = json.loads(json.dumps(WORKSHEET, ensure_ascii=False))
        taught["semantic_content"]["sections"][1]["heading"] = "Năm loại yêu cầu thường gặp"
        laid = lay_out_quiz(SAME_REQUESTS)[0]
        self.assertTrue(stem_copies_answer(laid, unit_teaching([taught], []), laid))

    def test_the_prompt_allows_practising_on_the_worksheets_data(self) -> None:
        self.assertIn("a worksheet's input data is no taught content: the question may practise on the same entries, "
                      "the worksheet gives no answer for them", flat(PRACTICE_RULES))


class TaughtNumberTests(unittest.TestCase):
    """Item 2: a number the lesson teaches as its rule is no copy of the worked example's figure."""

    RULE = html(section("Quy định phản hồi khiếu nại",
                        {"kind": "paragraph", "text": "Mọi khiếu nại cấp 3 phải được phản hồi trong 24 giờ kể từ khi "
                                                      "tiếp nhận."},
                        {"kind": "paragraph", "text": "Ví dụ: Chị Lan nhận khiếu nại cấp 3 lúc 9 giờ sáng thứ Hai và "
                                                      "phản hồi trong 24 giờ, trước 9 giờ sáng thứ Ba."}))
    QUESTION = quiz("Giả sử bạn tiếp nhận một khiếu nại cấp 3 vào chiều thứ Tư.\nBạn phải phản hồi trong thời hạn nào?",
                    [("Trong 48 giờ kể từ khi tiếp nhận", False), ("Trong 24 giờ kể từ khi tiếp nhận", True),
                     ("Trong 72 giờ kể từ khi tiếp nhận", False), ("Trước cuối tuần làm việc", False)])

    def test_the_rules_number_is_the_answer_not_a_copy(self) -> None:
        self.assertFalse(copies_example_measure(self.QUESTION, unit_teaching([self.RULE], [])))

    def test_a_number_only_the_example_states_still_counts(self) -> None:
        example_only = json.loads(json.dumps(self.RULE, ensure_ascii=False))
        example_only["semantic_content"]["sections"][0]["blocks"][0]["text"] = "Mọi khiếu nại cấp 3 được ưu tiên."
        self.assertTrue(copies_example_measure(self.QUESTION, unit_teaching([example_only], [])))


class SectionKindTests(unittest.TestCase):
    """Items 7 and 12: an input heading wins over the example words; a template that collects data is no input."""

    DATA = section("Dữ liệu minh họa", {"kind": "bullets", "items": REQUESTS})

    def test_input_data_named_with_an_example_word_stays_input(self) -> None:
        self.assertTrue(is_input_section(self.DATA))
        self.assertFalse(is_example_section(self.DATA))
        worksheet = html(WORKSHEET["semantic_content"]["sections"][0], self.DATA,
                         *WORKSHEET["semantic_content"]["sections"][2:])
        self.assertIsNone(missing_input(worksheet))

    def test_compaction_never_cuts_the_tasks_entries(self) -> None:
        worksheet = html(WORKSHEET["semantic_content"]["sections"][0], self.DATA,
                         *WORKSHEET["semantic_content"]["sections"][2:])
        # Only a candidate without the worked example fits; the five requests stay.
        compacted = compact_worksheet(worksheet, lambda value: "Ví dụ" not in json.dumps(value, ensure_ascii=False))
        assert compacted is not None
        data = next(item for item in compacted[0]["semantic_content"]["sections"]
                    if item["heading"] == "Dữ liệu minh họa")
        self.assertEqual(data["blocks"][0]["items"], REQUESTS)

    def test_a_template_that_collects_data_is_no_input(self) -> None:
        template = section("Bảng thu thập dữ liệu khách hàng", {"kind": "table", "rows": [
            {"label": "Việc cần làm", "value": "Khách hàng đang cố hoàn thành công việc gì mỗi ngày?"},
            {"label": "Nỗi đau", "value": "Điều gì khiến khách hàng bực bội hoặc lo sợ nhất?"}]})
        self.assertFalse(is_input_section(template))
        example = section("Ví dụ mẫu", {"kind": "table", "rows": [
            {"label": "Việc cần làm", "value": "Khách hàng đang cố hoàn thành công việc đối soát hóa đơn mỗi ngày."},
            {"label": "Nỗi đau", "value": "Điều khiến khách hàng bực bội nhất là sai số liệu."}]})
        worksheet = html(section("Nhiệm vụ", {"kind": "task", "text": "Điền bảng cho khách hàng của bạn."}),
                         template, example, WORKSHEET["semantic_content"]["sections"][-1])
        self.assertEqual(reused_entries(worksheet), [])


class SuppositionScopeTests(unittest.TestCase):
    """Item 3: "Giả sử " only before a declarative scenario sentence that opens with a past time and its comma."""

    TET = evidence_index(["Doanh số tháng trước Tết thường tăng mạnh nhờ nhu cầu quà biếu doanh nghiệp."])

    def test_questions_instructions_and_the_learners_past_stay(self) -> None:
        for text in ("Trong tuần qua, bạn đã nhận bao nhiêu phản hồi từ khách hàng? Hãy ghi lại ba phản hồi.",
                     "Trong tháng qua, hãy liệt kê ba khách hàng bạn đã gặp trực tiếp.",
                     "Năm ngoái, bạn đã ký được bao nhiêu hợp đồng mới.",
                     "Trong năm qua bạn ký ba hợp đồng."):
            self.assertEqual(as_supposition(text, "vi", None), (text, 0))

    def test_a_longer_time_or_one_the_facts_use_stays(self) -> None:
        text = "Trong tháng trước Tết, doanh số thường tăng mạnh nhờ quà biếu."
        self.assertEqual(as_supposition(text, "vi", self.TET), (text, 0))
        sourced = "Tháng trước, đội bán hàng gặp mười khách."
        self.assertEqual(as_supposition(sourced, "vi", evidence_index(["Số khách tháng trước là mười."])), (sourced, 0))

    def test_only_the_units_language(self) -> None:
        english = "In the past month, the team closed two deals."
        self.assertEqual(as_supposition(english, "vi", None), (english, 0))
        self.assertEqual(as_supposition(english, "en", None),
                         ("Suppose that in the past month, the team closed two deals.", 1))

    def test_a_declarative_scenario_still_opens_with_suppose(self) -> None:
        scenario = "Trong quý vừa qua, bộ phận kinh doanh áp dụng quy trình bốn bước."
        self.assertEqual(as_supposition(scenario, "vi", None), ("Giả sử t" + scenario[1:], 1))

    def test_a_worksheets_task_stays(self) -> None:
        semantic = {"sections": [section("Nhiệm vụ", {"kind": "task", "text": "Trong tuần qua, đội của bạn nhận năm "
                                                                             "yêu cầu mới."})]}
        counts: Counter[str] = Counter()
        self.assertIs(suppose_semantic(semantic, "vi", None, counts), semantic)
        self.assertEqual(counts[SUPPOSITION_ADDED], 0)


def rendered(*facts: str) -> str:
    return render_idm_source_locked_html("Tiêu đề", list(facts), locale="vi")


class SourceLockedTextTests(StoryboardEndpointTestCase):
    """Items 4, 10, 11 and bullet-only sources: the source-locked fallback keeps the source's numbers, operators and
    signs, keeps a quotation apart from the list item or heading before it, keeps names in a capitals heading, and a
    source of bullets still gives a valid whole-unit fallback."""

    def test_figures_operators_and_signs_stay(self) -> None:
        # (A lone "2024" line stays out: the shared cleaning reads it as a page or slide number.)
        html_text = rendered("TỶ LỆ KHÁCH HÀNG QUAY LẠI", "85%", "TĂNG TRƯỞNG", "+32", "MÔ HÌNH", "4P",
                             "Khách hàng quay lại sau ba tháng nhờ dịch vụ hậu mãi tốt.")
        for figure in ("<p>85%</p>", "<p>+32</p>", "<p>4P</p>"):
            self.assertIn(figure, html_text)
        self.assertIn("Doanh thu trung bình * Số năm", rendered(
            "Giá trị vòng đời khách hàng = Doanh thu trung bình * Số năm gắn bó với doanh nghiệp."))
        self.assertIn("<p>-5%</p>", rendered("Biên lợi nhuận theo nhóm khách hàng năm nay:",
                                              "Nhóm A | -5% | +3% so với cùng kỳ"))
        # A bullet star after a label still goes.
        self.assertNotIn("*", rendered("Về Chức năng (Functional Pain): * Bị chậm tiến độ vì chờ duyệt hồ sơ."))

    def test_a_quotation_is_never_joined_to_a_list_item_or_heading(self) -> None:
        listed = rendered("Ba mong muốn của khách hàng:", "- Khách hàng muốn được ghi nhận vì kết quả làm việc tốt",
                          "“Tôi muốn sếp thấy kết quả của mình.”", "Đây là nhu cầu xã hội của khách hàng.")
        self.assertIn("<p>“Tôi muốn sếp thấy kết quả của mình.”</p>", listed)
        self.assertNotIn("làm việc tốt “Tôi", listed)
        heading = rendered("CHÂN DUNG KHÁCH HÀNG", "“Tôi cần báo giá trong ngày để kịp trình sếp.”",
                           "Đây là lời của trưởng phòng mua hàng.")
        self.assertIn("<h3>Chân dung khách hàng</h3><p>“Tôi cần báo giá trong ngày để kịp trình sếp.”</p>", heading)

    def test_a_capitals_heading_keeps_its_names(self) -> None:
        self.assertIn("<h3>Vinamilk tại Hà Nội</h3>", rendered(
            "VINAMILK TẠI HÀ NỘI", "Doanh nghiệp mở rộng kênh phân phối tại các tỉnh phía Bắc trong năm nay."))
        self.assertIn("<h3>Chi nhánh Vinamilk tại Đà Nẵng</h3>", rendered(
            "CHI NHÁNH VINAMILK TẠI ĐÀ NẴNG", "Công ty Vinamilk mở thêm kênh phân phối cho các tỉnh miền Trung."))
        self.assertIn("<h3>Chiến lược của Tân Hiệp Phát</h3>", rendered(
            "CHIẾN LƯỢC CỦA TÂN HIỆP PHÁT", "Công ty Tân Hiệp Phát mở rộng kênh bán lẻ trong năm nay."))

    async def test_a_bullet_only_source_still_falls_back_whole(self) -> None:
        body = facts_of(severity_body(), dict(zip("abcde", ("- Khách hàng mục tiêu", "- Lợi ích cốt lõi",
                                                            "- Cách làm khác biệt", "- Bằng chứng tin cậy",
                                                            "- Lời kêu gọi hành động"), strict=True)))
        status, data, provider = await self.post({**body, "fallback_only": True})
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assertIn("- Khách hàng mục tiêu.", data["unit"]["components"][1]["question"])


def plan_body(*, sentence: str, context: str, criterion: str | None = None, facts: dict[str, str] | None = None
              ) -> dict[str, Any]:
    """The worksheet unit (no form in its facts) with its practice plan changed; ``facts`` replace the first facts."""

    body = facts_of(worksheet_body(), facts or {"a": "Câu hỏi thảo luận 5 phút: Nhóm bạn gặp khiếu nại nào khó nhất?"})
    for slot in body["unit_contract"]["idm_unit_brief"]["components"]:
        if slot.get("practice"):
            slot["practice"].update(sentence=sentence, context_input=context)
            if criterion is not None:
                slot["practice"]["feedback_focus"]["criterion"] = criterion
    return rehashed_brief(body)


class PlanWorksheetTests(StoryboardEndpointTestCase):
    """Item 5: the fallback worksheet built from the practice plan."""

    async def fallback(self, body: dict[str, Any]) -> dict[str, Any]:
        status, data, provider = await self.post({**body, "fallback_only": True})
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assertEqual(data["quality_state"], "review_required")
        return data

    async def test_input_the_plan_only_describes_is_named_in_the_note(self) -> None:
        data = await self.fallback(plan_body(sentence="Đánh giá 5 khiếu nại mẫu và xếp mỗi khiếu nại vào đúng cấp độ.",
                                             context="Danh sách 5 khiếu nại mẫu của khách hàng doanh nghiệp."))
        self.assertIn("Phiếu thực hành dự phòng ở khối 1 (Lý thuyết) chỉ mô tả dữ liệu đầu vào, chưa có dữ liệu để học "
                      "viên làm bài — tác giả cần bổ sung dữ liệu.", data["unit"]["idm_quality"]["author_note"])
        listed = await self.fallback(plan_body(
            sentence="Đánh giá 3 khiếu nại mẫu và xếp mỗi khiếu nại vào đúng cấp độ.",
            context="Ba khiếu nại: hàng trầy nhẹ, giao thiếu dưới 50 triệu đồng, thiết bị phát nhiệt."))
        self.assertNotIn("chỉ mô tả dữ liệu đầu vào", listed["unit"]["idm_quality"]["author_note"])

    async def test_suggested_criteria_stay_in_the_note_of_a_plan_fallback(self) -> None:
        data = await self.fallback(plan_body(sentence="Đánh giá một khiếu nại mẫu và chọn cấp độ phù hợp.",
                                             context="Một khiếu nại về hàng giao thiếu.",
                                             criterion=SUGGESTED))
        self.assertIn("Tự kiểm tra", data["unit"]["components"][0]["html"])
        self.assertIn("tiêu chí AI gợi ý", data["unit"]["idm_quality"]["author_note"])

    def test_statements_are_capped_and_never_classroom_instructions(self) -> None:
        plan = {"sentence": "Đánh giá các yêu cầu của khách hàng.", "context_input": "Một danh sách yêu cầu."}
        lines = [f"Quy định số {number} được áp dụng cho mọi yêu cầu của khách hàng doanh nghiệp." for number in
                 range(1, 19)]
        html_text = plan_worksheet_html("Phiếu", plan, "vi", ["Các Team được phát 5 yêu cầu và chọn 2 sáng kiến.",
                                                             *lines]) or ""
        self.assertNotIn("Các Team được phát", html_text)
        self.assertEqual(html_text.count("<li>Quy định số"), 6)


class NoteOrderTests(unittest.TestCase):
    """The unit note: what to check first ("Cần xem"), then what to polish, then what was done; the budget cuts the
    end."""

    def test_checks_lead_and_edit_notes_follow(self) -> None:
        note = build_unit_author_note(
            locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[], fallback_slots=[], ai_drafted=True,
            slot_types=["html", "problem"], review_slots=[1], slot_reasons={1: ["IDM_W5_ANSWER_LEAK"]},
            review_reasons=[REVIEW_REASON_KEPT_SLOTS], notices=[REVIEW_REASON_EVIDENCE],
            remaining=[SlotFinding("IDM_W5_ANSWER_LEAK", 1), SlotFinding("IDM_W5_SENTENCE_LENGTH", 0)],
            fixed_codes=["HTML_SOURCE_LOCATOR"], unexplained_acronyms=["TNA"])
        order = [note.index(part) for part in ("Cần tác giả xem lại vì", "Giữ bản AI", "Cần xem khối 2",
                                               "Lưu ý:", "Nên biên tập khối 1", "Tình huống minh hoạ",
                                               "Chữ viết tắt", "Đã tự sửa")]
        self.assertEqual(order, sorted(order))

    def test_the_budget_never_cuts_what_to_check(self) -> None:
        note = build_unit_author_note(
            locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[], fallback_slots=[], ai_drafted=True,
            slot_types=["html", "problem"], review_slots=[1], slot_reasons={1: ["IDM_W5_ANSWER_LEAK"]},
            review_reasons=[REVIEW_REASON_KEPT_SLOTS], remaining=[SlotFinding("IDM_W5_ANSWER_LEAK", 1)],
            source_flags=["Tài liệu ghi khác nhau ở phần “Ba cấp độ”: “Cấp 2 được phản hồi trong 48 giờ” (trang 7) "
                          "và “Cấp 2 được phản hồi trong 24 giờ” (trang 9). Hãy kiểm tra trước khi áp dụng."] * 3,
            unexplained_acronyms=[f"AB{number}" for number in range(8)],
            source_notes=["Tài liệu ghi chưa rõ ở phần “Ba cấp độ”: " + "rất dài " * 60 + "."] * 3,
            glued=["bộchân", "khảthi"], notices=[REVIEW_REASON_EVIDENCE], faq_items_dropped=1, faq_items_restated=1,
            faq_items_copied=1, faq_items_recall=1, faq_items_open=1, faq_items_invalid=1, callouts_to_prose=1,
            callouts_repeating=1, suppositions=2, framework_list_inserted=5, worksheet_compacted=True)
        self.assertLessEqual(len(note), 1500)
        self.assertIn("Cần xem khối 2 (Quiz)", note)
