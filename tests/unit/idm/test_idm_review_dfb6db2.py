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
from app.idm.prompts_course import w1_section_prompt
from app.idm.prompts_unit import unit_writer_prompt
from app.idm.qa import (
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_KEPT_SLOTS,
    JudgeOutcome,
    SlotFinding,
    build_unit_author_note,
)
from app.idm.quiz_letters import number_item_letters
from app.idm.quiz_text import EXPLANATION_NAMES_ONLY_CODE, _asks_early, explanation_report, lay_out_quiz
from app.idm.source_flags import REVIEW_FLAG_TYPES
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.text import evidence_index, glued_words
from app.idm.visual_pages import evidence_needs_check
from app.idm.worksheet import compact_worksheet, plan_worksheet_html
from app.idm.worksheet_input import is_example_section, is_input_section, missing_input, reused_entries
from tests.unit.idm.test_idm_qc_feedback import SUGGESTED
from tests.unit.idm.test_idm_qc_run_5496155c import facts_of
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    rehashed_brief,
    severity_body,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)


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


GUIDANCE = (" Đọc kỹ mô tả của khách hàng, ghi lại thiệt hại, số lần phàn nàn và các yếu tố an toàn, pháp lý, truyền "
            "thông trước khi chọn cấp độ cho từng trường hợp.")


class RepairOverBudgetTests(StoryboardEndpointTestCase):
    """Item 6: a repair that adds the worksheet's data and goes over the budget was reverted to the data-less
    draft; it is compacted first."""

    async def test_the_repair_with_the_data_is_compacted_not_reverted(self) -> None:
        body = worksheet_body()
        repaired = worksheet_writer(body)
        task, template, example, check = repaired["components"]["c0"]["semantic_content"]["sections"]
        complaints = task["blocks"].pop()
        for row in template["blocks"][0]["rows"]:
            row["value"] += GUIDANCE * 4
        repaired["components"]["c0"]["semantic_content"]["sections"] = [
            task, section("Dữ liệu đầu vào", complaints), template, example, check]
        writer = worksheet_writer(body)
        writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"].pop()
        self.assertIsNotNone(missing_input(writer["components"]["c0"]))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(repaired, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        slot = data["unit"]["components"][0]
        self.assertNotIn("source_locked_fallback", slot)
        data_section = next(item for item in slot["semantic_content"]["sections"]
                            if item["heading"] == "Dữ liệu đầu vào")
        self.assertEqual(len(data_section["blocks"][0]["items"]), 3)
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertNotIn("Cần xem khối 1", note)
        self.assertIn("Phiếu thực hành dài hơn giới hạn nên đã rút gọn", note)
        self.assertIn("Đã tự sửa: phiếu thực hành thiếu dữ liệu đầu vào", note)


class OwnDataTests(unittest.TestCase):
    """Item 8: a task on data the learner brings or makes needs no input section."""

    TEMPLATE = section("Mẫu cần điền", {"kind": "table", "rows": [{"label": "Đối tượng", "value": "Ai?"},
                                                                   {"label": "Lợi ích", "value": "Gì?"}]})
    CHECK = section("Tự kiểm tra", {"kind": "bullets", "items": ["Đủ ba phần", "Có số liệu"]})

    def worksheet(self, task: str) -> dict[str, Any]:
        return html(section("Nhiệm vụ", {"kind": "task", "text": task}), self.TEMPLATE, self.CHECK)

    def test_the_learners_own_data_is_no_missing_input(self) -> None:
        for task in ("Soạn đề xuất giá trị cho 3 khách hàng mục tiêu của chính bạn rồi so với tiêu chí.",
                     "Dựa trên danh sách bạn vừa lập ở bước trước, chọn ra hai ưu tiên.",
                     "Phân tích dữ liệu sau khi phỏng vấn để điền bảng.",
                     "Thu thập phản hồi từ 5 khách hàng của bạn trong tuần này rồi điền bảng."):
            with self.subTest(task=task):
                self.assertIsNone(missing_input(self.worksheet(task)))

    def test_given_data_is_still_required(self) -> None:
        reference = missing_input(self.worksheet("Đọc danh sách 5 yêu cầu dưới đây và chọn 2 yêu cầu ưu tiên."))
        self.assertIsNotNone(reference)
        self.assertIsNotNone(missing_input(self.worksheet("Phân loại 3 khách hàng mục tiêu theo dữ liệu sau.")))


class LowFindingsTests(unittest.TestCase):
    """Items 9, 13, 14, 15: a verb is no item label, a foreign word is no glued word, reported speech is no second
    question, an explanation in other words is no bare name."""

    def test_a_verb_choose_is_no_item_label(self) -> None:
        verb = quiz("Nhóm bạn có ba lựa chọn:\n- Lựa chọn A: Giảm giá ngay.\n- Lựa chọn B: Tặng quà.\n- Lựa chọn C: "
                    "Gọi lại sau.\nBạn nên lựa chọn A, B hay C?",
                    [("Lựa chọn A", False), ("Lựa chọn B", True), ("Lựa chọn C", False)])
        self.assertEqual(number_item_letters(verb), (verb, 0))
        noun = {**verb, "question": verb["question"].replace("Bạn nên lựa chọn A, B hay C?", "Lựa chọn nào đúng?")}
        self.assertEqual(number_item_letters(noun)[0]["question"].splitlines()[1], "- Cách 1: Giảm giá ngay.")

    def test_foreign_words_are_not_glued(self) -> None:
        self.assertEqual(glued_words("Khách hàng ở café Zürich gửi thư cho Nestlé và José."), [])
        self.assertEqual(glued_words("Lập bộchân dung và chọn sáng kiến khảthi."), ["bộchân", "khảthi"])

    def test_reported_speech_is_no_second_question(self) -> None:
        self.assertEqual(_asks_early(["Khách hàng gọi điện và hỏi: Khi nào thì hàng về kho?", "- Cách 1: a",
                                      "- Cách 2: b", "Bạn nên trả lời theo cách nào?"]), [])
        self.assertEqual(_asks_early(["Khách hàng gọi điện. Bạn làm gì trước?", "- Cách 1: a", "- Cách 2: b",
                                      "Cách nào đúng?"]), ["1"])

    def test_an_explanation_in_other_words_is_no_bare_name(self) -> None:
        items = quiz("Trưởng bộ phận đưa ra các phản hồi sau:\n- Phát biểu 1: Cần xuất báo cáo tồn kho mỗi chiều để "
                     "đối soát dữ liệu.\n- Phát biểu 2: Giải pháp tự động gợi ý đặt hàng sớm, mang lại kết quả bất ngờ."
                     "\nPhát biểu nào là một lợi ích kỳ vọng?",
                     [("Phát biểu 1", False), ("Phát biểu 2", True)],
                     "Nguyên tắc: lợi ích kỳ vọng là kết quả vượt mong đợi.\nA - Sai vì đây là việc cần làm thuộc chức "
                     "năng nhằm xử lý tác vụ hàng ngày.\nB - Đúng vì phương án này đúng tiêu chí.")
        self.assertEqual(explanation_report(lay_out_quiz(items)[0]).get(EXPLANATION_NAMES_ONLY_CODE),
                         ["the lines of B never say what their item is"])


def image(page: int | None, share: float | None, *, box: bool = True) -> dict[str, Any]:
    locator: dict[str, Any] = {} if page is None else {"page": page}
    if box:
        locator["bbox_normalized"] = [0.5, 0.2, 0.95, 0.9]
    region: dict[str, Any] = {"region_kind": "embedded_image", "asset_revision": "c" * 64, "locator": locator,
                              "observation": {"status": "unreviewed", "facts": []},
                              "inference": {"status": "not_performed", "claims": []}}
    if share is not None:
        region["page_area_ratio"] = share
    return region


class VisualWarningFallbackTests(unittest.TestCase):
    """The image warning falls back to the old behaviour when the pages cannot be told apart."""

    VISUAL = ("VISUAL_OBSERVATION_REQUIRED:" + "d" * 64,)

    def test_a_decorative_picture_with_its_measures_is_no_warning(self) -> None:
        self.assertFalse(evidence_needs_check([{"visual_regions": [image(14, 0.3)]}], self.VISUAL))

    def test_a_failed_bundle_or_an_unmeasured_region_keeps_the_warning(self) -> None:
        decorative = [{"visual_regions": [image(14, 0.3)]}]
        self.assertTrue(evidence_needs_check(decorative, ["EVIDENCE_BUNDLE_ASSEMBLY_FAILED"]))
        self.assertTrue(evidence_needs_check([{"visual_regions": [image(14, None, box=False)]}], self.VISUAL))
        self.assertTrue(evidence_needs_check([{"visual_regions": [image(None, 0.3)]}], self.VISUAL))


class UnclearDefinitionTests(unittest.TestCase):
    """W1: a rule a learner could apply in two ways is a conflict (it stays "Cần xem"); unclear is only text that
    cannot be followed as written (an edit note)."""

    def test_a_two_way_rule_is_a_conflict(self) -> None:
        prompt = flat(w1_section_prompt("vi", project_context={}, section_id="sec_001", section_title_path="A",
                                        facts=[("f1", "Một dòng", None)]))
        self.assertIn("conflict is two facts that say different things, or one rule, number or instruction a learner "
                      "could apply in two ways (the author must settle it); unclear is a passage a learner cannot "
                      "follow as written", prompt)
        self.assertIn("conflict", REVIEW_FLAG_TYPES)
        self.assertNotIn("unclear", REVIEW_FLAG_TYPES)


class Review4RendererTests(unittest.TestCase):
    """Review 4 of dfb6db2..266d5df: M1 (a quoted word inside a wrapped sentence goes on), M2 (a slide term
    capitalised mid-sentence is no name), L1 (a star after a bracket is an operator)."""

    def test_a_wrapped_sentence_with_a_quote_stays_one_paragraph(self) -> None:
        for facts in (("Khách hàng sẽ nhớ mãi một trải nghiệm",
                       "\"Wow\" ở tuyến trước nếu hậu trường vận hành chuẩn xác."),
                      ("Doanh nghiệp B2B thường gặp những", "“nút thắt” rủi ro trong khâu phê duyệt nội bộ kéo dài."),
                      ("Nhân viên tuyến đầu cần tránh câu", "“Việc này không thuộc bộ phận của tôi.”")):
            with self.subTest(first=facts[0]):
                self.assertEqual(rendered(*facts).count("<p>"), 1)
                self.assertNotIn("<h3>", rendered(*facts))

    def test_a_slide_term_capitalised_mid_sentence_is_no_name(self) -> None:
        self.assertIn("<h3>Triết lý cốt lõi (quan điểm quản trị)</h3>", rendered(
            "TRIẾT LÝ CỐT LÕI (QUAN ĐIỂM QUẢN TRỊ)",
            "Doanh nghiệp tối ưu từng Điểm chạm với khách hàng trong hành trình."))
        self.assertIn("<h3>Chiến lược giá và phân phối</h3>", rendered(
            "CHIẾN LƯỢC GIÁ VÀ PHÂN PHỐI", "Mô hình gồm Sản phẩm, Giá, Phân phối và Xúc tiến cho mỗi thị trường."))
        self.assertIn("<h3>Chi nhánh Vinamilk tại Đà Nẵng</h3>", rendered(
            "CHI NHÁNH VINAMILK TẠI ĐÀ NẴNG", "Công ty Vinamilk mở thêm kênh phân phối cho các tỉnh miền Trung."))

    def test_a_star_after_a_bracket_is_an_operator(self) -> None:
        self.assertIn("(Tác động + Mức khẩn) * Khả thi", rendered(
            "Điểm ưu tiên = (Tác động + Mức khẩn) * Khả thi của từng sáng kiến trong danh mục."))


class Review4GivenDataTests(OwnDataTests):
    """Review 4, M3: the "learner's own data" exception never swallows data the task says it gives."""

    def test_given_data_named_near_the_learner_is_still_required(self) -> None:
        for task in ("Đánh giá 3 đề xuất sau của nhóm bạn theo tiêu chí khả thi.",
                     "Xem 3 hồ sơ ứng viên bạn vừa nhận dưới đây và chọn một người.",
                     "Đọc danh sách dưới đây mà bạn đã nhận từ phòng kinh doanh và phân loại theo mức ưu tiên."):
            with self.subTest(task=task):
                self.assertIsNotNone(missing_input(self.worksheet(task)))
