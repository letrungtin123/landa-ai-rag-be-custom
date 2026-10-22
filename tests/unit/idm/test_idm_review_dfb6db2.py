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
from app.idm.quiz_text import lay_out_quiz
from app.idm.text import evidence_index
from app.idm.worksheet import compact_worksheet
from app.idm.worksheet_input import is_example_section, is_input_section, missing_input, reused_entries


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
