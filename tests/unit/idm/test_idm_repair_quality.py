"""Owner decisions of 2026-10-20 on W5/W6 repairs (pure functions; the endpoint flows are in
``test_idm_repair_quality_unit.py``).

* P2: a new-fact finding names its exact numbers and names; list labels ("Shift 1", "Trụ cột 01") and the count of the
  list that follows ("gồm 5 bước:" before five items) are layout; a new number inside an AI-drafted example is a notice.
* P3: the second targeted repair and its time gate.
* P7: an FAQ title matches the words the unit writes in the same word pair.
* P1: what makes a slot worse after a repair.

Nothing reaches the network.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.idm.budget import second_repair_fits
from app.idm.fidelity_checks import (
    NEW_FACT_CODE,
    FidelityChecker,
    KnownFacts,
    example_segments,
    label_heads,
    list_counts,
    masked_count,
    masked_labels,
)
from app.idm.prompts import new_fact_hint, second_repair_line
from app.idm.qa import FAQ_TITLE_MISMATCH_CODE, SENTENCE_COPY_CODE, faq_title_mismatch
from app.idm.qa_checks import ANSWER_LEAK_CODE
from app.idm.storyboard_repair import worse
from app.idm.text import evidence_index
from tests.unit.idm.test_idm_fidelity import FACTS, brief

FIRST = "semantic_content.sections[0].blocks[0]"


def block(kind: str, text: str | None = None, items: list[str] | None = None,
          rows: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": items or [],
            "rows": [{"label": label, "value": value} for label, value in rows or []]}


def html(*blocks: dict[str, Any], heading: str = "Công ty mong đợi gì ở bạn?") -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": heading, "blocks": list(blocks)}]}}


def checker(facts: dict[str, str] | None = None) -> FidelityChecker:
    """The explained teaching slot of ``test_idm_fidelity`` (facts f1-f5: numbers 2012 and 14 only)."""

    texts = facts or FACTS
    unit_brief = brief("explain")
    return FidelityChecker.build(unit_brief, texts, evidence_index(texts.values()), locale="vi",
                                 plan_texts=[unit_brief.course_title, unit_brief.lesson_title])


def tokens(component: dict[str, Any], facts: dict[str, str] | None = None) -> dict[str, tuple[str, ...]]:
    """Location -> the tokens that make it a finding."""

    return {hit.location: hit.review_tokens for hit in checker(facts).new_fact_hits(component) if hit.review_tokens}


class ListLabelTests(unittest.TestCase):
    def test_a_label_sequence_is_layout_at_the_start_of_an_item_and_anywhere_in_a_line(self) -> None:
        component = html(
            block("bullets", items=["Trụ cột 01: Tận tâm với khách hàng.", "Trụ cột 02: Sáng tạo trong công việc.",
                                    "Trụ cột 03: Hợp tác với đồng nghiệp."]),
            block("paragraph", "Khi đề xuất cách làm mới, bạn đang sống đúng trụ cột 02 của công ty."))
        self.assertEqual(tokens(component), {})
        self.assertNotIn(NEW_FACT_CODE, [code for code, _ in checker().findings([component, {}])])
        english = html(block("bullets", items=["Pillar 1: Care for every customer.", "Pillar 2: Create new ways.",
                                               "Pillar 3: Work as one team."]),
                       block("paragraph", "When you suggest a new way of working, you live pillar 2."))
        self.assertEqual(tokens(english), {})

    def test_labels_in_rows_and_headings_open_a_sequence_too(self) -> None:
        component = {"type": "html", "semantic_content": {"sections": [
            {"heading": "Shift 1: Nghĩ lớn", "blocks": [block("paragraph", "Shift 1 bắt đầu từ khát vọng.")]},
            {"heading": "Shift 2: Nghĩ khác", "blocks": [block("table", rows=[("Tháng 1 (khởi động)", "Gặp đội ngũ."),
                                                                             ("Tháng 2 (tăng tốc)", "Đo kết quả.")])]},
        ]}}
        self.assertEqual(tokens(component), {})

    def test_a_label_outside_its_sequence_stays_a_new_fact(self) -> None:
        cases = {
            # The sequence does not start at 1, or skips a number.
            "no_first": html(block("bullets", items=["Trụ cột 02: Sáng tạo.", "Trụ cột 03: Hợp tác."])),
            "gap": html(block("bullets", items=["Trụ cột 01: Tận tâm.", "Trụ cột 03: Hợp tác."])),
            # A label number after the last item of the list.
            "beyond": html(block("bullets", items=["Trụ cột 01: Tận tâm.", "Trụ cột 02: Sáng tạo."]),
                           block("paragraph", "Trụ cột 04 sẽ được giới thiệu ở bài sau.")),
            # A number after a word that opens no item ("tăng 7" is no label).
            "not_a_label": html(block("paragraph", "Doanh thu tăng 7 lần sau khi áp dụng.")),
        }
        expected = {"no_first": {f"{FIRST}.items[0]": ("02",), f"{FIRST}.items[1]": ("03",)},
                    "gap": {f"{FIRST}.items[0]": ("01",), f"{FIRST}.items[1]": ("03",)},
                    "beyond": {"semantic_content.sections[0].blocks[1]": ("04",)},
                    "not_a_label": {FIRST: ("7",)}}
        for name, component in cases.items():
            with self.subTest(case=name):
                self.assertEqual(tokens(component), expected[name])

    def test_a_word_the_facts_number_keeps_its_numbering_as_fact(self) -> None:
        # The facts number "Bậc 01" … "Bậc 03": a fourth rung the writer adds is an invented fact, not a list label.
        facts = {**FACTS, "f5": "Bậc 01 là gia công. Bậc 02 là chất lượng quốc gia. Bậc 03 là thương hiệu riêng."}
        component = html(block("bullets", items=["Bậc 01: Gia công.", "Bậc 02: Chất lượng quốc gia.",
                                                 "Bậc 03: Thương hiệu riêng.", "Bậc 04: Chuẩn toàn cầu."]))
        self.assertEqual(tokens(component, facts), {f"{FIRST}.items[3]": ("04",)})

    def test_label_helpers(self) -> None:
        self.assertEqual(label_heads(["SHIFT 1: THINK BIG", "Trụ cột 02 - Sáng tạo", "Tháng 3 (Act)", "Có 3 bước sau",
                                      "Kỷ nguyên 5.0 là gì", "Phase 2."]),
                         {"shift": {1}, "cot": {2}, "thang": {3}, "phase": {2}})
        self.assertEqual(masked_labels("Theo Shift 3 và Shift 6 của kỷ nguyên 5.0", {"shift": 5}),
                         "Theo x và Shift 6 của kỷ nguyên 5.0")


class ListCountTests(unittest.TestCase):
    def test_the_count_of_the_list_that_follows_is_layout(self) -> None:
        values = ["Tận tâm với khách hàng.", "Sáng tạo trong công việc.", "Hợp tác với đồng nghiệp."]
        self.assertEqual(tokens(html(block("paragraph", "Công ty có 3 giá trị cốt lõi sau:"),
                                     block("bullets", items=values))), {})
        self.assertEqual(tokens(html(block("paragraph", "Công ty có 3 giá trị cốt lõi sau:"),
                                     block("table", rows=[(value[:8], value) for value in values]))), {})
        self.assertEqual(tokens(html(block("paragraph", "The company has 3 core values:"),
                                     block("bullets", items=values))), {})

    def test_a_count_that_does_not_match_the_list_or_counts_a_unit_stays_a_new_fact(self) -> None:
        values = ["Tận tâm với khách hàng.", "Sáng tạo trong công việc.", "Hợp tác với đồng nghiệp."]
        cases = {
            "four_of_three": (html(block("paragraph", "Công ty có 4 giá trị cốt lõi sau:"),
                                   block("bullets", items=values)), ("4",)),
            "duration": (html(block("paragraph", "Bạn học trong 3 ngày như sau:"), block("bullets", items=values)),
                         ("3",)),
            "english_duration": (html(block("paragraph", "You learn it in 3 days:"), block("bullets", items=values)),
                                 ("3",)),
            "no_list": (html(block("paragraph", "Công ty có 3 giá trị cốt lõi."), block("paragraph", "Tận tâm.")),
                        ("3",)),
        }
        for name, (component, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(tokens(component), {FIRST: expected})

    def test_count_helpers(self) -> None:
        component = html(block("paragraph", "Gồm 2 bước:"), block("steps", items=["Đọc.", "Ghi."]),
                         block("paragraph", "Sau đó"), block("paragraph", "Hết."))
        self.assertEqual(list_counts(component), {FIRST: 2})
        self.assertEqual(masked_count("Gồm 2 bước trong 2 ngày", 2), "Gồm x bước trong 2 ngày")


class DraftedExampleNumberTests(unittest.TestCase):
    def test_a_new_number_inside_a_drafted_example_is_a_notice_not_a_finding(self) -> None:
        for text in ("Ví dụ minh hoạ: Một nhân viên mới nhận 3 cuộc gọi trong buổi sáng đầu tiên.",
                     "Illustrative example: A new employee answers 3 calls on the first morning."):
            with self.subTest(text=text):
                component = html(block("paragraph", text))
                hits = checker().new_fact_hits(component)
                self.assertEqual([(hit.numbers, hit.example, hit.review_tokens) for hit in hits], [(("3",), True, ())])
                self.assertEqual(checker().new_fact_locations(component), [])
                self.assertEqual(checker().example_numbers([component, {}]), {0: 1})
                self.assertNotIn(NEW_FACT_CODE, [code for code, _ in checker().findings([component, {}])])

    def test_the_list_a_drafted_example_introduces_is_part_of_it(self) -> None:
        component = html(block("paragraph", "Ví dụ minh hoạ: kế hoạch tuần đầu của một nhân viên mới:"),
                         block("table", rows=[("Tuần 1", "Gặp 3 phòng ban và đọc sổ tay.")]),
                         block("paragraph", "Dễ nhầm: 5 ngày đầu chưa cần báo cáo."))
        self.assertEqual(sorted(example_segments(component)),
                         [FIRST, "semantic_content.sections[0].blocks[1].rows[0]"])
        # The plan's number is a notice; the paragraph after the example is not part of it.
        self.assertEqual(tokens(component), {"semantic_content.sections[0].blocks[2]": ("5",)})
        self.assertEqual(checker().example_numbers([component, {}]), {0: 1})

    def test_names_inside_and_numbers_outside_a_drafted_example_stay_new_facts(self) -> None:
        cases = {
            "name_in_example": ("Ví dụ minh hoạ: Chị Mai ở Đà Nẵng nhận 3 cuộc gọi.", ("Mai", "Đà Nẵng")),
            "unlabelled_case": ("Giả sử bạn nhận 3 cuộc gọi trong buổi sáng đầu tiên.", ("3",)),
            "source_example_label": ("Ví dụ: công ty có 3 văn phòng trong cả nước.", ("3",)),
            "english_unlabelled": ("Suppose you answer 3 calls on the first morning.", ("3",)),
        }
        for name, (text, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(tokens(html(block("paragraph", text))), {FIRST: expected})
        # The drafted example's own numbers are never quoted for a repair.
        self.assertEqual(checker().example_numbers([html(block("paragraph", cases["name_in_example"][0])), {}]),
                         {0: 1})

    def test_known_tokens_are_reported_as_written(self) -> None:
        known = KnownFacts.of(["Bậc 01 là gia công."])
        self.assertEqual(known.new_tokens("Bậc 07 do Công ty Sao Mai đặt ra năm 1999."), (("07", "1999"), ("Sao Mai",)))
        self.assertEqual(known.new_tokens("Bước 2: Bậc 01 là gia công."), ((), ()))
        self.assertTrue(known.states_new_fact("Có 1.500 nhân viên."))
        self.assertFalse(known.states_new_fact("Bậc 1 là gia công."))


class RepairPromptTests(unittest.TestCase):
    def test_new_tokens_are_quoted_bounded_and_markup_free(self) -> None:
        self.assertEqual(new_fact_hint(["Think Fast", "2", "Think Fast"]), ' (new: "Think Fast", "2")')
        self.assertEqual(new_fact_hint([]), "")
        # Never markup or a quote (tokens are word characters by construction; anything else is dropped).
        self.assertEqual(new_fact_hint(['<b>"x"</b>', "ok"]), ' (new: "ok")')
        many = new_fact_hint(f"N{index}" for index in range(20))
        self.assertEqual(many.count('"') // 2, 8)
        self.assertIn('"' + "A" * 40 + '"', new_fact_hint(["A" * 60]))

    def test_second_repair_lines(self) -> None:
        self.assertTrue(second_repair_line("c1", ANSWER_LEAK_CODE).startswith(
            "c1 IDM_W5_ANSWER_LEAK: SECOND ATTEMPT: the first rewrite still gives the answer away. Set the question's "
            "case in a different industry AND at a different touchpoint"))
        self.assertIn("SECOND ATTEMPT", second_repair_line("c0", NEW_FACT_CODE))
        self.assertIn("SECOND ATTEMPT", second_repair_line("c1", "IDM_W5_ANSWER_CONTESTED"))
        self.assertEqual(second_repair_line("c0", SENTENCE_COPY_CODE), "")


class BudgetTests(unittest.TestCase):
    def test_the_second_repair_leaves_the_judge_its_time(self) -> None:
        # A low repair (20 s) plus the call headroom (5 s), plus the judge (25 s) when the unit is judged.
        self.assertTrue(second_repair_fits(50.0, judge=True))
        self.assertFalse(second_repair_fits(49.9, judge=True))
        # Without a judge the targeted-repair floor (30 s) holds.
        self.assertTrue(second_repair_fits(30.0, judge=False))
        self.assertFalse(second_repair_fits(29.9, judge=False))


class WorseTests(unittest.TestCase):
    def test_what_makes_a_slot_worse(self) -> None:
        none: frozenset[str] = frozenset()
        self.assertTrue(worse(frozenset({SENTENCE_COPY_CODE}), frozenset({NEW_FACT_CODE})))
        self.assertTrue(worse(none, frozenset({SENTENCE_COPY_CODE})))  # a code added, none fixed
        self.assertTrue(worse(frozenset({NEW_FACT_CODE}), frozenset({NEW_FACT_CODE, SENTENCE_COPY_CODE})))
        # A wording code that replaces another one, or a fix, is not worse.
        self.assertFalse(worse(frozenset({"IDM_W5_SENTENCE_LENGTH"}), frozenset({SENTENCE_COPY_CODE})))
        self.assertFalse(worse(frozenset({NEW_FACT_CODE}), frozenset({SENTENCE_COPY_CODE})))
        self.assertFalse(worse(frozenset({NEW_FACT_CODE}), none))
        self.assertFalse(worse(frozenset({ANSWER_LEAK_CODE}), frozenset({ANSWER_LEAK_CODE})))


def faq(title: str, *items: tuple[str, str]) -> dict[str, Any]:
    return {"type": "la_faq", "title": title, "items": [{"question": question, "answer": answer}
                                                        for question, answer in items]}


class FaqTitleTests(unittest.TestCase):
    ITEMS = (("Vì sao không nên bỏ qua kiểm toán ở giai đoạn đầu?", "Vì sẽ chọn sai điểm nghẽn cần xử lý trước."),
             ("Điều gì xảy ra nếu chưa có Quick-Win?", "Đội ngũ khó thấy kết quả sớm và dễ nản."))

    def test_a_title_naming_the_units_topic_in_the_units_words_matches(self) -> None:
        component = faq("Hiểu đúng lộ trình 90 ngày", *self.ITEMS)
        self.assertTrue(faq_title_mismatch(component))
        # The question slot or the FAQ's facts write "lộ trình", even when the html before the FAQ does not.
        self.assertFalse(faq_title_mismatch(component, "Lộ trình 90 ngày chia thành ba chặng."))

    def test_a_shared_syllable_in_another_word_pair_does_not_match(self) -> None:
        # "giao" of "đơn hàng giao trễ" is not "giao" of "họp giao ban" (Vietnamese words are syllables).
        component = faq("Câu hỏi thường gặp trong họp giao ban", *self.ITEMS)
        self.assertTrue(faq_title_mismatch(component, "Một khách hàng phàn nàn vì đơn hàng giao trễ."))
        self.assertFalse(faq_title_mismatch(component, "Trong họp giao ban, trưởng nhóm báo cáo kết quả tuần."))
        self.assertEqual(FAQ_TITLE_MISMATCH_CODE, "IDM_W5_FAQ_TITLE_MISMATCH")


if __name__ == "__main__":
    unittest.main()
