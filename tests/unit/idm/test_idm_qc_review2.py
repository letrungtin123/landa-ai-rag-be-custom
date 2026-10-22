"""Regressions from the second independent review (2c6d719..6666b8c, 2026-10-22): the stem-item renumbering and the
media brief completion. Each test fails on the code before its fix."""

from __future__ import annotations

import unittest
from typing import Any

from app.idm.mcq import normalize_single_choice
from app.idm.quiz_text import QUESTION_ITEM_LETTERS_CODE, lay_out_quiz, number_item_letters, question_report


def quiz(question: str, choices: list[tuple[str, bool]], explanation: str = "", **extra: Any) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question,
            "choices": [{"text": text, "correct": correct} for text, correct in choices], "explanation": explanation,
            **extra}


def renamed(component: dict[str, Any]) -> tuple[dict[str, Any], int]:
    return number_item_letters(component)


class ConservativeRenumberingTests(unittest.TestCase):
    """A1-A3, A8: only letters after a generic label noun are renamed, all of them (chains too) or none; a quiz the
    server cannot rename with certainty keeps its text and gets the edit note."""

    def test_names_that_hold_a_letter_are_never_renamed(self) -> None:
        cases = [
            ("Phân loại đám cháy:\n- Đám cháy loại A: chất rắn\n- Đám cháy loại B: chất lỏng\nBình nào dùng cho "
             "loại B?",
             [("Đám cháy loại A", False), ("Đám cháy loại B", True)]),
            ("Kho phân nhóm theo ABC:\n- Nhóm A: 20% mặt hàng\n- Nhóm B: 30% mặt hàng\n- Nhóm C: 50% mặt hàng\n"
             "Nhóm nào?",
             [("Nhóm A", True), ("Nhóm C", False)]),
            ("Các vitamin:\n- Vitamin A: tốt cho mắt\n- Vitamin B: chuyển hoá\nVitamin nào tốt cho mắt?",
             [("Vitamin A", True), ("Vitamin B", False)]),
            ("Nhóm máu:\n- Nhóm máu A: kháng nguyên A\n- Nhóm máu B: kháng nguyên B\nNhóm nào?",
             [("Nhóm máu A", True), ("Nhóm máu B", False)]),
            ("Bằng lái:\n- Hạng A: xe máy\n- Hạng B: ô tô\nBạn cần hạng nào?", [("Hạng A", False), ("Hạng B", True)]),
            ("Mô hình:\n- Mô hình A: B2B\n- Mô hình B: B2C\nChọn?", [("Mô hình A (B2B)", True), ("Mô hình B", False)]),
        ]
        for question, choices in cases:
            component = quiz(question, choices)
            self.assertEqual(renamed(component), (component, 0), question)
            self.assertNotIn(QUESTION_ITEM_LETTERS_CODE, question_report(component), question)

    def test_another_label_for_the_same_letter_keeps_everything(self) -> None:
        # A2: the stem says "Ở chi nhánh A", the choices "Chi nhánh A": not a generic label, nothing renamed.
        branch = quiz("Bối cảnh:\n- Ở chi nhánh A: doanh thu 100 triệu\n- Ở chi nhánh B: doanh thu 80 triệu\nChi nhánh "
                      "nào cần hỗ trợ?", [("Chi nhánh A", False), ("Chi nhánh B", True)],
                      "Nguyên tắc: x.\nA - Sai vì chi nhánh A cao hơn.\nB - Đúng vì chi nhánh B thấp hơn.")
        self.assertEqual(renamed(branch), (branch, 0))
        # A generic stem label, but a choice names the letter after another label: nothing renamed, the note instead.
        mixed = quiz("Bối cảnh:\n- Phương án A: giảm giá\n- Phương án B: tặng quà\nChọn?",
                     [("Lựa chọn A", False), ("Phương án B", True)])
        self.assertEqual(renamed(mixed), (mixed, 0))
        self.assertIn(QUESTION_ITEM_LETTERS_CODE, question_report(mixed))
        lower = quiz("Bối cảnh:\n- Phương án A: giảm\n- Phương án B: tặng\nChọn?",
                     [("phương án a", True), ("PHƯƠNG ÁN B", False)])
        self.assertEqual(renamed(lower), (lower, 0))
        bare = quiz("Bối cảnh:\n- Phương án A: giảm\n- Phương án B: tặng\n- Phương án C: chờ\nChọn?",
                    [("Phương án A", True), ("Phương án B", False), ("Phương án C", False)],
                    "Nguyên tắc: x.\nA - Đúng vì phương án A giữ khách.\nB - Sai vì C không giải quyết.\nC - Sai "
                    "vì chờ.")
        self.assertEqual(renamed(bare), (bare, 0))

    def test_chained_letters_are_renamed_all_together(self) -> None:
        component = quiz("Bối cảnh:\n- Phương án A: giảm giá\n- Phương án B: tặng quà\n- Phương án C: không làm gì\n"
                         "Nên chọn phương án A hay B?",
                         [("Kết hợp phương án A và B", True), ("Chỉ làm phương án C", False),
                          ("Phương án A, B và C", False), ("Phương án A/B", False)],
                         "Nguyên tắc: kết hợp ưu đãi.\nA - Đúng vì phương án A và B bổ trợ nhau.\nB - Sai vì phương "
                         "án C không giải quyết.\nC - Sai vì làm cả phương án A, B hoặc C là quá tốn.\nD - Sai vì "
                         "phương án A/B thiếu một bước.",
                         hints=["So sánh phương án A với phương án B"], title="Phương án A hay B?")
        out, count = renamed(component)
        self.assertGreater(count, 0)
        self.assertIn("Nên chọn phương án 1 hay 2?", out["question"])
        self.assertEqual([choice["text"] for choice in out["choices"]],
                         ["Kết hợp phương án 1 và 2", "Chỉ làm phương án 3", "Phương án 1, 2 và 3", "Phương án 1/2"])
        self.assertIn("A - Đúng vì phương án 1 và 2 bổ trợ nhau.", out["explanation"])
        self.assertIn("phương án 1, 2 hoặc 3", out["explanation"])
        self.assertEqual((out["hints"], out["title"]), (["So sánh phương án 1 với phương án 2"], "Phương án 1 hay 2?"))
        for ordinal in range(3):
            laid, _codes = lay_out_quiz(normalize_single_choice(out, "seed-x", ordinal)[0])
            self.assertEqual([line[:4] for line in str(laid["explanation"]).splitlines()[1:]],
                             ["A - ", "B - ", "C - ", "D - "])

    def test_semicolons_parentheses_and_an_english_article(self) -> None:
        inline = quiz("Có ba phương án: Phương án A: giảm giá; Phương án B: tặng quà; Phương án C: chờ. Chọn phương án "
                      "nào?", [("Phương án A", True), ("Phương án B", False), ("Phương án C", False)])
        self.assertEqual(renamed(inline)[0]["question"], "Có ba phương án: Phương án 1: giảm giá; Phương án 2: tặng "
                                                         "quà; Phương án 3: chờ. Chọn phương án nào?")
        dashed = quiz("Có ba phương án: - Phương án A: giảm giá - Phương án B: tặng quà - Phương án C: chờ. Chọn?",
                      [("Phương án A", True), ("Phương án C", False)])
        self.assertEqual(renamed(dashed)[0]["question"], "Có ba phương án: - Phương án 1: giảm giá - Phương án 2: "
                                                         "tặng quà - Phương án 3: chờ. Chọn?")
        tagged = quiz("Bối cảnh:\n- Phương án A (rẻ): giảm\n- Phương án B (nhanh): tặng\nChọn?",
                      [("Phương án A", True), ("Phương án B", False)])
        self.assertEqual(renamed(tagged)[0]["question"], "Bối cảnh:\n- Phương án 1 (rẻ): giảm\n- Phương án 2 (nhanh): "
                                                         "tặng\nChọn?")
        english = quiz("Two plans:\n- Plan A: cut prices\n- Plan B: add a gift\nWhich plan keeps margin?",
                       [("Plan A", False), ("Plan B", True)],
                       "Principle: x.\nA - Incorrect because plan A cuts margin; a plan a manager likes.\nB - Correct "
                       "because Plan B keeps it.")
        out, _count = renamed(english)
        self.assertIn("plan 1 cuts margin; a plan a manager likes.", out["explanation"])
        self.assertEqual([choice["text"] for choice in out["choices"]], ["Plan 1", "Plan 2"])

    def test_only_a_single_choice_whose_items_are_all_labelled(self) -> None:
        typed = quiz("Bối cảnh:\n- Phương án A: giảm giá\n- Phương án B: tặng quà\nGhi tên phương án giữ biên.", [],
                     answers=["Phương án B", "B"], problem_type="short_text")
        self.assertEqual(renamed(typed), (typed, 0))
        partial = quiz("Bối cảnh:\n- Phương án A: giảm giá\n- Một ý khác\n- Phương án B: tặng quà\nChọn?",
                       [("Phương án A", True), ("Phương án B", False)])
        self.assertEqual(renamed(partial), (partial, 0))
