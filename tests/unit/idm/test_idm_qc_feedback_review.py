"""Regressions from the independent review of the QC-feedback branch (2026-10-22).

Each test reproduces one finding of the review (its experiment in the review's scratch folder) and fails on the code
before its fix. Every provider is a local fake.
"""

from __future__ import annotations

import unittest

from app.idm.mcq import normalize_single_choice
from app.idm.qa_checks import repeated_callouts
from app.idm.quiz_text import (
    QUESTION_LAYOUT_CODE,
    lay_out_quiz,
    layout_explanation,
    layout_question,
    question_report,
    quiz_findings,
)
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
)


def words(text: str) -> list[str]:
    """The words of ``text`` without list markers (the layout must keep every word)."""

    return [word for word in text.split() if word not in {"-", "—", "–"}]  # noqa: RUF001 - dashes


class InlineListTests(unittest.TestCase):
    """M1: the inline-list split broke ranges ("10 - 20 triệu"), times ("8:00 - 12:00") and pairs ("Hà Nội - Hải
    Phòng"); a list is split only after an intro ending in ":" and at items that start alike ("Mức 1:", "Mức 2:")."""

    def test_ranges_stay_inside_their_item(self) -> None:
        stem = ("Các mức bồi thường như sau: - Mức 1: 10 - 20 triệu đồng - Mức 2: 20 - 50 triệu đồng. Khách hàng X "
                "thiệt hại 30 triệu. Áp dụng mức nào?")
        laid = layout_question(stem)
        self.assertEqual(laid.split("\n"), [
            "Các mức bồi thường như sau:", "- Mức 1: 10 - 20 triệu đồng", "- Mức 2: 20 - 50 triệu đồng.",
            "Khách hàng X thiệt hại 30 triệu.", "Áp dụng mức nào?"])
        self.assertEqual(words(laid), words(stem))
        self.assertNotIn(QUESTION_LAYOUT_CODE, question_report({"question": laid}))
        en_dash = "Các mức như sau: - Mức 1: từ 10 – 20 triệu - Mức 2: từ 20 – 50 triệu. Chọn mức nào?"  # noqa: RUF001
        self.assertEqual(layout_question(en_dash).split("\n")[1:3],
                         ["- Mức 1: từ 10 – 20 triệu", "- Mức 2: từ 20 – 50 triệu."])  # noqa: RUF001

    def test_times_and_place_pairs_stay_whole(self) -> None:
        shifts = "Các ca như sau: - Ca sáng: 8:00 - 12:00 - Ca chiều: 13:00 - 17:00. Bạn chọn ca nào?"
        self.assertEqual(layout_question(shifts).split("\n"), [
            "Các ca như sau:", "- Ca sáng: 8:00 - 12:00", "- Ca chiều: 13:00 - 17:00.", "Bạn chọn ca nào?"])
        routes = "Các tuyến như sau: - Tuyến 1: Hà Nội - Hải Phòng - Tuyến 2: Hà Nội - Nam Định. Chọn tuyến nào?"
        laid = layout_question(routes)
        self.assertEqual(laid.split("\n"), [
            "Các tuyến như sau:", "- Tuyến 1: Hà Nội - Hải Phòng", "- Tuyến 2: Hà Nội - Nam Định.",
            "Chọn tuyến nào?"])
        self.assertEqual(question_report({"question": laid}), {})

    def test_an_unsure_list_is_left_as_written_and_reported(self) -> None:
        for stem in ("Thông tin: - Khách B2B - Doanh nghiệp lớn - Khách B2C - Cá nhân. Bạn chọn kênh nào?",
                     "Bối cảnh: - Tỷ lệ giảm - 5% so với quý trước - Chi phí tăng. Bạn nên làm gì?"):
            laid = layout_question(stem)
            self.assertEqual(laid.split("\n")[0], stem.rsplit(". ", 1)[0] + ".")
            self.assertEqual(words(laid), words(stem))
            self.assertIn(QUESTION_LAYOUT_CODE, question_report({"question": laid}))
        # Items that start alike are cut; what follows the last label without a label of its own is reported.
        mixed = layout_question("Theo bảng: - Nhóm X: doanh thu cao - Nhóm Y: doanh thu thấp - Ghi chú - Xem phụ lục "
                                "2. Nhóm nào được ưu tiên?")
        self.assertEqual(mixed.splitlines()[:3], ["Theo bảng:", "- Nhóm X: doanh thu cao",
                                                  "- Nhóm Y: doanh thu thấp - Ghi chú - Xem phụ lục 2."])
        self.assertIn(QUESTION_LAYOUT_CODE, question_report({"question": mixed}))


KEY_LINES = ("Nguyên tắc: phân loại mức độ trước khi xử lý.\nA - Sai vì chỉ cấp 3 mới báo pháp chế.\n"
             "B - Đúng vì lần đầu, chưa thiệt hại là cấp 1.\nC - Sai vì không hứa bồi thường trước khi phân loại.\n"
             "D - Sai vì cũng bỏ qua phân loại như phương án A.")
FOUR = [{"text": "Báo ngay cho bộ phận pháp chế", "correct": False},
        {"text": "Phân loại cấp 1 và xử lý tại quầy", "correct": True},
        {"text": "Hứa bồi thường để giữ khách", "correct": False},
        {"text": "Chuyển khiếu nại cho quản lý vùng", "correct": False}]


def four(explanation: str) -> dict[str, object]:
    return {"type": "problem", "problem_type": "multiple_choice",
            "question": "Khách hàng phàn nàn lần đầu, chưa có thiệt hại. Bạn xử lý thế nào?",
            "choices": [dict(choice) for choice in FOUR], "explanation": explanation}


class LetterMentionTests(unittest.TestCase):
    """M2 / m7: a letter an explanation mentions ("như phương án A", "đáp án đúng là B", "Phương án A: sai") is no
    option label; in line form only a letter at a line start labels an option, and a closing line stays a line."""

    def test_shuffled_lines_follow_the_new_letters_whatever_they_mention(self) -> None:
        for seed in range(40):
            for ordinal in range(4):
                shuffled, _codes = normalize_single_choice(four(KEY_LINES), f"seed-{seed}", ordinal)
                laid, _laid = lay_out_quiz(shuffled)
                lines = str(laid["explanation"]).splitlines()
                self.assertEqual([line[:4] for line in lines[1:]], ["A - ", "B - ", "C - ", "D - "], (seed, ordinal))
                self.assertEqual(quiz_findings(laid), [], (seed, ordinal))
                key = "ABCD"[[choice["correct"] for choice in laid["choices"]].index(True)]
                self.assertTrue(lines[1 + "ABCD".index(key)].startswith(f"{key} - Đúng vì"))

    def test_a_principle_that_names_the_key_keeps_the_lines_in_order(self) -> None:
        explanation = ("Nguyên tắc: phân loại trước, nên đáp án đúng là B.\nA - Sai vì chỉ cấp 3 mới báo pháp chế.\n"
                       "B - Đúng vì lần đầu là cấp 1.\nC - Sai vì không hứa bồi thường.\nD - Sai vì chưa cần quản lý.")
        shuffled, _codes = normalize_single_choice(four(explanation), "seed-x", 0)
        laid, _laid = lay_out_quiz(shuffled)
        lines = str(laid["explanation"]).splitlines()
        self.assertEqual(lines[0], "Nguyên tắc: phân loại trước, nên đáp án đúng là A.")
        self.assertEqual([line[:4] for line in lines[1:]], ["A - ", "B - ", "C - ", "D - "])
        self.assertTrue(lines[1].startswith("A - Đúng vì lần đầu"))

    def test_a_letter_after_a_word_never_starts_a_segment(self) -> None:
        paragraph = ("Cần phân loại trước khi xử lý. Phương án A: sai vì chưa đến cấp 3. Phương án B: đúng vì đây là "
                     "cấp 1. Phương án C: sai vì hứa bồi thường trước khi phân loại. Phương án D: sai vì chưa cần quản "
                     "lý vùng.")
        for seed in range(40):
            for ordinal in range(4):
                shuffled, _codes = normalize_single_choice(four(paragraph), f"seed-{seed}", ordinal)
                laid, _laid = lay_out_quiz(shuffled)
                self.assertNotIn("Phương án.", laid["explanation"])
                self.assertEqual(str(laid["explanation"]).count("Phương án"), 4)
        for text in ("Đáp án đúng là B. A - Sai vì chưa phân loại mức độ. C - Sai vì cấp 3 cần escalate. D - Sai vì "
                     "không hứa bồi thường.",
                     "Cần phân loại trước khi xử lý. Phương án A: sai vì bỏ qua phân loại. Phương án B: đúng vì phân "
                     "loại trước. Phương án C: sai vì hứa bồi thường."):
            self.assertEqual(layout_explanation(text, 3 if "Phương" in text else 4), text)
        self.assertEqual(layout_explanation("Nguyên tắc: hạng C: khách mới cần tư vấn. A - Sai vì x; B - Đúng vì y; "
                                            "phương án C sai vì z.", 3),
                         "Nguyên tắc: hạng C: khách mới cần tư vấn. A - Sai vì x; B - Đúng vì y; phương án C sai vì z.")
        self.assertEqual(layout_explanation("Theo mục A. của quy trình, cần phân loại. A - Sai vì x; B - Đúng vì y; "
                                            "C - Sai vì z.", 3),
                         "Theo mục A. của quy trình, cần phân loại.\nA - Sai vì x.\nB - Đúng vì y.\nC - Sai vì z.")

    def test_a_closing_line_stays_its_own_line(self) -> None:
        lined = "Phân loại trước.\nA - Sai vì x.\nB - Đúng vì y.\nC - Sai vì z.\nLưu ý: luôn ghi nhận khiếu nại."
        self.assertEqual(layout_explanation(lined, 3), lined)
        shuffled, _codes = normalize_single_choice({**four(lined), "choices": [dict(item) for item in FOUR[:3]]},
                                                   "seed-x", 0)
        self.assertEqual(str(shuffled["explanation"]).splitlines()[-1], "Lưu ý: luôn ghi nhận khiếu nại.")


def section(*blocks: dict[str, object]) -> dict[str, object]:
    return {"heading": "H", "blocks": list(blocks)}


def text_block(kind: str, text: str) -> dict[str, object]:
    return {"kind": kind, "text": text, "items": [], "rows": []}


PARAGRAPH = "Nhân viên tuyệt đối không hứa bồi thường cho khách trước khi phân loại mức độ khiếu nại."


class CalloutRepeatTests(StoryboardEndpointTestCase):
    """M3 / m11: a callout with its own number, level or negation is no repeat; only the paragraphs before it in its
    section are compared, with word trigrams; the note counts the callouts dropped from the returned unit only."""

    def test_a_callout_with_its_own_number_level_or_negation_is_kept(self) -> None:
        kept = {
            "new number": [text_block("paragraph", "Nhân viên tuyệt đối không hứa bồi thường cho khách trước khi phân "
                                                   "loại mức độ khiếu nại theo bảng dưới đây."),
                           text_block("warning", "Không hứa bồi thường cho khách trước khi phân loại mức độ khiếu nại, "
                                                 "và phải báo trong 24 giờ.")],
            "negation flip": [text_block("paragraph", "Sau khi phân loại mức độ khiếu nại, nhân viên được hứa bồi "
                                                      "thường cho khách theo bảng mức."),
                              text_block("warning", "Không được hứa bồi thường cho khách trước khi phân loại mức độ "
                                                    "khiếu nại.")],
            "exception": [text_block("paragraph", "Khiếu nại cấp 3 phải chuyển cho quản lý ngay trong ngày."),
                          text_block("warning", "Khiếu nại cấp 3 không phải chuyển cho quản lý ngay trong ngày.")],
            "first time": [text_block("paragraph", "Cửa hàng áp dụng giảm giá 10% cho khách hàng mua hàng lần thứ hai "
                                                   "trở đi."),
                           text_block("warning", "Không áp dụng giảm giá 10% cho khách hàng mua hàng lần đầu.")],
            "deadline": [text_block("paragraph", "Khiếu nại cấp 2 phải được phản hồi cho khách hàng trong vòng 48 giờ "
                                                 "kể từ khi tiếp nhận."),
                         text_block("warning", "Khiếu nại cấp 3 phải được phản hồi cho khách hàng trong vòng 24 giờ kể "
                                               "từ khi tiếp nhận.")],
            "callout first": [text_block("warning", "Không hứa bồi thường trước khi phân loại mức độ."),
                              text_block("paragraph", PARAGRAPH)],
        }
        for name, blocks in kept.items():
            self.assertEqual(repeated_callouts({"sections": [section(*blocks)]}), [], name)
        other_section = {"sections": [section(text_block("paragraph", PARAGRAPH)),
                                      section(text_block("warning", PARAGRAPH.replace("Nhân viên tuyệt đối k", "K")))]}
        self.assertEqual(repeated_callouts(other_section), [])
        # The paragraph above in other words, nothing new: a repeat.
        repeat = {"sections": [section(text_block("paragraph", PARAGRAPH),
                                       text_block("warning", "Không hứa bồi thường cho khách trước khi phân loại mức "
                                                             "độ khiếu nại."))]}
        self.assertEqual(repeated_callouts(repeat), [(0, 1)])

    async def test_the_note_counts_the_returned_unit_only(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        blocks = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"]
        blocks.append(text_block("warning", "Cấp 3: liên quan an toàn, pháp lý hoặc truyền thông. Escalate ngay cho "
                                            "quản lý và bộ phận Pháp chế theo quy trình CR-04."))
        blocks.append(text_block("warning", "Doanh nghiệp không thay đổi tư duy sẽ có nguy cơ bị thị trường đào thải."))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Đã bỏ 1 khung trích dẫn/lưu ý chỉ nhắc lại đoạn văn có sẵn trong bài.", note)
