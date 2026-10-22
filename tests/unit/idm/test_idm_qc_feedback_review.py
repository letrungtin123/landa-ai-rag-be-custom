"""Regressions from the independent review of the QC-feedback branch (2026-10-22).

Each test reproduces one finding of the review (its experiment in the review's scratch folder) and fails on the code
before its fix. Every provider is a local fake.
"""

from __future__ import annotations

import unittest
from collections import Counter

from app.idm.artifacts import clean_question
from app.idm.contracts import IdmIssueV1
from app.idm.mcq import normalize_single_choice
from app.idm.notes import SkippedBlockNote, build_course_notes
from app.idm.qa import JudgeOutcome, build_unit_author_note
from app.idm.qa_checks import repeated_callouts
from app.idm.quiz_text import (
    EXPLANATION_VERDICT_CODE,
    QUESTION_LAYOUT_CODE,
    explanation_report,
    filler_phrases,
    lay_out_quiz,
    layout_explanation,
    layout_question,
    question_report,
    quiz_findings,
)
from app.idm.source_flags import SourceFact, flag_sentence, flag_sentences
from app.idm.source_locked import idm_source_grounded_single_choice
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


LONG_NAME = ("Quy trình tiếp nhận và phân loại khiếu nại của khách hàng theo mức độ nghiêm trọng (cấp 1, cấp 2, "
             "cấp 3)")
LONG_FACTS = {
    "f1": SourceFact("Row 3: Thời hạn phản hồi khiếu nại cấp 2 là 24 giờ kể từ khi bộ phận chăm sóc khách hàng tiếp "
                     "nhận thông tin từ khách hàng qua hotline.", 12),
    "f2": SourceFact("Theo quy định năm 2019, thời hạn phản hồi khiếu nại cấp 2 là 48 giờ làm việc, không tính ngày "
                     "nghỉ lễ và cuối tuần theo lịch công ty.", 37),
    "f3": SourceFact("Khiếu nại cấp 3 phải báo cáo cho Ban Giám đốc và bộ phận Pháp chế ngay trong ngày, kèm biên bản "
                     "làm việc với khách hàng.", 40),
}
CONFLICT3 = IdmIssueV1(type="conflict", note="x", fact_keys=["f1", "f2", "f3"])


class SourceFlagSizeTests(unittest.TestCase):
    """m6 / m12: the points to check are short, capped in the unit and course notes, and a conflict between two
    sections without facts names both sections."""

    def test_a_sentence_quotes_less_and_keeps_its_pages(self) -> None:
        sentence = str(flag_sentence(LONG_NAME, CONFLICT3, LONG_FACTS, "vi"))
        self.assertLessEqual(len(sentence), 300)
        self.assertEqual(sentence.count("“") - 1, 2)  # the block name and two quotes
        self.assertIn("(trang 12)", sentence)
        self.assertIn("(trang 37)", sentence)

    def test_the_unit_note_keeps_its_other_lines(self) -> None:
        flags = flag_sentences([(f"Phần {n}: {LONG_NAME}", [CONFLICT3]) for n in range(3)], LONG_FACTS, "vi")
        note = build_unit_author_note(
            locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[], fallback_slots=[1], ai_drafted=False,
            slot_types=["html", "problem"], review_reasons=["source_check", "fallback"], source_flags=flags,
            suggested_criteria=True)
        self.assertLessEqual(len(note), 1500)
        self.assertIn("1 khối dùng bản dự phòng dựng từ tài liệu", note)
        self.assertIn("AI gợi ý tiêu chí từ nội dung đã dạy", note)
        self.assertIn("chỗ khác cần kiểm tra", note)

    def test_the_course_note_keeps_its_other_sections(self) -> None:
        checks = flag_sentences([(f"Phần {n}: {LONG_NAME}", [CONFLICT3]) for n in range(30)], LONG_FACTS, "vi")
        removed = [SkippedBlockNote(f"Khối đã loại {n}", "Không phục vụ Must Do.") for n in range(3)]
        note = build_course_notes(locale="vi", ai_proposed=False, total_minutes=30, lesson_count=3, module_count=1,
                                  reference_block_count=0, excluded_block_count=3, fallback_stages=[],
                                  removed=removed, source_checks=checks)
        self.assertIn("• Nội dung đã loại khỏi khoá (Remove) — 3 khối, kèm lý do:", note)
        self.assertEqual(note.count("Khối đã loại"), 3)
        checks_part = note.split("• Chỗ tài liệu cần kiểm tra trước khi áp dụng:\n", 1)[1].split("\n• ")[0]
        self.assertLessEqual(len(checks_part), 1600)

    def test_a_conflict_between_two_sections_names_both(self) -> None:
        note = "Hai mục nêu thời hạn khác nhau."
        conflict = IdmIssueV1(type="conflict", note=note, fact_keys=[])
        sentences = flag_sentences([("Thời hạn phản hồi", [conflict]), ("Escalate khiếu nại", [conflict])], {}, "vi")
        self.assertEqual(sentences, ["Tài liệu ghi khác nhau giữa phần “Thời hạn phản hồi” và phần “Escalate khiếu "
                                     "nại”. Hãy kiểm tra trước khi áp dụng."])
        alone = flag_sentences([("Thời hạn phản hồi", [conflict])], {}, "en")
        self.assertEqual(alone, ["The source differs between “Thời hạn phản hồi” and another part of the source. "
                                 "Check this before applying it."])


THREE = [{"text": "Báo trưởng ca ngay lập tức để xử lý", "correct": False},
         {"text": "Phân loại mức độ khiếu nại trước khi xử lý", "correct": True},
         {"text": "Hứa bồi thường để giữ chân khách hàng", "correct": False}]


class SmallFixTests(unittest.TestCase):
    """m8 / m9 / m10: a fallback reason cut at a word, a row label that kept its line break, the verdicts and the
    empty-praise list."""

    def test_a_fallback_reason_is_cut_at_a_word(self) -> None:
        rows = ["Row 1: Mức độ vi phạm của nhân viên trong quá trình phục vụ khách hàng tại quầy | Hình thức xử lý kỷ "
                "luật áp dụng theo nội quy công ty",
                "Row 2: Vi phạm lần đầu, không gây thiệt hại cho khách hàng và uy tín công ty | Nhắc nhở bằng lời và "
                "ghi nhận vào sổ theo dõi của trưởng ca",
                "Row 3: Vi phạm lần thứ hai trong vòng ba tháng hoặc gây phàn nàn chính thức | Cảnh cáo bằng văn bản "
                "và trừ điểm thi đua của tháng",
                "Row 4: Vi phạm nghiêm trọng gây thiệt hại tài chính hoặc ảnh hưởng uy tín | Đình chỉ công việc và "
                "chuyển hồ sơ cho Ban Giám đốc xem xét",
                "Row 5: Tái phạm nhiều lần sau khi đã bị cảnh cáo bằng văn bản | Chấm dứt hợp đồng lao động theo quy "
                "định của pháp luật"]
        question = idm_source_grounded_single_choice("Kỷ luật", rows, locale="vi")
        assert question is not None
        source_words = {word.strip(",.|") for row in rows for word in row.split()}
        for line in str(question["explanation"]).splitlines()[1:]:
            last = line.rstrip(".").split()[-1]
            self.assertTrue(last == "…" or last.endswith("…") or last in source_words, line)

    def test_a_row_label_keeps_the_line_break(self) -> None:
        fixes: Counter[str] = Counter()
        cleaned = clean_question({"question": "Look at row 3:\n- Group X: 10 units\n- Group Y: 20 units\nWhich group?",
                                  "explanation": "x"}, set(), fixes)
        self.assertEqual(str(cleaned["question"]).splitlines()[1:], ["- Group X: 10 units", "- Group Y: 20 units",
                                                                     "Which group?"])

    def test_negative_verdicts_and_true_filler(self) -> None:
        for wrong in ("Không đúng vì", "Chưa chính xác vì", "Sai, vì"):
            explanation = (f"Nguyên tắc: phân loại trước khi xử lý.\nA - {wrong} chưa phân loại mức độ.\nB - Đúng vì "
                           f"phân loại trước tiên.\nC - {wrong} hứa bồi thường khi chưa phân loại.")
            self.assertEqual(explanation_report({"problem_type": "multiple_choice", "choices": THREE,
                                                 "explanation": explanation}), {}, wrong)
        for wrong in ("Not correct because", "Incorrect:", "Incorrect because"):
            explanation = (f"Principle: classify first.\nA - {wrong} it skips classification.\nB - Correct because it "
                           f"classifies first.\nC - {wrong} it promises compensation.")
            self.assertEqual(explanation_report({"problem_type": "multiple_choice", "choices": THREE,
                                                 "explanation": explanation}), {}, wrong)
        flipped = ("Nguyên tắc: phân loại trước khi xử lý.\nA - Không đúng vì chưa phân loại mức độ.\n"
                   "B - Không đúng vì phân loại trước tiên.\nC - Sai vì hứa bồi thường khi chưa phân loại.")
        self.assertIn(EXPLANATION_VERDICT_CODE, explanation_report({"problem_type": "multiple_choice",
                                                                     "choices": THREE, "explanation": flipped}))
        self.assertEqual(filler_phrases("A - Incorrect because you must tell your superior first."), [])
        self.assertEqual(filler_phrases("A - Sai vì mức xuất sắc chỉ dành cho nhân viên đạt 100% KPI."), [])
        self.assertEqual(filler_phrases("B - Đúng vì đây là lựa chọn dũng cảm."), ["dũng cảm"])
