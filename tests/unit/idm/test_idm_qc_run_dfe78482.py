"""Owner-approved fixes from the real AI ID run dfe78482 (AI d812b90, 2026-10-22).

The fixture ``tests/fixtures/qc_run_dfe78482.json`` holds the AI-written cases of that run: the quiz of
chapter_3.lesson_1.unit_3 (stem items labelled "Phương án A … D"), the three stems that were marked "Cần xem" only for
their length, and the AI text of the four media briefs before the server's completion. Every provider is a local fake.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.mcq import normalize_single_choice
from app.idm.media_detail import complete_media_brief, media_brief_gaps
from app.idm.quiz_text import (
    QUESTION_LENGTH_CODE,
    QUESTION_LIMIT_CODE,
    QUIZ_STYLE_CODES,
    lay_out_quiz,
    number_item_letters,
    question_report,
    utf16_length,
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

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "qc_run_dfe78482.json"
RUN = json.loads(FIXTURE.read_text(encoding="utf-8"))
LETTERED = "chapter_3.lesson_1.unit_3.component_2"
LETTER_LABEL = re.compile(r"(?i)phương án [A-H](?!\w)")


def problem(path: str) -> dict[str, Any]:
    case = RUN["problems"][path]
    return {"type": "problem", "problem_type": "multiple_choice", "question": case["question"],
            "choices": [dict(choice) for choice in case["choices"]], "explanation": case["explanation"]}


class LetteredItemTests(unittest.TestCase):
    """Fix 1: stem items labelled with option letters read as answer letters after the shuffle ("A: Phương án C: …");
    the server numbers them in the stem, the choices and the explanation, never touching the "A - " prefixes."""

    def test_the_real_stem_gets_numbered_items_everywhere(self) -> None:
        component = problem(LETTERED)
        numbered, count = number_item_letters(component)
        self.assertGreater(count, 0)
        for seed, ordinal in (("seed-x", 0), ("seed-y", 2), ("seed-z", 3)):
            shuffled, _codes = normalize_single_choice(numbered, seed, ordinal)
            laid, _laid = lay_out_quiz(shuffled)
            texts = [laid["question"], *(choice["text"] for choice in laid["choices"]), laid["explanation"]]
            self.assertFalse(any(LETTER_LABEL.search(text) for text in texts), texts)
            stem = str(laid["question"]).splitlines()
            self.assertEqual([line[:14] for line in stem[1:5]],
                             ["- Phương án 1:", "- Phương án 2:", "- Phương án 3:", "- Phương án 4:"])
            key = next(choice["text"] for choice in laid["choices"] if choice["correct"])
            self.assertTrue(key.startswith("Phương án 2: Thiết kế tài liệu tóm tắt"), key)
            lines = str(laid["explanation"]).splitlines()
            self.assertEqual([line[:4] for line in lines[1:]], ["A - ", "B - ", "C - ", "D - "])
            letter = "ABCD"[[choice["correct"] for choice in laid["choices"]].index(True)]
            self.assertTrue(lines[1 + "ABCD".index(letter)].startswith(f"{letter} - Đúng vì"))
        # Nothing but the item letters changes.
        before = re.sub(r"(?i)(phương án) [A-D](?!\w)", r"\1 #", json.dumps(component, ensure_ascii=False))
        after = re.sub(r"(?i)(phương án) [1-4](?!\w)", r"\1 #", json.dumps(numbered, ensure_ascii=False))
        self.assertEqual(before, after)

    def test_a_stem_without_letter_labels_is_left_alone(self) -> None:
        component = problem("chapter_3.lesson_2.unit_1.component_2")
        self.assertEqual(number_item_letters(component), (component, 0))
        mention = {**problem(LETTERED), "question": "Bốn phương án được nêu. Phương án nào đúng?"}
        self.assertEqual(number_item_letters(mention)[1], 0)


LONG_READABLE = ("chapter_2.lesson_2.unit_3.component_2", LETTERED, "chapter_3.lesson_2.unit_1.component_2")
# A stem whose context runs long (90 words of context and question), under Node's 1000 characters.
WORDY_STEM = ("Tại một doanh nghiệp, khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ ba ngày so với "
              "cam kết ban đầu, chưa có thiệt hại tài chính nào được ghi nhận nhưng khách hàng tỏ ra rất bực bội và "
              "yêu cầu được gặp người phụ trách ngay trong buổi sáng hôm nay để giải quyết dứt điểm sự việc trước khi "
              "họ chuyển sang dùng dịch vụ của một nhà cung cấp khác trong khu vực. Theo bảng cấp độ đã học trong bài "
              "này, khiếu nại này thuộc cấp độ nào?")


class StemLengthTests(StoryboardEndpointTestCase):
    """Fix 2: a stem over the readability cap is an edit note, never "Cần xem"; a short context, a list of 4-5 scenario
    items and the question are a normal stem; only Node's 1000 UTF-16 units stay blocking."""

    def test_the_three_readable_list_stems_are_normal(self) -> None:
        for path in LONG_READABLE:
            question = problem(path)["question"]
            self.assertGreater(len(question.split()), 120, path)
            self.assertEqual(question_report({"question": question}), {}, path)

    def test_the_hard_limit_is_its_own_blocking_finding(self) -> None:
        over = "Khách hàng phàn nàn. " * 52 + "Khiếu nại thuộc cấp nào?"
        self.assertGreater(utf16_length(over), 1000)
        report = question_report({"question": over})
        self.assertIn(QUESTION_LIMIT_CODE, report)
        self.assertNotIn(QUESTION_LIMIT_CODE, QUIZ_STYLE_CODES)
        self.assertIn(QUESTION_LENGTH_CODE, QUIZ_STYLE_CODES)
        self.assertIn(QUESTION_LENGTH_CODE, question_report({"question": WORDY_STEM}))

    async def test_a_wordy_stem_is_an_edit_note_without_a_review_mark(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        writer["components"]["c1"]["question"] = WORDY_STEM
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(data["unit"]["components"][1]["question"].split("\n")[-1],
                         "Theo bảng cấp độ đã học trong bài này, khiếu nại này thuộc cấp độ nào?")
        self.assertEqual(data["quality_state"], "validated")
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 2 (Quiz): đề câu hỏi dài", note)
        self.assertNotIn("Cần tác giả xem lại", note)


THIN_POINTS = ["Cấp 1: ảnh hưởng thấp, nhân viên tự xử lý", "Cấp 2: thiệt hại dưới 50 triệu đồng, báo trưởng nhóm"]
LONG_POINT = ("Cấp 3: khiếu nại liên quan an toàn, pháp lý hoặc truyền thông phải được escalate ngay cho quản lý và bộ "
              "phận Pháp chế theo quy trình CR-04 trong cùng ngày tiếp nhận")


def brief(case: dict[str, Any], **changes: Any) -> ArchitectureMediaBriefV2:
    fields = {key: case[key] for key in ("type", "title", "content_points", "context_description", "rationale")}
    return ArchitectureMediaBriefV2.model_validate({**fields, **changes})


class MediaBriefCompletionTests(unittest.TestCase):
    """Fix 3: the completion ran on a detailed AI brief ("Nửa dưới gồm 4 cột … Khung 1 đến Khung 4") and appended "5
    khung" with cut quotes. Only a really thin brief is completed, never with a numbering of its own when the AI
    numbered its frames or scenes, and a point is quoted whole or not at all."""

    def test_the_four_briefs_of_the_run_stay_as_the_ai_wrote_them(self) -> None:
        for case in RUN["media_briefs"]:
            original = brief(case)
            self.assertEqual(media_brief_gaps(original), [], case["lesson_key"])
            self.assertIs(complete_media_brief(original, "vi"), original)

    def test_a_thin_brief_quotes_whole_short_points_only(self) -> None:
        case = RUN["media_briefs"][0]
        thin = brief(case, context_description="Một bảng kiểm treo tại phòng họp.",
                     content_points=[*THIN_POINTS, LONG_POINT])
        completed = complete_media_brief(thin, "vi")
        assert completed is not None
        text = completed.context_description
        self.assertIn(f"Khung 1: “{THIN_POINTS[0]}”", text)
        self.assertIn(f"Khung 2: “{THIN_POINTS[1]}”", text)
        self.assertIn("Khung 3: ý 3 của nội dung", text)
        self.assertNotIn("…", text)
        self.assertLessEqual(len(text), 1000)

    def test_a_short_brief_that_numbers_its_frames_gets_no_second_numbering(self) -> None:
        case = RUN["media_briefs"][0]
        numbered = brief(case, context_description="Khung 1 ghi Cấp 1. Khung 2 ghi Cấp 2.",
                         content_points=THIN_POINTS)
        self.assertEqual(media_brief_gaps(numbered), [])
        self.assertIs(complete_media_brief(numbered, "vi"), numbered)
