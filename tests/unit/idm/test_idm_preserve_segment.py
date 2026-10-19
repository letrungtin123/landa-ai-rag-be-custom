"""QLT-6 preserve_source: the segmentation call, its validation, the deterministic segmentation and the units."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from app.idm.policy import (
    IDM_PRESERVE_MAX_UNITS_PER_LESSON,
    IDM_PRESERVE_SEGMENT_MAX_OUTPUT_TOKENS,
    IDM_PRESERVE_TITLE_MAX_CHARS,
)
from app.idm.preserve_outline import (
    SEGMENTATION_INVALID_CODE,
    TOO_LARGE_CODE,
    build_outline,
    suffixed,
)
from app.idm.preserve_segment import (
    ChapterRange,
    IdmPreserveSegmentationV1,
    LessonRange,
    fallback_segmentation,
    ranges_of,
    segment_document,
    segment_prompt,
    validate_segmentation,
)
from app.idm.runtime import IdmProviderError, IdmStageError
from tests import idm_golden as g
from tests.golden import idm_preserve_golden as pg
from tests.support.idm_test_support import _run_sync, make_runtime

TEXTS = pg.content_texts()
PAGES: list[int | None] = [fact.source_page for fact in pg.source_facts()
                           if fact.fact_key not in {*pg.PAGE_NUMBER_KEYS, *pg.REPEATED_HEADER_KEYS}]


def answer(change: Any = None) -> IdmPreserveSegmentationV1:
    payload = copy.deepcopy(pg.SEGMENTATION)
    if change is not None:
        change(payload)
    return IdmPreserveSegmentationV1.model_validate(payload)


def codes(change: Any) -> list[str]:
    return [issue.code for issue in validate_segmentation(answer(change), TEXTS)]


def segment(provider: g.FakeIdmProvider, texts: list[str] = TEXTS, pages: list[int | None] = PAGES) -> Any:
    return _run_sync(lambda: segment_document(make_runtime(provider), texts, pages))


def lesson(chapter: int, index: int) -> Any:
    def pick(payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = payload["chapters"][chapter]["lessons"][index]
        return result
    return pick


class PromptTests(unittest.TestCase):
    def test_lines_are_numbered_with_hints_and_data_is_fenced(self) -> None:
        prompt = segment_prompt(TEXTS)
        self.assertIn("1|H|KHOÁ HỌC: KỸ NĂNG PHẢN HỒI TRONG CÔNG VIỆC", prompt)
        self.assertIn("3|H1|MODUN 1: VÌ SAO PHẢN HỒI QUAN TRỌNG", prompt)
        self.assertIn("4|H2|Bài 1: Phản hồi là gì", prompt)
        self.assertIn("22|H2|3.1 Ví dụ áp dụng", prompt)
        self.assertIn("lines 1 to\n27", prompt.replace(" 27", "\n27"))
        self.assertIn("<SOURCE_LINES>", prompt)
        self.assertIn("You write NO\ntext", prompt.replace("NO text", "NO\ntext"))
        long_line = "Một câu rất dài " * 30
        self.assertIn("…", segment_prompt([long_line]))
        self.assertNotIn(long_line.strip(), segment_prompt([long_line]))


class ValidationTests(unittest.TestCase):
    def test_the_golden_answer_is_valid(self) -> None:
        self.assertEqual(validate_segmentation(answer(), TEXTS), [])
        chapters = ranges_of(answer())
        self.assertEqual([chapter.title for chapter in chapters], [3, 16])
        self.assertEqual(chapters[1].lessons[0], LessonRange(17, 16, 23, (22,)))

    def test_ranges_must_cover_every_line_once_in_order(self) -> None:
        def gap(payload: dict[str, Any]) -> None:
            lesson(0, 1)(payload)["first_ordinal"] = 11
        self.assertIn("IDM_PRESERVE_RANGE_NOT_CONTIGUOUS", codes(gap))

        def overlap(payload: dict[str, Any]) -> None:
            lesson(0, 0)(payload)["last_ordinal"] = 10
        self.assertIn("IDM_PRESERVE_RANGE_NOT_CONTIGUOUS", codes(overlap))

        def short(payload: dict[str, Any]) -> None:
            lesson(1, 1)(payload)["last_ordinal"] = 26
            payload["chapters"][1]["last_ordinal"] = 26
        self.assertEqual(codes(short), ["IDM_PRESERVE_RANGE_NOT_COVERED"])

        def beyond(payload: dict[str, Any]) -> None:
            lesson(1, 1)(payload)["last_ordinal"] = 40
        self.assertEqual(codes(beyond), ["IDM_PRESERVE_RANGE_INVALID"])

        def reordered(payload: dict[str, Any]) -> None:
            payload["chapters"].reverse()
        self.assertIn("IDM_PRESERVE_RANGE_NOT_CONTIGUOUS", codes(reordered))

    def test_titles_must_be_heading_lines_of_their_range(self) -> None:
        def outside(payload: dict[str, Any]) -> None:
            lesson(0, 1)(payload)["title_ordinal"] = 4
        self.assertIn("IDM_PRESERVE_TITLE_OUTSIDE_RANGE", codes(outside))

        def deep(payload: dict[str, Any]) -> None:
            lesson(1, 1)(payload)["title_ordinal"] = 27
        self.assertIn("IDM_PRESERVE_TITLE_OUTSIDE_RANGE", codes(deep))

        def sentence(payload: dict[str, Any]) -> None:
            lesson(0, 1)(payload)["title_ordinal"] = 12  # a table row
        self.assertIn("IDM_PRESERVE_TITLE_NOT_HEADING", codes(sentence))

        def chapter_after_lesson(payload: dict[str, Any]) -> None:
            payload["chapters"][0]["title_ordinal"] = 5
        self.assertIn("IDM_PRESERVE_CHAPTER_TITLE_INVALID", codes(chapter_after_lesson))

        def chapter_range(payload: dict[str, Any]) -> None:
            payload["chapters"][0]["last_ordinal"] = 14
        self.assertIn("IDM_PRESERVE_CHAPTER_RANGE_INVALID", codes(chapter_range))

    def test_unit_titles_follow_the_lesson_title_inside_the_lesson(self) -> None:
        for bad in ([17], [24], [22, 22], [19]):
            def units(payload: dict[str, Any], value: list[int] = bad) -> None:
                lesson(1, 0)(payload)["unit_title_ordinals"] = value
            self.assertIn("IDM_PRESERVE_UNIT_TITLE_INVALID", codes(units), bad)

    def test_out_of_range_numbers_never_read_past_the_document(self) -> None:
        def wild(payload: dict[str, Any]) -> None:
            lesson(0, 0)(payload)["title_ordinal"] = 999
            payload["chapters"][0]["title_ordinal"] = 500
            lesson(0, 1)(payload)["unit_title_ordinals"] = [700]
        found = codes(wild)
        self.assertIn("IDM_PRESERVE_TITLE_OUTSIDE_RANGE", found)
        self.assertIn("IDM_PRESERVE_CHAPTER_TITLE_INVALID", found)
        self.assertIn("IDM_PRESERVE_UNIT_TITLE_INVALID", found)

    def test_the_first_lesson_of_the_course_may_start_with_a_preface(self) -> None:
        self.assertEqual(validate_segmentation(answer(), TEXTS), [])  # lines 1-2 precede "MODUN 1" in lesson 1


class SegmentCallTests(unittest.TestCase):
    def test_one_low_thinking_call_answers_line_numbers(self) -> None:
        provider = pg.preserve_provider()
        result = segment(provider)
        self.assertEqual((result.origin, result.fallback), ("provider", None))
        self.assertEqual(len(provider.calls), 1)
        call = provider.calls[0]
        self.assertEqual((call["thinking_level"], call["max_output_tokens"]),
                         ("low", IDM_PRESERVE_SEGMENT_MAX_OUTPUT_TOKENS))
        self.assertEqual(result.chapters, ranges_of(answer()))

    def test_an_invalid_answer_gets_one_repair(self) -> None:
        broken = copy.deepcopy(pg.SEGMENTATION)
        broken["chapters"][0]["lessons"][1]["first_ordinal"] = 11
        provider = pg.preserve_provider(broken, pg.SEGMENTATION)
        result = segment(provider)
        self.assertEqual(result.origin, "provider")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("IDM_PRESERVE_RANGE_NOT_CONTIGUOUS", provider.calls[1]["prompt"])
        self.assertEqual(result.codes["IDM_PRESERVE_RANGE_NOT_CONTIGUOUS"], 1)

    def test_two_invalid_answers_fall_back_to_the_document_headings(self) -> None:
        broken = copy.deepcopy(pg.SEGMENTATION)
        broken["chapters"][0]["title_ordinal"] = 12
        result = segment(pg.preserve_provider(broken, "not json"))
        self.assertEqual((result.origin, result.fallback), ("deterministic_fallback", "headings"))
        self.assertEqual([chapter.title for chapter in result.chapters], [3, 16])
        self.assertEqual([[item.title for item in chapter.lessons] for chapter in result.chapters], [[4, 10], [17, 24]])

    def test_a_transient_provider_failure_falls_back_and_an_exhausted_key_stops(self) -> None:
        result = segment(pg.preserve_provider(IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False)))
        self.assertEqual(result.origin, "deterministic_fallback")
        with self.assertRaises(IdmProviderError) as raised:
            segment(pg.preserve_provider(IdmProviderError("AI_PROVIDER_QUOTA_EXHAUSTED", terminal=True)))
        self.assertEqual(raised.exception.code, "AI_PROVIDER_QUOTA_EXHAUSTED")

    def test_no_budget_left_falls_back_without_a_call(self) -> None:
        provider = pg.preserve_provider()
        result = _run_sync(lambda: segment_document(make_runtime(provider, seconds=1.0), TEXTS, PAGES))
        self.assertEqual((result.origin, provider.calls), ("deterministic_fallback", []))


class FallbackTests(unittest.TestCase):
    def test_single_level_chapter_markers_make_one_lesson_each(self) -> None:
        texts = ["Chương 1: Mở đầu", "Nội dung một.", "Chương 2: Thân bài", "Nội dung hai."]
        chapters, kind = fallback_segmentation(texts, [None] * 4, "vi")
        self.assertEqual(kind, "headings")
        self.assertEqual(chapters, (ChapterRange(1, (LessonRange(1, 1, 2),)), ChapterRange(3, (LessonRange(3, 3, 4),))))

    def test_single_level_lesson_markers_share_the_document_title(self) -> None:
        texts = ["Sổ tay bán hàng", "Lời mở đầu.", "Bài 1: Chào hỏi", "Nội dung.", "Bài 2: Chốt đơn", "Nội dung."]
        chapters, _kind = fallback_segmentation(texts, [None] * 6, "vi")
        self.assertEqual(chapters, (ChapterRange(1, (LessonRange(3, 1, 4), LessonRange(5, 5, 6))),))

    def test_numbering_levels_without_markers(self) -> None:
        texts = ["1. Giới thiệu", "Nội dung.", "1.1 Khái niệm", "Nội dung.", "2. Áp dụng", "2.1 Bước đầu", "Nội dung."]
        chapters, kind = fallback_segmentation(texts, [None] * 7, "en")
        self.assertEqual(kind, "headings")
        self.assertEqual(chapters, (ChapterRange(1, (LessonRange(3, 1, 4),)), ChapterRange(5, (LessonRange(6, 5, 7),))))

    def test_three_levels_make_units(self) -> None:
        texts = ["Chương 1", "Bài 1", "Mục 1", "Nội dung.", "Mục 2", "Nội dung."]
        chapters, _kind = fallback_segmentation(texts, [None] * 6, "vi")
        self.assertEqual(chapters, (ChapterRange(1, (LessonRange(2, 1, 6, (3, 5)),)),))

    def test_list_numbering_alone_and_no_headings_split_by_pages_and_size(self) -> None:
        sentence = "Một câu nội dung có mười từ để đo độ dài bài."
        texts = ["1. Bước một", "2. Bước hai", *[sentence] * 200, *[f"{sentence} Trang hai." for _ in range(200)]]
        pages: list[int | None] = [1] * 202 + [2] * 200
        chapters, kind = fallback_segmentation(texts, pages, "vi")
        self.assertEqual(kind, "size")
        lessons = chapters[0].lessons
        self.assertEqual([(item.first, item.last) for item in lessons], [(1, 202), (203, 402)])
        self.assertEqual((lessons[0].title, lessons[0].label), (1, None))
        self.assertEqual((lessons[1].title, lessons[1].label), (0, "Phần 2"))

    def test_more_chapter_markers_than_chapters_become_lessons(self) -> None:
        texts = [line for number in range(1, 31) for line in (f"Tuần {number}", "Nội dung.")]
        chapters, _kind = fallback_segmentation(texts, [None] * len(texts), "vi")
        self.assertEqual(len(chapters), 1)
        self.assertEqual(len(chapters[0].lessons), 30)


class OutlineTests(unittest.TestCase):
    def test_golden_units_follow_the_lesson_and_its_sub_heading(self) -> None:
        outline = build_outline(ranges_of(answer()), TEXTS, "vi")
        units = [(unit.title, unit.lines) for chapter in outline.chapters for item in chapter.lessons
                 for unit in item.units]
        self.assertEqual(units, [("Bài 1: Phản hồi là gì", tuple(range(1, 10))),
                                 ("Bài 2: Lợi ích của phản hồi", tuple(range(10, 16))),
                                 ("Bài 3: Mô hình SBI", tuple(range(16, 22))), ("3.1 Ví dụ áp dụng", (22, 23)),
                                 ("Bài 4: Những lỗi thường gặp", tuple(range(24, 28)))])
        self.assertEqual([chapter.title for chapter in outline.chapters],
                         ["MODUN 1: VÌ SAO PHẢN HỒI QUAN TRỌNG", "MODUN 2: CÁCH ĐƯA RA PHẢN HỒI"])

    def test_a_long_lesson_continues_in_units_at_line_boundaries(self) -> None:
        paragraph = " ".join(["từ"] * 100) + "."
        texts = ["Bài 1: Dài", *[paragraph] * 12]
        chapters = (ChapterRange(1, (LessonRange(1, 1, len(texts)),)),)
        units = build_outline(chapters, texts, "vi").chapters[0].lessons[0].units
        self.assertEqual([unit.title for unit in units], ["Bài 1: Dài", "Bài 1: Dài (tiếp theo)",
                                                          "Bài 1: Dài (tiếp theo)"])
        self.assertEqual([line for unit in units for line in unit.lines], list(range(1, len(texts) + 1)))
        english = build_outline(chapters, texts, "en").chapters[0].lessons[0].units
        self.assertEqual(english[1].title, "Bài 1: Dài (continued)")

    def test_a_continued_unit_is_titled_by_its_sub_heading(self) -> None:
        paragraph = " ".join(["từ"] * 100) + "."
        texts = ["Bài 1: Dài", *[paragraph] * 5, "1.1 Mục con", *[paragraph] * 9]
        chapters = (ChapterRange(1, (LessonRange(1, 1, len(texts)),)),)
        units = build_outline(chapters, texts, "vi").chapters[0].lessons[0].units
        self.assertEqual(units[0].lines[-1], 6, "the sub-heading is not left at the end of a unit")
        self.assertEqual(units[1].lines[0], 7)
        self.assertEqual([unit.title for unit in units], ["Bài 1: Dài", "1.1 Mục con", "1.1 Mục con (tiếp theo)"])

    def test_a_lesson_too_long_for_one_lesson_continues_as_parts(self) -> None:
        paragraph = " ".join(["từ"] * 400) + "."
        texts = ["Bài 1: Rất dài", *[paragraph] * (IDM_PRESERVE_MAX_UNITS_PER_LESSON * 5)]
        chapters = (ChapterRange(1, (LessonRange(1, 1, len(texts)),)),)
        outline = build_outline(chapters, texts, "vi")
        lessons = outline.chapters[0].lessons
        self.assertEqual([item.title for item in lessons], ["Bài 1: Rất dài", "Bài 1: Rất dài (phần 2)"])
        self.assertTrue(all(len(item.units) <= IDM_PRESERVE_MAX_UNITS_PER_LESSON for item in lessons))
        self.assertEqual(outline.split_lessons, 1)

    def test_a_chapter_with_too_many_lessons_continues_as_parts(self) -> None:
        texts = [line for number in range(1, 33) for line in (f"Bài {number}", "Nội dung.")]
        lessons = tuple(LessonRange(index * 2 + 1, index * 2 + 1, index * 2 + 2) for index in range(32))
        outline = build_outline((ChapterRange(1, lessons),), texts, "en")
        self.assertEqual([chapter.title for chapter in outline.chapters], ["Bài 1", "Bài 1 (part 2)"])
        self.assertEqual(outline.split_chapters, 1)

    def test_bounds_and_coverage_are_enforced(self) -> None:
        texts = [line for number in range(1, 26) for line in (f"Chương {number}", "Nội dung.")]
        chapters = tuple(ChapterRange(index * 2 + 1, (LessonRange(index * 2 + 1, index * 2 + 1, index * 2 + 2),))
                         for index in range(25))
        with self.assertRaises(IdmStageError) as raised:
            build_outline(chapters, texts, "vi")
        self.assertEqual(raised.exception.code, TOO_LARGE_CODE)
        with self.assertRaises(IdmStageError) as raised:
            build_outline((ChapterRange(1, (LessonRange(1, 1, 3),)),), texts, "vi")
        self.assertEqual(raised.exception.code, SEGMENTATION_INVALID_CODE)
        with self.assertRaises(IdmStageError):
            build_outline((ChapterRange(0, (LessonRange(0, 1, len(texts)),)),), texts, "vi")

    def test_label_titles_and_suffix_bounds(self) -> None:
        chapters = (ChapterRange(0, (LessonRange(0, 1, 1, (), "Phần 1"),), "Phần 1"),)
        outline = build_outline(chapters, ["một dòng văn bản rất dài không phải tiêu đề " * 6], "vi")
        self.assertEqual((outline.chapters[0].title, outline.label_titles), ("Phần 1", 2))
        long_title = "Tiêu đề " * 30
        self.assertLessEqual(len(suffixed(long_title.strip(), " (tiếp theo)")), IDM_PRESERVE_TITLE_MAX_CHARS)
        self.assertTrue(suffixed(long_title.strip(), " (tiếp theo)").endswith(" (tiếp theo)"))

    def test_logged_answer_shapes_are_counts_only(self) -> None:
        from app.idm.preserve_segment import segmentation_log  # noqa: PLC0415 - one helper for this test

        logged = json.loads(segmentation_log(segment(pg.preserve_provider())))
        self.assertEqual(logged, {"chapters": 2, "codes": {}, "fallback": None, "lessons": 4, "origin": "provider"})


if __name__ == "__main__":
    unittest.main()
