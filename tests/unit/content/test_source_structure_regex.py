"""Table-of-contents and chapter lines are read in linear time with the result of the backtracking patterns.

``REFERENCE_CHAPTER_HEADING_RE`` is the chapter pattern as it was at 76a5cbb; ``TOC_LINE_RE`` is unchanged and is
the reference of ``toc_line_parts`` (security audit 2026-10-24, follow-up of H2: every line of an uploaded document
goes through both).
"""

from __future__ import annotations

import random
import re
import time
import unittest
from types import SimpleNamespace
from typing import Any

from app.content import source_structure
from app.content.source_structure import CHAPTER_HEADING_RE, TOC_LINE_RE, analyze_source_structure, toc_line_parts

REFERENCE_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chương|chuong|chapter|phần|phan|part)\s+([0-9IVXLCDM]+)\s*[:.)-]?\s*(.*?)\s*$",
    re.IGNORECASE | re.UNICODE,
)
EM = "\N{EM SPACE}"  # white space that _clean_line keeps
ARABIC_ONE, ARABIC_TWO = "\N{ARABIC-INDIC DIGIT ONE}", "\N{ARABIC-INDIC DIGIT TWO}"
TOC_ALPHABET = ". . .  " + EM + "\t\n1234" + ARABIC_ONE + "ab\N{LATIN CAPITAL LETTER D WITH STROKE}x:)-"
CHAPTER_TOKENS = ("chương", "Chuong", "CHAPTER", "phần", "part", " ", "  ", EM, "\t", "\n", "1", "IV", "x", ":",
                  ".", ")", "-", "abc", "Đ")


def reference_toc(line: str) -> tuple[str | Any, ...] | None:
    match = TOC_LINE_RE.match(line)
    return match.groups() if match else None


class TocLineTests(unittest.TestCase):
    def test_hand_cases_give_the_pattern_groups(self) -> None:
        lines = ("Mục 1 ........ 12", "Mục 1 12", "Mục 1  12", f"Mục 1{EM}{EM} 7 ", "..5", "a...5", "a. ..5",
                 "   12", EM * 3 + "12", "Chương 1 .... 12345", "Tiêu đề\n..  3", "\n  .. 5", " x .. 1\n",
                 "Muc luc", "", "1", "a.." + ARABIC_ONE + ARABIC_TWO)
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(toc_line_parts(line), reference_toc(line))
        self.assertEqual(toc_line_parts("Mục 1 ........ 12"), ("Mục 1 ", "12"))
        self.assertEqual(toc_line_parts("   12"), (" ", "12"))

    def test_random_lines_give_the_pattern_groups(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        for _ in range(60_000):
            line = "".join(generator.choices(TOC_ALPHABET, k=generator.randint(0, 22)))
            if toc_line_parts(line) != reference_toc(line):
                self.fail(f"toc_line_parts differs from TOC_LINE_RE on {line!r}")

    def test_cleaned_lines_give_the_pattern_groups(self) -> None:
        generator = random.Random(4096)  # noqa: S311 - seeded test data, not security
        for _ in range(20_000):
            raw = "".join(generator.choices(TOC_ALPHABET + "•_", k=generator.randint(0, 22)))
            line = source_structure._clean_line(raw)
            self.assertEqual(toc_line_parts(line), reference_toc(line), line)


class ChapterHeadingTests(unittest.TestCase):
    def test_random_lines_give_the_reference_match(self) -> None:
        generator = random.Random(8192)  # noqa: S311 - seeded test data, not security
        for _ in range(60_000):
            line = "".join(generator.choices(CHAPTER_TOKENS, k=generator.randint(0, 9)))
            new, old = CHAPTER_HEADING_RE.match(line), REFERENCE_CHAPTER_HEADING_RE.match(line)
            if (new and (new.span(), new.groups())) != (old and (old.span(), old.groups())):
                self.fail(f"CHAPTER_HEADING_RE differs from its reference on {line!r}")

    def test_title_is_the_line_up_to_its_last_character(self) -> None:
        match = CHAPTER_HEADING_RE.match(f"Chương 2: Quản lý  rủi ro {EM} ")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.groups(), ("Chương", "2", "Quản lý  rủi ro"))


class LongLineTimingTests(unittest.TestCase):
    def test_megabyte_lines_finish_fast(self) -> None:
        # Before: a line of 8,000 dots took 0.5 s (quadratic) and 2,000 EM SPACE after a chapter title 11 s (cubic).
        lines = {
            "dots": "Muc 1" + "." * 1_000_000 + "x",
            "EM SPACE after a chapter title": "chương 1: abc" + EM * 1_000_000 + "x",
            "EM SPACE before a page number": "Muc 1" + EM * 1_000_000 + "12",
            "EM SPACE and dots": "abc" + (EM + ".") * 500_000 + "x",
        }
        for name, line in lines.items():
            with self.subTest(name):
                cleaned = source_structure._clean_line(line)
                started = time.perf_counter()
                toc_line_parts(cleaned)
                CHAPTER_HEADING_RE.match(cleaned)
                self.assertLess(time.perf_counter() - started, 0.5)
                started = time.perf_counter()
                analyze_source_structure([SimpleNamespace(text="1. Mục lục\n2. Hai\n" + line, page=1, section=None,
                                                          metadata={})])
                self.assertLess(time.perf_counter() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
