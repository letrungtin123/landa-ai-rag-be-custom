"""sanitize_source_fact_for_learner in linear time gives byte-for-byte the output of the backtracking version.

``reference_sanitize`` is a verbatim copy of the function at 76a5cbb (security audit 2026-10-24, H2), with the
regexes it ran inline and the source-locator pattern as they were then; the furniture patterns it shares with the
module did not change. Each rewritten pattern is also compared alone with its plain form on raw text (tabs, line
breaks and other white space included), so the proof does not depend on what the steps before it leave.
"""

from __future__ import annotations

import html
import random
import re
import time
import unittest
from typing import Any

from app.content import learner_content_purity as purity
from app.content.instructional_quality import clean_source_facts
from app.content.learner_content_purity import sanitize_source_fact_for_learner

REFERENCE_SOURCE_LOCATOR_RE = re.compile(
    r"\b(?:trang|page|slide|chunk|đoạn\s+nguồn|doan\s+nguon|mục\s+nguồn|muc\s+nguon)"
    r"\s*(?:số|so|number|no\.?|#)?\s*[:#-]?\s*\d{1,6}\b",
    re.IGNORECASE,
)
# (new pattern, the plain pattern it replaces, replacement)
REWRITTEN_STEPS = (
    (purity._SPACE_BEFORE_PUNCTUATION_RE, re.compile(r"\s+([,.;:!?])"), r"\1"),
    (purity._SEPARATOR_RUN_RE, re.compile(r"(?:\s*[,;:|]\s*){2,}"), "; "),
    (purity._TRAILING_DASH_RE, re.compile(r"\s+(?:[-–—]\s*)$"), ""),  # noqa: RUF001 - the glyph is intended
    (purity._EDGE_SEPARATORS_RE, re.compile(r"^(?:\s*[-–—,:;|]\s*)+|(?:\s*[-–—,:;|]\s*)+$"), ""),  # noqa: RUF001
)


def reference_visible_text(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def reference_sanitize(value: Any) -> str:
    text = reference_visible_text(value)
    if not text or purity._ATTRIBUTION_ONLY_RE.fullmatch(text):
        return ""
    match = purity._LEADING_ATTRIBUTION_RE.match(text)
    if match:
        body = match.group("body").strip()
        if not body:
            return ""
        body = body[:1].upper() + body[1:]
        text = f"{match.group('label') or ''}{body}".strip()
    text = purity._WRAPPED_SOURCE_FURNITURE_RE.sub(" ", text)
    text = REFERENCE_SOURCE_LOCATOR_RE.sub(" ", text)
    text = purity._SOURCE_ATTRIBUTION_RE.sub(" ", text)
    text = re.sub(r"[\[(]\s*[\])]", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?:\s*[,;:|]\s*){2,}", "; ", text)
    text = re.sub(r"\s+(?:[-–—]\s*)$", "", text)  # noqa: RUF001 - the glyph is intended
    text = re.sub(r"^(?:\s*[-–—,:;|]\s*)+|(?:\s*[-–—,:;|]\s*)+$", "", text)  # noqa: RUF001 - the glyph is intended
    text = re.sub(r"\s+", " ", text).strip()
    if not text or purity._DANGLING_PROVENANCE_ONLY_RE.fullmatch(text.strip(" .,:;!?-–—")):  # noqa: RUF001
        return ""
    return text


# Words and furniture that reach every step: removed furniture leaves space runs, separators and dashes meet them.
FACT_TOKENS = (
    " ", "  ", "\t", "\n", "-", "–", "—", " - ", " – ", ",", ";", ":", "|", " , ", " ; ", " | ", ".", "!", "?",  # noqa: RUF001
    "[]", "( )", "[ ]", "(trang 1)", "(page 2)", "(slide 3, 4)", "(Nguồn: tài liệu.pdf)", "trang", "trang 5", "page 12",
    "số", "no.", "#", "Trang số 7", "đoạn nguồn 2", "theo tài liệu", "nguồn:", "Nguồn: ", "Theo tài liệu nguồn,",
    "Giải thích: ", "tài liệu nêu rằng", "xem", "tại", "<b>", "</b>", "<br>", "&nbsp;", "&amp;", "a", "b", "Bước",
    "1", "12", "2024", "Quy trình", "kết thúc", "ệ", "Đ", "ữ", "x", "Row 1:", "=", "→", "trang thiết bị",
)
FACT_CHARS = " \t-–—,;:|.!?[]()abĐệ1#"  # noqa: RUF001 - the glyph is intended
RAW_CHARS = " \t\n\x0b  -–—,;:|.!?ab1"  # noqa: RUF001 - the glyph is intended
LOCATOR_TOKENS = ("trang", "Trang", "page", "SLIDE", "chunk", "đoạn nguồn", "muc  nguon", "số", "so", "number", "no.",
                  "no", "#", ":", "-", " ", "  ", "   ", "\t", "\n", "1", "12", "1234567", "x", "a", "_", "(", ")")


def random_fact(generator: random.Random) -> str:
    if generator.random() < 0.5:
        return "".join(generator.choices(FACT_CHARS, k=generator.randint(0, 24)))
    tokens = generator.choices(FACT_TOKENS, k=generator.randint(0, 12))
    return "".join(token + generator.choice(("", "", " ")) for token in tokens)


class SanitizeSourceFactRegexTests(unittest.TestCase):
    def test_random_facts_give_the_reference_output(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        for _ in range(60_000):
            fact = random_fact(generator)
            if sanitize_source_fact_for_learner(fact) != reference_sanitize(fact):
                self.fail(f"sanitize_source_fact_for_learner differs from the reference on {fact!r}")

    def test_known_facts_give_the_reference_output(self) -> None:
        facts = (
            "Bước 1 - - - - kết thúc", "Quy trình -", " - , Quy trình , ; kết thúc | - ", "a [] [] [] , b",
            "Theo tài liệu nguồn, cần đánh giá rủi ro.", "trang (trang 1) (trang 2) x", "Trang 5: Mục tiêu",
            "Nội dung (Nguồn: tài liệu.pdf, trang 3)", "Kế hoạch ,, ; | hành động", "Hạn chót —", "| a | b |",
        )
        for fact in facts:
            with self.subTest(fact=fact):
                self.assertEqual(sanitize_source_fact_for_learner(fact), reference_sanitize(fact))
        self.assertEqual(sanitize_source_fact_for_learner("Bước 1 - - - - kết thúc"), "Bước 1 - - - - kết thúc")
        self.assertEqual(sanitize_source_fact_for_learner("Kế hoạch ,, ; | hành động"), "Kế hoạch; hành động")

    def test_each_rewritten_pattern_matches_its_plain_form_on_raw_text(self) -> None:
        generator = random.Random(1024)  # noqa: S311 - seeded test data, not security
        for new, plain, replacement in REWRITTEN_STEPS:
            for _ in range(50_000):
                text = "".join(generator.choices(RAW_CHARS, k=generator.randint(0, 20)))
                if new.sub(replacement, text) != plain.sub(replacement, text):
                    self.fail(f"{new.pattern!r} differs from {plain.pattern!r} on {text!r}")

    def test_source_locator_matches_its_plain_form(self) -> None:
        generator = random.Random(2048)  # noqa: S311 - seeded test data, not security
        for _ in range(50_000):
            text = "".join(generator.choices(LOCATOR_TOKENS, k=generator.randint(0, 10)))
            new = [match.span() for match in purity._SOURCE_LOCATOR_RE.finditer(text)]
            if new != [match.span() for match in REFERENCE_SOURCE_LOCATOR_RE.finditer(text)]:
                self.fail(f"the source-locator pattern differs from its plain form on {text!r}")

    def test_adversarial_lines_finish_fast(self) -> None:
        # Before: 20 spaced dashes took 0.2 s and each 2 more took 4 times longer (40: days); a 32 KB line of
        # "[]" took about 5 s and one of "(trang 1)" after "trang" several minutes.
        lines = {
            "40 spaced dashes": "Bước 1" + " -" * 40 + " x",
            "2,000 spaced dashes": "Bước 1" + " -" * 2_000 + " x",
            "2,000 dashes with two spaces": "a" + " -  " * 2_000 + "x",
            "32 KB of removed brackets": "a" + " []" * 10_900 + " b",
            "32 KB of locators after trang": "trang" + " (trang 1)" * 3_270 + " x",
            "32 KB of spaced pipes": "a" + " |" * 16_000,
        }
        for name, line in lines.items():
            with self.subTest(name):
                started = time.perf_counter()
                sanitize_source_fact_for_learner(line)
                clean_source_facts([line])
                self.assertLess(time.perf_counter() - started, 0.05)


if __name__ == "__main__":
    unittest.main()
