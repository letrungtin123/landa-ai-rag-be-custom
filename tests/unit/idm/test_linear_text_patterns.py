"""Super-linear regexes on AI output and document text, replaced by linear code with the same results (wave 3 of the
hardening, after the regex scan of 2026-10-24).

Each reference below is the regex as it was, kept verbatim (characters a reader could confuse are spelled with chr):
50,000 random inputs built from pieces that exercise its edge cases must give the same result, and the input that
made it quadratic must now take milliseconds at 1 MB.
"""

from __future__ import annotations

import random
import re
import time
import unittest
from collections.abc import Callable
from typing import Any

from app import lesson_content_observation
from app.content import provider_json, source_structure
from app.idm import (
    acronyms,
    answer_checks,
    example_roles,
    faq_checks,
    mcq,
    media_detail,
    preserve_lines,
    quiz_letters,
    quiz_text,
    worksheet,
    worksheet_input,
)
from app.idm.text import visible_end, visible_start
from app.services.retrieval import source_coverage
from app.services.staged_unit import validation

FUZZ_CASES = 50_000
MAX_SECONDS = 0.05
MEGABYTE = 1_000_000
EN_DASH, EM_DASH, ELLIPSIS, MIDDLE_DOT = chr(0x2013), chr(0x2014), chr(0x2026), chr(0xB7)
LEFT_QUOTE, RIGHT_QUOTE, LEFT_GUILLEMET, RIGHT_GUILLEMET = chr(0x201C), chr(0x201D), chr(0xAB), chr(0xBB)
APOSTROPHE, NO_BREAK_SPACE, IDEOGRAPHIC_SPACE, BULLET = chr(0x2019), chr(0xA0), chr(0x3000), chr(0x2022)
TIMES, TAB, VERTICAL_TAB, LONG_S, KELVIN = chr(0xD7), "\t", chr(0x0B), chr(0x17F), chr(0x212A)
SPACES = (" ", " ", "  ", TAB, "\n", NO_BREAK_SPACE, IDEOGRAPHIC_SPACE, VERTICAL_TAB)


# --- references: the regexes before the change ---------------------------------------------------------------
def old_dashed(text: str, token: str) -> list[str]:
    dashed = (rf"([^\W\d_][\w'{APOSTROPHE}]*(?:\s+[^\W\d_][\w'{APOSTROPHE}&/]*)+)\s+[-{EN_DASH}{EM_DASH}]\s+"
              rf"{re.escape(token)}(?![\w&])")
    return [match.group(1) for match in re.finditer(dashed, text)]


OLD_TOC_LEADER = re.compile(rf"(?:\.{{3,}}|{ELLIPSIS}{{2,}}|{MIDDLE_DOT}{{3,}}|_{{3,}})\s*\d{{1,4}}\s*$")
OLD_TRAILING_ELLIPSIS = re.compile(rf"\s*(?:\.{{2,}}|{ELLIPSIS})\s*$")
OLD_CELL_SPLIT = re.compile(r"\s*[;|]\s*")
OLD_HEADING_LABEL = re.compile(
    rf"\s*(?:[(\[]\s*{worksheet._LABEL}\s*[)\]]|[{worksheet._DASHES}:]\s*{worksheet._LABEL})\s*$", re.IGNORECASE)
OLD_WHEN_VI = re.compile(r"(?<!\w)khi(?!\s+nào)(?!\w)[^?]*?(?<!\w)thì(?!\w)")
OLD_WHEN_FOLDED = re.compile(r"(?<!\w)khi(?!\s+nao)(?!\w)[^?]*?(?<!\w)thi(?!\w)")
OLD_WHEN_EN = re.compile(r"(?<!\w)when(?!\w)[^?,]+,")
OLD_INLINE_MARKER = re.compile(rf"[ \t]+[-{EN_DASH}{BULLET}][ \t]+(?=\S)")
OLD_SPEECH = re.compile(r"(?<!\w)(?<![Cc]âu )(?i:hỏi|nói|thắc mắc|than|phàn nàn|chia sẻ|nhắn|viết|asks?|asked"
                        r"|says?|said|writes?|wrote)(?!\w)[ \t]*:[^?\n]*\?")
OLD_QUOTED = re.compile(f'"[^"]*"|{LEFT_QUOTE}[^{RIGHT_QUOTE}]*{RIGHT_QUOTE}|{LEFT_GUILLEMET}[^{RIGHT_GUILLEMET}]*'
                        f"{RIGHT_GUILLEMET}")
OLD_WORD_BEFORE = re.compile(r"(\w+)[ \t]+$")
OLD_CONTACT = re.compile(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", re.IGNORECASE)
OLD_OBJECT_SPAN = re.compile(r"\{.*\}", flags=re.DOTALL)
OLD_WORKED_CALCULATION = re.compile(rf"\d+\s*[x{TIMES}*]\s*\d+\s*=\s*\d+")
OLD_EXAMPLE_LABEL = re.compile(
    r"^\s*[\"'(]*\s*(?:ví dụ minh hoạ|ví dụ minh họa|ví dụ|illustrative example|worked example|for example|example)"
    r"\s*[:,]\s*", re.IGNORECASE)
OLD_HEAD_END = re.compile(rf"\s+[-{EN_DASH}{EM_DASH}|]\s+|\s*:\s*|\s*\(")
OLD_ROLE_EXAMPLE_LABEL = re.compile(r"^\s*[\"'(]*\s*(?:ví dụ minh hoạ|ví dụ minh họa|illustrative example)\s*:\s*",
                                    re.IGNORECASE)
OLD_ENTRY_SPLIT = re.compile(r"\s*(?:[,;]|\bvà\b|\band\b)\s*", re.IGNORECASE)
OLD_PRESERVE_CELL_SPLIT = re.compile(f"[{preserve_lines.WS_CLASS}]+\\|[{preserve_lines.WS_CLASS}]+")
OLD_RANGE_SUFFIX = re.compile(
    r"\s*\(\s*(?:(?:từ|tu|from)\s+)?(?:slides?|trang|pages?)\s+\d{1,4}\s*"
    r"(?:(?:đến|den|to)\s+(?:(?:slides?|trang|pages?)\s+)?\d{1,4}"
    r"|[-\u2013\u2014]\s*(?:(?:slides?|trang|pages?)\s+)?\d{1,4})\s*\)\s*$",
    re.IGNORECASE | re.UNICODE,
)


def bracket(match: re.Match[str]) -> str:
    return f"<{match.group(0)[1:-1]}>"


def word_before(text: str) -> list[str | None]:
    reverse = text[::-1]
    return [quiz_letters.word_before(text, reverse, end) for end in range(len(text) + 1)]


def old_word_before(text: str) -> list[str | None]:
    return [match.group(1) if (match := OLD_WORD_BEFORE.search(text, 0, end)) else None
            for end in range(len(text) + 1)]


def entry_count(text: str, pattern: re.Pattern[str]) -> int:
    return len([part for part in pattern.split(text) if part.strip()])


def span(match: re.Match[str] | None) -> tuple[int, int] | None:
    return match.span() if match else None


# (name, new, old, pieces, the input piece that made the regex quadratic)
Case = tuple[str, Callable[[str], Any], Callable[[str], Any], tuple[str, ...], str]
WORDS = ("Customer", "Experience", "a", "b", "1abc", "R&D", "a/b", "x&", "&y", "_z", "'q", f"{APOSTROPHE}q", "Đ", "ữ",
         "²", "12", ".", ",", "(", ")", "-", "CX", "CXY", "CX&", "CX'")
CASES: tuple[Case, ...] = (
    ("dashed_phrases", lambda text: list(acronyms.dashed_phrases(text, "CX")), lambda text: old_dashed(text, "CX"),
     (*WORDS, *SPACES, " - CX", f" {EN_DASH} CX", f" {EM_DASH} CX", " - ", "- CX"), "xx "),
    ("ends_with_toc_leader", source_coverage.ends_with_toc_leader,
     lambda text: bool(OLD_TOC_LEADER.search(text)),
     (".", ".", "...", ELLIPSIS, ELLIPSIS * 2, MIDDLE_DOT, MIDDLE_DOT * 3, "_", "___", "1", "12", "1234", "12345",
      "a", "x", *SPACES), ELLIPSIS),
    ("trailing ellipsis", worksheet.without_trailing_ellipsis, lambda text: OLD_TRAILING_ELLIPSIS.sub("", text),
     (".", ".", "..", ELLIPSIS, "a", "!", "]", *SPACES), "."),
    ("split_cells", worksheet.split_cells, OLD_CELL_SPLIT.split, (";", "|", "a", "b c", ".", *SPACES), " "),
    ("heading label", lambda text: worksheet._HEADING_LABEL_RE.sub("", text),
     lambda text: OLD_HEADING_LABEL.sub("", text),
     ("(", "[", ")", "]", "-", EN_DASH, EM_DASH, ":", "tiêu chí gợi ý", "Tiêu Chí Gợi Ý", "suggested criteria",
      f"{LONG_S}uggested criterion", "suggested criterion", "x", "Tự kiểm tra", *SPACES), "\n"),
    ("when ... thì", lambda text: faq_checks._when_then(text, faq_checks._WHEN_VI, faq_checks._THEN_VI),
     lambda text: bool(OLD_WHEN_VI.search(text)),
     ("khi", "thì", "nào", "khi nào", "?", "x", "khix", "xthì", "thìx", ",", *SPACES), "khi "),
    ("when ... thi", lambda text: faq_checks._when_then(text, faq_checks._WHEN_FOLDED, faq_checks._THEN_FOLDED),
     lambda text: bool(OLD_WHEN_FOLDED.search(text)), ("khi", "thi", "nao", "?", "x", ",", *SPACES), "khi "),
    ("when ... ,", faq_checks._when_comma, lambda text: bool(OLD_WHEN_EN.search(text)),
     ("when", "When", "whenever", ",", "?", "x", *SPACES), "when "),
    ("inline markers", lambda text: [match.span() for match in quiz_text._INLINE_MARKER_RE.finditer(text)],
     lambda text: [match.span() for match in OLD_INLINE_MARKER.finditer(text)],
     ("-", EN_DASH, BULLET, " ", "  ", TAB, "A", "1", "x", "\n", NO_BREAK_SPACE), "\t"),
    ("last visible char", lambda text: [quiz_text._last_visible_char(text, end) for end in range(len(text) + 1)],
     lambda text: [text[:end].rstrip()[-1:] for end in range(len(text) + 1)], ("a", "-", *SPACES), " "),
    ("speech", quiz_text._without_speech, lambda text: OLD_SPEECH.sub("", text),
     ("hỏi", "Hỏi", "nói", "asks", "asked", "ask", "said", "says", "câu ", "Câu ", ":", "?", "\n", "x", "xnói",
      " ", TAB, "thắc mắc"), "nói:"),
    ("quoted (callable)", lambda text: quiz_text.sub_quoted(bracket, text), lambda text: OLD_QUOTED.sub(bracket, text),
     ('"', LEFT_QUOTE, RIGHT_QUOTE, LEFT_GUILLEMET, RIGHT_GUILLEMET, "a", " ", "?"), LEFT_QUOTE),
    ("quoted (text)", lambda text: quiz_text.sub_quoted("", text), lambda text: OLD_QUOTED.sub("", text),
     ('"', LEFT_QUOTE, RIGHT_QUOTE, LEFT_GUILLEMET, RIGHT_GUILLEMET, "a", "?"), LEFT_GUILLEMET),
    ("word_before", word_before, old_word_before, ("ab", "1", "_", "Ý", " ", TAB, "\n", "-", "x y"), "1"),
    ("contact copy", validation.contains_contact_copy, lambda text: bool(OLD_CONTACT.search(text)),
     ("http://", "HTTPS://", "www.", "WWW.", "@", "a", "b.", ".", "+", "-", "co", "vn", "x1", LONG_S, KELVIN,
      "htt", " "), "ha"),
    ("object_span", provider_json.object_span,
     lambda text: (match.group(0) if (match := OLD_OBJECT_SPAN.search(text)) else None),
     ("{", "}", "a", "\n", '"x"', "{}"), "{"),
    ("worked calculation", lambda text: bool(lesson_content_observation._WORKED_CALCULATION.search(text)),
     lambda text: bool(OLD_WORKED_CALCULATION.search(text)),
     ("1", "12", "x", TIMES, "*", "=", " ", "\n", "a", "٣"), "11"),
    ("example label", lambda text: span(answer_checks._EXAMPLE_LABEL_RE.match(text)),
     lambda text: span(OLD_EXAMPLE_LABEL.match(text)),
     ('"', "'", "(", "Ví dụ", "ví dụ minh họa", "VÍ DỤ MINH HOẠ", "example", "worked example", "for example", ":", ",",
      "x", *SPACES), "\n "),
    ("head end", lambda text: answer_checks._HEAD_END_RE.split(text, maxsplit=1)[0],
     lambda text: OLD_HEAD_END.split(text, maxsplit=1)[0],
     ("Bậc", "01", "-", EN_DASH, EM_DASH, "|", ":", "(", "x", *SPACES), "\t"),
    ("role example label", lambda text: span(example_roles._EXAMPLE_LABEL_RE.match(text)),
     lambda text: span(OLD_ROLE_EXAMPLE_LABEL.match(text)),
     ('"', "(", "ví dụ minh hoạ", "Illustrative Example", ":", "x", *SPACES), "  "),
    ("entries", lambda text: entry_count(text, worksheet_input._ENTRY_SEPARATOR_RE),
     lambda text: entry_count(text, OLD_ENTRY_SPLIT),
     (",", ";", "và", "VÀ", "and", "AND", "vàx", "xand", "a", "b c", *SPACES), "  \n"),
    ("preserve cells", preserve_lines._CELL_SPLIT_RE.split, OLD_PRESERVE_CELL_SPLIT.split,
     ("|", "a", "b c", *SPACES, chr(0x2028), chr(0xFEFF)), " "),
    ("range suffix", lambda text: source_structure.SOURCE_RANGE_SUFFIX_RE.sub("", text),
     lambda text: OLD_RANGE_SUFFIX.sub("", text),
     ("(", ")", "slide", "Slides", "trang", "page", "từ", "from", "to", "đến", "-", EN_DASH, "3", "12", "a", "!",
      *SPACES), IDEOGRAPHIC_SPACE),
)


def fuzz_cases(pieces: tuple[str, ...], seed: int) -> list[str]:
    generator = random.Random(seed)  # noqa: S311 - seeded test data, not security
    return ["".join(generator.choices(pieces, k=generator.randint(0, 24))) for _ in range(FUZZ_CASES)]


def fastest_seconds(function: Callable[[str], Any], text: str) -> float:
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        function(text)
        timings.append(time.perf_counter() - started)
    return min(timings)


class LinearPatternParityTests(unittest.TestCase):
    def test_random_inputs_give_the_regex_results(self) -> None:
        for seed, (name, new, old, pieces, _adversarial) in enumerate(CASES):
            with self.subTest(name=name):
                for text in fuzz_cases(pieces, 20261024 + seed):
                    if new(text) != old(text):
                        self.fail(f"{name} differs from the regex on {text!r}: {new(text)!r} != {old(text)!r}")

    def test_the_acronym_check_still_finds_a_dashed_full_form(self) -> None:
        self.assertTrue(acronyms.explained("MOT", None, ["Bước 1: Moments of Truth - MOT."]))
        self.assertTrue(acronyms.explained("CX", None, ["Trải nghiệm khách hàng - CX"],
                                           source_texts=["Trải nghiệm khách hàng (CX)"]))
        self.assertFalse(acronyms.explained("NPS", None, ["Ba chỉ số cần theo dõi hằng tuần - NPS"]))

    def test_the_faq_type_reads_when_clauses_as_before(self) -> None:
        self.assertEqual(faq_checks.question_type("Khi giao hàng trễ thì ai gọi khách"), "what_if")
        self.assertEqual(faq_checks.question_type("When a delivery is late, who calls the customer?"), "what_if")
        self.assertEqual(faq_checks.question_type("khi nao giao hang thi ai goi"), "recall")  # "khi nào" asks the rule


class LinearPatternTimingTests(unittest.TestCase):
    def test_a_megabyte_of_the_worst_input_takes_milliseconds(self) -> None:
        # Before (20 kB of the input): dashed 4.9 s, TOC leader 3.5 s, trailing ellipsis 2.4 s, cells 1.0 s, heading
        # label 1.8 s, when-clauses 0.9 s, inline markers 1.1 s, speech 0.16 s, quotes 0.17 s, word before 2.2 s,
        # contact 0.95 s, object span 0.07 s, worked calculation 2.3 s, example labels 7.2 s and 4.1 s, head end 1.7 s,
        # entries 8.7 s, preserve cells 1.0 s, range suffix 0.8 s; each four times longer at 80 kB, minutes at 1 MB.
        for name, function, _old, _pieces, piece in CASES:
            if name in {"word_before", "last visible char"}:
                continue  # one call per position: timed below
            text = piece * (MEGABYTE // len(piece))
            for variant in (text, text + "!", "a " + text):
                with self.subTest(name=name, variant=variant[-1]):
                    self.assertLess(fastest_seconds(function, variant), MAX_SECONDS)

    def test_a_megabyte_of_white_space_takes_milliseconds(self) -> None:
        # Long runs of spaces were the other quadratic input of most of these patterns (and of the " - TOKEN" search).
        runs = {"spaces": " " * MEGABYTE, "word, tabs, word": "Ab" + TAB * MEGABYTE + "CX",
                "word, line breaks, dash": "Moments of" + "\n" * MEGABYTE + "- CX!",
                "ideographic spaces": "a" + IDEOGRAPHIC_SPACE * MEGABYTE + "!"}
        for name, function, _old, _pieces, _piece in CASES:
            if name in {"word_before", "last visible char"}:
                continue
            # The speech verbs start with two lookbehinds and a case-insensitive alternation: the regex engine tries
            # every position (about 85 ms a megabyte, as before the change): linear, held to the linear bound.
            bound = 0.5 if name == "speech" else MAX_SECONDS
            for run_name, text in runs.items():
                with self.subTest(name=name, run=run_name):
                    self.assertLess(fastest_seconds(function, text), bound)

    def test_word_before_and_the_marker_context_read_each_character_once(self) -> None:
        text = "x " + "1" * MEGABYTE + " \t"
        reverse = text[::-1]
        self.assertLess(fastest_seconds(lambda value: quiz_letters.word_before(value, reverse, len(value)), text),
                        MAX_SECONDS)
        line = "A" + " " * MEGABYTE
        self.assertLess(fastest_seconds(lambda value: quiz_text._last_visible_char(value, len(value)), line), 0.5)


# --- loops that copied the text before every match (quadratic in the text's length times the matches) --------------
def old_segment_starts(text: str, count: int) -> list[tuple[int, int]]:
    letters = mcq.OPTION_LETTERS[:count]
    lined = "\n" in text.strip()
    starts = []
    for match in mcq._SEGMENT_RE.finditer(text):
        if match.group(1) not in letters:
            continue
        before = text[:match.start()].rstrip(" \t")
        if not before or (before.endswith("\n") if lined else before[-1] in mcq._SEGMENT_AFTER):
            starts.append(match.span())
    return starts


def old_numbered(text: str) -> bool:
    for match in media_detail._NUMBERED_RE.finditer(text):
        if match.group("clear"):
            return True
        before = text[:match.start()].rstrip(" \t")
        words = before.split()
        if not before or before[-1] in media_detail._PHRASE_MARKS or before.endswith("\n") or (
                words and words[-1].casefold() in media_detail._PLACING_WORDS):
            return True
    return False


def old_acronym_guide(text: str) -> tuple[dict[str, str], frozenset[str]]:
    """AcronymGuide.build([text], "vi") as it was: (expansions, ignored)."""

    found: dict[str, str] = {}
    ignored: set[str] = set()
    for match in acronyms._EXPANDED_RE.finditer(text):
        token, full = match.group(1), " ".join(match.group(2).split())
        words = full.split()
        if (acronyms._MIN_EXPANSION_WORDS <= len(words) <= acronyms._MAX_EXPANSION_WORDS
                and acronyms.idm_fold(words[0])[:1] == acronyms.idm_fold(token)[:1]
                and not acronyms.ACRONYM_RE.fullmatch(full)):
            found.setdefault(token, full)
    for run in acronyms._CAPS_RUN_RE.findall(text):
        ignored.update(run.split())
    for match in acronyms.ACRONYM_RE.finditer(text):
        token = match.group(1)
        if re.match(acronyms._NAME_AFTER_RE, text[match.end():]) and token not in found:
            ignored.add(token)
        before = text[:match.start()].rstrip()
        if before.endswith("(") and text[match.end():].lstrip().startswith(")"):
            ignored.add(token)
    expansions = {**dict(acronyms._COMMON_VI.items()), **found}
    return expansions, frozenset(ignored - set(expansions))


def acronym_guide(text: str) -> tuple[dict[str, str], frozenset[str]]:
    guide = acronyms.AcronymGuide.build([text], "vi")
    return dict(guide.expansions), guide.ignored


LOOP_CASES: tuple[Case, ...] = (
    ("visible_end", lambda text: [visible_end(text, end) for end in range(len(text) + 1)],
     lambda text: [len(text[:end].rstrip()) for end in range(len(text) + 1)], ("a", "-", *SPACES), " "),
    ("visible_end blanks", lambda text: [visible_end(text, end, " \t") for end in range(len(text) + 1)],
     lambda text: [len(text[:end].rstrip(" \t")) for end in range(len(text) + 1)], ("a", *SPACES), " "),
    ("visible_start", lambda text: [visible_start(text, start) for start in range(len(text) + 1)],
     lambda text: [len(text) - len(text[start:].lstrip()) for start in range(len(text) + 1)], ("a", *SPACES), " "),
    ("segment_starts", lambda text: [match.span() for match in mcq.segment_starts(text, 4)],
     lambda text: old_segment_starts(text, 4),
     ("A", "B", "D", "E", " - ", "-", ": ", ".", ") ", " ", "\t", "\n", "x", ";", ",", "(", EN_DASH, "như"), "A - "),
    ("_numbered", media_detail._numbered, old_numbered,
     ("khung 1", "Hàng 2", "ô 3", "tầng 2", "kho hàng 2 tầng", "ở", "trên", "cột", "số", "1", "12", "%", " ", "\t",
      "\n", ".", ",", "x", "-", "(", "người"), "hàng 2 "),
    ("acronym guide", acronym_guide, old_acronym_guide,
     ("ROI", "SPD", "FPT", "(", ")", " ", "Software", "DEEP C", "x", "1", "CR-04", "ERA5.0", "Standard",
      "Product", "Dev", "(Standard Product Dev)", "THINK BIG", *SPACES), "ROI "),
)


class LoopCopyTests(unittest.TestCase):
    def test_random_inputs_give_the_old_results(self) -> None:
        for seed, (name, new, old, pieces, _adversarial) in enumerate(LOOP_CASES):
            with self.subTest(name=name):
                for text in fuzz_cases(pieces, 4242 + seed):
                    if new(text) != old(text):
                        self.fail(f"{name} differs from the old code on {text!r}: {new(text)!r} != {old(text)!r}")

    def test_many_matches_in_a_long_text_take_milliseconds(self) -> None:
        # Before: every match copied the text before it (and _numbered split it): 0.1 MB of "hàng 2 " took seconds.
        for name, function in (("segment_starts", lambda text: mcq.segment_starts(text, 4)),
                               ("_numbered", media_detail._numbered), ("acronym guide", acronym_guide)):
            piece = next(case[4] for case in LOOP_CASES if case[0] == name)
            text = "Đ " + piece * (100_000 // len(piece))
            with self.subTest(name=name):
                self.assertLess(fastest_seconds(function, text), 0.5)


if __name__ == "__main__":
    unittest.main()
