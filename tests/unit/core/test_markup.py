"""app.core.markup: the results of the tag and element regexes every call site used before, in linear time.

``replace_tags`` gives ``re.sub(r"<[^>]+>", ...)`` (security audit 2026-10-24, H1) and, with ``include_empty``,
``re.sub(r"<[^>]*>", ...)``; ``remove_elements`` gives the ``<style[^>]*>[\\s\\S]*?</style>`` removal of the Node
mirror (review of wave 1, L-5). The regexes stay here as the references.
"""

from __future__ import annotations

import random
import re
import time
import unittest
from pathlib import Path

from app.core.markup import remove_elements, replace_tags

REFERENCE_TAG_RE = re.compile(r"<[^>]+>")
REFERENCE_TAG_OR_EMPTY_RE = re.compile(r"<[^>]*>")
ELEMENT_NAMES = ("style", "script", "iframe", "object", "embed")
REFERENCE_ELEMENT_RES = {name: re.compile(rf"<{name}[^>]*>[\s\S]*?</{name}>", re.IGNORECASE | re.ASCII)
                         for name in ELEMENT_NAMES}
# Letters an ASCII-only case folding must not take for s / i / k: LONG S, DOTTED CAPITAL I, KELVIN SIGN.
LONG_S, DOTTED_I, KELVIN = chr(0x17F), chr(0x130), chr(0x212A)
APP_ROOT = Path(__file__).resolve().parents[3] / "app"
# "<" and ">" weigh most so tags, "<>", unclosed "<" and stray ">" are all frequent; Vietnamese letters and line
# breaks are inside "[^>]" as well.
ALPHABET = "<<<<>>>> \n\tab/=\"'\\ệĐữỳ"
HAND_CASES = (
    "", "<", ">", "<>", "><", "<<>>", "<<>", "<>>", "<a>", "<a", "a>", "<<a>", "<a<b>", "<a>b>", "a < b and c > d",
    "x <> y <z>", "<\n>", "<p>Xin chào <b>bạn</b></p>", "<<<<", "<a" * 5, "<" + "x" * 9, "<>" * 4 + "<",
    "<\\1>", "Đường <i>số</i> 3 > 2 < 5", "> <", "<>a>", "<a><>", "<<<>>>", "text without tags",
)
# Pieces of element markup: opening and closing tags in every case, cut ones, unclosed ones, letters the ASCII-only
# case folding must not take for s/t/y/l/e/c/r/i/p (U+017F LONG S, U+0130, U+212A KELVIN SIGN) and plain text.
ELEMENT_PIECES = (
    "<style", "<STYLE", "<Style", "<style>", "<style type=\"text/css\">", "</style>", "</STYLE>", "</Style >",
    "</style", "<styl", "tyle>", "<script", "<SCRIPT>", "<Script src=x>", "</script>", "</SCRIPT>", "</scrip",
    f"<{LONG_S}tyle>", f"</{LONG_S}tyle>", f"<scr{DOTTED_I}pt>", f"<{KELVIN}>", ">", ">", "<", "</", "<b>", "x",
    " ", "\n", "Đ", "ữ", "{a:b}", "<iframe>", "</iframe>", "<OBJECT data=x>", "</object>", "<embed", "</embed>",
)
ELEMENT_HAND_CASES = (
    "", "<style>a</style>", "x<style>a</style>y<style>b</style>z", "<style>a", "<style", "<style a</style>",
    "<style>a</style", "<style><style>a</style></style>", "<STYLE>a</style>", "<style>a</STYLE>", "<stylex>a</style>",
    "<style>a</style>" * 3 + "<style>", "<style" * 20 + ">" + "</style>", "<style>" * 5, "</style><style>",
    f"<{LONG_S}tyle>a</{LONG_S}tyle>", "<script>alert(1)</script><style>p{}</style>", "<style\n>\n</style>",
)


def reference(text: str, replacement: str) -> str:
    return REFERENCE_TAG_RE.sub(replacement, text)


def reference_elements(text: str, name: str) -> str:
    return REFERENCE_ELEMENT_RES[name].sub("", text)


class ReplaceTagsTests(unittest.TestCase):
    def test_hand_cases_match_the_regex(self) -> None:
        for text in HAND_CASES:
            for replacement in (" ", ""):
                with self.subTest(text=text, replacement=replacement):
                    self.assertEqual(replace_tags(text, replacement), reference(text, replacement))
                    self.assertEqual(replace_tags(text, replacement, include_empty=True),
                                     REFERENCE_TAG_OR_EMPTY_RE.sub(replacement, text))
        self.assertEqual(replace_tags("a<>b", " "), "a<>b")
        self.assertEqual(replace_tags("a<>b", " ", include_empty=True), "a b")
        self.assertEqual(replace_tags("a < b and c > d", " "), "a   d")
        self.assertEqual(replace_tags("<<a>", " "), " ")

    def test_random_strings_match_the_regex(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        for _ in range(50_000):
            text = "".join(generator.choices(ALPHABET, k=generator.randint(0, 40)))
            replacement = generator.choice((" ", ""))
            if replace_tags(text, replacement) != reference(text, replacement):
                self.fail(f"replace_tags differs from the regex on {text!r} with {replacement!r}")
            if replace_tags(text, replacement, include_empty=True) != REFERENCE_TAG_OR_EMPTY_RE.sub(replacement, text):
                self.fail(f"replace_tags(include_empty=True) differs from <[^>]*> on {text!r} with {replacement!r}")

    def test_replacement_is_literal_text(self) -> None:
        self.assertEqual(replace_tags("a<b>c", "\\1"), "a\\1c")
        self.assertEqual(replace_tags("a<b>c", "\\"), "a\\c")
        self.assertEqual(replace_tags("a<>c", "\\1", include_empty=True), "a\\1c")

    def test_adversarial_megabyte_inputs_finish_fast(self) -> None:
        # The regex took 0.5 s on 40,000 "<" and grows with the square of the length (about 6 minutes for 1 MB).
        inputs = {
            "1 MB of <": "<" * 1_000_000,
            "<a repeated": "<a" * 500_000,
            "< then text without >": "<" + "x" * 1_000_000,
            "1 MB of < then >": "<" * 1_000_000 + ">",
            "<> repeated then <": "<>" * 500_000 + "<",
            "> then 1 MB of <": ">" + "<" * 1_000_000,
            "<<>> repeated": "<<>>" * 250_000,
        }
        for name, text in inputs.items():
            for include_empty in (False, True):
                with self.subTest(name, include_empty=include_empty):
                    started = time.perf_counter()
                    replace_tags(text, " ", include_empty=include_empty)
                    self.assertLess(time.perf_counter() - started, 0.5)

    def test_no_call_site_runs_a_quadratic_tag_regex_again(self) -> None:
        patterns = (r"<[^>]+>", r"<[^>]*>")
        offenders = [f"{path.relative_to(APP_ROOT).as_posix()}: {pattern}"
                     for path in sorted(APP_ROOT.rglob("*.py")) if path.name != "markup.py"
                     for pattern in patterns if pattern in path.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [])


class RemoveElementsTests(unittest.TestCase):
    def test_hand_cases_match_the_regex(self) -> None:
        for text in ELEMENT_HAND_CASES:
            for name in ELEMENT_NAMES:
                with self.subTest(text=text, name=name):
                    self.assertEqual(remove_elements(text, name), reference_elements(text, name))
        self.assertEqual(remove_elements("a<STYLE x>b</Style>c", "style"), "ac")
        self.assertEqual(remove_elements("a<style>b", "style"), "a<style>b")
        unchanged = f"<{LONG_S}tyle>b</{LONG_S}tyle>"
        self.assertEqual(remove_elements(unchanged, "style"), unchanged)

    def test_random_markup_matches_the_regexes(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        for _ in range(50_000):
            text = "".join(generator.choices(ELEMENT_PIECES, k=generator.randint(0, 30)))
            name = generator.choice(ELEMENT_NAMES)
            if remove_elements(text, name) != reference_elements(text, name):
                self.fail(f"remove_elements differs from the regex on {text!r} ({name})")
            both = remove_elements(remove_elements(text, "style"), "script")
            if both != reference_elements(reference_elements(text, "style"), "script"):
                self.fail(f"style then script differs from the regexes on {text!r}")

    def test_unclosed_elements_take_linear_time(self) -> None:
        # The regex scans to the end of the text from every unclosed "<style": 20,000 of them took seconds.
        inputs = {
            "<style repeated": "<style" * 200_000,
            "<style> repeated": "<style>" * 150_000,
            "<STYLE x> then text": "<STYLE x>" * 100_000 + "x" * 100_000,
            "closed then unclosed": "<style>a</style>" * 50_000 + "<style>" * 50_000,
            "<script> repeated": "<script>" * 150_000,
        }
        for name, text in inputs.items():
            with self.subTest(name):
                started = time.perf_counter()
                remove_elements(remove_elements(text, "style"), "script")
                self.assertLess(time.perf_counter() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
