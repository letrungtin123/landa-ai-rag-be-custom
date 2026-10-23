"""app.core.markup.replace_tags: the result of ``re.sub(r"<[^>]+>", ...)`` for every string, in linear time.

The regex is the one every call site used before (security audit 2026-10-24, H1); it stays here as the reference.
"""

from __future__ import annotations

import random
import re
import time
import unittest
from pathlib import Path

from app.core.markup import replace_tags

REFERENCE_TAG_RE = re.compile(r"<[^>]+>")
APP_ROOT = Path(__file__).resolve().parents[3] / "app"
# "<" and ">" weigh most so tags, "<>", unclosed "<" and stray ">" are all frequent; Vietnamese letters and line
# breaks are inside "[^>]" as well.
ALPHABET = "<<<<>>>> \n\tab/=\"'\\ệĐữỳ"
HAND_CASES = (
    "", "<", ">", "<>", "><", "<<>>", "<<>", "<>>", "<a>", "<a", "a>", "<<a>", "<a<b>", "<a>b>", "a < b and c > d",
    "x <> y <z>", "<\n>", "<p>Xin chào <b>bạn</b></p>", "<<<<", "<a" * 5, "<" + "x" * 9, "<>" * 4 + "<",
    "<\\1>", "Đường <i>số</i> 3 > 2 < 5", "> <", "<>a>", "<a><>", "<<<>>>", "text without tags",
)


def reference(text: str, replacement: str) -> str:
    return REFERENCE_TAG_RE.sub(replacement, text)


class ReplaceTagsTests(unittest.TestCase):
    def test_hand_cases_match_the_regex(self) -> None:
        for text in HAND_CASES:
            for replacement in (" ", ""):
                with self.subTest(text=text, replacement=replacement):
                    self.assertEqual(replace_tags(text, replacement), reference(text, replacement))
        self.assertEqual(replace_tags("a<>b", " "), "a<>b")
        self.assertEqual(replace_tags("a < b and c > d", " "), "a   d")
        self.assertEqual(replace_tags("<<a>", " "), " ")

    def test_random_strings_match_the_regex(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        for _ in range(50_000):
            text = "".join(generator.choices(ALPHABET, k=generator.randint(0, 40)))
            replacement = generator.choice((" ", ""))
            if replace_tags(text, replacement) != reference(text, replacement):
                self.fail(f"replace_tags differs from the regex on {text!r} with {replacement!r}")

    def test_replacement_is_literal_text(self) -> None:
        self.assertEqual(replace_tags("a<b>c", "\\1"), "a\\1c")
        self.assertEqual(replace_tags("a<b>c", "\\"), "a\\c")

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
            with self.subTest(name):
                started = time.perf_counter()
                replace_tags(text, " ")
                self.assertLess(time.perf_counter() - started, 0.5)

    def test_no_call_site_runs_the_quadratic_regex_again(self) -> None:
        offenders = [path.relative_to(APP_ROOT).as_posix() for path in sorted(APP_ROOT.rglob("*.py"))
                     if path.name != "markup.py" and r"<[^>]+>" in path.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
