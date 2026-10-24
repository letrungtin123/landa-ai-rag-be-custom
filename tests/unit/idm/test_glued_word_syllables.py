"""Glued-word syllables in linear time with the regex's result (re-score of 2026-10-24, N1).

``(?:syllable){2,}`` matched whole tried every way of cutting a long run of vowels into syllables before failing on
the next letter: "đ" + 28 times "a" + "baz" took 0.36 s and every three more letters six times as long, holding the GIL.
"""

from __future__ import annotations

import random
import re
import time
import unittest

from app.idm import text as idm_text
from app.idm.text import glued_words

OLD_SYLLABLES = re.compile(r"(?:(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnprstvx])?[aeiouy]{1,3}"
                           r"(?:ch|ng|nh|[cmnpt])?){2,}")
LETTERS = "aeiouybcdđghklmnprstvxqzfjw"
PIECES = ("ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr", "a", "e", "i", "o", "u", "y", "đ",
          "c", "t", "m", "n", "p", "x", "z", "f", "aa", "uye", "ooo")


def random_words(seed: int, count: int) -> list[str]:
    generator = random.Random(seed)  # noqa: S311 - test data
    words = ["", "a", "aa", "ba", "baba", "nghiêng", "đaaaa", "quyen", "giai", "chanh", "tranh", "nhanhnhanh"]
    for _ in range(count):
        if generator.random() < 0.5:
            words.append("".join(generator.choice(LETTERS) for _ in range(generator.randint(1, 16))))
        else:
            words.append("".join(generator.choice(PIECES) for _ in range(generator.randint(1, 7))))
    return words


class GluedSyllableParityTests(unittest.TestCase):
    def test_the_walk_gives_the_regex_result(self) -> None:
        for word in random_words(20261024, 120_000):
            if idm_text._two_syllables_or_more(word) != bool(OLD_SYLLABLES.fullmatch(word)):
                self.fail(f"{word!r}: {idm_text._two_syllables_or_more(word)} != {bool(OLD_SYLLABLES.fullmatch(word))}")

    def test_glued_words_are_still_found(self) -> None:
        self.assertEqual(glued_words("Lập bộhồ sơ, khiến họphải từchối và bỏlỡmột cơ hội"),
                         ["bộhồ", "họphải", "từchối", "bỏlỡmột"])


class GluedSyllableTimingTests(unittest.TestCase):
    def test_a_long_run_of_vowels_takes_milliseconds(self) -> None:
        for word in ("đ" + "a" * 10_000 + "baz", "đ" + "ua" * 5_000 + "x", "đ" + "a" * 100_000):
            started = time.perf_counter()
            idm_text._two_syllables_or_more(word)
            glued_words(f"Từ {word} cuối")
            self.assertLess(time.perf_counter() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
