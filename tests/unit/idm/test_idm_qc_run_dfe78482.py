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

from app.idm.mcq import normalize_single_choice
from app.idm.quiz_text import lay_out_quiz, number_item_letters

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
