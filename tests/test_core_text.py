"""app.core.text: the shared accent folds give byte-for-byte the output of the copies they replaced.

The ``_reference_*`` functions are verbatim copies of the private implementations as they were
before refactor wave 2 (step 2.2) replaced them with ``app.core.text``. Each one is compared, over
the corpus below, with the shared variant and with the live call site that now uses it.
"""

from __future__ import annotations

import re
import unicodedata
import unittest
from html import unescape

from app import lesson_content_observation, lesson_prompt_policy
from app.content import instructional_quality, learner_content_purity
from app.core.text import casefold_strip_marks, casefold_strip_marks_and_d
from app.idm import html_rules
from app.idm.text import idm_fold


# app/content/instructional_quality.py::_fold (app/instructional_quality.py before step 2.1)
def _reference_instructional_quality_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


# app/idm/html_rules.py::_fold
def _reference_html_rules_fold(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold())
    return " ".join("".join(char for char in text if unicodedata.category(char) != "Mn").split())


# app/content/learner_content_purity.py::_fold (app/learner_content_purity.py before step 2.1)
def _reference_learner_content_purity_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


# app/lesson_prompt_policy.py::instructional_action_intents
def _reference_instructional_action_intents(text: str) -> list[str]:
    folded = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    patterns = {
        "calculation": r"\b(?:calculate|compute|tinh toan|tinh diem)\b",
        "analysis": r"\b(?:analyse|analyze|phan tich)\b",
        "selection": r"\b(?:select|choose|lua chon|chon bien phap)\b",
        "procedure": r"\b(?:perform|thuc hien)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, folded)]


# app/lesson_content_observation.py::_normalize
def _reference_lesson_content_observation_normalize(text: str) -> str:
    text = unescape(re.sub(r"<[^>]*>", " ", text)).casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return " ".join(re.findall(r"\w+", "".join(c for c in text if unicodedata.category(c) != "Mn")))


BASE_TEXTS = (
    "", " ", "\t\n  \r", "đ", "Đ", "đĐdD", "Đường đi", "ĐƯỜNG ĐI", "đường đi khó", "Đ\u0301", "đ\u0303",
    "Tiếng Việt", "TIẾNG VIỆT", "Hà Nội, Việt Nam!", "Bước 1: Đọc kỹ đề bài.", "bước 2 - buoc 3", "Câu hỏi thường gặp?",
    "Người học, nhân viên; học viên.", "  khoảng   trắng  thừa  ", "Ê-đê", "ặ ằ ẳ ẵ ắ Ặ Ằ Ẳ Ẵ Ắ", "Ữ Ự Ừ Ử Ứ ữ ự ừ ử ứ",
    "ỷ ỹ ý ỳ ỵ Ỷ Ỹ Ý Ỳ Ỵ", "ơ ư Ơ Ư ô Ô â Â ă Ă ê Ê", "Quản lý rủi ro (QLRR)", "Thuế GTGT 10% và 8%",
    "e\u0301 a\u0323\u0302 o\u031b\u0300", "Dấu\u00a0cách cứng", "Không\u200brộng", "tab\tdòng\nmới",
    "C++ / C# & .NET", "x² + y³ = z½", "café naïve résumé façade", "Straße", "\u0130stanbul \u0131",
    "\ufb01le \ufb02ow", "Σίσυφος", "\u01c4 \u01c5 \u01c6", "\u212b \u2126 \u212a",
    "①② \uff26\uff55\uff4c\uff4c", "✅ đã xong", "日本語のテキスト", "한국어 문장", "हिन\u094dदी पाठ",
    "\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd", "<b>Đề</b> &amp; bài", "&lt;p&gt;Đúng&lt;/p&gt; <br/>Sai",
    "<p class='x'>Tính toán</p>", "Tính toán thuế", "PHÂN TÍCH dữ liệu", "Lựa chọn phương án", "Thực hiện quy trình",
    "Chọn biện pháp an toàn", "Tính điểm", "calculate, ANALYSE, choose, perform", "analyze-select",
)


def corpus() -> list[str]:
    """Every base text as written, upper, lower, case-folded, NFC, NFD, NFKC and NFKD (duplicates removed)."""

    texts: list[str] = []
    for text in BASE_TEXTS:
        forms = (text, text.upper(), text.lower(), text.casefold(), text.title())
        for form in forms:
            for variant in (form, *(unicodedata.normalize(kind, form) for kind in ("NFC", "NFD", "NFKC", "NFKD"))):
                if variant not in texts:
                    texts.append(variant)
    return texts


class SharedFoldMatchesReplacedCopiesTests(unittest.TestCase):
    def test_corpus_covers_the_cases_that_tell_the_variants_apart(self) -> None:
        texts = corpus()
        self.assertGreater(len(texts), 300)
        for needle in ("đ", "Đ", "", "Đường đi", unicodedata.normalize("NFD", "Đường đi"), "ĐƯỜNG ĐI", "Straße"):
            self.assertIn(needle, texts)

    def test_casefold_strip_marks_matches_the_copies_it_replaced(self) -> None:
        for text in corpus():
            with self.subTest(text=text):
                expected = _reference_instructional_quality_fold(text)
                self.assertEqual(casefold_strip_marks(text), expected)
                self.assertEqual(instructional_quality._fold(text), expected)
                self.assertEqual(html_rules._fold(text), _reference_html_rules_fold(text))

    def test_casefold_strip_marks_and_d_matches_the_copies_it_replaced(self) -> None:
        for text in corpus():
            with self.subTest(text=text):
                expected = _reference_learner_content_purity_fold(text)
                self.assertEqual(casefold_strip_marks_and_d(text), expected)
                self.assertEqual(learner_content_purity._fold(text), expected)
                self.assertEqual(lesson_prompt_policy.instructional_action_intents(text),
                                 _reference_instructional_action_intents(text))
                self.assertEqual(lesson_content_observation._normalize(text),
                                 _reference_lesson_content_observation_normalize(text))

    def test_the_corpus_reaches_every_intent_of_the_replaced_policy_copy(self) -> None:
        found = {intent for text in corpus() for intent in _reference_instructional_action_intents(text)}
        self.assertEqual(found, {"calculation", "analysis", "selection", "procedure"})

    def test_replaced_names_are_the_shared_functions(self) -> None:
        self.assertIs(instructional_quality._fold, casefold_strip_marks)
        self.assertIs(learner_content_purity._fold, casefold_strip_marks_and_d)


class FoldVariantsStaySeparateTests(unittest.TestCase):
    """The variants are not interchangeable; these cases pin where they differ."""

    def test_d_handling(self) -> None:
        for text in ("Đường", unicodedata.normalize("NFD", "Đường"), "ĐƯỜNG", "đường"):
            with self.subTest(text=text):
                self.assertEqual(casefold_strip_marks(text), "đuong")
                self.assertEqual(casefold_strip_marks_and_d(text), "duong")
        self.assertEqual(casefold_strip_marks(""), "")
        self.assertEqual(casefold_strip_marks_and_d(""), "")

    def test_neither_variant_collapses_whitespace_or_drops_punctuation(self) -> None:
        self.assertEqual(casefold_strip_marks("  Bước 1:  Đọc! "), "  buoc 1:  đoc! ")
        self.assertEqual(casefold_strip_marks_and_d("  Bước 1:  Đọc! "), "  buoc 1:  doc! ")

    def test_idm_fold_is_a_different_variant(self) -> None:
        # idm_fold: NFC, lower() (not casefold), "đ" to "d", whitespace collapsed and trimmed.
        text = "  Straße   Đường "
        self.assertEqual(idm_fold(text), "straße duong")
        self.assertEqual(casefold_strip_marks_and_d(text), "  strasse   duong ")
        self.assertEqual(casefold_strip_marks(text), "  strasse   đuong ")


if __name__ == "__main__":
    unittest.main()
