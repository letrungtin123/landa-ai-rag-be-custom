"""Text helpers owned by the IDM pipeline.

The legacy ``_fold`` helpers do not map ``đ`` to ``d``; IDM matching rules are
written against a fold that does (spec §7.1), so IDM keeps its own variant and
never changes the legacy ones (spec §11.6).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from app.idm.policy import (
    CAPABILITY_LEAD_INS,
    GENERIC_TITLE_PATTERN,
    GENERIC_TITLES,
    UNMEASURABLE_VERBS,
)

_WHITESPACE_RE: Final = re.compile(r"\s+")
_GENERIC_TITLE_RE: Final = re.compile(GENERIC_TITLE_PATTERN)
_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)
_ELLIPSIS: Final = "…"
# Vietnamese verbs are matched with diacritics: folding would make "hiệu" (as in
# "hiệu chỉnh", to calibrate) collide with the unmeasurable verb "hiểu".
_UNMEASURABLE_VI: Final = ("hiểu rõ", "hiểu", "biết", "nắm được", "nắm rõ", "làm quen")
_LEARNER_SUBJECTS: Final = (
    "người học", "học viên", "nhân viên", "the learner", "learners", "learner", "participants", "staff",
)


def idm_fold(value: str) -> str:
    """NFC-normalise, lower-case, strip diacritics, map đ→d and collapse spaces."""

    text = unicodedata.normalize("NFC", value).lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _cut(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max(0, max_length - 1)].rstrip() + _ELLIPSIS


def sanitize_author_text(value: str, max_length: int) -> str:
    """Make generated author/learner text safe for the workspace editor (R10).

    The editor rejects ``<`` and ``>`` outside HTML payloads, so they become the
    single angle quotation marks U+2039/U+203A. Runs of spaces collapse; line
    breaks and a short list indentation survive.
    """

    printable = "".join(char for char in value if char in "\n\t" or unicodedata.category(char) != "Cc")
    replaced = printable.replace("<", "‹").replace(">", "›")  # noqa: RUF001 - deliberate lookalike glyphs
    lines = []
    for line in replaced.strip("\n").splitlines():
        indent = min(len(line) - len(line.lstrip(" ")), _MAX_INDENT)
        lines.append(" " * indent + " ".join(line.split()) if line.strip() else "")
    return _cut("\n".join(lines).strip(), max_length)


def single_line(value: str, max_length: int) -> str:
    """Collapse all whitespace (including line breaks) and cut to ``max_length``."""

    return _cut(sanitize_author_text(" ".join(value.split()), max_length + 1).replace("\n", " "), max_length)


_MAX_INDENT: Final = 4


def is_generic_title(value: str) -> bool:
    folded = idm_fold(value).strip(" .:;-")
    return folded in GENERIC_TITLES or _GENERIC_TITLE_RE.fullmatch(folded) is not None


def _after_lead_in(text: str) -> str:
    for subject in _LEARNER_SUBJECTS:
        if text.startswith(subject + " "):
            text = text[len(subject) + 1 :]
            break
    for lead_in in (*CAPABILITY_LEAD_INS, "có thể"):
        match = re.search(rf"(?:^|\s){re.escape(lead_in)}\s+", text)
        if match is not None:
            return text[match.end() :]
    return text


def first_main_verb(statement: str) -> str:
    """Return the statement text that follows the learner subject and capability lead-in."""

    text = unicodedata.normalize("NFC", statement).lower().strip()
    return " ".join(_after_lead_in(text).split())


def is_unmeasurable_objective(statement: str) -> bool:
    """True when the main verb of an objective is not observable (spec §7.3)."""

    remainder = first_main_verb(statement)
    if any(remainder == verb or remainder.startswith(verb + " ") for verb in _UNMEASURABLE_VI):
        return True
    folded = idm_fold(remainder)
    return any(folded == verb or folded.startswith(verb + " ") for verb in _UNMEASURABLE_EN)


# English entries of the spec list; the Vietnamese entries are matched with diacritics above.
_UNMEASURABLE_EN: Final = tuple(sorted(verb for verb in UNMEASURABLE_VERBS if verb in {
    "understand", "know", "learn", "be aware", "appreciate",
}))


def char_ngrams(value: str, size: int) -> set[str]:
    folded = idm_fold(value)
    if len(folded) <= size:
        return {folded} if folded else set()
    return {folded[index : index + size] for index in range(len(folded) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def word_ngrams(value: str, size: int) -> list[tuple[str, ...]]:
    words = _WORD_RE.findall(idm_fold(value))
    return [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]


def ngram_overlap(candidate: str, source: str, size: int = 8) -> float:
    """Share of the candidate's word ``size``-grams that also appear in ``source``."""

    candidate_grams = word_ngrams(candidate, size)
    if not candidate_grams:
        return 0.0
    source_grams = set(word_ngrams(source, size))
    return sum(gram in source_grams for gram in candidate_grams) / len(candidate_grams)
