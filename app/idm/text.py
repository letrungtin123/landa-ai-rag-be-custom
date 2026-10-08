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


# Sentences the deterministic W1-reduce fallback writes (content_map.fallback_w1_reduce). The
# topic (a cleaned source heading or chapter title) is the only part fit for a module or lesson
# title (QC course 234653: every chapter was titled "Người học có thể áp dụng … trong công việc").
# The pre-2026-10-08 templates are still recognised for designs written before the change.
_FALLBACK_OBJECTIVE_VI: Final = "áp dụng các điểm chính của {topic} vào một tình huống công việc cụ thể"
_FALLBACK_OBJECTIVE_EN: Final = "apply the key points of {topic} to a specific work situation"
_FALLBACK_MUST_DO_VI: Final = "Áp dụng {topic} vào một tình huống công việc"
_FALLBACK_MUST_DO_EN: Final = "Apply {topic} to a work situation"
_FALLBACK_OBJECTIVE_RE: Final = re.compile(
    r"^(?:áp dụng các điểm chính của (?P<vi>.+) vào một tình huống công việc cụ thể"
    r"|apply the key points of (?P<en>.+) to a specific work situation"
    r"|áp dụng (?P<vi_old>.+) trong công việc|apply (?P<en_old>.+) at work)\.?$",
    re.IGNORECASE,
)
_FALLBACK_MUST_DO_RE: Final = re.compile(
    r"^(?:Áp dụng (?P<vi>.+) vào một tình huống công việc|Apply (?P<en>.+) to a work situation"
    r"|Thực hiện đúng (?P<vi_old>.+)|Correctly carry out (?P<en_old>.+))$",
)


def fallback_objective_statement(topic: str, locale: str) -> str:
    """The deterministic objective for a source topic: one observable action (apply), not a title."""

    if locale == "vi":
        return "Người học có thể " + _FALLBACK_OBJECTIVE_VI.format(topic=topic)
    return "The learner can " + _FALLBACK_OBJECTIVE_EN.format(topic=topic)


def fallback_must_do_statement(topic: str, locale: str) -> str:
    """The deterministic Must Do for a source topic (an action, never "Thực hiện đúng {topic}")."""

    return (_FALLBACK_MUST_DO_VI if locale == "vi" else _FALLBACK_MUST_DO_EN).format(topic=topic)


def _topic(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value)


def _capitalized(value: str) -> str:
    value = value.strip(" .")
    return value[:1].upper() + value[1:]


def objective_title(statement: str) -> str:
    """A module title from an objective: the action phrase, never "Người học có thể …".

    "Người học có thể phân tích bốn lực đẩy …" → "Phân tích bốn lực đẩy …"; a fallback
    objective ("… áp dụng các điểm chính của {topic} vào …") → "{topic}".
    """

    text = " ".join(unicodedata.normalize("NFC", statement).split())
    lowered = text.lower()
    start = 0
    for subject in _LEARNER_SUBJECTS:
        if lowered.startswith(subject + " "):
            start = len(subject) + 1
            break
    for lead_in in (*CAPABILITY_LEAD_INS, "có thể"):
        if lowered.startswith(lead_in + " ", start):
            start += len(lead_in) + 1
            break
    remainder = text[start:] if len(lowered) == len(text) else text
    fallback = _FALLBACK_OBJECTIVE_RE.fullmatch(remainder)
    if fallback is not None:
        remainder = _topic(fallback)
    return _capitalized(remainder) or text


def must_do_title(statement: str) -> str:
    """A lesson title from a Must Do: a fallback Must Do ("Áp dụng {topic} vào …") becomes "{topic}";
    a client objective copied as Must Do loses its learner lead-in ("Người học có thể …")."""

    text = " ".join(unicodedata.normalize("NFC", statement).split())
    fallback = _FALLBACK_MUST_DO_RE.fullmatch(text)
    return _capitalized(_topic(fallback)) if fallback is not None else objective_title(text)


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
