"""Server-side hygiene of single-choice questions written by the W5 writer (QC course 364564, N2).

The writer put the correct answer at position B in 8 of 9 questions and made it the longest
option in 8 of 9; 3 of 9 questions prefixed their options with "A./B." while every explanation
named the options by letter ("A - ..."). The ``problem`` payload stores one explanation for the
whole question (``choices`` hold only ``text``/``correct``, Node renders one ``<solution>``), so
feedback cannot be kept per option. After every writer or repair answer, before validation:

* leading option labels ("A.", "B)", "(C)", "D -") are removed from the option text;
* the options are put in a reproducible order drawn from the slot's ``component_plan_id``
  (SHA-256 Fisher-Yates: the same plan always gives the same order, across runs and Python
  versions);
* every option letter the explanation uses is rewritten to the option's new letter, and an
  explanation made of one "X - ..." segment per option is re-ordered A, B, C ...

``answer_length_cue`` is the non-blocking review check: the correct option is far longer than
every other option, so a learner can guess it by length.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from app.idm.policy import IDM_MCQ_LENGTH_CUE_RATIO

OPTION_LETTERS: Final = "ABCDEF"
SHUFFLED_CODE: Final = "w5_problem_options_shuffled"
LABELS_STRIPPED_CODE: Final = "w5_problem_option_labels_stripped"
RELABELLED_CODE: Final = "w5_problem_explanation_relabelled"
ANSWER_LENGTH_CUE_CODE: Final = "IDM_W5_ANSWER_LENGTH_CUE"
_SINGLE_CHOICE_TYPES: Final = frozenset({"multiple_choice", "single_choice"})
_MIN_OPTIONS: Final = 2
_MIN_LABELLED_OPTIONS: Final = 2
_LENGTH_CUE_MIN_OPTIONS: Final = 3
_DRAW_BYTES: Final = 8

# "A. text", "B) text", "C: text", "D - text", "(A) text"; a space must follow, so "A.I. ..." stays.
_CHOICE_LABEL_RE: Final = re.compile(r"^\s*(?:\(([A-Fa-f])\)|([A-Fa-f])\s*(?:[.):—–]|-))\s+(?=\S)")  # noqa: RUF001
# A single capital letter standing alone ("A", not "A1", "Ab" or "XA").
_LETTER_RE: Final = re.compile(r"(?<!\w)([A-F])(?!\w)")
# Contexts that make a standalone letter an option label.
_MARKER_AFTER_RE: Final = re.compile(r"\s*(?:[—–:)]|\.(?!\w)|-(?=\s))")  # noqa: RUF001 - dashes
_KEYWORD_BEFORE_RE: Final = re.compile(
    r"(?:phương án|đáp án|lựa chọn|câu trả lời|options?|answers?|choices?)\s*\(?\s*$", re.IGNORECASE)
_VERB_AFTER_RE: Final = re.compile(r"\s+(?:là|is|was|đúng|sai|chưa|không|cũng|vì)(?!\w)", re.IGNORECASE)
_JOINER: Final = r"\s*(?:,|/|&|và|and|or|hoặc)\s*"
_LIST_AFTER_RE: Final = re.compile(_JOINER + r"[A-F](?!\w)", re.IGNORECASE)
_LIST_BEFORE_RE: Final = re.compile(r"(?<!\w)[A-F]" + _JOINER + r"$", re.IGNORECASE)
# Words after which a capital letter is a name ("loại A", "nhóm B", "Plan C"), never an option.
_NAME_BEFORE_RE: Final = re.compile(
    r"(?:loại|nhóm|hạng|cấp|mức|khối|vùng|giai đoạn|kế hoạch|phương thức|mẫu|type|class|grade|group|level|"
    r"tier|plan|phase|vitamin|model)\s*$", re.IGNORECASE)
# The start of one "X - ..." segment per option.
_SEGMENT_RE: Final = re.compile(r"(?:^|(?<=[\s;,.(]))([A-F])\s*(?:[—–:.)]|-(?=\s))")  # noqa: RUF001
# How the per-option segments of an explanation may end: one shared mark, re-joined as given.
_SEGMENT_JOINERS: Final = {";": "; ", ",": ", ", ".": " ", "!": " ", "?": " "}
_SENTENCE_BREAK_RE: Final = re.compile(r"[.!?]\s+\S")
_CONTEXT_CHARS: Final = 40


def option_order(seed: str, count: int) -> list[int]:
    """A reproducible permutation of ``range(count)``: ``order[new_position] = old_index``."""

    order = list(range(count))
    for top in range(count - 1, 0, -1):
        draw = hashlib.sha256(f"{seed}:{top}".encode()).digest()[:_DRAW_BYTES]
        pick = int.from_bytes(draw, "big") % (top + 1)
        order[top], order[pick] = order[pick], order[top]
    return order


def _strip_labels(choices: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop "A." style prefixes when they label the options in order (at least two of them)."""

    cuts: list[int | None] = []
    for position, choice in enumerate(choices):
        text = choice.get("text")
        match = _CHOICE_LABEL_RE.match(text) if isinstance(text, str) else None
        letter = (match.group(1) or match.group(2)).upper() if match else ""
        cuts.append(match.end() if match and letter == OPTION_LETTERS[position] else None)
    if sum(cut is not None for cut in cuts) < _MIN_LABELLED_OPTIONS:
        return [dict(choice) for choice in choices], 0
    stripped = [{**choice, "text": str(choice["text"])[cut:]} if cut is not None else dict(choice)
                for choice, cut in zip(choices, cuts, strict=True)]
    return stripped, sum(cut is not None for cut in cuts)


def _is_label(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - _CONTEXT_CHARS):start]
    after = text[end:end + _CONTEXT_CHARS]
    if _NAME_BEFORE_RE.search(before):
        return False
    return bool(_MARKER_AFTER_RE.match(after) or _KEYWORD_BEFORE_RE.search(before) or _VERB_AFTER_RE.match(after)
                or _LIST_AFTER_RE.match(after) or _LIST_BEFORE_RE.search(before))


def relabel_explanation(text: str, new_letter: Mapping[str, str]) -> str:
    """Rewrite every option letter used as a label ("A -", "phương án B", "C và D") through ``new_letter``."""

    def swap(match: re.Match[str]) -> str:
        letter = match.group(1)
        if letter in new_letter and _is_label(text, match.start(), match.end()):
            return new_letter[letter]
        return letter

    return _LETTER_RE.sub(swap, text)


def labelled_letters(text: str) -> set[str]:
    """Option letters ``text`` uses as labels ("A -", "phương án B", "A và C", "Đáp án đúng là D."); the same
    reading ``relabel_explanation`` applies after the options were shuffled (QC course 364564, N10)."""

    normalized = unicodedata.normalize("NFC", text)
    return {match.group(1) for match in _LETTER_RE.finditer(normalized)
            if _is_label(normalized, match.start(), match.end())}


def _ordered_segments(text: str, relabelled: str, count: int, new_letter: Mapping[str, str]) -> str | None:
    """``relabelled`` with its per-option segments in A, B, C ... order, when that is safe.

    Safe means: the original explanation names each option exactly once at a segment start, in
    order A, B, C ..., and its last segment is one sentence (a closing remark after it would
    otherwise travel with that option).
    """

    starts = [match for match in _SEGMENT_RE.finditer(text) if match.group(1) in OPTION_LETTERS[:count]]
    if [match.group(1) for match in starts] != list(OPTION_LETTERS[:count]) or any(
            relabelled[match.start()] != new_letter[match.group(1)] for match in starts):
        return None
    bounds = [match.start() for match in starts] + [len(text)]
    bodies = [relabelled[bounds[index]:bounds[index + 1]].strip() for index in range(count)]
    marks = {body[-1:] for body in bodies[:-1]}
    if _SENTENCE_BREAK_RE.search(bodies[-1]) or len(marks) != 1 or next(iter(marks)) not in _SEGMENT_JOINERS:
        return None
    mark = next(iter(marks))
    joiner = _SEGMENT_JOINERS[mark]
    closing = bodies[-1][-1:] if bodies[-1][-1:] in {".", "!", "?"} else ""
    clauses = mark in {";", ","}
    if not clauses and not closing:
        return None  # sentence segments whose last one has no full stop would run together
    if clauses:
        # Clause-separated segments: the last one's full stop moves to the new last segment.
        bodies = [body.rstrip(";,").rstrip() for body in bodies]
        bodies = [body[:-1] if closing and body.endswith(closing) else body for body in bodies]
    new_order = sorted(range(count), key=lambda index: new_letter[OPTION_LETTERS[index]])
    joined = joiner.join(bodies[index] for index in new_order)
    return relabelled[:bounds[0]] + joined + (closing if clauses else "")


def normalize_single_choice(component: dict[str, Any], seed: str) -> tuple[dict[str, Any], Counter[str]]:
    """``component`` with unlabeled options in a seeded order and an explanation that follows them."""

    codes: Counter[str] = Counter()
    choices = component.get("choices")
    kind = str(component.get("problem_type") or "multiple_choice").strip().lower()
    if (kind not in _SINGLE_CHOICE_TYPES or not isinstance(choices, list)
            or not _MIN_OPTIONS <= len(choices) <= len(OPTION_LETTERS)
            or not all(isinstance(choice, dict) for choice in choices)):
        return component, codes
    stripped, labels = _strip_labels(choices)
    if labels:
        codes[LABELS_STRIPPED_CODE] += labels
    order = option_order(seed, len(stripped))
    if order != sorted(order):
        codes[SHUFFLED_CODE] += 1
    result = {**component, "choices": [stripped[old] for old in order]}
    explanation = component.get("explanation")
    if isinstance(explanation, str) and order != sorted(order):
        text = unicodedata.normalize("NFC", explanation)
        new_letter = {OPTION_LETTERS[old]: OPTION_LETTERS[new] for new, old in enumerate(order)}
        relabelled = relabel_explanation(text, new_letter)
        rewritten = _ordered_segments(text, relabelled, len(order), new_letter) or relabelled
        if rewritten != explanation:
            codes[RELABELLED_CODE] += 1
            result["explanation"] = rewritten
    return result, codes


def answer_length_cue(component: Mapping[str, Any]) -> bool:
    """The only correct option is far longer than every other option (non-blocking review, N2)."""

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    if len(choices) < _LENGTH_CUE_MIN_OPTIONS:
        return False
    lengths = [len(" ".join(str(choice.get("text") or "").split())) for choice in choices]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    if len(correct) != 1:
        return False
    others = [length for index, length in enumerate(lengths) if index != correct[0]]
    return lengths[correct[0]] > IDM_MCQ_LENGTH_CUE_RATIO * max(others)


__all__ = [
    "ANSWER_LENGTH_CUE_CODE", "LABELS_STRIPPED_CODE", "OPTION_LETTERS", "RELABELLED_CODE", "SHUFFLED_CODE",
    "answer_length_cue", "labelled_letters", "normalize_single_choice", "option_order", "relabel_explanation",
]
