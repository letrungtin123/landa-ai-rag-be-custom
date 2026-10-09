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

from app.idm.contracts import IdmLessonDesignV1
from app.idm.policy import IDM_MCQ_LENGTH_CUE_RATIO

OPTION_LETTERS: Final = "ABCDEF"
SHUFFLED_CODE: Final = "w5_problem_options_shuffled"
LABELS_STRIPPED_CODE: Final = "w5_problem_option_labels_stripped"
RELABELLED_CODE: Final = "w5_problem_explanation_relabelled"
BALANCED_CODE: Final = "w5_problem_key_position_balanced"
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
# question_ordinal: "chapter_2.lesson_3.unit_1"; steps per chapter and per lesson (see its docstring).
_UNIT_PATH_RE: Final = re.compile(r"^chapter_(\d+)\.lesson_(\d+)\.unit_(\d+)$")
_CHAPTER_STEP: Final = 3
_LESSON_STEP: Final = 1
_UNIT_STEP: Final = 2
# encode_key_letters: four letters per cycle; practice ids pt_1..pt_9 (IdmPracticeTaskV1.practice_id).
_KEY_CYCLE: Final = 4
_MAX_PRACTICE_NUMBER: Final = 9
_PRACTICE_ID_RE: Final = re.compile(r"^pt_([1-9])$")


def option_order(seed: str, count: int) -> list[int]:
    """A reproducible permutation of ``range(count)``: ``order[new_position] = old_index``."""

    order = list(range(count))
    for top in range(count - 1, 0, -1):
        draw = hashlib.sha256(f"{seed}:{top}".encode()).digest()[:_DRAW_BYTES]
        pick = int.from_bytes(draw, "big") % (top + 1)
        order[top], order[pick] = order[pick], order[top]
    return order


def question_ordinal(unit_path: str) -> int | None:
    """A stable number per question from its place in the course: ``3 x chapter + lesson + 2 x unit``.

    QC run 8de1c76b (Q4): an independent seeded shuffle per question put the key at D in 6 of 9 questions. Every
    unit is written alone, so a question cannot see the others; its letter (ordinal modulo the option count)
    comes from ``chapter_c.lesson_l.unit_u`` (a unit holds at most one problem). The lessons of a chapter cycle
    A, B, C, D; the next unit of a lesson moves two letters; any step to the next lesson is odd, so two
    consecutive lessons never share the key's letter. Measured on 3,000 simulated courses (1-3 units per lesson,
    the question mostly in the last unit) against the independent shuffle: the most frequent letter 37% instead
    of 40% of the questions, a repeated letter 10% instead of 27% of consecutive questions, three in a row in 7%
    instead of 49% of the courses.
    """

    match = _UNIT_PATH_RE.match(unit_path)
    if match is None:
        return None
    chapter, lesson, unit = (int(part) for part in match.groups())
    return _CHAPTER_STEP * chapter + _LESSON_STEP * lesson + _UNIT_STEP * unit


def practice_key_shift(practice_id: str | None) -> int:
    """The key-letter shift W4 encoded in a practice id (``pt_1`` -> 0, ``pt_2`` -> 1, ...; see
    ``encode_key_letters``); 0 without a practice, so a design written before the encoding keeps the plain
    ``question_ordinal`` letters."""

    match = _PRACTICE_ID_RE.match(practice_id or "")
    return int(match.group(1)) - 1 if match else 0


def encode_key_letters(lessons: Sequence[IdmLessonDesignV1], chapter_order: int, first_letter: int,
                       ) -> tuple[list[IdmLessonDesignV1], int]:
    """W4 lays out the keys of a chapter's practice questions A, B, C, D, A ... in course order (Q4).

    A unit is written alone and only knows its place in the course, so ``question_ordinal`` alone revisits a
    letter when lessons put their question in different units. W4 sees every lesson of the chapter: it gives the
    practice of the k-th practice question the id whose ``practice_key_shift`` moves that question's
    ``question_ordinal`` letter to ``first_letter + k``. The id is internal (``pt_1``-``pt_9``, never shown to
    learners); only the numbering of a lesson's practices changes. Lessons whose ids cannot be chosen keep
    theirs. Returns the lessons and how many practices were renumbered.
    """

    renumbered = 0
    position = first_letter
    result: list[IdmLessonDesignV1] = []
    for lesson_number, lesson in enumerate(lessons, start=1):
        shifts: dict[str, int] = {}
        for unit_number, unit in enumerate(lesson.units, start=1):
            for component in unit.components:
                practice_id = component.practice_id
                if component.type != "problem" or practice_id is None or practice_id in shifts:
                    continue
                base = question_ordinal(f"chapter_{chapter_order + 1}.lesson_{lesson_number}.unit_{unit_number}")
                shifts[practice_id] = (position - (base or 0)) % _KEY_CYCLE
                position += 1
        mapping = _practice_numbers(shifts, [practice.practice_id for practice in lesson.practice_tasks])
        if mapping is None or all(old == new for old, new in mapping.items()):
            result.append(lesson)
            continue
        renumbered += sum(old != new for old, new in mapping.items())
        result.append(lesson.model_copy(update={
            "practice_tasks": [practice.model_copy(update={"practice_id": mapping[practice.practice_id]})
                               for practice in lesson.practice_tasks],
            "units": [unit.model_copy(update={"components": [
                component.model_copy(update={"practice_id": mapping.get(component.practice_id or "",
                                                                        component.practice_id)})
                for component in unit.components]}) for unit in lesson.units],
        }))
    return result, renumbered


def _practice_numbers(shifts: Mapping[str, int], practice_ids: Sequence[str]) -> dict[str, str] | None:
    """New ids: ``pt_{shift + 1 + 4j}`` for the practices of questions, the lowest free ids for the others."""

    used: set[int] = set()
    mapping: dict[str, str] = {}
    for practice_id, shift in shifts.items():
        number = next((value for value in range(shift + 1, _MAX_PRACTICE_NUMBER + 1, _KEY_CYCLE)
                       if value not in used), None)
        if number is None:
            return None
        used.add(number)
        mapping[practice_id] = f"pt_{number}"
    for practice_id in practice_ids:
        if practice_id in mapping:
            continue
        number = next((value for value in range(1, _MAX_PRACTICE_NUMBER + 1) if value not in used), None)
        if number is None:
            return None
        used.add(number)
        mapping[practice_id] = f"pt_{number}"
    return mapping


def balanced_order(seed: str, correct: int, count: int, target: int) -> list[int]:
    """``order[new_position] = old_index`` with the correct option at ``target`` and the distractors in the
    seeded order of ``option_order``."""

    distractors = [index for index in range(count) if index != correct]
    seeded = [distractors[position] for position in option_order(seed, len(distractors))]
    return [*seeded[:target], correct, *seeded[target:]]


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


def normalize_single_choice(component: dict[str, Any], seed: str,
                            ordinal: int | None = None) -> tuple[dict[str, Any], Counter[str]]:
    """``component`` with unlabeled options in a seeded order and an explanation that follows them.

    With ``ordinal`` (``question_ordinal``) the only correct option goes to letter ``ordinal`` modulo the option
    count and the distractors keep their seeded order (QC run 8de1c76b, Q4: balanced keys across the course).
    """

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
    correct = [index for index, choice in enumerate(stripped) if choice.get("correct") is True]
    if ordinal is not None and len(correct) == 1:
        order = balanced_order(seed, correct[0], len(stripped), ordinal % len(stripped))
        codes[BALANCED_CODE] += 1
    else:
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


def option_lengths(component: Mapping[str, Any]) -> tuple[int, int] | None:
    """(length of the only correct option, length of the longest other option) in characters, or ``None``."""

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    if len(choices) < _LENGTH_CUE_MIN_OPTIONS:
        return None
    lengths = [len(" ".join(str(choice.get("text") or "").split())) for choice in choices]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    if len(correct) != 1:
        return None
    return lengths[correct[0]], max(length for index, length in enumerate(lengths) if index != correct[0])


def answer_length_cue(component: Mapping[str, Any]) -> bool:
    """The only correct option is far longer than every other option (N2): the W5 repair rewrites the options to
    comparable lengths once, and a cue the repair leaves is a review note (QC run 8de1c76b, Q4)."""

    lengths = option_lengths(component)
    return lengths is not None and lengths[0] > IDM_MCQ_LENGTH_CUE_RATIO * lengths[1]


__all__ = [
    "ANSWER_LENGTH_CUE_CODE", "BALANCED_CODE", "LABELS_STRIPPED_CODE", "OPTION_LETTERS", "RELABELLED_CODE",
    "SHUFFLED_CODE", "answer_length_cue", "balanced_order", "encode_key_letters", "labelled_letters",
    "normalize_single_choice", "option_lengths", "option_order", "practice_key_shift", "question_ordinal",
    "relabel_explanation",
]
