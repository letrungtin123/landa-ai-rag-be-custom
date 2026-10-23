"""Stem items labelled with option letters (owner decision 2026-10-22, run dfe78482; review 2 of 2026-10-22).

"- Phương án A: …" reads as an answer letter once the options are shuffled ("A: Phương án C: …"). The server numbers
such items ("Phương án 1", "Phương án 2" …) in the stem, the choices, the explanation, the hints and the title, but
only when it is certain (review 2: a wrong rename is worse than none):

* the label is a generic one (``_LABELS``: "phương án", "tình huống", "option", "plan" …), never a name that holds a
  letter ("Vitamin A", "Nhóm máu A", "Hạng B", "loại C", "Nhóm A/B/C", "B2B");
* the stem's labelled items are lettered A, B, C … in order after that one label, and every listed item ("- …") is
  one of them;
* every capital item letter of the quiz is right after that label, or chained to one that is ("phương án A và B",
  "A, B và C", "A hay B", "A hoặc B", "A or B", "A/B"), or an option prefix of the explanation ("A - Đúng vì …").

A letter after another label ("Lựa chọn A"), a lowercase one ("phương án a"), a bare one ("B không giải quyết") or the
label with a number already ("phương án 2") leaves the whole quiz as written; its stem gets an edit note instead
(``quiz_text.QUESTION_ITEM_LETTERS_CODE``). The English article ("a plan a manager likes") is not a letter.

Run 5496155c (owner decision 2026-10-22): "A: Phương án 2" still confused learners, because the options A-D are the
"phương án" of the question. When the label is the options' own noun ("phương án", "lựa chọn", "option"), the renaming
also writes a neutral noun at exactly the places it renumbers ("Phương án A" -> "Cách 1", "phương án A và B" -> "cách 1
và 2", "Option A" -> "Approach 1"), keeping a capital where there was one; the noun alone ("Phương án nào đúng?") stays.
Safety: the same certain positions as the letters, all of them or none.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Final

from app.idm.mcq import segment_starts

ITEM_LETTERS_NUMBERED: Final = "w5_problem_item_letters_numbered"
_ITEM_LETTERS: Final = "ABCDEFGH"
_MIN_ITEMS: Final = 2
_SINGLE_CHOICE_TYPES: Final = frozenset({"multiple_choice", "single_choice"})
# The text fields of a single-choice quiz a renaming covers (choices and hints are lists).
_TEXT_FIELDS: Final = ("title", "question", "explanation")
# Generic label nouns and their canonical form (an English plural is its singular).
_LABELS: Final = {
    "phương án": "phương án", "lựa chọn": "lựa chọn", "tình huống": "tình huống", "trường hợp": "trường hợp",
    "cách": "cách", "đề xuất": "đề xuất", "kịch bản": "kịch bản",
    "option": "option", "options": "option", "case": "case", "cases": "case", "scenario": "scenario",
    "scenarios": "scenario", "plan": "plan", "plans": "plan", "proposal": "proposal", "proposals": "proposal",
}
_ENGLISH: Final = frozenset({"option", "case", "scenario", "plan", "proposal"})
# Labels that are the options' own noun, and the neutral noun a renamed item gets instead.
_NEUTRAL_NOUNS: Final = {"phương án": "cách", "lựa chọn": "cách", "option": "approach"}
_LABEL: Final = (r"(?i:phương[ \t]+án|lựa[ \t]+chọn|tình[ \t]+huống|trường[ \t]+hợp|cách|đề[ \t]+xuất|kịch[ \t]+bản"
                 r"|options?|cases?|scenarios?|plans?|proposals?)")
_LABEL_RE: Final = re.compile(rf"(?<!\w)({_LABEL})(?!\w)")
# A stem item: the label, a capital letter, an optional note in brackets ("Phương án A (rẻ):") and a colon.
_ITEM_RE: Final = re.compile(rf"(?<!\w)({_LABEL})[ \t]+([A-H])(?:[ \t]+\([^()\n]{{1,80}}\))?[ \t]*:")
_LIST_MARKERS: Final = frozenset({"-", "–", "•", "*"})  # noqa: RUF001 - the en dash is a marker
_LINE_MARKER_RE: Final = re.compile(r"^[ \t]*[-–•*][ \t]+(?=\S)")  # noqa: RUF001 - the en dash is a marker
_INLINE_MARKER_RE: Final = re.compile(r"(?<=\S)[ \t]+[-–•][ \t]+(?=\S)")  # noqa: RUF001 - the en dash is a marker
# The first letter after a label, then the letters chained to it.
_FIRST_LETTER_RE: Final = re.compile(r"[ \t]+([A-Ha-h])(?!\w)")
_CHAIN_RE: Final = re.compile(r"(?:[ \t]*[,/&][ \t]*(?:(?i:và|hoặc|hay|and|or)[ \t]+)?|[ \t]+(?i:và|hoặc|hay|and|or)"
                              r"[ \t]+)([A-Ha-h])(?!\w)")
_NUMBER_AFTER_RE: Final = re.compile(r"[ \t]+\d")
_LETTER_RE: Final = re.compile(r"(?<!\w)[A-Ha-h](?!\w)")
# The English article: "a" (or "A" at a sentence start) before a lowercase word that is not a verb of a letter label
# ("A is cheaper" names plan A).
_ARTICLE_AFTER_RE: Final = re.compile(r"[ \t]+([a-z]+)(?!\w)")
_LABEL_VERBS: Final = frozenset({"is", "was", "has", "had", "and", "or", "vs", "would", "will", "can", "could",
                                 "should", "may", "might", "must", "does", "did", "costs", "keeps", "gives", "works"})
_SENTENCE_STARTS: Final = frozenset({"", ".", "!", "?", ":"})
# Review of dfb6db2 (item 9): "lựa chọn"/"chọn" after a modal or a subject is a verb ("Bạn nên lựa chọn A, B hay C?"),
# no item label: such a quiz is left as written (an edit note), never "Bạn nên cách 1, 2 hay 3?".
_VERB_BEFORE_RE: Final = re.compile(r"(?<!\w)(?i:nên|hãy|sẽ|cần|phải|muốn|bạn|để|được|thể|will|should|would|to|you)"
                                    r"[ \t]+$")


def _canonical(label: str) -> str:
    return _LABELS[" ".join(label.casefold().split())]


def _stem_items(question: str) -> list[tuple[str, str, int, str]]:
    """(canonical label, letter, line number, place) of each stem item named with a generic label and a capital letter;
    place "line" after a list marker at a line start, "inline" after a marker inside a line ("…: - Phương án A: … -
    Phương án B: …"), "text" at a line start or after a colon or a semicolon ("…: Phương án A: …; Phương án B: …")."""

    items = []
    for match in _ITEM_RE.finditer(question):
        line_start = question.rfind("\n", 0, match.start()) + 1
        before = question[line_start:match.start()].strip()
        if before in _LIST_MARKERS:
            place = "line"
        elif before[-1:] in _LIST_MARKERS and before[-2:-1].isspace():
            place = "inline"
        elif not before or before.endswith((":", ";")):
            place = "text"
        else:
            continue
        items.append((_canonical(match.group(1)), match.group(2), question.count("\n", 0, match.start()), place))
    return items


def stem_item_letters(component: Mapping[str, Any]) -> list[str]:
    """The letters of the stem items of ``component`` named with a generic label ("- Phương án A: …"), when there are
    at least two (one lettered item reads as no answer letter)."""

    question = component.get("question")
    if not isinstance(question, str):
        return []
    letters = [item[1] for item in _stem_items(unicodedata.normalize("NFC", question))]
    return letters if len(letters) >= _MIN_ITEMS else []


def _listed_alone(question: str, items: list[tuple[str, str, int, str]]) -> bool:
    """Whether every listed entry of the stem is one of its items: a line that starts with a list marker starts with
    one, and on a line with items after inline markers every inline marker starts one."""

    for number, line in enumerate(question.split("\n")):
        places = [place for _label, _letter, at, place in items if at == number]
        if _LINE_MARKER_RE.match(line) and "line" not in places:
            return False
        if "inline" in places and len(_INLINE_MARKER_RE.findall(line)) != places.count("inline"):
            return False
    return True


def _plan(question: str) -> tuple[str, dict[str, str]] | None:
    """The label and the letter -> number map of the stem items: at least two, lettered A, B, C … in order after one
    label, every listed entry one of them; ``None`` otherwise."""

    items = _stem_items(question)
    labels = {item[0] for item in items}
    letters = [item[1] for item in items]
    if (len(labels) != 1 or len(letters) < _MIN_ITEMS or letters != list(_ITEM_LETTERS[:len(letters)])
            or not _listed_alone(question, items)):
        return None
    return labels.pop(), {letter: str(position) for position, letter in enumerate(letters, start=1)}


def _article(text: str, start: int, end: int) -> bool:
    """Whether the letter at ``start``..``end`` of an English text is the article "a" ("a plan a manager likes", "A
    manager …" at a sentence start)."""

    after = _ARTICLE_AFTER_RE.match(text, end)
    if after is None or after.group(1) in _LABEL_VERBS:
        return False
    if text[start] == "a":
        return True
    before = text[:start].rstrip(" \t")
    return text[start] == "A" and (before[-1:] in _SENTENCE_STARTS or before.endswith("\n"))


def _noun(label: str, neutral: str) -> str:
    """``neutral`` with the capital of ``label`` ("Phương án" -> "Cách", "phương án" -> "cách")."""

    return neutral[:1].upper() + neutral[1:] if label[:1].isupper() else neutral


def _scan(text: str, label: str, numbers: Mapping[str, str], prefixes: set[int]) -> list[tuple[int, int, str]] | None:
    """(start, end, replacement) of each item letter of ``text`` to rename, and of the label noun before it when it is
    the options' own noun (``_NEUTRAL_NOUNS``); ``None`` when a letter cannot be mapped with certainty. ``prefixes``:
    the positions of the explanation's option prefixes ("A - …")."""

    english = label in _ENGLISH
    neutral = _NEUTRAL_NOUNS.get(label)
    renames: list[tuple[int, int, str]] = []
    for found in _LABEL_RE.finditer(text):
        same = _canonical(found.group(1)) == label
        if same and _NUMBER_AFTER_RE.match(text, found.end()):
            return None
        end, pattern = found.end(), _FIRST_LETTER_RE
        while (letter := pattern.match(text, end)) is not None:
            char = letter.group(1)
            if english and char == "a" and _article(text, letter.start(1), letter.end(1)):
                break
            if char.upper() in numbers and (char.islower() or not same):
                return None
            if same and char not in numbers:
                return None
            if same:
                if neutral and pattern is _FIRST_LETTER_RE and _VERB_BEFORE_RE.search(text, 0, found.start(1)):
                    return None
                if neutral and pattern is _FIRST_LETTER_RE:
                    renames.append((found.start(1), found.end(1), _noun(found.group(1), neutral)))
                renames.append((letter.start(1), letter.end(1), numbers[char]))
            end, pattern = letter.end(), _CHAIN_RE
    claimed = {start for start, _end, _value in renames}
    for token in _LETTER_RE.finditer(text):
        start, char = token.start(), token.group()
        if (start in claimed or start in prefixes or char not in numbers
                or (english and _article(text, start, token.end()))):
            continue
        return None
    return renames


def _texts(component: Mapping[str, Any]) -> list[tuple[str, int | None, str]]:
    """(field, position in its list or ``None``, text) of every learner text of the quiz."""

    texts: list[tuple[str, int | None, str]] = [
        (key, None, value) for key in _TEXT_FIELDS if isinstance(value := component.get(key), str)]
    for key, field in (("choices", "text"), ("hints", None)):
        values = component.get(key)
        for position, value in enumerate(values if isinstance(values, list) else ()):
            text = value.get(field) if field and isinstance(value, dict) else value
            if isinstance(text, str):
                texts.append((key, position, text))
    return texts


def number_item_letters(component: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """``component`` with its stem items numbered everywhere ("Phương án 1" …) when the renaming is certain (see the
    module docstring), and how many letters changed; ``(component, 0)`` otherwise."""

    kind = str(component.get("problem_type") or "multiple_choice").strip().lower()
    question = component.get("question")
    choices = component.get("choices")
    if kind not in _SINGLE_CHOICE_TYPES or not isinstance(question, str) or not isinstance(choices, list):
        return component, 0
    plan = _plan(unicodedata.normalize("NFC", question))
    if plan is None:
        return component, 0
    label, numbers = plan
    renamed: list[tuple[str, int | None, str]] = []
    count = 0
    for key, position, text in _texts(component):
        normal = unicodedata.normalize("NFC", text)
        prefixes = ({match.start(1) for match in segment_starts(normal, len(choices))} if key == "explanation"
                    else set())
        renames = _scan(normal, label, numbers, prefixes)
        if renames is None:
            return component, 0
        for start, end, value in sorted(renames, reverse=True):
            normal = f"{normal[:start]}{value}{normal[end:]}"
        if renames:
            renamed.append((key, position, normal))
            count += sum(value.isdigit() for _start, _end, value in renames)
    if not count:
        return component, 0
    result = {**component, "choices": [dict(choice) if isinstance(choice, dict) else choice for choice in choices]}
    if isinstance(component.get("hints"), list):
        result["hints"] = list(component["hints"])
    for key, position, text in renamed:
        if position is None:
            result[key] = text
        elif key == "choices":
            result[key][position]["text"] = text
        else:
            result[key][position] = text
    return result, count


# Run 2f8781d6 (c2.l3.u3): the writer and its repair both answered options that only named a stem item ("Ý 1"), shorter
# than the shared validator's 8 characters (PROBLEM_CHOICE_INVALID, a rule Node shares), and the whole unit fell back.
# A short item label of the stem is written out in a longer noun that still is not the options' own ("phương án").
# Review 5 (M1): only the stem's item labels ("- Ý 3:" at a line start) and the options that are exactly such a name
# change, never a word inside a sentence ("cần lưu ý 2 điều" became "lưu ý kiến 2 điều").
ITEM_LABELS_EXPANDED: Final = "w5_problem_item_labels_expanded"
# The shared validator's shortest choice (``validation.validate_staged_unit_content``, PROBLEM_CHOICE_INVALID).
_MIN_CHOICE_CHARS: Final = 8
_LONGER_LABELS: Final = {"ý": "ý kiến", "mục": "mục số", "câu": "câu số", "bước": "bước số", "item": "item no.",
                         "point": "point no."}
_SHORT_LABEL: Final = r"(?i:ý|mục|câu|bước|item|point)"
_SHORT_ITEM_RE: Final = re.compile(rf"^([ \t]*[-–•*][ \t]+)({_SHORT_LABEL})[ \t]+(\d{{1,2}})([ \t]*:)",  # noqa: RUF001
                                   re.MULTILINE)
_SHORT_USE_RE: Final = re.compile(rf"({_SHORT_LABEL})[ \t]+(\d{{1,2}})")


def expand_short_item_labels(component: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """``component`` with the short label of its stem items ("- Ý 1:", "- Mục 2:") written out ("- Ý kiến 1:", "- Mục
    số 2:") at the start of the stem's item lines and in the options that are only such a name ("Ý 1"), and how many
    of them changed; ``(component, 0)`` unless an option is such a name under 8 characters, the stem lists the items
    1, 2 … under that one label, and every option that is such a name names a listed item. Nothing else of the quiz
    changes (review 5, M1)."""

    kind = str(component.get("problem_type") or "multiple_choice").strip().lower()
    question = component.get("question")
    choices = component.get("choices")
    if kind not in _SINGLE_CHOICE_TYPES or not isinstance(question, str) or not isinstance(choices, list):
        return component, 0
    stem = unicodedata.normalize("NFC", question)
    listed = [(match.group(2).lower(), int(match.group(3))) for match in _SHORT_ITEM_RE.finditer(stem)]
    labels = {label for label, _number in listed}
    numbers = [number for _label, number in listed]
    if len(labels) != 1 or len(numbers) < _MIN_ITEMS or numbers != list(range(1, len(numbers) + 1)):
        return component, 0
    label = labels.pop()
    longer = _LONGER_LABELS[label]

    def noun(written: str) -> str:
        return longer[:1].upper() + longer[1:] if written[:1].isupper() else longer

    # The options that are only an item name of the label ("Ý 3", "Ý 3."), with the name's place in their text.
    named: dict[int, re.Match[str]] = {}
    for position, choice in enumerate(choices):
        text = unicodedata.normalize("NFC", str(choice.get("text") or "")) if isinstance(choice, dict) else ""
        match = _SHORT_USE_RE.fullmatch(text.strip(" .:;,"))
        if match is not None and match.group(1).lower() == label:
            named[position] = match
    if (not any(len(match.group(0)) < _MIN_CHOICE_CHARS for match in named.values())
            or any(int(match.group(2)) not in numbers for match in named.values())):
        return component, 0
    result = {**component, "choices": [dict(choice) if isinstance(choice, dict) else choice for choice in choices]}
    result["question"], count = _SHORT_ITEM_RE.subn(
        lambda match: f"{match.group(1)}{noun(match.group(2))} {match.group(3)}{match.group(4)}"
        if match.group(2).lower() == label else match.group(0), stem)
    for position, match in named.items():
        text = unicodedata.normalize("NFC", str(choices[position].get("text") or "")).strip(" .:;,")
        result["choices"][position]["text"] = f"{noun(match.group(1))} {match.group(2)}{text[match.end():]}"
        count += 1
    return result, count


__all__ = ["ITEM_LABELS_EXPANDED", "ITEM_LETTERS_NUMBERED", "expand_short_item_labels", "number_item_letters",
           "stem_item_letters"]
