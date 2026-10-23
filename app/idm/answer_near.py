"""A key the unit's own words give away, short of the leak check (run 2f8781d6, B6).

QC of run 2f8781d6: 6 of 12 questions were answered by matching the words shown right before them, and ``ANSWER_LEAK``
(``app.idm.answer_checks``, ``qa_checks.copied_options`` and ``reworded_key``) caught none: the copies were shorter or
looser than its measures, or sat in the case itself. Two soft findings read the same material more loosely; each gets
one targeted repair (keep the principle, change the case details and the wording) and then stays an edit note, never a
fallback:

* ``ANSWER_NEAR_EXAMPLE_CODE`` (``near_example``), when the correct option, or the listed item it names (``resolved_
  choices``), is read against the passages of the html slots before the question (sentences, list items, table rows;
  never a worksheet's input data, which a question may practise on):

  - it shares a run of at least ``IDM_ANSWER_NEAR_RUN_WORDS`` words with one passage
    (``IDM_ANSWER_NEAR_RUN_CONTENT_WORDS`` of them content words), longer than any distractor's (c2.l4.u2: "Ban Truyền
    thông và Ban Giao thương" from the Back-stage row);
  - or most of its content words are those of one passage (``IDM_ANSWER_NEAR_MIN_SHARE``) and clearly more than any
    distractor's (``IDM_ANSWER_NEAR_MIN_MARGIN``; c1.l2.u1: "căng thẳng và bực bội vì … mất kiểm soát" from the row of
    the key's layer);
  - or the case copies a passage (``IDM_ANSWER_NEAR_CASE_RUN_WORDS`` words in a row) that holds a word pair only the key
    has (c2.l1.u3: the worked example's company "nhận thấy … bản demo" decided the purchase);
  - or the case names the key's need in the key's own words (``IDM_ANSWER_NEAR_STEM_RUN_WORDS`` in a row) while no
    distractor shares more than two (c2.l2.u3: "kiểm soát tiến độ tự động" / "báo cáo tiến độ tự động").

* ``ANSWER_ITEM_LENGTH_CODE`` (``item_length_cue``): the options only name the case's listed items and the item of the
  key is clearly the longest one (QC: 5 of 6 lists; c1.l1.u1 106 against 97 characters, c2.l1.u3 70 against 61).

Calibrated on the run's 11 provider questions (the 5 leaking ones caught, the good c1.l2.u2 and c1.l1.u2 not) and
checked on the 23 questions of two earlier runs of the same source. Pure functions; thresholds in ``app.idm.policy``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, Literal

from app.idm.answer_checks import _labelled_key, resolved_choices, unit_teaching
from app.idm.policy import (
    IDM_ANSWER_NEAR_CASE_RUN_WORDS,
    IDM_ANSWER_NEAR_MIN_KEY_WORDS,
    IDM_ANSWER_NEAR_MIN_MARGIN,
    IDM_ANSWER_NEAR_MIN_SHARE,
    IDM_ANSWER_NEAR_RUN_CONTENT_WORDS,
    IDM_ANSWER_NEAR_RUN_WORDS,
    IDM_ANSWER_NEAR_STEM_OTHER_MAX_WORDS,
    IDM_ANSWER_NEAR_STEM_RUN_WORDS,
    IDM_MCQ_ITEM_LENGTH_CUE_MIN_CHARS,
    IDM_MCQ_ITEM_LENGTH_CUE_RATIO,
)
from app.idm.quiz_text import named_items
from app.idm.text import content_words, idm_fold
from app.idm.worksheet_input import is_input_section

ANSWER_NEAR_EXAMPLE_CODE: Final = "IDM_W5_ANSWER_NEAR_EXAMPLE"
ANSWER_ITEM_LENGTH_CODE: Final = "IDM_W5_ANSWER_ITEM_LENGTH_CUE"
_WORD_RE: Final = re.compile(r"\w+")
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?;])\s+")
_MIN_OPTIONS: Final = 2
_MIN_ITEMS: Final = 3
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})
NearKind = Literal["option", "item", "case", "stem"]


def _tokens(text: str) -> list[str]:
    """Lower-case words with their accents (a folded "độ" would meet "do")."""

    return _WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def _run(first: Sequence[str], second: Sequence[str]) -> tuple[int, int]:
    """(length, start in ``first``) of the longest run of words ``first`` and ``second`` share."""

    best, start = 0, 0
    previous = [0] * (len(second) + 1)
    for index, word in enumerate(first):
        current = [0]
        for position, other in enumerate(second):
            current.append(previous[position] + 1 if word == other else 0)
            if current[-1] > best:
                best, start = current[-1], index - current[-1] + 1
        previous = current
    return best, start


def _bigrams(text: str) -> set[tuple[str, str]]:
    words = _tokens(text)
    return set(pairwise(words))


def passages(html_before: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sentences, list items and table rows of the html slots before a question (a worksheet's input data left
    out: a question may practise on it, review of dfb6db2)."""

    found: list[str] = []
    for component in html_before:
        semantic = component.get("semantic_content")
        sections = [section for section in (semantic.get("sections") or [] if isinstance(semantic, dict) else [])
                    if isinstance(section, dict)]
        blocks = [block for section in sections for block in section.get("blocks") or [] if isinstance(block, dict)]
        worksheet = any(block.get("kind") == "task" for block in blocks)
        for section in sections:
            if worksheet and is_input_section(section):
                continue
            for block in (block for block in section.get("blocks") or [] if isinstance(block, dict)):
                if block.get("kind") in _TEXT_KINDS:
                    found.extend(part for part in _SENTENCE_RE.split(str(block.get("text") or "")) if part.strip())
                found.extend(str(item) for item in block.get("items") or [])
                found.extend(f"{row.get('label') or ''}: {row.get('value') or ''}" for row in block.get("rows") or []
                             if isinstance(row, dict))
    return found


def _options(component: Mapping[str, Any]) -> tuple[list[str], int | None, bool]:
    """(option texts read as their items, the key's index, whether the key names an item)."""

    read = resolved_choices(component)
    served = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    choices = [choice for choice in read.get("choices") or [] if isinstance(choice, dict)]
    keys = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    key = keys[0] if len(keys) == 1 else None
    texts = [str(choice.get("text") or "") for choice in choices]
    named = key is not None and key < len(served) and str(served[key].get("text") or "") != texts[key]
    return texts, key, named


def _case(question: str) -> str:
    """The stem without its listed items: the case the question gives."""

    return " ".join(line for line in question.splitlines() if not line.strip().startswith("-"))


def near_example(component: Mapping[str, Any], html_before: Sequence[Mapping[str, Any]]) -> NearKind | None:
    """What gives the key away by its words (module docstring): the key option, the listed item it names, the case
    that copies the passage holding it, or the stem that names it; ``None`` when nothing does."""

    texts, key, named = _options(component)
    # A key that opens with a taught label applies its rule to the case by design ("Cấp 2 vì khách phàn nàn lần thứ
    # hai"), as for the leak check (``answer_checks._labelled_key``).
    if key is None or len(texts) < _MIN_OPTIONS or _labelled_key(texts, key, unit_teaching(list(html_before), [])):
        return None
    shown = [_tokens(passage) for passage in passages(html_before)]
    shown_texts = passages(html_before)
    others = [index for index in range(len(texts)) if index != key]
    words = [_tokens(text) for text in texts]

    def longest(index: int) -> tuple[int, list[str]]:
        best: tuple[int, list[str]] = (0, [])
        for passage in shown:
            length, start = _run(words[index], passage)
            if length > best[0]:
                best = (length, words[index][start:start + length])
        return best

    key_run, run_words = longest(key)
    if (key_run >= IDM_ANSWER_NEAR_RUN_WORDS and len(content_words(" ".join(run_words)))
            >= IDM_ANSWER_NEAR_RUN_CONTENT_WORDS and key_run > max(longest(index)[0] for index in others)):
        return "item" if named else "option"
    content = [set(content_words(text)) for text in texts]

    def share(index: int) -> float:
        return max((len(content[index] & set(content_words(passage))) / len(content[index])
                    for passage in shown_texts), default=0.0) if content[index] else 0.0

    if (len(content[key]) >= IDM_ANSWER_NEAR_MIN_KEY_WORDS and share(key) >= IDM_ANSWER_NEAR_MIN_SHARE
            and share(key) - max(share(index) for index in others) >= IDM_ANSWER_NEAR_MIN_MARGIN):
        return "item" if named else "option"
    case = _tokens(_case(str(component.get("question") or "")))
    own = _bigrams(texts[key]) - set().union(*(_bigrams(texts[index]) for index in others))
    if any(_run(case, passage)[0] >= IDM_ANSWER_NEAR_CASE_RUN_WORDS and own & _bigrams(text)
           for passage, text in zip(shown, shown_texts, strict=True)):
        return "case"
    stem = [_run(words[index], case)[0] for index in range(len(texts))]
    if (stem[key] >= IDM_ANSWER_NEAR_STEM_RUN_WORDS
            and max(stem[index] for index in others) <= IDM_ANSWER_NEAR_STEM_OTHER_MAX_WORDS):
        return "stem"
    return None


def item_length_cue(component: Mapping[str, Any]) -> tuple[int, int] | None:
    """(characters of the key's item, of the longest other item) when the options only name the case's listed items
    and the key's item is clearly the longest one; ``None`` otherwise."""

    items = named_items(str(component.get("question") or ""))
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    names = [idm_fold(str(choice.get("text") or "")).strip(" .:;,") for choice in choices]
    keys = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    if len(items) < _MIN_ITEMS or len(keys) != 1 or not names or not all(name in items for name in names):
        return None
    lengths = [len(" ".join(items[name].split())) for name in names]
    key = lengths[keys[0]]
    longest = max(length for index, length in enumerate(lengths) if index != keys[0])
    if key >= IDM_MCQ_ITEM_LENGTH_CUE_RATIO * longest and key - longest >= IDM_MCQ_ITEM_LENGTH_CUE_MIN_CHARS:
        return key, longest
    return None


__all__ = ["ANSWER_ITEM_LENGTH_CODE", "ANSWER_NEAR_EXAMPLE_CODE", "item_length_cue", "near_example", "passages"]
