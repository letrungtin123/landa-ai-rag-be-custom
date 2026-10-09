"""Single-choice checks against what the unit itself shows (QC run ab8d67e1, R2).

The 4-gram copy check and the distinctive-pair check (``qa.copied_options``, ``qa.reworded_key``) need long keys
with many distinctive word pairs. The run showed three more ways a question gives itself away or has no single
answer, each read against the passages of the html slots before the question (a sentence of a paragraph or
quotation, a list item, a table row) and the unit's FAQ answers:

* a key that paraphrases ONE passage in the same word order (c4.l1.u1 restated its blockquote);
* a short key that is the only option the html shows, or a case that copies an example passage holding the key;
* a contested key: the options name labels the unit teaches ("Bậc 01", "Bậc 02" …) and the question's case meets
  another label's criteria at least as well as the key's (c1.l2.u2: "Bậc 01" for a case with ISO and CE while
  the FAQ right after says ISO, CE, FDA only reach "Bậc 02").

Pure functions on the writer's payload; thresholds and their calibration live in ``app.idm.policy``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.policy import (
    IDM_ANSWER_CONTESTED_MIN_WORDS,
    IDM_ANSWER_QUOTE_MIN_KEY_WORDS,
    IDM_ANSWER_QUOTE_MIN_MARGIN,
    IDM_ANSWER_QUOTE_MIN_SHARE,
    IDM_ANSWER_SHORT_DISTRACTOR_MAX_SHARE,
    IDM_ANSWER_SHORT_KEY_MAX_WORDS,
    IDM_ANSWER_STEM_COPY_MIN_SHARE,
    IDM_ANSWER_STEM_PASSAGE_MIN_WORDS,
)
from app.idm.text import content_words, idm_fold

_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?;])\s+")
# The head of a taught label: "Bậc 01 - Vietnam Manufacturing" -> "Bậc 01", "Hiện đại hóa: …" -> "Hiện đại hóa".
_HEAD_END_RE: Final = re.compile(r"\s+[-–—|]\s+|\s*:\s*|\s*\(")  # noqa: RUF001 - dashes
# An option that starts with a numbered label ("Cấp 2 vì …", "Bậc 01 - …") applies a taught rule to the case.
_NUMBERED_LABEL_RE: Final = re.compile(r"^\s*[^\W\d_]+(?:\s+[^\W\d_]+)?\s+0?\d{1,2}(?!\d)")
_MIN_HEAD_CHARS: Final = 3
# A label is looked for at the start of an option (its first words), never in its reason.
_HEAD_SEARCH_CHARS: Final = 60
_MIN_OPTIONS: Final = 2
_LIST_KINDS: Final = frozenset({"bullets", "steps"})
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})


@dataclass(frozen=True)
class UnitTeaching:
    """What a question can be read against: html passages before it, taught (label, criteria) pairs, FAQ text."""

    passages: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]
    faq_sentences: tuple[str, ...]


def _blocks(component: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    for section in sections if isinstance(sections, list) else []:
        for block in section.get("blocks") or [] if isinstance(section, dict) else []:
            if isinstance(block, dict):
                yield block


def unit_teaching(html_before: Sequence[Mapping[str, Any]], faq: Sequence[Mapping[str, Any]]) -> UnitTeaching:
    """The passages and taught labels of the html slots before a question, and the sentences of the unit's FAQ."""

    passages: list[str] = []
    rows: list[tuple[str, str]] = []
    for component in html_before:
        for block in _blocks(component):
            kind = block.get("kind")
            if kind in _TEXT_KINDS:
                passages.extend(part for part in _SENTENCE_RE.split(str(block.get("text") or "")) if part.strip())
            elif kind in _LIST_KINDS:
                for item in block.get("items") or []:
                    passages.append(str(item))
                    label, separator, value = str(item).partition(":")
                    if separator and value.strip():
                        rows.append((label, value))
            elif kind == "table":
                for row in block.get("rows") or []:
                    if isinstance(row, dict):
                        label, value = str(row.get("label") or ""), str(row.get("value") or "")
                        passages.append(f"{label}: {value}")
                        rows.append((label, value))
    sentences = [part for item in faq for part in _SENTENCE_RE.split(str(item.get("answer") or "")) if part.strip()]
    return UnitTeaching(tuple(passages), tuple(rows), tuple(sentences))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(idm_fold(text))


def _ordered_share(part: Sequence[str], whole: Sequence[str]) -> float:
    """Share of ``part``'s words found in ``whole`` in the same order (longest common subsequence)."""

    if not part or not whole or not set(part) & set(whole):
        return 0.0
    previous = [0] * (len(whole) + 1)
    for word in part:
        current = [0]
        for position, other in enumerate(whole):
            current.append(previous[position] + 1 if word == other else max(previous[position + 1], current[-1]))
        previous = current
    return previous[-1] / len(part)


def _choices(component: Mapping[str, Any]) -> tuple[list[str], int | None]:
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    return [str(choice.get("text") or "") for choice in choices], (correct[0] if len(correct) == 1 else None)


def _head(label: str) -> str:
    return idm_fold(_HEAD_END_RE.split(label.strip(), maxsplit=1)[0]).strip(" .")


def _names(folded: str, head: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(head)}(?!\w)", folded) is not None


def labelled_options(texts: Sequence[str], rows: Sequence[tuple[str, str]]) -> dict[int, str]:
    """Option index -> the head of the one taught label it starts with ("bac 01")."""

    heads = {head for label, _value in rows if len(head := _head(label)) >= _MIN_HEAD_CHARS}
    mapped: dict[int, str] = {}
    for index, text in enumerate(texts):
        start = idm_fold(text)[:_HEAD_SEARCH_CHARS]
        named = [head for head in heads if _names(start, head)]
        longest = max(named, key=len) if named else None
        # One label only: an option naming two ("Bậc 02 chưa tới Bậc 03") is not a label of either.
        if longest is not None and all(head in longest for head in named):
            mapped[index] = longest
    return mapped


def quoted_key(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key gives itself away against ONE passage shown before the question (R2):

    * it restates the passage in the same word order (a quotation, a list item, a row) far more than any distractor
      restates any passage (keys of 8 words or more that do not name a taught label);
    * a short key is shown word for word while no distractor appears in the html (the only familiar option);
    * the question's case copies an example passage that holds the key and no distractor.
    """

    texts, key = _choices(component)
    passages = [tokens for passage in teaching.passages if (tokens := _tokens(passage))]
    if key is None or len(texts) < _MIN_OPTIONS or not passages:
        return False
    options = [_tokens(text) for text in texts]
    best = [max(_ordered_share(option, passage) for passage in passages) for option in options]
    others = max(best[index] for index in range(len(options)) if index != key)
    words = len(options[key])
    if words <= IDM_ANSWER_SHORT_KEY_MAX_WORDS and best[key] == 1.0 and others <= IDM_ANSWER_SHORT_DISTRACTOR_MAX_SHARE:
        return True
    # A key that names a taught label restates its rule by design ("Cấp 2 vì khách phàn nàn lần thứ hai").
    labelled = (key in labelled_options(texts, teaching.rows)
                or sum(bool(_NUMBERED_LABEL_RE.match(text)) for text in texts) >= _MIN_OPTIONS)
    if (not labelled and words >= IDM_ANSWER_QUOTE_MIN_KEY_WORDS and best[key] >= IDM_ANSWER_QUOTE_MIN_SHARE
            and best[key] - others >= IDM_ANSWER_QUOTE_MIN_MARGIN):
        return True
    stem = _tokens(str(component.get("question") or ""))
    for passage in passages:
        holds = [_ordered_share(option, passage) >= IDM_ANSWER_QUOTE_MIN_SHARE for option in options]
        if (len(passage) >= IDM_ANSWER_STEM_PASSAGE_MIN_WORDS and holds[key] and sum(holds) == 1
                and _ordered_share(passage, stem) >= IDM_ANSWER_STEM_COPY_MIN_SHARE):
            return True
    return False


def contested_option(component: Mapping[str, Any], teaching: UnitTeaching) -> int | None:
    """A distractor whose taught criteria the question's case meets at least as well as the key's (R2), or None.

    Only for options that name labels the unit teaches: the criteria of a label are its row values and the
    sentences of the html and of the unit's FAQ answers that name it and no other option's label.
    """

    texts, key = _choices(component)
    mapped = labelled_options(texts, teaching.rows)
    if key is None or key not in mapped or len(set(mapped.values())) < _MIN_OPTIONS:
        return None
    heads = set(mapped.values())
    criteria: dict[str, list[str]] = {head: [] for head in heads}
    for label, value in teaching.rows:
        if (head := _head(label)) in criteria:
            criteria[head].append(value)
    for sentence in (*teaching.passages, *teaching.faq_sentences):
        folded = idm_fold(sentence)
        named = [head for head in heads if _names(folded, head)]
        if len(named) == 1:
            criteria[named[0]].append(sentence)
    label_words = {word for head in heads for word in content_words(head)}
    case = set(content_words(str(component.get("question") or "")))

    def score(head: str) -> int:
        return max((len((set(content_words(text)) - label_words) & case) for text in criteria[head]), default=0)

    key_score = score(mapped[key])
    rivals = [(score(head), index) for index, head in mapped.items() if head != mapped[key]]
    best, rival = max(rivals) if rivals else (0, None)
    return rival if best >= max(IDM_ANSWER_CONTESTED_MIN_WORDS, key_score) else None


__all__ = ["UnitTeaching", "contested_option", "labelled_options", "quoted_key", "unit_teaching"]
