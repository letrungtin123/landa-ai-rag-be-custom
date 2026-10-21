"""The specific details of the organization profile and the source that states them (QC run of 2026-10-21).

The profile is background, never lesson knowledge. A production run showed the path an invented detail took: the
context documents only say "IRC", the drafted profile said "xin IRC/ERC", the course design copied "Xin ERC (nếu cần)"
from the profile and the unit writer taught it; the unit's NEW_FACT check let it through because the design text
counts as known. ``ground_profile`` closes the first step (the distillation, after the final profile): an item
of a list (and the whole learner group or glossary entry) that holds a specific detail the source text given to
the model does not state is dropped; in ``organization_summary`` and ``culture_tone`` only the sentences that hold
one are dropped. Nothing is rewritten or added.

A specific detail (``specific_tokens``) is a whitespace-separated token, split on "/" and without the punctuation
around it, that contains a digit (a number, an amount, a code: "20.000", "110kV", "40:2021") or that is an acronym or
code of at least two letters written mostly in capitals ("IRC", "I-REC", "SCADA"). Ordinary words are never checked.
A source states one (``SourceText.states``) when it occurs there as a whole token, comparing NFC, case-folded,
whitespace-collapsed text, or, for one with digits, when a token of the source has the same digits once the
separators are removed ("3.400" and "3400").
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.org_profile import IdmOrgProfileV1

_COMPOUND_SEPARATOR: Final = "/"
_MIN_ACRONYM_LETTERS: Final = 2
# Every part of a source token that holds a digit, as written (its digits are compared without separators). It starts
# only where a part starts, so a long run without a digit costs one pass, not one per position.
_DIGIT_PART_RE: Final = re.compile(r"(?<![^\s/])[^\s/\d]*\d[^\s/]*")
# A sentence of organization_summary or culture_tone ends at . ! ? or … followed by a space (lines end one too).
_SENTENCE_END_RE: Final = re.compile(r"(?<=[.!?…])\s+")


def _trimmed(part: str) -> str:
    """``part`` without the punctuation around it ("(IRC)," -> "IRC")."""

    start, end = 0, len(part)
    while start < end and not part[start].isalnum():
        start += 1
    while end > start and not part[end - 1].isalnum():
        end -= 1
    return part[start:end]


def _is_specific(token: str) -> bool:
    if any(char.isdecimal() for char in token):
        return True
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= _MIN_ACRONYM_LETTERS and 2 * sum(char.isupper() for char in letters) > len(letters)


def specific_tokens(text: str) -> list[str]:
    """The specific details of ``text`` in reading order (see the module docstring), as written."""

    found: list[str] = []
    for word in unicodedata.normalize("NFC", text).split():
        for part in word.split(_COMPOUND_SEPARATOR):
            token = _trimmed(part)
            if token and _is_specific(token):
                found.append(token)
    return found


def _folded(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _digits(token: str) -> str:
    return "".join(char for char in token if char.isdecimal())


def _word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


@dataclass(frozen=True)
class SourceText:
    """A source read once: its folded text (NFC, case-folded, whitespace collapsed) and the digit sequences of its
    tokens."""

    text: str
    digits: frozenset[str]

    @classmethod
    def of(cls, texts: Iterable[str]) -> SourceText:
        folded = _folded("\n".join(texts))
        return cls(folded, frozenset(_digits(part) for part in _DIGIT_PART_RE.findall(folded)))

    def states(self, token: str) -> bool:
        """``token`` occurs in the source as a whole token, or has the digits of one of its tokens."""

        key = _folded(token)
        start = self.text.find(key) if key else -1
        while start >= 0:
            end = start + len(key)
            if ((start == 0 or not _word_char(self.text[start - 1]))
                    and (end == len(self.text) or not _word_char(self.text[end]))):
                return True
            start = self.text.find(key, start + 1)
        digits = _digits(token)
        return bool(digits) and digits in self.digits


# --- The distillation: drop what the source does not state ----------------------------------------------------------
def _kept_sentences(text: str, grounded: Callable[[str], bool]) -> tuple[str, int]:
    """``text`` without its sentences that ``grounded`` refuses, and how many were dropped; unchanged when none is."""

    lines: list[str] = []
    dropped = 0
    for line in text.split("\n"):
        sentences = _SENTENCE_END_RE.split(line)
        kept = [sentence for sentence in sentences if grounded(sentence)]
        dropped += len(sentences) - len(kept)
        if len(kept) == len(sentences):
            lines.append(line)
        elif kept:
            lines.append(" ".join(kept))
    return ("\n".join(lines), dropped) if dropped else (text, 0)


def ground_profile(profile: IdmOrgProfileV1, source_texts: Sequence[str]) -> tuple[IdmOrgProfileV1, dict[str, int]]:
    """``profile`` without the items and sentences that hold a specific detail ``source_texts`` (the text the model
    was given) does not state, and the number dropped per field (only fields that lost something). Nothing is
    rewritten; a profile left empty is returned as it is (authors edit the draft)."""

    source = SourceText.of(source_texts)
    known: dict[str, bool] = {}

    def grounded(*texts: str) -> bool:
        for text in texts:
            for token in specific_tokens(text):
                if token not in known:
                    known[token] = source.states(token)
                if not known[token]:
                    return False
        return True

    removed: Counter[str] = Counter()

    def kept[T](name: str, items: Sequence[T], texts: Callable[[T], tuple[str, ...]]) -> list[T]:
        result = [item for item in items if grounded(*texts(item))]
        removed[name] += len(items) - len(result)
        return result

    summary, summary_dropped = _kept_sentences(profile.organization_summary, grounded)
    tone, tone_dropped = _kept_sentences(profile.culture_tone, grounded)
    removed.update(organization_summary=summary_dropped, culture_tone=tone_dropped)
    fields: dict[str, Any] = {
        "organization_summary": summary,
        "products_services": kept("products_services", profile.products_services, lambda item: (item,)),
        "learner_groups": [group.model_dump(mode="json") for group in kept(
            "learner_groups", profile.learner_groups, lambda group: (group.name, group.description))],
        "glossary": [entry.model_dump(mode="json") for entry in kept(
            "glossary", profile.glossary, lambda entry: (entry.term, entry.meaning))],
        "key_processes": kept("key_processes", profile.key_processes, lambda item: (item,)),
        "culture_tone": tone,
        "design_conventions": kept("design_conventions", profile.design_conventions, lambda item: (item,)),
    }
    counts = {name: count for name, count in sorted(removed.items()) if count}
    return (IdmOrgProfileV1.model_validate(fields), counts) if counts else (profile, {})

