"""Terms a course is about and the blocks that define them (QC course 364564, N4).

The course "BiC 5.0" never defined BiC: W1 put the formula "BiC = Lean System + Efficiency →
Preventive & Proactive" in one block with the programme's history, and W2 removed the block as
history. A block whose facts define a term of the course title, an objective or a Must Do is never
left out; the definition is what the rest of the course builds on.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Final

from app.idm.text import idm_fold

_WORD_RE: Final = re.compile(r"[^\W_]+(?:[.\-][^\W_]+)*", re.UNICODE)
# Folded words that open a definition without being the term ("Mô hình BiC là …", "The BiC model means …").
_DEFINITION_HEADS: Final = frozenset({
    "mo", "hinh", "khai", "niem", "dinh", "nghia", "phuong", "phap", "he", "thong", "cong", "thuc", "thuat", "ngu",
    "the", "a", "an", "model", "concept", "definition", "of", "term", "formula",
})
# A definition: "<subject> = …", "<subject>: …", "<subject> là …", "<subject> is/means/stands for …".
_SEPARATOR_RE: Final = re.compile(
    r"\s*(?:=|:|≡)\s*|\s+(?:là|nghĩa là|có nghĩa là|được hiểu là|được định nghĩa là|được gọi là|viết tắt của"
    r"|is|means|stands for|refers to|is defined as)\s+", re.IGNORECASE)
_CLAUSE_RE: Final = re.compile(r"(?<=[.!?;])\s+")
_PARENTHETICAL_RE: Final = re.compile(r"\([^()]*\)")
_SUBJECT_MAX_WORDS: Final = 6
# A subject names the term plus at most this many words ("BiC 5.0").
_SUBJECT_EXTRA_WORDS: Final = 2
_MIN_TERM_CHARS: Final = 2
_ACRONYM_CAPITALS: Final = 2
_MAX_LABEL_WORDS: Final = 3


def _is_acronym(word: str) -> bool:
    """"BiC", "ERA", "CEO", "5Why": two capitals, or a capital and a digit."""

    capitals = sum(char.isupper() for char in word)
    return len(word) >= _MIN_TERM_CHARS and (capitals >= _ACRONYM_CAPITALS
                                             or (capitals >= 1 and any(char.isdigit() for char in word)))


def _sentence_words(text: str) -> list[list[str]]:
    return [_WORD_RE.findall(clause) for clause in re.split(r"[.:;!?]\s+", unicodedata.normalize("NFC", text))]


def course_terms(titles: Iterable[str], statements: Iterable[str]) -> list[str]:
    """Terms of the course: acronyms of the title texts (course title hint, document names) and of the
    objective and Must Do statements, plus the capitalised names inside those statements ("Tam Doanh")."""

    terms: list[str] = []
    for title in titles:
        terms.extend(word for clause in _sentence_words(title) for word in clause if _is_acronym(word))
    for statement in statements:
        for words in _sentence_words(statement):
            run: list[str] = []
            for position, word in enumerate([*words, ""]):
                acronym = _is_acronym(word)
                if acronym:
                    terms.append(word)
                if position > 0 and not acronym and word[:1].isupper():
                    run.append(word)
                    continue
                if run:
                    terms.append(" ".join(run))
                run = []
    unique = list(dict.fromkeys(term for term in terms if len(term) >= _MIN_TERM_CHARS))
    return sorted(unique, key=lambda term: (-len(term), term))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(idm_fold(text).replace(".", " "))


def _subject_names(subject: str, term_words: Sequence[str]) -> bool:
    words = _words(_PARENTHETICAL_RE.sub(" ", subject))
    while words and words[0] in _DEFINITION_HEADS and words[:len(term_words)] != list(term_words):
        words = words[1:]
    return (0 < len(words) <= _SUBJECT_MAX_WORDS and words[:len(term_words)] == list(term_words)
            and len(words) - len(term_words) <= _SUBJECT_EXTRA_WORDS)


def defines(text: str, term: str) -> bool:
    """``text`` defines ``term``: a clause (or the clause after a short "Label:") whose subject is the term."""

    term_words = _words(term)
    if not term_words:
        return False
    for clause in _CLAUSE_RE.split(unicodedata.normalize("NFC", " ".join(text.split()))):
        candidates = [clause.lstrip("-•*–—\"'“‘« ")]  # noqa: RUF001 - bullets and quotes a PDF line starts with
        label = re.match(r"^[^:=]{1,40}:\s*", candidates[0])
        if label is not None and len(_words(label.group(0))) <= _MAX_LABEL_WORDS:
            candidates.append(candidates[0][label.end():])
        for candidate in candidates:
            separator = _SEPARATOR_RE.search(candidate)
            if separator is not None and separator.start() > 0 and _subject_names(candidate[:separator.start()],
                                                                                    term_words):
                return True
    return False


def defined_terms(fact_texts: Iterable[str], terms: Sequence[str]) -> list[str]:
    """The ``terms`` that one of the facts defines, in ``terms`` order."""

    texts = list(fact_texts)
    return [term for term in terms if any(defines(text, term) for text in texts)]


def mentions(text: str, term: str) -> bool:
    """``text`` contains ``term`` as whole words (accent-folded)."""

    words, term_words = _words(text), _words(term)
    size = len(term_words)
    return bool(size) and any(words[index:index + size] == term_words for index in range(len(words) - size + 1))


__all__ = ["course_terms", "defined_terms", "defines", "mentions"]
