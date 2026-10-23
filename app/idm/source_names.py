"""Names of tools, models and frameworks that must come from the source (run 2f8781d6, B9).

QC of run 2f8781d6 (T2): the course called the source's three-layer table of jobs, pains and gains "ma trận thấu cảm",
a name the source never uses (it has "Bản đồ thấu cảm" and "Hồ sơ khách hàng B2B", two different tools), in lesson
and unit titles, headings and objectives. A named tool phrase of a title or a heading ("ma trận …", "bản đồ …", "khung
…", "mô hình …", "bộ lọc …", "sơ đồ …", "… Canvas", "… Map", "… Matrix", "… Framework") must appear in the unit's
facts or the titles of its source sections; one that does not is a soft finding of its slot (one targeted repair with
a rename hint, then an edit note naming it), and in the unit's or the lesson's title a note line for the author.

Kept narrow on purpose: the title is read with its accents ("số đo", "bán đồ" are no tools); the qualifier is the two
words after a Vietnamese tool noun (one compound word, stopping at a function word, a number or punctuation) or the
capitalised words before an English one; the source is read without spaces, accents or punctuation (the PDF glues
words: "Bản đồThấu cảm", "Bộlọc Dự án"); a phrase whose next words name a code the source states ("khung quản trị
G-STIC") or that a source name in brackets follows ("sơ đồ dịch vụ (Service Blueprint)") is the source's; a time frame
is no tool ("khung thời gian"); "khung" and "mô hình" name a tool only before a capitalised word.

A unit only sees its own facts and its lesson's: "Bản đồ hành trình" (the source's overview page names it so) titles a
unit whose pages say "Lộ trình trải nghiệm … hành trình trải nghiệm khách hàng". A Vietnamese phrase whose qualifier
the source writes ("hành trình") is a plain description, unless the source gives that qualifier another tool noun
("bản đồ thấu cảm" for "ma trận thấu cảm": the name of a different tool); an English name is a proper name and must be
written as it is. On the run's titles: "ma trận thấu cảm" is caught; "Bản đồ hành trình", "Bộ lọc Dự án", "Ma trận ưu
tiên", "Service Blueprint" and "G-STIC" are not.

Review 5 (M9): a tool noun turned into a verb ("sơ đồ hóa", "bản đồ hoá") names no tool; an English tool noun names one
only after a proper name of two capitalised words at least, once the generic words of a Title Case heading are left out
("Value Proposition Canvas"; never "A Practical Framework", "Our Decision Framework", "Building Your Prioritization
Matrix", "… a Simple Map"). Review 6 (L2): inside a sentence that is no Title Case heading, one proper word is enough
("Hãy dùng Eisenhower Matrix …", "Bạn nên vẽ Empathy Map …"); a capitalised Vietnamese word before it is left out.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

from app.idm.text import idm_fold

NAME_NOT_IN_SOURCE_CODE: Final = "IDM_W5_NAME_NOT_IN_SOURCE"
# Read with accents ("số đo", "bán đồ" are no tools); the title is AI-written, so never glued.
_VI_TOOLS: Final = ("ma trận", "bản đồ", "khung", "mô hình", "bộ lọc", "sơ đồ")
_EN_TOOLS: Final = frozenset({"canvas", "map", "matrix", "framework"})
# Function words that end a qualifier (with accents: "cấp độ" goes on, "đó" stops).
_STOPS: Final = frozenset({
    "và", "của", "cho", "trong", "theo", "với", "để", "gồm", "là", "khi", "từ", "tới", "đến", "giữa", "các", "những",
    "một", "này", "đó", "do", "nào", "ở", "về", "qua", "bằng", "trên", "dưới", "sau", "trước", "thành", "vào", "như",
    "hay", "hoặc", "được", "bị", "sẽ", "đã", "đang", "cần", "phải", "có", "không", "chưa", "and", "or", "of", "for",
    "the", "a", "an", "to", "in", "on", "with",
})
# "Khung", "mô hình" are also plain words ("mô hình đóng gói giải pháp", "khung biểu mẫu"): they name a tool only with a
# capitalised name after them ("Khung G-STIC", "Mô hình Kinh doanh").
_WEAK_TOOLS: Final = frozenset({"khung", "mô hình"})
# Qualifiers that make the noun no tool ("khung thời gian", "khung giờ").
_GENERIC: Final = frozenset({"thời gian", "giờ"})
_QUALIFIER_WORDS: Final = 2
_NAME_WINDOW: Final = 4
_EN_NAME_WORDS: Final = 3
# Review 5 (M9): a Vietnamese tool noun as a verb, and the generic words of an English Title Case heading.
_VERB_MARKS: Final = frozenset({"hóa", "hoá"})
_EN_GENERIC: Final = frozenset({
    "a", "an", "the", "our", "your", "their", "my", "his", "her", "its", "this", "that", "these", "those", "each",
    "every", "any", "all", "one", "new", "simple", "practical", "basic", "good", "great", "effective", "quick", "easy",
    "complete", "full", "key", "main", "core", "general", "standard", "custom", "final", "first", "next", "build",
    "building", "use", "using", "create", "creating", "apply", "applying", "map", "mapping", "design", "designing",
    "make", "making", "choose", "choosing", "how", "why", "what", "when", "step", "steps", "decision", "action",
})
_EN_MIN_NAME_WORDS: Final = 2
_SHORT_TOKEN_CHARS: Final = 2
# A described qualifier says something ("hành trình", "dịch vụ"): no one or two letters.
_MIN_DESCRIBED_CHARS: Final = 4
_CLAUSE_RE: Final = re.compile(r"[:;,()?!.\[\]\"“”]|\s[-–—]\s")  # noqa: RUF001 - the dashes end a clause
_TOKEN_RE: Final = re.compile(r"[^\W_][\w'’&/.-]*")  # noqa: RUF001 - the apostrophe of a word
_DIGIT_RE: Final = re.compile(r"\d")
# A name in brackets right after a clause, starting with a capital ("(Service Blueprint)", "(G-STIC)").
_GLOSS_RE: Final = re.compile(r"\s*\(\s*([^()]{2,80}?)\s*\)")


def _key(text: str) -> str:
    """``text`` without accents, spaces and punctuation, for a containment test that ignores glued words."""

    return re.sub(r"[\W_]+", "", idm_fold(text))


def _code_like(token: str) -> bool:
    """A word that names a code ("G-STIC", "CJM", "2-S"): a capital after its first letter, a digit or a hyphen."""

    return any(char.isupper() for char in token[1:]) or bool(_DIGIT_RE.search(token)) or "-" in token.strip("-")


class _Phrase(NamedTuple):
    text: str  # the named tool phrase as written
    after: list[str]  # the words after the tool noun
    name: str  # the English proper name after a Vietnamese noun ("sơ đồ Service Blueprint"), else ""
    noun: str  # the Vietnamese tool noun, "" for an English name
    qualifier: str  # the words after the Vietnamese tool noun


def _phrases(clause: str) -> list[_Phrase]:
    """The named tool phrases of one clause of a title or heading."""

    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFC", clause))
    lowered = [token.lower() for token in tokens]
    # Review 6 (L2): one proper word names a tool inside a sentence ("Hãy dùng Eisenhower Matrix …"), never in a Title
    # Case heading, where every word is capitalised ("Building Your Prioritization Matrix").
    title_case = all(token[:1].isupper() for token in tokens
                     if token[:1].isalpha() and token.lower() not in _STOPS and len(token) > _SHORT_TOKEN_CHARS)
    found: list[_Phrase] = []
    for index in range(len(tokens)):
        for noun in _VI_TOOLS:
            words = noun.split()
            if lowered[index:index + len(words)] != words:
                continue
            start = index + len(words)
            qualifier: list[str] = []
            for token, key in zip(tokens[start:], lowered[start:], strict=True):
                if len(qualifier) == _QUALIFIER_WORDS or key in _STOPS or _DIGIT_RE.search(token):
                    break
                qualifier.append(token)
            nested = any(" ".join(lowered[start:start + len(other.split())]) == other for other in _VI_TOOLS)
            if (not qualifier or nested or " ".join(qualifier).lower() in _GENERIC or lowered[start] in _VERB_MARKS
                    or (noun in _WEAK_TOOLS and not qualifier[0][:1].isupper())):
                continue
            # A tool the source names in English after it ("sơ đồ Service Blueprint") is named by that name.
            name = qualifier if all(token.isascii() and token[:1].isupper() for token in qualifier) else []
            found.append(_Phrase(" ".join(tokens[index:start + len(qualifier)]), tokens[start:start + _NAME_WINDOW],
                                 " ".join(name), noun, " ".join(qualifier)))
        if lowered[index] in _EN_TOOLS and index:
            before: list[str] = []
            for token in reversed(tokens[max(0, index - _EN_NAME_WORDS):index]):
                if not token[:1].isupper() or token.lower() in _STOPS:
                    break
                before.insert(0, token)
            # Review 6 (L2): a capitalised Vietnamese word opens the sentence, not the name ("Dùng Eisenhower Matrix").
            while before and (before[0].lower() in _EN_GENERIC or not before[0].isascii()):
                before.pop(0)
            if len(before) >= _EN_MIN_NAME_WORDS or (len(before) == 1 and index > 1 and not title_case):
                found.append(_Phrase(" ".join([*before, tokens[index]]), [], "", "", ""))
    return found


def _distinct(phrases: Iterable[str]) -> list[str]:
    """``phrases`` without one already written in another case ("Ma trận thấu cảm", "ma trận thấu cảm")."""

    seen: dict[str, str] = {}
    for phrase in phrases:
        seen.setdefault(phrase.lower(), phrase)
    return list(seen.values())


@dataclass(frozen=True)
class NameGuide:
    """The source text of a unit (its facts and the titles of their sections), read without spaces or accents."""

    source: str

    @classmethod
    def build(cls, texts: Iterable[str]) -> NameGuide:
        return cls(" ".join(_key(text) for text in texts if text))

    def missing(self, text: str) -> list[str]:
        """The named tool phrases of ``text`` the source does not write."""

        missing: list[str] = []
        text, start = text or "", 0
        for match in [*_CLAUSE_RE.finditer(text), None]:
            end = match.start() if match is not None else len(text)
            clause, start = text[start:end], match.end() if match is not None else end
            # "Sơ đồ dịch vụ (Service Blueprint)": a plain description glossed by the source's own name.
            gloss = _GLOSS_RE.match(text, end)
            if gloss is not None and gloss.group(1)[:1].isupper() and _key(gloss.group(1)) in self.source:
                continue
            for phrase in _phrases(clause):
                if (_key(phrase.text) in self.source or (phrase.name and _key(phrase.name) in self.source)
                        or any(_code_like(token) and _key(token) in self.source for token in phrase.after)
                        or self.describes(phrase)):
                    continue
                missing.append(phrase.text)
        return _distinct(missing)

    def describes(self, phrase: _Phrase) -> bool:
        """A Vietnamese tool noun before words the source writes, which no other tool noun of the source takes: a
        plain description ("bản đồ hành trình" where the source has "hành trình trải nghiệm")."""

        qualifier = _key(phrase.qualifier)
        return (bool(phrase.noun) and len(qualifier) >= _MIN_DESCRIBED_CHARS and qualifier in self.source
                and not any(_key(other) + qualifier in self.source for other in _VI_TOOLS if other != phrase.noun))

    def slot_names(self, component: Mapping[str, Any]) -> list[str]:
        """The phrases of a slot's title and html headings the source does not write."""

        texts = [str(component.get("title") or "")]
        semantic = component.get("semantic_content")
        for section in semantic.get("sections") or [] if isinstance(semantic, dict) else []:
            if isinstance(section, dict):
                texts.append(str(section.get("heading") or ""))
        return _distinct([phrase for text in texts for phrase in self.missing(text)])


__all__ = ["NAME_NOT_IN_SOURCE_CODE", "NameGuide"]
