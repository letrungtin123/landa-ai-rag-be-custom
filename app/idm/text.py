"""Text helpers owned by the IDM pipeline.

The legacy ``_fold`` helpers do not map ``đ`` to ``d``; IDM matching rules are
written against a fold that does (spec §7.1), so IDM keeps its own variant and
never changes the legacy ones (spec §11.6).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal

from app.idm.policy import (
    CAPABILITY_LEAD_INS,
    CASE_DECISION_VERBS,
    GENERIC_TITLE_PATTERN,
    GENERIC_TITLES,
    IDM_FAQ_MIN_PAIR_SUPPORT,
    IDM_FAQ_MIN_WORD_SUPPORT,
    IDM_FAQ_SENTENCE_MIN_PAIR_SUPPORT,
    IDM_FAQ_SENTENCE_MIN_WORD_SUPPORT,
    IDM_FAQ_SENTENCE_MIN_WORDS,
    OUTPUT_VERBS_EN,
    OUTPUT_VERBS_VI,
    UNMEASURABLE_VERBS,
)

_WHITESPACE_RE: Final = re.compile(r"\s+")
_GENERIC_TITLE_RE: Final = re.compile(GENERIC_TITLE_PATTERN)
_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)
_ELLIPSIS: Final = "…"
# A boundary cut keeps at least this share of the bound; an earlier boundary would drop too much text.
_MIN_BOUNDARY_SHARE: Final = 0.5
_TRAILING_JOINERS: Final = " ,;:-–—"  # noqa: RUF001 - dashes a clause may end with before the cut
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
    """``value`` within ``max_length`` characters, ending in an ellipsis after the last whole word.

    QC run ab8d67e1 (R5): a cut in the middle of a word ("Best-in") reads as a typo. The cut falls on the last space
    that keeps at least half of the bound; only a text without such a space is cut inside a word.
    """

    if len(value) <= max_length:
        return value
    room = max(0, max_length - len(_ELLIPSIS))
    space = value.rfind(" ", 0, room + 1)
    head = value[:space] if space >= int(max_length * _MIN_BOUNDARY_SHARE) else value[:room]
    return head.rstrip(_TRAILING_JOINERS) + _ELLIPSIS


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
_SENTENCE_STOP_RE: Final = re.compile(r"[.!?…](?=\s)")


def trim_at_boundary(value: str, max_length: int) -> str:
    """One line of at most ``max_length`` characters, cut at a sentence end or else a word boundary.

    QC course 364564 (N1): one ``learner_action`` of 300+ characters rejected a whole W4 repair
    answer. Over-long author text is shortened instead: the last full sentence that keeps at least
    half of the bound, otherwise the last whole word followed by an ellipsis.
    """

    text = sanitize_author_text(" ".join(value.split()), len(value) + 1).replace("\n", " ")
    if len(text) <= max_length:
        return text
    floor = int(max_length * _MIN_BOUNDARY_SHARE)
    # A stop counts only when a space follows it inside the bound, so the cut never ends mid-word.
    stops = [match.end() for match in _SENTENCE_STOP_RE.finditer(text[: max_length + 1])]
    if stops and stops[-1] >= floor:
        return text[: stops[-1]]
    room = max_length - len(_ELLIPSIS)
    space = text.rfind(" ", 0, room + 1)
    head = text[:space] if space >= floor else text[:room]
    return head.rstrip(_TRAILING_JOINERS) + _ELLIPSIS


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


def produces_output(statement: str, kind: str) -> bool:
    """A Must Do whose action yields a work product (fill in, draft, map, sign ...).

    Classifying, identifying or choosing is applying a rule to a case even when W1 marked it "do"
    (spec §10.1): a scenario question practises it, a worksheet is not expected. A Must Do marked
    "decide" still produces an output when its leading verb does ("Ký cam kết …", QC run 8de1c76b, Q5).
    """

    remainder = first_main_verb(statement)
    folded = idm_fold(remainder)
    if kind == "do":
        return not any(folded == verb or folded.startswith(verb + " ") for verb in CASE_DECISION_VERBS)
    if kind == "decide":
        return (any(remainder == verb or remainder.startswith(verb + " ") for verb in OUTPUT_VERBS_VI)
                or any(folded == verb or folded.startswith(verb + " ") for verb in OUTPUT_VERBS_EN))
    return False


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


# --- Evidence grounding (FAQ guard, worksheet self-check) ---------------------------------------
# Folded function words: connectives, pronouns, copulas and quantifiers carry no claim, so an answer
# may add them freely ("Vì vậy", "Điều này giúp …"). Vietnamese compounds are two syllables, so word
# pairs are the unit of support; a pair of two function words never counts.
_FUNCTION_WORDS: Final = frozenset({
    "la", "cua", "va", "cac", "nhung", "co", "khong", "duoc", "cho", "de", "voi", "thi", "ma", "mot", "nhu", "nay",
    "do", "khi", "neu", "vi", "trong", "tren", "tu", "den", "ra", "vao", "se", "da", "dang", "can", "phai", "hay",
    "hoac", "cung", "rat", "nhat", "moi", "tat", "ca", "chi", "con", "lai", "nen", "thay", "boi", "ve", "theo",
    "tai", "bang", "qua", "sau", "truoc", "giua", "hon", "ban", "nguoi", "viec", "cach", "nao", "gi", "sao", "ai",
    "day", "doi", "khac", "nhieu", "it", "dieu", "chung", "the", "van", "luon", "minh", "ho", "toi", "ta", "ay",
    "kia", "bi", "o", "vay", "tuc", "gom", "them", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "by", "as", "at", "this", "that", "from", "not", "no", "but", "if", "then", "so", "than",
    "their", "its", "will", "should", "must", "does", "was", "were", "has", "have", "had", "which", "who", "what",
    "when", "how", "why", "these", "those", "there", "here", "also", "only", "more", "most", "such", "into", "about",
    "you", "your", "they",
})
_NUMBER_RE: Final = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE_END_RE: Final = re.compile(r"(?<=[.!?;:])\s+")
_MIN_CONTENT_WORD_CHARS: Final = 2
GroundingVerdict = Literal["grounded", "numbers", "support", "sentence"]


def _normal_number(token: str) -> str:
    whole, _, rest = token.replace(",", ".").partition(".")
    return (whole.lstrip("0") or "0") + (f".{rest}" if rest else "")


def _numbers(folded: str) -> set[str]:
    return {_normal_number(token) for token in _NUMBER_RE.findall(folded)}


def _content(word: str) -> bool:
    return len(word) >= _MIN_CONTENT_WORD_CHARS and word not in _FUNCTION_WORDS and not word.isdigit()


def content_words(text: str) -> list[str]:
    """Folded words of ``text`` that carry meaning (no function words, digits or one-letter words)."""

    return [word for word in _WORD_RE.findall(idm_fold(text)) if _content(word)]


def content_pairs(text: str) -> set[tuple[str, str]]:
    """Adjacent folded word pairs of ``text`` whose two words both carry meaning: a Vietnamese compound
    ("phong trao", "ton kem") or a content phrase, never a connective ("chi la")."""

    tokens = _WORD_RE.findall(idm_fold(text))
    return {pair for pair in pairwise(tokens) if _content(pair[0]) and _content(pair[1])}


@dataclass(frozen=True)
class EvidenceIndex:
    """Folded words, adjacent word pairs and numbers of the facts a writer was given."""

    words: frozenset[str]
    pairs: frozenset[tuple[str, str]]
    numbers: frozenset[str]


def evidence_index(texts: Iterable[str], *, number_texts: Iterable[str] = ()) -> EvidenceIndex:
    """Index the evidence once; ``number_texts`` (the approved plan) only add numbers."""

    words: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    numbers: set[str] = set()
    for value in texts:
        folded = idm_fold(value)
        tokens = _WORD_RE.findall(folded)
        words.update(tokens)
        pairs.update(pairwise(tokens))
        numbers.update(_numbers(folded))
    for value in number_texts:
        numbers.update(_numbers(idm_fold(value)))
    return EvidenceIndex(frozenset(words), frozenset(pairs), frozenset(numbers))


def claim_support(text: str, index: EvidenceIndex) -> tuple[float, float, int]:
    """(content-word share, content-pair share, content-word count) of ``text`` found in the evidence."""

    tokens = _WORD_RE.findall(idm_fold(text))
    content = [word for word in tokens if _content(word)]
    pairs = [pair for pair in pairwise(tokens) if _content(pair[0]) or _content(pair[1])]
    word_share = sum(word in index.words for word in content) / len(content) if content else 1.0
    pair_share = sum(pair in index.pairs for pair in pairs) / len(pairs) if pairs else word_share
    return word_share, pair_share, len(content)


def grounding_verdict(text: str, index: EvidenceIndex) -> GroundingVerdict:
    """Whether ``text`` only restates the evidence: no new number, enough supported wording overall
    and in every long sentence. Thresholds and their calibration live in ``app.idm.policy``."""

    if _numbers(idm_fold(text)) - index.numbers:
        return "numbers"
    words, pairs, _count = claim_support(text, index)
    if pairs < IDM_FAQ_MIN_PAIR_SUPPORT and words < IDM_FAQ_MIN_WORD_SUPPORT:
        return "support"
    for sentence in _SENTENCE_END_RE.split(" ".join(text.split())):
        words, pairs, count = claim_support(sentence, index)
        if (count >= IDM_FAQ_SENTENCE_MIN_WORDS and pairs < IDM_FAQ_SENTENCE_MIN_PAIR_SUPPORT
                and words < IDM_FAQ_SENTENCE_MIN_WORD_SUPPORT):
            return "sentence"
    return "grounded"
