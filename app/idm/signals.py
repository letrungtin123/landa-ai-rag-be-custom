"""Deterministic source signals and W1 sectioning (spec §7.1).

Nothing here calls a model. Flags help the W1 provider see structure before it
groups facts; sections bound the size of one W1-map call.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.idm.policy import (
    IDM_DUPLICATE_JACCARD_THRESHOLD,
    IDM_DUPLICATE_NGRAM,
    IDM_HEADING_ALL_CAPS_RATIO,
    IDM_HEADING_MAX_CHARS,
    IDM_STEP_RUN_MIN_LINES,
    IDM_W1_MAX_SECTIONS,
    IDM_W1_SECTION_MAX_CHARS,
    IDM_W1_SECTION_MAX_FACTS,
    IDM_W1_SECTION_MIN_CHARS_BEFORE_HEADING_CUT,
    IDM_W1_SECTION_TARGET_CHARS,
)
from app.idm.runtime import IdmError
from app.idm.text import char_ngrams, idm_fold, jaccard
from app.instructional_quality import (
    ORDERED_STEP_RE,
    TABLE_ROW_RE,
    ordered_source_steps,
    source_relationship_pairs,
    source_term_definitions,
)
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

_SECTION_HEADING_RE: Final = re.compile(r"^(chuong|phan|muc|chapter|part|section)\s*[0-9ivx]+")
_NUMBERED_HEADING_RE: Final = re.compile(r"^[0-9]+(\.[0-9]+)*\.?\s+\S")
_STEP_LINE_RE: Final = re.compile(r"^(?:(bước|buoc|step)\s*[0-9]+|[0-9]+[.)]\s)", re.IGNORECASE)
_RULE_RE: Final = re.compile(
    r"\b(luon luon|luon|khong duoc|bat buoc|phai|nghiem cam|cam|tru (khi|truong hop)|chi khi|toi da|toi thieu"
    r"|must|must not|never|always|unless|only if|required|prohibited|shall)\b",
)
_EXAMPLE_RE: Final = re.compile(r"\b(vi du|chang han|tinh huong|case|example|e\.g\.|minh hoa|truong hop)\b")
_EMPHASIS_RE: Final = re.compile(r"\b(luu y|quan trong|chu y|canh bao|note|important|warning|critical)\b")
_CAPS_RUN_MIN_WORDS: Final = 3
_CAPS_WORD_MIN_LETTERS: Final = 2
_VISUAL_RE: Final = re.compile(r"\b(hinh|so do|bieu do|figure|diagram|chart|image)\b")
_TERM_IS_RE: Final = re.compile(r"^(.{2,80}?)\s+(?:là|la|means|is|refers to)\s+(.{20,})$", re.IGNORECASE)
_TERM_ABBR_RE: Final = re.compile(
    r"^(.{2,40}?)\s*\((?:viết tắt của|viet tat cua|short for|abbreviation of)\s+(.{3,})\)\s*(.*)$",
    re.IGNORECASE,
)
_MAX_DUPLICATE_CANDIDATES: Final = 400
_MAX_TERM_WORDS: Final = 8
_MIN_CROSSWORD_ANSWER: Final = 2
_MAX_CROSSWORD_ANSWER: Final = 24
_MIN_DEFINITION_CHARS: Final = 20
_MAX_DEFINITIONS: Final = 10


class IdmCapacityError(IdmError):
    """The source is larger than one IDM course_skeleton task can design."""


@dataclass(frozen=True)
class IdmSection:
    section_id: str
    document_id: str
    title_path: str
    fact_keys: tuple[str, ...]
    content_chars: int


@dataclass
class SectionPlan:
    sections: list[IdmSection]
    warnings: list[str] = field(default_factory=list)


def _letters(value: str) -> list[str]:
    return [char for char in value if char.isalpha()]


def _is_heading(fact: SourceSnapshotFactV2, folded: str) -> bool:
    text = fact.fact_text.strip()
    if not text or len(text) > IDM_HEADING_MAX_CHARS or text.endswith((".", ";")):
        return False
    titles = [fact.locator.get("scope_title"), fact.source_ref]
    heading_path = fact.locator.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        titles.append(heading_path[-1])
    if any(isinstance(title, str) and title.strip() and idm_fold(title) == folded for title in titles):
        return True
    letters = _letters(text)
    if letters and sum(char.isupper() for char in letters) / len(letters) >= IDM_HEADING_ALL_CAPS_RATIO:
        return True
    return _SECTION_HEADING_RE.match(folded) is not None or _NUMBERED_HEADING_RE.match(text) is not None


def _has_caps_run(text: str) -> bool:
    """At least three consecutive fully upper-case words (Unicode aware)."""

    run = 0
    for word in re.findall(r"\w+", text):
        letters = _letters(word)
        if len(letters) >= _CAPS_WORD_MIN_LETTERS and all(char.isupper() for char in letters):
            run += 1
            if run >= _CAPS_RUN_MIN_WORDS:
                return True
        else:
            run = 0
    return False


def _step_number(text: str) -> int | None:
    """The number of a numbered step line, or ``None``."""

    stripped = text.strip()
    match = ORDERED_STEP_RE.match(stripped)
    if match is not None:
        return int(match.group(1))
    number = re.match(r"^(?:(?:bước|buoc|step)\s*)?(\d{1,3})", stripped, re.IGNORECASE)
    return int(number.group(1)) if number is not None and _STEP_LINE_RE.match(stripped) else None


def compute_fact_signals(facts: Sequence[SourceSnapshotFactV2]) -> dict[str, list[str]]:
    """Return the sorted flag list of every fact, keyed by fact key."""

    flags: dict[str, set[str]] = {fact.fact_key: set() for fact in facts}
    numbers = [_step_number(fact.fact_text) for fact in facts]
    index = 0
    while index < len(facts):
        if numbers[index] is None:
            index += 1
            continue
        # A run is consecutive numbering (n, n+1, ...): "5. HEADING" followed by
        # "Bước 1" starts a new run, so numbered section headings stay headings.
        end = index + 1
        while (end < len(facts) and numbers[end] is not None and facts[end].document_id == facts[index].document_id
               and numbers[end] == (numbers[end - 1] or 0) + 1):
            end += 1
        if end - index >= IDM_STEP_RUN_MIN_LINES:
            for position in range(index, end):
                flags[facts[position].fact_key].add("S")
        index = max(end, index + 1)

    sizes: list[int] = []
    seen: list[tuple[int, set[str]]] = []
    exact: set[str] = set()
    for fact in facts:
        text = fact.fact_text
        folded = idm_fold(text)
        current = flags[fact.fact_key]
        if _is_heading(fact, folded):
            current.add("H")
        if TABLE_ROW_RE.match(text.strip()):
            current.add("T")
        if _RULE_RE.search(folded):
            current.add("R")
        if _EXAMPLE_RE.search(folded):
            current.add("E")
        if _EMPHASIS_RE.search(folded) or _has_caps_run(text):
            current.add("M")
        visual_regions = fact.locator.get("visual_regions")
        if (isinstance(visual_regions, list) and visual_regions) or _VISUAL_RE.search(folded):
            current.add("V")
        if _is_duplicate(folded, text, sizes, seen, exact):
            current.add("D")
    for value in flags.values():
        # A numbered step or table row written in capitals is structure, not a heading.
        if "S" in value or "T" in value:
            value.discard("H")
    return {key: sorted(value) for key, value in flags.items()}


def _is_duplicate(
    folded: str,
    text: str,
    sizes: list[int],
    seen: list[tuple[int, set[str]]],
    exact: set[str],
) -> bool:
    """Jaccard of character 5-grams against earlier facts of similar length.

    Jaccard >= t implies a size ratio >= t, so only earlier facts inside that size
    window can match; the window is capped to keep the pass linear in practice.
    """

    if not folded:
        return False
    if folded in exact:
        return True
    grams = char_ngrams(text, IDM_DUPLICATE_NGRAM)
    size = len(grams)
    duplicate = False
    if size:
        low = bisect.bisect_left(sizes, int(size * IDM_DUPLICATE_JACCARD_THRESHOLD))
        high = bisect.bisect_right(sizes, int(size / IDM_DUPLICATE_JACCARD_THRESHOLD) + 1)
        for candidate_index in range(low, min(high, low + _MAX_DUPLICATE_CANDIDATES)):
            if jaccard(grams, seen[candidate_index][1]) >= IDM_DUPLICATE_JACCARD_THRESHOLD:
                duplicate = True
                break
        position = bisect.bisect_left(sizes, size)
        sizes.insert(position, size)
        seen.insert(position, (size, grams))
    exact.add(folded)
    return duplicate


def plan_idm_sections(
    facts: Sequence[SourceSnapshotFactV2],
    signals: dict[str, list[str]],
    document_names: dict[str, str],
    *,
    max_sections: int = IDM_W1_MAX_SECTIONS,
) -> SectionPlan:
    """Cut the ordered facts into W1 sections (spec §7.1 step 3)."""

    sections: list[IdmSection] = []
    warnings: list[str] = []
    current: list[SourceSnapshotFactV2] = []
    current_chars = 0
    heading = ""
    section_title = ""

    def flush() -> None:
        nonlocal current, current_chars, section_title
        if not current:
            return
        document_id = current[0].document_id
        title = section_title or document_names.get(document_id) or document_id
        sections.append(IdmSection(
            section_id=f"sec_{len(sections) + 1:03d}",
            document_id=document_id,
            title_path=" ".join(title.split())[:240],
            fact_keys=tuple(fact.fact_key for fact in current),
            content_chars=current_chars,
        ))
        current, current_chars, section_title = [], 0, ""

    previous_flags: list[str] = []
    for fact in facts:
        flags = signals.get(fact.fact_key, [])
        length = len(fact.fact_text)
        if current and fact.document_id != current[0].document_id:
            flush()
            heading = ""
        if current:
            is_heading_cut = "H" in flags and current_chars >= IDM_W1_SECTION_MIN_CHARS_BEFORE_HEADING_CUT
            over_target = current_chars + length > IDM_W1_SECTION_TARGET_CHARS
            over_max = (current_chars + length > IDM_W1_SECTION_MAX_CHARS
                        or len(current) + 1 > IDM_W1_SECTION_MAX_FACTS)
            inside_run = any(flag in flags and flag in previous_flags for flag in ("T", "S"))
            if over_max:
                if inside_run:
                    warnings.append("IDM_W1_SECTION_SPLIT_INSIDE_RUN")
                flush()
            elif (is_heading_cut or over_target) and not inside_run:
                flush()
        if "H" in flags:
            heading = fact.fact_text.strip()
        if not section_title and (not current or "H" in flags):
            # The heading a section opens with (or inherits) names it, not a later one.
            section_title = heading
        current.append(fact)
        current_chars += length
        previous_flags = flags
    flush()
    if len(sections) > max_sections:
        raise IdmCapacityError("IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY")
    return SectionPlan(sections=sections, warnings=warnings)


def idm_has_ordered_steps(texts: Iterable[str], *, locale: str) -> bool:
    return len(ordered_source_steps(list(texts), locale=locale)) >= IDM_STEP_RUN_MIN_LINES


def _crossword_answer(term: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", idm_fold(term).upper())


def idm_term_definitions(texts: Iterable[str]) -> list[tuple[str, str]]:
    """Explicit term/definition pairs, including "X là Y" and "X means Y" (spec §11.1)."""

    values = [text.strip() for text in texts if text and text.strip()]
    result: list[tuple[str, str]] = list(source_term_definitions(values))
    answers = {answer for answer, _definition in result}
    for value in values:
        if len(result) >= _MAX_DEFINITIONS:
            break
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value):
            continue
        match = _TERM_ABBR_RE.match(value) or _TERM_IS_RE.match(value)
        if match is None:
            continue
        term = match.group(1).strip(" .;,-")
        definition = " ".join(part for part in match.groups()[1:] if part).strip()
        answer = _crossword_answer(term)
        if (len(term.split()) > _MAX_TERM_WORDS or len(definition) < _MIN_DEFINITION_CHARS
                or not _MIN_CROSSWORD_ANSWER <= len(answer) <= _MAX_CROSSWORD_ANSWER or answer in answers):
            continue
        answers.add(answer)
        result.append((answer, definition[:500]))
    return result


def idm_relationship_pairs(texts: Iterable[str]) -> list[tuple[str, str, str | None]]:
    return source_relationship_pairs(list(texts))
