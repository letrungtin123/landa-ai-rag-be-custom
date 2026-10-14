"""Units of a kept document: lessons split at line boundaries, never inside a line (QLT-6 ``preserve_source``).

Each lesson range is cut into units in reading order: at its explicit sub-heading lines first, then by the reading
budget (``IDM_PRESERVE_UNIT_*``). A unit is titled by the lesson title (its first unit), by its sub-heading line, or
"<title> (tiếp theo)" / "(continued)" when the same part continues. A lesson that would need more units than a lesson
holds is first given a larger budget, then split into consecutive lessons "<title> (phần 2)"; a chapter with more
lessons than a chapter holds continues the same way. Text is never cut: only the course bounds can refuse a document
(``IDM_PRESERVE_SOURCE_TOO_LARGE``).
"""

from __future__ import annotations

import html as html_lib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from app.idm.policy import (
    IDM_PRESERVE_MAX_CHAPTERS,
    IDM_PRESERVE_MAX_LESSONS,
    IDM_PRESERVE_MAX_LESSONS_PER_CHAPTER,
    IDM_PRESERVE_MAX_UNITS,
    IDM_PRESERVE_MAX_UNITS_PER_LESSON,
    IDM_PRESERVE_TITLE_MAX_CHARS,
    IDM_PRESERVE_UNIT_MAX_CHARS,
    IDM_PRESERVE_UNIT_MAX_FACTS,
    IDM_PRESERVE_UNIT_MAX_HTML_CHARS,
    IDM_PRESERVE_UNIT_MAX_WORDS,
    IDM_PRESERVE_UNIT_TARGET_WORDS,
    IDM_SHARD_MAX_SOURCE_CHARS,
)
from app.idm.preserve_lines import is_render_heading, normalize_line, table_cells, title_eligible, visible_text
from app.idm.preserve_segment import ChapterRange, LessonRange
from app.idm.runtime import IdmStageError
from app.idm.text import single_line

TOO_LARGE_CODE: Final = "IDM_PRESERVE_SOURCE_TOO_LARGE"
SEGMENTATION_INVALID_CODE: Final = "IDM_PRESERVE_SEGMENTATION_INVALID"
# Markup around one html block (<p></p>, <h3></h3>) and around one table cell (<td></td>).
_BLOCK_MARKUP_CHARS: Final = 9
_CELL_MARKUP_CHARS: Final = 9


@dataclass(frozen=True)
class UnitPlan:
    title: str
    lines: tuple[int, ...]


@dataclass(frozen=True)
class LessonPlan:
    title: str
    units: tuple[UnitPlan, ...]


@dataclass(frozen=True)
class ChapterPlan:
    title: str
    lessons: tuple[LessonPlan, ...]


@dataclass(frozen=True)
class Outline:
    chapters: tuple[ChapterPlan, ...]
    # Lessons and chapters continued as "(phần 2)", and plain "Phần N" titles of a document without headings.
    split_lessons: int = 0
    split_chapters: int = 0
    label_titles: int = 0


@dataclass
class _Size:
    words: int = 0
    facts: int = 0
    chars: int = 0
    html: int = 0
    lines: list[int] = field(default_factory=list)


def _cost(text: str) -> tuple[int, int, int]:
    """Words, source characters and an upper estimate of the rendered html characters of one line."""

    visible = visible_text(text)
    cells = table_cells(text)
    markup = _BLOCK_MARKUP_CHARS + (_CELL_MARKUP_CHARS * len(cells) if cells else 0)
    return len(visible.split()), len(text), len(html_lib.escape(visible, quote=True)) + markup


def suffixed(title: str, suffix: str) -> str:
    """``title`` + ``suffix`` within the title bound (the title is shortened at a word, never the suffix)."""

    if len(title) + len(suffix) <= IDM_PRESERVE_TITLE_MAX_CHARS:
        return title + suffix
    return single_line(title, IDM_PRESERVE_TITLE_MAX_CHARS - len(suffix)) + suffix


def _chunks(lines: Sequence[int], texts: Sequence[str], target_words: int) -> list[list[int]]:
    """Consecutive lines in units within the budget; a heading line never ends a unit it does not start."""

    chunks: list[list[int]] = []
    current = _Size()
    for line in lines:
        words, chars, html = _cost(texts[line - 1])
        continuing_table = bool(current.lines) and table_cells(texts[current.lines[-1] - 1]) is not None \
            and table_cells(texts[line - 1]) is not None
        word_limit = IDM_PRESERVE_UNIT_MAX_WORDS if continuing_table else target_words
        if current.lines and (current.words + words > word_limit or current.facts + 1 > IDM_PRESERVE_UNIT_MAX_FACTS
                              or current.chars + chars > IDM_PRESERVE_UNIT_MAX_CHARS
                              or current.html + html > IDM_PRESERVE_UNIT_MAX_HTML_CHARS):
            carried: list[int] = []
            if len(current.lines) > 1 and is_render_heading(texts[current.lines[-1] - 1]):
                carried = [current.lines.pop()]
            chunks.append(current.lines)
            current = _Size()
            for kept in carried:
                kept_words, kept_chars, kept_html = _cost(texts[kept - 1])
                current.words, current.chars, current.html = kept_words, kept_chars, kept_html
                current.facts = 1
                current.lines.append(kept)
        current.lines.append(line)
        current.words += words
        current.facts += 1
        current.chars += chars
        current.html += html
    if current.lines:
        chunks.append(current.lines)
    return chunks


def _sub_heading(line: int, texts: Sequence[str]) -> bool:
    return is_render_heading(texts[line - 1]) and title_eligible(texts[line - 1])


def _continued_title(first: int, bound: int, part_title: str, texts: Sequence[str], continued: str) -> str:
    """Title of a unit that continues a part: the sub-heading it starts with (exact text), else the nearest
    sub-heading above it in the same part, else the part title, each followed by "(tiếp theo)"/"(continued)"."""

    if _sub_heading(first, texts):
        return normalize_line(texts[first - 1])
    above = next((line for line in range(first - 1, bound, -1) if _sub_heading(line, texts)), None)
    return suffixed(normalize_line(texts[above - 1]) if above is not None else part_title, continued)


def _lesson_units(lesson: LessonRange, title: str, texts: Sequence[str], target_words: int,
                  continued: str) -> list[UnitPlan]:
    starts = [lesson.first, *lesson.unit_titles]
    units: list[UnitPlan] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else lesson.last
        part_title = title if index == 0 else normalize_line(texts[start - 1])
        # Sub-headings that may title a continued unit come after the part's own title line.
        bound = (lesson.title or lesson.first - 1) if index == 0 else start
        for position, chunk in enumerate(_chunks(range(start, end + 1), texts, target_words)):
            unit_title = part_title if position == 0 else _continued_title(chunk[0], bound, part_title, texts,
                                                                           continued)
            units.append(UnitPlan(unit_title, tuple(chunk)))
    return units


def _lesson_parts(units: Sequence[UnitPlan], texts: Sequence[str]) -> list[list[UnitPlan]]:
    """Units into lessons of at most the unit limit and one chapter shard of source characters."""

    parts: list[list[UnitPlan]] = []
    current: list[UnitPlan] = []
    chars = 0
    for unit in units:
        size = sum(len(texts[line - 1]) for line in unit.lines)
        if current and (len(current) == IDM_PRESERVE_MAX_UNITS_PER_LESSON or chars + size > IDM_SHARD_MAX_SOURCE_CHARS):
            parts.append(current)
            current, chars = [], 0
        current.append(unit)
        chars += size
    if current:
        parts.append(current)
    return parts


def _title(line: int, label: str | None, texts: Sequence[str]) -> str:
    if line:
        return normalize_line(texts[line - 1])
    if label is None:
        raise IdmStageError(SEGMENTATION_INVALID_CODE)
    return label


def _check_cover(chapters: Sequence[ChapterRange], total: int) -> None:
    expected = 1
    for chapter in chapters:
        for lesson in chapter.lessons:
            if lesson.first != expected or lesson.last < lesson.first:
                raise IdmStageError(SEGMENTATION_INVALID_CODE)
            expected = lesson.last + 1
    if expected != total + 1:
        raise IdmStageError(SEGMENTATION_INVALID_CODE)


def build_outline(chapters: Sequence[ChapterRange], texts: Sequence[str], locale: str) -> Outline:
    """Chapters, lessons and units of the kept document, within the course bounds."""

    _check_cover(chapters, len(texts))
    vi = locale == "vi"
    continued = " (tiếp theo)" if vi else " (continued)"
    planned: list[ChapterPlan] = []
    split_lessons = split_chapters = labels = 0
    for chapter in chapters:
        chapter_title = _title(chapter.title, chapter.label, texts)
        labels += chapter.title == 0
        lessons: list[LessonPlan] = []
        for lesson in chapter.lessons:
            title = _title(lesson.title, lesson.label, texts)
            labels += lesson.title == 0
            units: list[UnitPlan] = []
            target = IDM_PRESERVE_UNIT_TARGET_WORDS
            while True:
                units = _lesson_units(lesson, title, texts, target, continued)
                if len(units) <= IDM_PRESERVE_MAX_UNITS_PER_LESSON or target >= IDM_PRESERVE_UNIT_MAX_WORDS:
                    break
                target = min(IDM_PRESERVE_UNIT_MAX_WORDS, target * 2)
            parts = _lesson_parts(units, texts)
            split_lessons += len(parts) > 1
            for number, part in enumerate(parts, start=1):
                lesson_title = title if number == 1 else suffixed(title, f" (phần {number})" if vi
                                                                  else f" (part {number})")
                lessons.append(LessonPlan(lesson_title, tuple(part)))
        size = IDM_PRESERVE_MAX_LESSONS_PER_CHAPTER
        groups = [lessons[start:start + size] for start in range(0, len(lessons), size)]
        split_chapters += len(groups) > 1
        for number, group in enumerate(groups, start=1):
            title = chapter_title if number == 1 else suffixed(chapter_title, f" (phần {number})" if vi
                                                                 else f" (part {number})")
            planned.append(ChapterPlan(title, tuple(group)))
    lesson_total = sum(len(chapter.lessons) for chapter in planned)
    unit_total = sum(len(lesson.units) for chapter in planned for lesson in chapter.lessons)
    if (len(planned) > IDM_PRESERVE_MAX_CHAPTERS or lesson_total > IDM_PRESERVE_MAX_LESSONS
            or unit_total > IDM_PRESERVE_MAX_UNITS):
        raise IdmStageError(TOO_LARGE_CODE)
    return Outline(tuple(planned), split_lessons, split_chapters, labels)
