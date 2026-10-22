"""Sentence-case titles (QLT-7, QC run fae66524).

The 4 chapter and 15 lesson titles and 36 of the 62 unit and block titles of the run were in Title Case, copied from
the slide headings of the source ("Bối Cảnh Mới Và Tuyên Bố Sứ Mệnh Made-in-World"), while chapter 2 was not.
Vietnamese titles are now put in sentence case deterministically, in Python, before Node accepts the design or the
unit:

* a title in Title Case (more of its Vietnamese words after the first are capitalised than not) keeps the capital of
  its first word and of the first word after ":", ".", "?", "!", "(" or a dash; every other capitalised Vietnamese word
  is lowered;
* any other title only has its runs of three or more capitalised words lowered when most of them are Vietnamese (the
  html label rule of ``artifacts.sentence_case_vi``);
* never lowered: acronyms and codes (BiC, CEO, AI, ERA5.0, ISO/CE, DEEP C), hyphenated terms (Made-in-World,
  Best-in-Class, 90-Day, Quick-Win), English words (Challenge, Think Big, See Different), personal and place names,
  and the proper names the facts and the plan write capitalised inside a sentence ("bộ lọc Tam Hóa", "công ty Tân
  Hiệp Phát"): ``TitleCaser.from_texts``.

English titles keep their case (Title Case is an English convention). The preserve_source path never calls this: its
titles are lines of the document, verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.idm.artifacts import name_positions
from app.idm.contracts import IdmLessonDesignV1, IdmW4CourseResponseV1

_WORD_SPLIT_RE: Final = re.compile(r"(\s+)")
_STRIP: Final = ".,;:!?()[]{}\"'“”‘’«»…*"  # noqa: RUF001 - quotes
_SENTENCE_ENDS: Final = (":", ".", "?", "!")
_DASHES: Final = frozenset({"-", "–", "—", "|", "•", "/"})  # noqa: RUF001 - dashes
# A segment of a fact or plan text: a sentence, a table cell or a dash-separated part.
_SEGMENT_RE: Final = re.compile(r"[.!?;:|•\n]+|\s[-–—]\s")  # noqa: RUF001 - dashes
# An unaccented Vietnamese syllable: an initial, a vowel group that exists without diacritics, a final ("Doanh",
# "Quan", "Tam"); English words such as "See", "Lean", "Big" or "Think" do not fit.
_ASCII_SYLLABLE_RE: Final = re.compile(
    r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"(?:oai|oay|uay|uya|uyu|ai|ao|au|ay|eo|ia|iu|oa|oe|oi|ua|ue|ui|uy|a|e|i|o|u|y)(?:ch|ng|nh|[cmnpt])?")
_RUN_MIN_WORDS: Final = 3
_PROTECTED_MIN_WORDS: Final = 2
# A sentence-case segment has at least this many lower-case Vietnamese words after its first word.
_SENTENCE_MIN_LOWER: Final = 2


def _core(word: str) -> str:
    return unicodedata.normalize("NFC", word).strip(_STRIP)


def vietnamese_word(core: str) -> bool:
    """A Vietnamese syllable: a letter outside ASCII (a diacritic, đ), or an unaccented syllable Vietnamese has."""

    return core.isalpha() and (any(not char.isascii() for char in core)
                               or _ASCII_SYLLABLE_RE.fullmatch(core.lower()) is not None)


def capitalised(core: str) -> bool:
    """"Hàng", "Ô": a capital first letter and only lower-case letters after it (never an acronym such as "AI")."""

    if not core or not core.isalpha() or not core[0].isupper():
        return False
    return core[1:].islower() if len(core) > 1 else not core.isascii()


def _proper_runs(text: str) -> set[tuple[str, ...]]:
    """Runs of two or more capitalised Vietnamese words inside a sentence-case segment of ``text`` (not its first
    word): the proper names the source or the plan writes ("bộ lọc Tam Hóa")."""

    runs: set[tuple[str, ...]] = set()
    for segment in _SEGMENT_RE.split(unicodedata.normalize("NFC", text)):
        raw = [word for word in segment.split() if _core(word)]
        words = [_core(word) for word in raw]
        vietnamese = [(position, word) for position, word in enumerate(words) if position and vietnamese_word(word)]
        capitals = sum(capitalised(word) for _position, word in vietnamese)
        lowers = sum(word[0].islower() for _position, word in vietnamese)
        if lowers < _SENTENCE_MIN_LOWER or capitals >= lowers:
            continue
        run: list[str] = []
        for position, word in enumerate([*words, ""]):
            if position and word and capitalised(word) and vietnamese_word(word):
                run.append(word.lower())
                # A list ("Chuyên gia, AI, Vốn, Thị trường") is no name: a comma ends the run.
                if not raw[position].endswith(","):
                    continue
            if len(run) >= _PROTECTED_MIN_WORDS:
                runs.add(tuple(run))
            run = []
    return runs


@dataclass(frozen=True)
class TitleCaser:
    """Sentence case for the titles of one stage, with the proper names its facts and plan write capitalised."""

    protected: frozenset[tuple[str, ...]] = frozenset()

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> TitleCaser:
        found: set[tuple[str, ...]] = set()
        for text in texts:
            found |= _proper_runs(text)
        return cls(frozenset(found))

    def name_positions(self, lowered: Sequence[str]) -> set[int]:
        """The positions of ``lowered`` (lower-case words) that belong to a known name or a name the texts write."""

        return self._kept(lowered)

    def _kept(self, lowered: Sequence[str]) -> set[int]:
        kept = set(name_positions(list(lowered)))
        for phrase in self.protected:
            size = len(phrase)
            for start in range(len(lowered) - size + 1):
                if tuple(lowered[start:start + size]) == phrase:
                    kept.update(range(start, start + size))
        return kept

    def title(self, text: str) -> str:
        """``text`` in Vietnamese sentence case (see the module docstring); unchanged when nothing applies."""

        parts = _WORD_SPLIT_RE.split(text)
        positions = [index for index, part in enumerate(parts) if part and not part.isspace()]
        cores = [_core(parts[index]) for index in positions]
        starts = {0}
        for number in range(1, len(positions)):
            previous = parts[positions[number - 1]]
            if (previous.rstrip("\"'”’)").endswith(_SENTENCE_ENDS) or previous in _DASHES  # noqa: RUF001
                    or parts[positions[number]].startswith("(")):
                starts.add(number)
        kept = self._kept([core.lower() for core in cores])
        capitals = [number for number, core in enumerate(cores)
                    if number not in starts and capitalised(core) and vietnamese_word(core)]
        lowers = [number for number, core in enumerate(cores)
                  if number not in starts and core[:1].islower() and vietnamese_word(core)]
        if len(capitals) >= _PROTECTED_MIN_WORDS and len(capitals) > len(lowers):
            targets = set(capitals)
        else:
            targets = self._run_targets(cores, starts)
        result = list(parts)
        for number in sorted(targets - kept):
            index = positions[number]
            word = parts[index]
            at = word.find(cores[number])
            result[index] = word[:at] + word[at].lower() + word[at + 1:]
        return "".join(result)

    @staticmethod
    def _run_targets(cores: Sequence[str], starts: set[int]) -> set[int]:
        """Vietnamese words of runs of 3+ capitalised words (a number does not end a run; a sentence start begins a
        new one) that are mostly Vietnamese, sentence starts excepted."""

        targets: set[int] = set()
        run: list[int] = []

        def flush() -> None:
            words = [position for position in run if vietnamese_word(cores[position])]
            if len(run) >= _RUN_MIN_WORDS and 2 * len(words) > len(run):
                targets.update(position for position in words if position not in starts)
            run.clear()

        for number, core in enumerate(cores):
            if number and number in starts:
                flush()
            if capitalised(core):
                run.append(number)
            elif not (core.isdigit() and run):
                flush()
        flush()
        return targets


def sentence_case_plan(plan: IdmW4CourseResponseV1, caser: TitleCaser, *, locale: str,
                       author_title: str | None) -> tuple[IdmW4CourseResponseV1, int]:
    """The W4 course plan with its course, chapter and lesson titles in sentence case, and how many changed. The
    course title stays as the author wrote it when it is the author's course name."""

    if locale != "vi":
        return plan, 0
    changed = 0

    def fix(text: str) -> str:
        nonlocal changed
        fixed = caser.title(text)
        changed += fixed != text
        return fixed

    keep_author = author_title is not None and " ".join(author_title.split()).casefold() == " ".join(
        plan.course_title.split()).casefold()
    updated = plan.model_copy(update={
        "course_title": plan.course_title if keep_author else fix(plan.course_title),
        "modules": [module.model_copy(update={
            "title": fix(module.title),
            "lessons": [lesson.model_copy(update={"title": fix(lesson.title)}) for lesson in module.lessons],
        }) for module in plan.modules],
    })
    return updated, changed


def sentence_case_lessons(lessons: Sequence[IdmLessonDesignV1], caser: TitleCaser, *, locale: str,
                          plan_titles: Mapping[str, str]) -> tuple[list[IdmLessonDesignV1], int]:
    """Lesson, unit and component titles of a module design in sentence case, and how many changed. A lesson title
    that only differs from its W4 title in case takes the W4 title (already in sentence case), so the course plan and
    the module design name the lesson alike."""

    if locale != "vi":
        return list(lessons), 0
    changed = 0

    def fix(text: str) -> str:
        nonlocal changed
        fixed = caser.title(text)
        changed += fixed != text
        return fixed

    def lesson_title(lesson: IdmLessonDesignV1) -> str:
        nonlocal changed
        planned = plan_titles.get(lesson.lesson_key)
        if planned is not None and " ".join(planned.split()).casefold() == " ".join(lesson.title.split()).casefold():
            # The W4 title is in sentence case already (a design written before QLT-7 gets it here).
            title = caser.title(planned)
            changed += title != lesson.title
            return title
        return fix(lesson.title)

    result = [lesson.model_copy(update={
        "title": lesson_title(lesson),
        "units": [unit.model_copy(update={
            "title": fix(unit.title),
            "components": [component.model_copy(update={"title": fix(component.title)})
                           for component in unit.components],
        }) for unit in lesson.units],
    }) for lesson in lessons]
    return result, changed


__all__ = [
    "TitleCaser", "capitalised", "sentence_case_lessons", "sentence_case_plan", "vietnamese_word",
]
