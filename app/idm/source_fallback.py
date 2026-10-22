"""The last resort of the whole-unit fallback of an IDM unit, and its note lines.

The lead's verification of run 5496155c (2026-10-22): 9 of the 10 real unit requests answered ``fallback_only`` with 422
``ORCHESTRATION_V2_UNIT_FALLBACK_INVALID`` (already at db022e7): their question slot (and some FAQ slots) rest on prose,
slide headings and classroom exercises, which no source structure turns into a question (``app.idm.source_locked``), so
the whole unit had no fallback and a provider failure broke it. The whole-unit fallback (never the fallback of one slot
of a provider draft) then uses complete sentences the lesson states, without adding a claim: a question asks whether
the lesson states one (the two wrong options say what the lesson does not: the opposite, or nothing), an FAQ answers
with them. The unit is marked for the author's review and its note says these slots are placeholders to rewrite, and
names the words the PDF glued together that a fallback slot shows (never split: where the space goes is a guess).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final

from app.content.instructional_quality import faq_answer_is_complete
from app.content.learner_content_purity import component_learner_text
from app.idm.source_locked import (
    _BULLET_RE,
    _FAQ_MAX_ITEMS,
    _FAQ_MIN_ITEMS,
    _FAQ_UNSAFE_RE,
    EXERCISE_RE,
    STATEMENT_END_RE,
    _choices,
    _faq_item,
    _feedback,
    _option_key,
    _prepare_lines,
    _sentence_like,
    _structural,
)
from app.idm.text import glued_words, idm_fold

# Glued words a fallback note names (the rest are counted).
_MAX_GLUED_WORDS: Final = 5
# A statement is a complete sentence of at least this many words and at most this many characters (never cut).
_STATEMENT_MIN_WORDS: Final = 8
_STATEMENT_MAX_CHARS: Final = 300
# A list item read as a statement has at least this many words ("Khách hàng mục tiêu").
_ITEM_MIN_WORDS: Final = 2
# A sentence opens with a capital letter, after its opening quotation marks; a lowercase or bracketed start is the rest
# of a line the PDF cut ("\"nút thắt\" rủi ro …", "(Lý tính - Việc làm) | …").
_OPENING_MARKS: Final = "\"'" + "".join(map(chr, (0x201C, 0x2018)))


def _statements(values: Iterable[Any], *, exercises: bool = False) -> list[str]:
    """The complete sentences of ``values`` a learner can read as a statement of the lesson (wrapped lines joined), in
    source order: sentence case, at least ``_STATEMENT_MIN_WORDS`` words, a closing full stop, no question, no table
    row or list item, and no exercise prompt unless ``exercises``."""

    found = []
    for line in _prepare_lines("", values):
        text = " ".join(line.split())
        opening = text.lstrip(_OPENING_MARKS)[:1]
        if (_structural(text) or "?" in text or (not exercises and EXERCISE_RE.match(idm_fold(text)))
                or " | " in text or not opening.isupper() or text.count("(") != text.count(")")
                or not _sentence_like(text)
                or not STATEMENT_END_RE.search(text) or len(text.split()) < _STATEMENT_MIN_WORDS
                or len(text) > _STATEMENT_MAX_CHARS or not faq_answer_is_complete(text)
                or _FAQ_UNSAFE_RE.search(text)):
            continue
        found.append(text)
    return list(dict.fromkeys(found))


def _items(values: Iterable[Any]) -> list[str]:
    """The bullet items of ``values`` (read before the shared cleaning drops their markers) a learner can read as a
    statement: at least ``_ITEM_MIN_WORDS`` words, no question, no exercise prompt, balanced brackets; a full stop
    closes one that has no closing punctuation."""

    found = []
    for line in (line for value in values for line in str(value).splitlines()):
        match = _BULLET_RE.match(" ".join(line.split()))
        text = match.group(1).strip() if match else ""
        if (len(text.split()) < _ITEM_MIN_WORDS or "?" in text or EXERCISE_RE.match(idm_fold(text))
                or text.count("(") != text.count(")") or len(text) > _STATEMENT_MAX_CHARS):
            continue
        found.append(text if STATEMENT_END_RE.search(text) else f"{text}.")
    return list(dict.fromkeys(found))


def lesson_statements(*groups: Iterable[Any]) -> list[str]:
    """The statements of ``groups`` (the slot's facts, then the unit's, then those of the lesson's earlier units), in
    that order, those without words the PDF glued together first; the exercise prompts only when there is nothing
    else, and the list items of a bullet-only source after them."""

    materialized = [list(values) for values in groups]
    found = list(dict.fromkeys(text for values in materialized for text in _statements(values)))
    # A unit that only shows a classroom exercise of the slides ("Thực hành …: Các nhóm được phát …") states that.
    found = found or list(dict.fromkeys(text for values in materialized
                                        for text in _statements(values, exercises=True)))
    # Review of dfb6db2: a source of bullets without punctuation states its items (with a closing full stop).
    found = found or list(dict.fromkeys(text for values in materialized for text in _items(values)))
    return sorted(found, key=lambda text: bool(glued_words(text)))


def idm_statement_question(title: str, statements: Sequence[str], *, locale: str) -> dict[str, Any] | None:
    """A single-choice question on the first of ``statements`` (a sentence the lesson states): is it what the lesson
    says? The key is the option that it is; the two others (the lesson says the opposite, the lesson does not say it)
    are false because the sentence is the lesson's own. ``None`` without a statement."""

    if not statements:
        return None
    statement = statements[0]
    en = locale == "en"
    options = (["True: the lesson states exactly this.", "False: the lesson states the opposite.",
                "The lesson does not mention this."] if en else
               ["Đúng: bài học nêu đúng như vậy.", "Sai: bài học nêu điều ngược lại.",
                "Bài học không nói đến điều này."])
    choices = _choices(options, options[0], f"statement:{title}:{statement}")
    stated = "the lesson states this sentence" if en else "bài học nêu đúng câu này"
    reasons = {_option_key(options[0]): stated,
               _option_key(options[1]): stated + (", not its opposite" if en else ", không nêu điều ngược lại"),
               _option_key(options[2]): stated}
    question = (f"Read this statement:\n- {statement}\nAccording to what you studied, is it true?" if en
                else f"Đọc nhận định sau:\n- {statement}\nTheo nội dung đã học, nhận định này có đúng không?")
    return {"problem_type": "multiple_choice", "question": question, "choices": choices,
            "explanation": _feedback(f"The lesson states: {statement}" if en else f"Bài học nêu: {statement}",
                                     choices, reasons, locale)}


def idm_statement_faq(title: str, statements: Sequence[str], *, locale: str) -> list[dict[str, str]] | None:
    """FAQ items whose answers are ``statements`` (sentences the lesson states) under the source-locked "what to
    remember" questions; ``None`` below two items."""

    items: list[dict[str, str]] = []
    for statement in statements:
        if (item := _faq_item(items, "", statement, title, locale, distinct=True)) is not None:
            items.append(item)
        if len(items) == _FAQ_MAX_ITEMS:
            break
    return items if len(items) >= _FAQ_MIN_ITEMS else None


def last_resort_builders(fact_texts: Sequence[str], brief: Mapping[str, Any] | None,
                         ) -> tuple[Callable[..., dict[str, Any] | None], Callable[..., list[dict[str, str]] | None]]:
    """The question and FAQ builders of the whole-unit fallback for the slots no source structure rebuilds: they read
    the statements of the slot's facts, then of ``fact_texts`` (the unit's), then of the lesson's earlier units (the
    brief's ``lesson_context_facts``)."""

    context = [str(item.get("fact_text") or "") for item in (brief or {}).get("lesson_context_facts") or []
               if isinstance(item, Mapping)]

    def question(title: str, values: Iterable[Any], *, locale: str) -> dict[str, Any] | None:
        return idm_statement_question(title, lesson_statements(values, fact_texts, context), locale=locale)

    def faq(title: str, values: Iterable[Any], *, locale: str, unit_values: Iterable[Any] = ()) -> (
            list[dict[str, str]] | None):
        del unit_values  # the unit's facts are ``fact_texts``
        return idm_statement_faq(title, lesson_statements(values, fact_texts, context), locale=locale)

    return question, faq


def placeholder_line(where: str, vi: bool) -> str:
    """The slots (``where``: "khối 2 (Quiz), khối 3 (Hỏi đáp)") a whole-unit fallback filled with sentences of the
    lesson: the author writes them again."""

    return (f"Tài liệu không đủ dữ kiện để dựng {where} nên bản dự phòng chỉ nhắc lại câu của bài — tác giả cần soạn "
            "lại phần này." if vi
            else f"The source holds nothing to build {where} from, so the fallback only repeats sentences of the "
                 "lesson — the author must write this part again.")


def glued_line(words: Sequence[str], vi: bool) -> str:
    """Words of the PDF without their spaces, kept as written in a fallback slot (never split by guessing)."""

    shown = ", ".join(f"«{word}»" for word in list(dict.fromkeys(words))[:_MAX_GLUED_WORDS])
    rest = len(dict.fromkeys(words)) - _MAX_GLUED_WORDS
    more = (f" và {rest} chữ khác" if vi else f" and {rest} more") if rest > 0 else ""
    return (f"Bản dự phòng giữ nguyên chữ bị dính liền trong tài liệu PDF, cần sửa tay: {shown}{more}." if vi
            else f"The fallback keeps words the PDF glued together, to fix by hand: {shown}{more}.")


def fallback_glued_words(components: Sequence[Any], fallback_slots: Sequence[int]) -> list[str]:
    """The glued words (``text.glued_words``) the learner text of the fallback slots shows, in order."""

    texts = [component_learner_text(component) for index, component in enumerate(components)
             if index in fallback_slots and isinstance(component, dict)]
    return glued_words(" ".join(texts))


__all__ = ["fallback_glued_words", "glued_line", "idm_statement_faq", "idm_statement_question", "last_resort_builders",
           "lesson_statements", "placeholder_line"]
