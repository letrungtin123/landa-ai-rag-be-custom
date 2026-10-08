"""Author-facing notes built from templates (spec §8.4). No model writes these.

Notes reach the workspace as ``implementation_notes``: they must stay within the
UI limits, never contain ``<``/``>`` and never expose internal identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Final, Literal, NamedTuple

from app.idm.policy import (
    IDM_NOTES_COURSE_MAX_CHARS,
    IDM_NOTES_LESSON_MAX_CHARS,
    IDM_NOTES_MAX_HOLD_ITEMS,
    IDM_NOTES_MAX_NICE_TO_KNOW,
    IDM_NOTES_MAX_SME_QUESTIONS,
    IDM_NOTES_MODULE_MAX_CHARS,
    IDM_NOTES_SUMMARY_CHARS,
    IDM_PIPELINE_VERSION,
)
from app.idm.text import sanitize_author_text, single_line

Locale = Literal["vi", "en"]


class HoldNote(NamedTuple):
    """One Hold item for the course notes: Must Dos are statements, never ids."""

    name: str
    reason: str
    question: str
    blocked_must_dos: tuple[str, ...] = ()


class SkippedBlockNote(NamedTuple):
    """A Nice to Know block left out of the lessons: name and a one-line summary."""

    name: str
    summary: str

_INTERNAL_ID_RE: Final = re.compile(
    r"\[?\b(?:cb_\d{4}|sec_\d{3}|lo_\d{1,2}|md_\d{1,2}|lsn_\d{3}|mod_\d{2}|pt_\d|idmcb_[0-9a-f]{32}"
    r"|cp2_[0-9a-f]{32}|ao2_[0-9a-f]{32}|scope3_[0-9a-z_]+)\b\]?",
)
_BLOOM_VI: Final = {
    "remember": "Ghi nhớ", "understand": "Hiểu", "apply": "Vận dụng",
    "analyze": "Phân tích", "evaluate": "Đánh giá", "create": "Sáng tạo",
}


def scrub_internal_ids(value: str, known_ids: Iterable[str] = ()) -> str:
    """Remove internal identifiers (fact keys, block/scope ids) from author text."""

    text = _INTERNAL_ID_RE.sub("", value)
    for identifier in sorted({item for item in known_ids if item}, key=len, reverse=True):
        text = text.replace(f"[{identifier}]", "").replace(identifier, "")
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    return text.replace(" .", ".").replace(" ,", ",").strip()


def clean_note(value: str, max_length: int, known_ids: Iterable[str] = ()) -> str:
    return sanitize_author_text(scrub_internal_ids(value, known_ids), max_length)


def bloom_label(bloom: str, locale: Locale) -> str:
    return _BLOOM_VI.get(bloom, bloom) if locale == "vi" else bloom.capitalize()


def _sentence(value: str) -> str:
    """Author text without its closing punctuation, so the template adds exactly one."""

    return value.strip().rstrip(" .;:")


def _hold_line(index: int, hold: HoldNote, vi: bool) -> str:
    reason = _sentence(hold.reason)
    # W2 rows without a reason or question are stored as "-" (blueprint.hold_items).
    line = f"  {index}. {hold.name}" + (f" — {reason}." if reason.strip(" -") else ".")
    if hold.question.strip(" -"):
        label = "Câu hỏi cho SME" if vi else "Question for the SME"
        line += f" {label}: {hold.question.strip()}"
    blocked = [_sentence(item) for item in hold.blocked_must_dos if item.strip()]
    if blocked:
        line += (" Must Do chưa dạy được: " if vi else " Must Do not taught yet: ") + "; ".join(blocked) + "."
    return line


class _Budget:
    """Lines of one note within a hard character budget; sections are added in priority order."""

    def __init__(self, head: list[str], tail: list[str], limit: int) -> None:
        self.head, self.tail = head, tail
        self.body: list[str] = []
        self.left = limit - len("\n".join([*head, *tail]))

    def section(self, title: str, items: Sequence[str], more: Callable[[int], str], cap: int) -> None:
        """Add ``title`` and as many ``items`` as fit (at most ``cap``); the rest become one ``more`` line."""

        shown: list[str] = []
        cost = len(title) + 1
        for position, item in enumerate(items[:cap]):
            rest = len(items) - position - 1
            reserve = len(more(rest)) + 1 if rest else 0
            if cost + len(item) + 1 + reserve > self.left:
                break
            shown.append(item)
            cost += len(item) + 1
        if not shown:
            return
        lines = [title, *shown]
        if len(shown) < len(items):
            lines.append(more(len(items) - len(shown)))
        self.body.extend(lines)
        self.left -= len("\n".join(lines)) + 1

    def text(self) -> str:
        return "\n".join([*self.head, *self.body, *self.tail])


def build_course_notes(
    *,
    locale: Locale,
    ai_proposed: bool,
    total_minutes: int,
    lesson_count: int,
    module_count: int,
    reference_block_count: int,
    excluded_block_count: int,
    holds: Sequence[HoldNote | tuple[str, str, str]],
    sme_questions: Sequence[str],
    fallback_stages: Sequence[str],
    known_ids: Iterable[str] = (),
    pending_objectives: Sequence[str] = (),
    nice_to_know: Sequence[SkippedBlockNote | tuple[str, str]] = (),
) -> str:
    """Course ``implementation_notes`` (spec §8.4).

    Every Hold item (block, reason, SME question, Must Dos it blocks), every objective that waits
    for the SME and the Nice to Know blocks reach the author (QC course 234653, R3/R4). Sections
    are filled in that order within the 7,000-character limit; whatever does not fit is counted
    ("và N mục khác") and stays complete in the workspace panels.
    """

    vi = locale == "vi"
    head = [f"[Thiết kế theo quy trình ID — {IDM_PIPELINE_VERSION}]" if vi
            else f"[Designed with the ID workflow — {IDM_PIPELINE_VERSION}]"]
    if ai_proposed:
        head.append("• Mục tiêu học tập và đối tượng học: do AI đề xuất từ tài liệu — cần xác nhận." if vi
                    else "• Learning objectives and target audience: proposed by AI from the source — please confirm.")
    head.append(f"• Thời lượng ước tính: {total_minutes} phút ({lesson_count} mục, {module_count} chương)." if vi
                else f"• Estimated duration: {total_minutes} minutes "
                     f"({lesson_count} sections, {module_count} chapters).")
    head.append(
        f"• Nội dung chuyển sang Tài liệu tra cứu/Job Aid: {reference_block_count} khối. "
        f"Nội dung loại khỏi khoá học: {excluded_block_count} khối (Nice to Know/Remove)." if vi else
        f"• Content moved to Reference/Job Aid: {reference_block_count} blocks. "
        f"Content left out of the course: {excluded_block_count} blocks (Nice to Know/Remove).")
    tail = []
    if fallback_stages:
        tail.append("• Một số bước thiết kế dùng phương án dự phòng tự động — cần rà soát kỹ." if vi
                    else "• Some design steps used the automatic fallback — review carefully.")
    tail.append("• Cấu trúc được thiết kế theo Must Do, không theo mục lục tài liệu." if vi
                else "• The structure follows the Must Dos, not the source table of contents.")
    budget = _Budget(head, tail, IDM_NOTES_COURSE_MAX_CHARS)

    def more(count: int) -> str:
        return f"  và {count} mục khác" if vi else f"  and {count} more"

    hold_notes = [item if isinstance(item, HoldNote) else HoldNote(*item) for item in holds]
    budget.section(
        f"• Cần chuyên gia bổ sung (Hold) — chờ SME xác nhận ({len(hold_notes)}), chưa đưa vào bài học:" if vi
        else f"• Needs SME input (Hold) — awaiting SME confirmation ({len(hold_notes)}), not in the lessons yet:",
        [_hold_line(index, hold, vi) for index, hold in enumerate(hold_notes, start=1)], more,
        IDM_NOTES_MAX_HOLD_ITEMS)
    budget.section(
        "• Mục tiêu học tập chờ SME (chưa hiển thị là kết quả đầu ra của khoá học):" if vi
        else "• Learning objectives awaiting the SME (not shown as course outcomes):",
        [f"  - {_sentence(item)}." for item in dict.fromkeys(pending_objectives) if item.strip()], more,
        IDM_NOTES_MAX_HOLD_ITEMS)
    skipped = [item if isinstance(item, SkippedBlockNote) else SkippedBlockNote(*item) for item in nice_to_know]
    budget.section(
        f"• Nội dung tham khảo đã lược (Nice to know) — {len(skipped)} khối, tác giả có thể bổ sung thủ công:" if vi
        else f"• Reference content left out (Nice to know) — {len(skipped)} blocks the author may add back:",
        [f"  - {item.name}: {single_line(item.summary, IDM_NOTES_SUMMARY_CHARS)}" for item in skipped], more,
        IDM_NOTES_MAX_NICE_TO_KNOW)
    questions = [question for question in dict.fromkeys(sme_questions) if question]
    budget.section("• Câu hỏi khác cho SME:" if vi else "• Other questions for the SME:",
                   [f"  - {question}" for question in questions], more, IDM_NOTES_MAX_SME_QUESTIONS)
    return clean_note(budget.text(), IDM_NOTES_COURSE_MAX_CHARS, known_ids)


def build_module_notes(
    *,
    locale: Locale,
    performance_goal: str,
    objectives: Sequence[str],
    lesson_count: int,
    total_minutes: int,
    known_ids: Iterable[str] = (),
) -> str:
    vi = locale == "vi"
    lines = [f"Mục tiêu hiệu suất: {performance_goal}" if vi else f"Performance goal: {performance_goal}"]
    if objectives:
        lines.append("• Mục tiêu học tập phục vụ:" if vi else "• Learning objectives served:")
        lines.extend(f"  - {objective}" for objective in objectives)
    lines.append(f"• {lesson_count} mục, ước tính {total_minutes} phút." if vi
                 else f"• {lesson_count} sections, about {total_minutes} minutes.")
    return clean_note("\n".join(lines), IDM_NOTES_MODULE_MAX_CHARS, known_ids)


def build_lesson_notes(
    *,
    locale: Locale,
    bloom: str | None,
    est_screens: int,
    est_minutes: int,
    ordering_rationale: str,
    job_aid: bool,
    known_ids: Iterable[str] = (),
) -> str:
    vi = locale == "vi"
    parts = []
    if job_aid:
        parts.append("Job Aid — tài liệu hỗ trợ khi làm việc, không đánh giá" if vi
                     else "Job Aid — on-the-job support, not assessed")
    elif bloom:
        parts.append(f"Bloom: {bloom_label(bloom, locale)}")
    parts.append(f"Ước tính {est_screens} khối / {est_minutes} phút" if vi
                 else f"Estimated {est_screens} blocks / {est_minutes} minutes")
    parts.append(f"Thứ tự: {ordering_rationale}" if vi else f"Order: {ordering_rationale}")
    return clean_note(" · ".join(parts), IDM_NOTES_LESSON_MAX_CHARS, known_ids)
