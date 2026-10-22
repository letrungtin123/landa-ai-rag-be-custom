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
    IDM_NOTES_MAX_REMOVED,
    IDM_NOTES_MODULE_MAX_CHARS,
    IDM_NOTES_SOURCE_CHECKS_MAX_CHARS,
    IDM_NOTES_SUMMARY_CHARS,
    IDM_PIPELINE_VERSION,
)
from app.idm.text import sanitize_author_text, single_line

Locale = Literal["vi", "en"]


class SkippedBlockNote(NamedTuple):
    """A block left out of the lessons (Remove): name and the W2 reason."""

    name: str
    summary: str


class KeptDefinitionNote(NamedTuple):
    """A block W2 proposed to leave out that the server kept because it defines a course term (N4)."""

    name: str
    terms: tuple[str, ...]
    proposed: str

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


class _Budget:
    """Lines of one note within a hard character budget; sections are added in priority order."""

    def __init__(self, head: list[str], tail: list[str], limit: int) -> None:
        self.head, self.tail = head, tail
        self.body: list[str] = []
        self.left = limit - len("\n".join([*head, *tail]))

    def section(self, title: str, items: Sequence[str], more: Callable[[int], str], cap: int,
                room: int | None = None) -> None:
        """Add ``title`` and as many ``items`` as fit (at most ``cap``, within ``room`` characters when given, so the
        sections after it keep theirs); the rest become one ``more`` line."""

        shown: list[str] = []
        cost = len(title) + 1
        left = self.left if room is None else min(self.left, room)
        for position, item in enumerate(items[:cap]):
            rest = len(items) - position - 1
            reserve = len(more(rest)) + 1 if rest else 0
            if cost + len(item) + 1 + reserve > left:
                break
            shown.append(item)
            cost += len(item) + 1
        if not shown:
            # Nothing fits: the section still says how many items wait elsewhere, when even that fits.
            pointer = [title, more(len(items))] if items else []
            if pointer and len("\n".join(pointer)) + 1 <= self.left:
                self.body.extend(pointer)
                self.left -= len("\n".join(pointer)) + 1
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
    fallback_stages: Sequence[str],
    known_ids: Iterable[str] = (),
    removed: Sequence[SkippedBlockNote | tuple[str, str]] = (),
    kept_definitions: Sequence[KeptDefinitionNote] = (),
    purpose_line: str = "",
    pattern_label: str = "",
    template_gaps: Sequence[str] = (),
    kept_background: Sequence[KeptDefinitionNote] = (),
    brief_lines: Sequence[str] = (),
    audience_entered: bool = False,
    source_checks: Sequence[str] = (),
    source_unclear: Sequence[str] = (),
    loose_must_dos: Sequence[str] = (),
    unserved_must_dos: Sequence[str] = (),
) -> str:
    """Course ``implementation_notes`` (spec §8.4).

    The definition blocks kept against the W2 proposal and the removed blocks with their reason reach the author (QC
    course 234653 R4, 364564 N4). Sections are filled in that order within the 7,000-character limit; whatever does not
    fit is counted ("và N mục khác") and stays complete in the workspace panels. Owner decisions 2026-10-22 (QC
    feedback A5, A6): there is no Hold list, no objective waiting for an SME, no Nice to Know list and no question for
    an SME any more (the design never holds a block and the AI writes no SME question).

    QLT-5: the head says which course purpose the design followed (``purpose_line``); then come the required parts of
    the purpose's course pattern the source lacks (``template_gaps``, plain notes the AI wrote no content for), and
    after the kept definitions the onboarding background kept against the W2 proposal.

    QLT-6: right after the workflow tag, what the author entered when starting AI ID (``brief_lines``: the learner
    audience and the course purpose, as entered); with an entered audience (``audience_entered``) only the
    objectives are marked as proposed by AI.

    ``source_checks`` (owner decision 2026-10-22): one plain sentence per contradictory or out-of-date point of a taught
    block (``app.idm.source_flags``), the first section after the head; ``source_unclear`` (run 5496155c): the unclear
    points, a softer section right after it. Review m20: then the Must Dos attached
    to a block without a relation ("statement → “block”", ``loose_must_dos``) and those with nothing to teach in the
    course (``unserved_must_dos``; the design never blocks a Must Do, contract K2).
    """

    vi = locale == "vi"
    head = [f"[Thiết kế theo quy trình ID — {IDM_PIPELINE_VERSION}]" if vi
            else f"[Designed with the ID workflow — {IDM_PIPELINE_VERSION}]"]
    head.extend(line for line in brief_lines if line.strip())
    if purpose_line:
        head.append(purpose_line)
    if ai_proposed and audience_entered:
        head.append("• Mục tiêu học tập: do AI đề xuất theo mục đích và đối tượng học tác giả nhập — cần xác nhận."
                    if vi else "• Learning objectives: proposed by AI for the purpose and audience the author entered "
                               "— please confirm.")
    elif ai_proposed:
        head.append("• Mục tiêu học tập và đối tượng học: do AI đề xuất từ tài liệu — cần xác nhận." if vi
                    else "• Learning objectives and target audience: proposed by AI from the source — please confirm.")
    head.append(f"• Thời lượng ước tính: {total_minutes} phút ({lesson_count} mục, {module_count} chương)." if vi
                else f"• Estimated duration: {total_minutes} minutes "
                     f"({lesson_count} sections, {module_count} chapters).")
    head.append(
        f"• Nội dung chuyển sang Tài liệu tra cứu/Job Aid: {reference_block_count} khối. "
        f"Nội dung loại khỏi khoá học: {excluded_block_count} khối." if vi else
        f"• Content moved to Reference/Job Aid: {reference_block_count} blocks. "
        f"Content left out of the course: {excluded_block_count} blocks.")
    tail = []
    if fallback_stages:
        tail.append("• Một số bước thiết kế dùng phương án dự phòng tự động — cần rà soát kỹ." if vi
                    else "• Some design steps used the automatic fallback — review carefully.")
    tail.append("• Cấu trúc được thiết kế theo Must Do, không theo mục lục tài liệu." if vi
                else "• The structure follows the Must Dos, not the source table of contents.")
    budget = _Budget(head, tail, IDM_NOTES_COURSE_MAX_CHARS)

    def more(count: int) -> str:
        return f"  và {count} mục khác" if vi else f"  and {count} more"

    _source_sections(budget, source_checks, source_unclear, vi, more)
    budget.section("• Cần xem — Must Do chưa có phần riêng trong tài liệu, đã gắn vào phần gần nhất:" if vi
                   else "• Review needed — Must Dos without content of their own, attached to the closest part:",
                   [f"  - {item}" for item in loose_must_dos], more, len(loose_must_dos))
    budget.section("• Cần xem — Must Do chưa có nội dung nào để dạy trong khoá:" if vi
                   else "• Review needed — Must Dos with nothing to teach in the course:",
                   [f"  - {item}" for item in unserved_must_dos], more, len(unserved_must_dos))
    gaps = [item for item in dict.fromkeys(template_gaps) if item.strip()]
    budget.section(
        f"• Khung khoá học ({pattern_label}): tài liệu còn thiếu {len(gaps)} phần — AI không tự viết nội dung cho "
        "các phần này:" if vi else
        f"• Course pattern ({pattern_label}): the source lacks {len(gaps)} parts — the AI wrote no content for them:",
        [f"  - {item}" for item in gaps], more, IDM_NOTES_MAX_HOLD_ITEMS)
    proposed_vi = {"remove": "Remove"}
    budget.section(
        f"• Giữ lại vì định nghĩa thuật ngữ cốt lõi (Week 2 đề xuất loại) — {len(kept_definitions)} khối:" if vi
        else f"• Kept because they define a core term (Week 2 proposed to leave them out) — "
             f"{len(kept_definitions)} blocks:",
        [f"  - {item.name}: {', '.join(item.terms)} "
         + (f"(đề xuất ban đầu: {proposed_vi.get(item.proposed, item.proposed)})." if vi
            else f"(first proposed: {proposed_vi.get(item.proposed, item.proposed)}).")
         for item in kept_definitions], more, IDM_NOTES_MAX_REMOVED)
    budget.section(
        f"• Giữ lại vì là nội dung hội nhập nhân viên mới cần (Week 2 đề xuất loại) — {len(kept_background)} khối:"
        if vi else f"• Kept as onboarding content a new employee needs (Week 2 proposed to leave them out) — "
                   f"{len(kept_background)} blocks:",
        [f"  - {item.name} " + (f"(đề xuất ban đầu: {proposed_vi.get(item.proposed, item.proposed)})." if vi
                                else f"(first proposed: {proposed_vi.get(item.proposed, item.proposed)}).")
         for item in kept_background], more, IDM_NOTES_MAX_REMOVED)
    # QC feedback 2026-10-21 (A6): no "Nice to know" list; W2 stores such blocks as Remove (``blueprint``), listed here.
    dropped = [item if isinstance(item, SkippedBlockNote) else SkippedBlockNote(*item) for item in removed]
    budget.section(
        f"• Nội dung đã loại khỏi khoá (Remove) — {len(dropped)} khối, kèm lý do:" if vi
        else f"• Content removed from the course (Remove) — {len(dropped)} blocks, with the reason:",
        [f"  - {item.name}: {single_line(item.summary, IDM_NOTES_SUMMARY_CHARS)}" for item in dropped], more,
        IDM_NOTES_MAX_REMOVED)
    return clean_note(budget.text(), IDM_NOTES_COURSE_MAX_CHARS, known_ids)


def _source_sections(budget: _Budget, checks: Sequence[str], unclear: Sequence[str], vi: bool,
                     more: Callable[[int], str]) -> None:
    """The points to check in the source (contradictory or out of date), then the unclear ones (run 5496155c: an edit
    note, never a check before applying), each within its own room."""

    titles = (("• Chỗ tài liệu cần kiểm tra trước khi áp dụng:",
               "• Chỗ tài liệu ghi chưa rõ, nên đọc lại khi biên tập:") if vi
              else ("• Points to check in the source:",
                    "• Points where the source is unclear, to reread when editing:"))
    for title, items in zip(titles, (checks, unclear), strict=True):
        shown = [item for item in dict.fromkeys(items) if item.strip()]
        budget.section(title, [f"  - {item}" for item in shown], more, len(shown), IDM_NOTES_SOURCE_CHECKS_MAX_CHARS)


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
