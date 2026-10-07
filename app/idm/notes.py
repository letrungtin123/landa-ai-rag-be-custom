"""Author-facing notes built from templates (spec §8.4). No model writes these.

Notes reach the workspace as ``implementation_notes``: they must stay within the
UI limits, never contain ``<``/``>`` and never expose internal identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final, Literal

from app.idm.policy import (
    IDM_NOTES_COURSE_MAX_CHARS,
    IDM_NOTES_LESSON_MAX_CHARS,
    IDM_NOTES_MAX_HOLD_ITEMS,
    IDM_NOTES_MAX_SME_QUESTIONS,
    IDM_NOTES_MODULE_MAX_CHARS,
    IDM_PIPELINE_VERSION,
)
from app.idm.text import sanitize_author_text

Locale = Literal["vi", "en"]

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


def build_course_notes(
    *,
    locale: Locale,
    ai_proposed: bool,
    total_minutes: int,
    lesson_count: int,
    module_count: int,
    reference_block_count: int,
    excluded_block_count: int,
    holds: Sequence[tuple[str, str, str]],
    sme_questions: Sequence[str],
    fallback_stages: Sequence[str],
    known_ids: Iterable[str] = (),
) -> str:
    """Course ``implementation_notes`` (``holds`` = (name, reason, question))."""

    vi = locale == "vi"
    lines = [f"[Thiết kế theo quy trình ID — {IDM_PIPELINE_VERSION}]" if vi
             else f"[Designed with the ID workflow — {IDM_PIPELINE_VERSION}]"]
    if ai_proposed:
        lines.append("• Mục tiêu học tập và đối tượng học: do AI đề xuất từ tài liệu — cần xác nhận." if vi
                     else "• Learning objectives and target audience: proposed by AI from the source — please confirm.")
    lines.append(f"• Thời lượng ước tính: {total_minutes} phút ({lesson_count} mục, {module_count} chương)." if vi
                 else f"• Estimated duration: {total_minutes} minutes "
                      f"({lesson_count} sections, {module_count} chapters).")
    lines.append(
        f"• Nội dung chuyển sang Tài liệu tra cứu/Job Aid: {reference_block_count} khối. "
        f"Nội dung loại khỏi khoá học: {excluded_block_count} khối (Nice to Know/Remove)." if vi else
        f"• Content moved to Reference/Job Aid: {reference_block_count} blocks. "
        f"Content left out of the course: {excluded_block_count} blocks (Nice to Know/Remove).")
    if holds:
        lines.append(f"• Chờ SME xác nhận ({len(holds)}):" if vi else f"• Awaiting SME confirmation ({len(holds)}):")
        for index, (name, reason, question) in enumerate(holds[:IDM_NOTES_MAX_HOLD_ITEMS], start=1):
            lines.append(f"  {index}. {name} — {reason}. Câu hỏi: {question}" if vi
                         else f"  {index}. {name} — {reason}. Question: {question}")
        if len(holds) > IDM_NOTES_MAX_HOLD_ITEMS:
            rest = len(holds) - IDM_NOTES_MAX_HOLD_ITEMS
            lines.append(f"  và {rest} mục khác" if vi else f"  and {rest} more")
    questions = [question for question in dict.fromkeys(sme_questions) if question][:IDM_NOTES_MAX_SME_QUESTIONS]
    if questions:
        lines.append("• Câu hỏi khác cho SME:" if vi else "• Other questions for the SME:")
        lines.extend(f"  - {question}" for question in questions)
    if fallback_stages:
        lines.append("• Một số bước thiết kế dùng phương án dự phòng tự động — cần rà soát kỹ." if vi
                     else "• Some design steps used the automatic fallback — review carefully.")
    lines.append("• Cấu trúc được thiết kế theo Must Do, không theo mục lục tài liệu." if vi
                 else "• The structure follows the Must Dos, not the source table of contents.")
    return clean_note("\n".join(lines), IDM_NOTES_COURSE_MAX_CHARS, known_ids)


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
