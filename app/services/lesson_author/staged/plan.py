"""Staged writer limits, workflow deadline and per-unit component plans."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from fastapi import HTTPException

from app.core.config import settings
from app.services.lesson_author.proposal_validation import (
    MIN_LESSON_AUTHOR_HTML_TEXT_CHARS,
    _normalize_structural_title,
)

# A chapter can exceed the single-response JSON budget even when its source
# text is compact. Keep the threshold below the observed 6.5K-character
# chapter scope so multi-lesson chapters use bounded generation batches.
STAGED_LESSON_AUTHOR_CONTEXT_THRESHOLD = 6000


STAGED_LESSON_AUTHOR_MIN_OUTPUT_TOKENS = 4096


# A source-locked fallback must meet the same learner-facing depth floor as
# model output. A short raw fact is evidence, not a complete lesson unit.
MIN_SOURCE_LOCKED_HTML_TEXT_CHARS = MIN_LESSON_AUTHOR_HTML_TEXT_CHARS


# The skeleton must have enough room for every lesson/unit title and its
# source-fact coverage map before content generation is split into batches.
STAGED_LESSON_AUTHOR_SKELETON_TOKENS = 4096


# A recovery skeleton intentionally stays small. It only partitions source
# facts; component selection is recalculated deterministically afterwards.
STAGED_LESSON_AUTHOR_RECOVERY_UNITS = 4


# Generate one unit per provider call.  A multi-unit JSON array is fragile in
# structured generation: one truncated or renamed item invalidates the whole
# chapter and forces an oversized fallback response.
STAGED_LESSON_AUTHOR_UNITS_PER_BATCH = 1


STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS = 8000
STAGED_LESSON_CONTENT_PROVIDER_TIMEOUT_MAX_MS = 300_000
STAGED_LESSON_WORKFLOW_TIMEOUT_MAX_MS = 480_000


class StagedLessonWorkflowDeadline:
    """Request-scoped budget shared by all Stage-2 content batches.

    This is intentionally a local orchestration guard, not a new retry loop.
    It leaves at least 120 seconds of the existing Node 600-second request
    envelope for routing, retrieval, response handling, and safe failure.
    """

    def __init__(self, remaining_budget_ms: int | None = None) -> None:
        self.started_at = perf_counter()
        self.timeout_ms = min(
            max(1, settings.staged_lesson_workflow_timeout_ms),
            STAGED_LESSON_WORKFLOW_TIMEOUT_MAX_MS,
        )
        if remaining_budget_ms is not None:
            self.timeout_ms = min(self.timeout_ms, max(0, remaining_budget_ms))

    def remaining_ms(self) -> int:
        elapsed_ms = max(0, round((perf_counter() - self.started_at) * 1000))
        return max(0, self.timeout_ms - elapsed_ms)

    def stage_two_provider_timeout_ms(self) -> tuple[int, int]:
        remaining_ms = self.remaining_ms()
        if remaining_ms <= 0:
            raise HTTPException(
                status_code=504,
                detail={
                    "code": "AI_STAGED_LESSON_WORKFLOW_TIMEOUT",
                    "message": "AI provider phản hồi quá lâu. Vui lòng thử lại sau.",
                },
            )
        provider_timeout_ms = min(
            max(1, settings.staged_lesson_content_provider_timeout_ms),
            STAGED_LESSON_CONTENT_PROVIDER_TIMEOUT_MAX_MS,
            remaining_ms,
        )
        return provider_timeout_ms, remaining_ms


def staged_lesson_content_output_tokens(*, request_max_output_tokens: int, max_facts_per_unit: int) -> int:
    """Retain the existing Stage-2 output-token calculation as a testable contract."""
    required_content_tokens = max(8_192, min(65_536, max(1, max_facts_per_unit) * 1_200))
    return min(request_max_output_tokens, required_content_tokens)


STAGED_COMPONENT_TYPE_ALIASES = {
    "html": "html",
    "problem": "problem",
    "quiz": "problem",
    "question": "problem",
    "multiple_choice": "problem",
    "multiple-select": "problem",
    "multiple_select": "problem",
    "multi_choice": "problem",
    "multi_select": "problem",
    "mcq": "problem",
    "dropdown": "problem",
    "select": "problem",
    "numerical": "problem",
    "numeric": "problem",
    "short_text": "problem",
    "short-answer": "problem",
    "short_answer": "problem",
    "la_problem": "problem",
    "la_faq": "la_faq",
    "faq": "la_faq",
    "la_sortable": "la_sortable",
    "sortable": "la_sortable",
    "ordering": "la_sortable",
    "la_crossword": "la_crossword",
    "crossword": "la_crossword",
    "vocabulary": "la_crossword",
    "la_diagram": "la_diagram",
    "diagram": "la_diagram",
    "flowchart": "la_diagram",
    "mindmap": "la_diagram",
}


def normalize_staged_component_type(value: Any) -> str | None:
    raw = str(value or "").strip().casefold().replace(" ", "_")
    return STAGED_COMPONENT_TYPE_ALIASES.get(raw)


def _staged_unit_fact_text(
    unit: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> tuple[list[str], str]:
    expected_ids = [
        str(fact_id).strip()
        for fact_id in unit.get("source_fact_ids", [])
        if str(fact_id).strip()
    ]
    expected_set = set(expected_ids)
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in expected_set
    ]
    text = " ".join(
        re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
        for fact in facts
    ).strip()
    return expected_ids, text


def build_staged_component_plan(
    unit: dict[str, Any],
    manifest: dict[str, Any] | None,
    locale: str = "vi",
) -> list[dict[str, Any]]:
    """Create an evidence-gated learning-format plan before content generation.

    The model may suggest formats, but deterministic source signals decide which
    formats are allowed. This prevents a generic html-only response while also
    preventing unsupported activities from being invented for a source.
    """
    fact_ids, fact_text = _staged_unit_fact_text(unit, manifest)
    folded = fact_text.casefold()
    semantic_folded = f"{str(unit.get('title') or '')} {fact_text}".casefold()
    raw_plan = unit.get("component_plan")
    if not isinstance(raw_plan, list):
        raw_plan = unit.get("components") if isinstance(unit.get("components"), list) else []
    requested_types = {
        normalized
        for item in raw_plan
        if isinstance(item, dict)
        for normalized in [normalize_staged_component_type(item.get("type"))]
        if normalized
    }

    has_process_signal = bool(re.search(
        r"\b(?:bước|buoc|step|steps|giai đoạn|giai doan|phase|phases|"
        r"quy trình|quy trinh|process|workflow|trình tự|trinh tu|sequence|"
        r"sau đó|sau do|tiếp theo|tiep theo|then|next|finally)\b",
        semantic_folded,
        flags=re.IGNORECASE,
    ))
    has_model_signal = bool(re.search(
        r"\b(?:mô hình triển khai|mo hinh trien khai|implementation model|"
        r"framework|chu trình|chu trinh|cycle)\b",
        semantic_folded,
        flags=re.IGNORECASE,
    ))
    # A numbered taxonomy (for example, risk categories) is not an ordered
    # activity. Only expose diagram/sortable when the source also describes a
    # process or an explicitly ordered implementation model.
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in set(fact_ids)
    ]
    explicit_ordered_items = _source_locked_ordered_items(facts)
    # A process title is not enough: the source must expose at least three
    # complete, explicit steps before we produce a learner ordering activity.
    has_ordered_evidence = (
        len(explicit_ordered_items) >= 3
        and (has_process_signal or has_model_signal)
    )
    # A procedure is normally explanatory. Ordering becomes an interaction
    # only when Stage A explicitly asks the learner to reconstruct the order;
    # source order by itself must never force la_sortable.
    has_ordering_practice_request = "la_sortable" in requested_types
    has_relationship_evidence = bool(re.search(
        r"(?:mối quan hệ|moi quan he|relationship|liên kết|lien ket|"
        r"hệ thống|he thong|system|phân cấp|phan cap|hierarchy|"
        r"luồng|luong|flow|workflow|phụ thuộc|phu thuoc|dependency)",
        semantic_folded,
        flags=re.IGNORECASE,
    )) and len(explicit_ordered_items) >= 2
    # A quiz is an assessment format, not a generic synonym for a list of
    # facts. Only select it when the source itself contains assessment/Q&A
    # signals or the model explicitly proposed it. This avoids forcing a
    # broken quiz into definition-only units and keeps the proposal grounded.
    has_assessable_evidence = len(fact_ids) >= 2 and bool(re.search(
        r"(?:\?|câu hỏi|cau hoi|đáp án|dap an|lựa chọn|lua chon|bài kiểm tra|bai kiem tra|"
        r"kiểm tra kiến thức|kiem tra kien thuc|"
        r"quiz|assessment|question|answer|đúng\s+(?:hay|hoặc)\s+sai|"
        r"dung\s+(?:hay|hoac)\s+sai|true\s*(?:or|/)\s*false)",
        folded,
        flags=re.IGNORECASE,
    ))
    has_faq_evidence = bool(re.search(r"(?:\?|hỏi đáp|hoi dap|faq|câu hỏi thường gặp|cau hoi thuong gap)", folded, flags=re.IGNORECASE))
    has_crossword_evidence = bool(re.search(r"(?:ô chữ|o chu|crossword|thuật ngữ|thuat ngu)", folded, flags=re.IGNORECASE))

    allowed = {"html"}
    if has_assessable_evidence:
        allowed.add("problem")
    if has_relationship_evidence:
        allowed.add("la_diagram")
    if has_ordered_evidence and has_ordering_practice_request:
        allowed.add("la_sortable")
    if has_faq_evidence:
        allowed.add("la_faq")
    if has_crossword_evidence:
        allowed.add("la_crossword")

    selected = ["html"]
    for component_type in ("problem", "la_diagram", "la_sortable", "la_faq", "la_crossword"):
        if component_type in allowed and (component_type in requested_types or component_type == "problem"):
            selected.append(component_type)

    rationale_text = {
        "html": (
            "Giải thích đầy đủ các ý, điều kiện và thuật ngữ có trong tài liệu nguồn."
            if locale != "en" else
            "Explains the source facts, conditions, and terminology in full."
        ),
        "problem": (
            "Kiểm tra mức độ hiểu các ý chính bằng câu hỏi chỉ dựa trên fact của tài liệu nguồn."
            if locale != "en" else
            "Checks understanding of the key source facts without adding external facts."
        ),
        "la_diagram": (
            "Biểu diễn trực quan quy trình hoặc mô hình có thứ tự đã xuất hiện trong tài liệu nguồn."
            if locale != "en" else
            "Visualizes the ordered process or model explicitly present in the source."
        ),
        "la_sortable": (
            "Cho người học sắp xếp lại các bước theo đúng trình tự được nêu trong tài liệu nguồn."
            if locale != "en" else
            "Lets learners reorder the steps using the sequence stated in the source."
        ),
        "la_faq": (
            "Chuyển các cặp hỏi-đáp rõ ràng trong tài liệu nguồn thành học liệu tra cứu."
            if locale != "en" else
            "Turns explicit source question-and-answer pairs into a reference activity."
        ),
        "la_crossword": (
            "Luyện nhớ các thuật ngữ đã được nêu rõ trong tài liệu nguồn."
            if locale != "en" else
            "Practices terminology explicitly present in the source."
        ),
    }
    purpose_by_type = {
        "html": "explain",
        "problem": "assess",
        "la_diagram": "relationship",
        "la_sortable": "sequence",
        "la_faq": "clarify",
        "la_crossword": "terminology",
    }
    requirement_by_type = {
        "html": "Explain every assigned source fact accurately and preserve required source structure.",
        "problem": "Assess understanding of the assigned source facts without adding unsupported facts.",
        "la_diagram": "Show the source-supported relationship or flow represented by the assigned facts.",
        "la_sortable": "Preserve the source-supported order of the assigned procedure.",
        "la_faq": "Clarify source-grounded questions using the assigned source facts.",
        "la_crossword": "Practice only source-supported terminology represented by the assigned facts.",
    }
    return [
        {
            "type": component_type,
            "rationale": rationale_text[component_type],
            "purpose": purpose_by_type[component_type],
            "source_fact_ids": fact_ids,
            "content_requirements": [requirement_by_type[component_type]],
        }
        for component_type in selected[:4]
    ]


def _merge_staged_unit_title(left: str, right: str, locale: str) -> str:
    """Keep merged source units readable without inventing a new topic."""
    joiner = " and " if locale == "en" else " và "
    candidate = f"{left.strip()}{joiner}{right.strip()}".strip()
    fallback = "Course content" if locale == "en" else "Nội dung khóa học"
    return _normalize_structural_title(candidate[:180], fallback)


def _merge_staged_units(
    left: dict[str, Any],
    right: dict[str, Any],
    locale: str,
) -> dict[str, Any]:
    source_fact_ids: list[str] = []
    seen_fact_ids: set[str] = set()
    for value in [*left.get("source_fact_ids", []), *right.get("source_fact_ids", [])]:
        fact_id = str(value or "").strip()
        if fact_id and fact_id not in seen_fact_ids:
            seen_fact_ids.add(fact_id)
            source_fact_ids.append(fact_id)
    return {
        "title": _merge_staged_unit_title(
            str(left.get("title") or ""),
            str(right.get("title") or ""),
            locale,
        ),
        "source_fact_ids": source_fact_ids,
        # Plans are recomputed after every merge from the combined evidence.
        "component_plan": [],
    }


def consolidate_staged_thin_units(
    skeleton: dict[str, Any],
    manifest: dict[str, Any] | None,
    locale: str,
) -> dict[str, Any]:
    """Merge adjacent thin units within a lesson before authoring.

    A fact fragment may be authoritative but cannot produce a useful standalone
    lesson. Merging only adjacent units in the same lesson preserves the source
    order and all fact IDs while allowing a complete explanation to be drafted.
    """
    for chapter_value in skeleton.get("chapters", []):
        if not isinstance(chapter_value, dict):
            continue
        for lesson_value in chapter_value.get("lessons", []):
            if not isinstance(lesson_value, dict):
                continue
            raw_units = [
                dict(unit)
                for unit in lesson_value.get("units", [])
                if isinstance(unit, dict)
            ]
            if len(raw_units) < 2:
                continue

            merged_units: list[dict[str, Any]] = []
            pending: dict[str, Any] | None = None
            for unit in raw_units:
                if pending is None:
                    pending = unit
                    continue
                _fact_ids, pending_text = _staged_unit_fact_text(pending, manifest)
                if len(pending_text) < MIN_LESSON_AUTHOR_HTML_TEXT_CHARS:
                    pending = _merge_staged_units(pending, unit, locale)
                else:
                    merged_units.append(pending)
                    pending = unit

            if pending is not None:
                _fact_ids, pending_text = _staged_unit_fact_text(pending, manifest)
                if len(pending_text) < MIN_LESSON_AUTHOR_HTML_TEXT_CHARS and merged_units:
                    pending = _merge_staged_units(merged_units.pop(), pending, locale)
                merged_units.append(pending)
            lesson_value["units"] = merged_units
    return skeleton


def ensure_staged_chapter_component_diversity(
    units: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    locale: str,
) -> None:
    """Add grounded retrieval checks when a substantive chapter is all HTML."""
    if any(unit.get("component_plan_locked") is True for unit in units):
        return
    if any(
        component_type != "html"
        for unit in units
        for component_type in unit.get("component_types", [])
    ):
        return

    candidates: list[tuple[int, int, dict[str, Any], list[str]]] = []
    for unit in units:
        fact_ids, fact_text = _staged_unit_fact_text(unit, manifest)
        if fact_ids and len(fact_text) >= MIN_LESSON_AUTHOR_HTML_TEXT_CHARS:
            candidates.append((len(fact_text), len(fact_ids), unit, fact_ids))
    if not candidates:
        return

    rationale = (
        "Checks recall of the source facts with a short-answer question; the answer must remain grounded in the assigned facts."
        if locale == "en"
        else "Kiểm tra việc nhớ các fact nguồn bằng câu hỏi trả lời ngắn; đáp án phải bám đúng các fact đã gán."
    )
    # Keep the chapter usable: enrich distinct substantive units but never turn
    # every source fragment into an assessment. The source order is retained.
    for _text_length, _fact_count, candidate, fact_ids in candidates[:3]:
        if len(candidate.get("component_plan", [])) >= 4:
            continue
        candidate["component_plan"] = [
            *candidate.get("component_plan", []),
            {
                "type": "problem",
                "rationale": rationale,
                "source_fact_ids": fact_ids,
            },
        ]
        candidate["component_types"] = [
            *candidate.get("component_types", []),
            "problem",
        ]


def _source_locked_ordered_items(facts: list[dict[str, Any]]) -> list[str]:
    """Extract ordered labels without adding facts that are absent from raw KB."""
    items: list[str] = []
    item_prefix = r"(?:\d+\s*[.)-]\s+|(?:bước|buoc|step)\s*\d+\s*[:.)-]\s+|[-•●▪◦]\s+)"
    for fact in facts:
        raw_text = str(fact.get("text") or "").strip()
        if not raw_text:
            continue
        # Preserve line boundaries before whitespace normalization: DOCX/PDF
        # extraction frequently represents a procedure as bullet-only lines.
        raw_segments = re.split(rf"(?={item_prefix})|[\r\n]+", raw_text)
        extracted = [
            re.sub(rf"^{item_prefix}", "", re.sub(r"\s+", " ", segment)).strip(" -:")
            for segment in raw_segments
            if re.match(rf"^\s*{item_prefix}", segment, flags=re.IGNORECASE)
        ]
        # Never promote arbitrary prose or visual line fragments into ordered
        # learner items. Only explicit bullets/numbered steps can be safely
        # reconstructed without an LLM semantic pass.
        items.extend(extracted)

    unique_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique_items.append(normalized[:500])
    return unique_items[:12]
