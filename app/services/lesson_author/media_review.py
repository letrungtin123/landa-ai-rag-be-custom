"""Blueprint media review: proposed media from approved evidence only."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal

from app.media_brief import build_media_brief
from app.services.lesson_author.errors import LessonAuthorProposalValidationError

MEDIA_REVIEW_VERSION = "media-review-v1"
MEDIA_DECISION_PROPOSED = "PROPOSED"
MEDIA_DECISION_NOT_NEEDED = "NOT_NEEDED"
MEDIA_DECISION_SOURCE_GAP = "SOURCE_GAP"
MEDIA_DECISION_FAILED = "FAILED"
MEDIA_DECISION_NOT_EVALUATED = "NOT_EVALUATED"


MEDIA_DECISION_STATUSES = {
    MEDIA_DECISION_PROPOSED,
    MEDIA_DECISION_NOT_NEEDED,
    MEDIA_DECISION_SOURCE_GAP,
    MEDIA_DECISION_FAILED,
    MEDIA_DECISION_NOT_EVALUATED,
}


MEDIA_VIDEO_INTENTS = {"procedure", "worked_example", "scenario"}
MEDIA_INFOGRAPHIC_INTENTS = {"comparison", "relationship_visualization", "warning"}


def _bounded_media_evidence_excerpt(value: Any, maximum: int = 145) -> str:
    """Make one presentation-safe evidence excerpt; this is not a fact resolver."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= maximum:
        return text
    boundary = text.rfind(" ", 0, maximum - 1)
    return (text[:boundary if boundary > 32 else maximum - 1].rstrip() + "…").strip()


def _media_plan_from_approved_evidence(
    *,
    unit_title: str,
    media_type: Literal["video", "static_infographic"],
    evidence_texts: list[str],
    locale: str,
) -> dict[str, str]:
    """Create a bounded recommendation brief from already-approved evidence.

    This is deliberately deterministic and never turns a missing source fact
    into a recommendation. It is a review brief, not an asset, script, URL,
    or a claim that every fact in a dense unit fits into a visual.
    """
    excerpts = [_bounded_media_evidence_excerpt(text) for text in evidence_texts if text][:3]
    if not excerpts:
        raise LessonAuthorProposalValidationError(
            "MEDIA_CANDIDATE_EVIDENCE_UNRESOLVED",
            "A media candidate requires resolved approved evidence text.",
        )
    is_english = locale == "en"
    evidence_panels = ("Source excerpts (original language): " if is_english else "Trích nguồn (ngôn ngữ gốc): ") + "; ".join(excerpts)
    if media_type == "video":
        return {
            "type": "video",
            "title": (f"Proposed walkthrough: {unit_title}" if is_english else f"Video đề xuất: {unit_title}")[:180],
            "content_outline": (
                f"Opening: state the learner action for {unit_title}. Sequence the approved evidence into observable steps: {evidence_panels}. Close with the learner check before continuing."
                if is_english else
                f"Mở đầu nêu hành động học tập của {unit_title}. Trình bày lần lượt evidence đã duyệt thành các bước quan sát được: {evidence_panels}. Kết thúc bằng điểm tự kiểm trước khi chuyển tiếp."
            )[:600],
            "rationale": (
                "A step-by-step visual can make the approved procedure easier to follow without creating a new asset."
                if is_english else
                "Trình bày tuần tự giúp người học theo dõi quy trình đã có evidence mà không tạo tài sản mới."
            )[:240],
        }
    return {
        "type": "static_infographic",
        "title": (f"Proposed visual summary: {unit_title}" if is_english else f"Infographic đề xuất: {unit_title}")[:180],
        "content_outline": (
            f"Panels: learning focus for {unit_title}; the approved relationship, comparison, or warning; source-grounded takeaways: {evidence_panels}; final decision cue for the learner."
            if is_english else
            f"Các panel: trọng tâm {unit_title}; mối quan hệ, so sánh hoặc cảnh báo đã duyệt; điểm chính bám nguồn: {evidence_panels}; gợi ý quyết định cho người học."
        )[:600],
        "rationale": (
            "A single visual can make the approved relationship or condition easier to compare without creating a new asset."
            if is_english else
            "Một visual tĩnh giúp người học so sánh mối quan hệ hoặc điều kiện đã có evidence mà không tạo tài sản mới."
        )[:240],
    }


def enrich_lesson_author_blueprint_media_review(
    blueprint: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    locale: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Attach explicit, source-bounded media decisions to a new Blueprint.

    Candidate selection is deterministic from approved semantic roles. A
    candidate without resolvable fact text remains SOURCE_GAP; it is never
    downgraded to NOT_NEEDED and no fake media object is produced. Existing
    provider media plans remain supported but are reclassified explicitly.
    """
    enriched = deepcopy(blueprint)
    facts_by_id = {
        str(fact.get("fact_id") or "").strip(): fact
        for fact in (source_coverage_manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    }
    existing_plan_count = sum(
        1
        for chapter in enriched.get("chapters", []) if isinstance(chapter, dict)
        for lesson in chapter.get("lessons", []) if isinstance(lesson, dict)
        for unit in lesson.get("units", []) if isinstance(unit, dict)
        if isinstance(unit.get("media_plan"), dict)
    )
    plan_capacity = max(0, 12 - existing_plan_count)
    decisions: list[dict[str, str]] = []
    metrics = {status: 0 for status in MEDIA_DECISION_STATUSES}
    for chapter_index, chapter in enumerate(enriched.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                if isinstance(unit.get("media_plan"), dict):
                    status, reason_code = MEDIA_DECISION_PROPOSED, "ARCHITECTURE_MEDIA_PLAN_VALIDATED"
                else:
                    intents = {
                        str(block.get("intent") or "").strip()
                        for block in unit.get("learning_blocks", [])
                        if isinstance(block, dict)
                    }
                    media_type: Literal["video", "static_infographic"] | None = (
                        "video" if intents & MEDIA_VIDEO_INTENTS else
                        "static_infographic" if intents & MEDIA_INFOGRAPHIC_INTENTS else None
                    )
                    if media_type is None:
                        status, reason_code = MEDIA_DECISION_NOT_NEEDED, "NO_SOURCE_BACKED_VISUAL_CANDIDATE"
                    else:
                        unit_fact_ids = [
                            str(fact_id).strip()
                            for fact_id in unit.get("source_fact_ids", [])
                            if isinstance(fact_id, str) and str(fact_id).strip()
                        ]
                        owned_media_ids = set(unit_fact_ids)
                        evidence_texts = [str(row.get("text") or "").strip()
                                          for fact_id, row in facts_by_id.items() if fact_id in owned_media_ids]
                        if not evidence_texts:
                            status, reason_code = MEDIA_DECISION_SOURCE_GAP, "MEDIA_CANDIDATE_EVIDENCE_UNRESOLVED"
                        elif plan_capacity <= 0:
                            status, reason_code = MEDIA_DECISION_FAILED, "MEDIA_RECOMMENDATION_CAPACITY_EXCEEDED"
                        else:
                            brief = build_media_brief(str(unit.get("title") or "").strip()[:180], media_type, evidence_texts, locale)
                            if brief is None:
                                status, reason_code = MEDIA_DECISION_SOURCE_GAP, "MEDIA_CANDIDATE_EVIDENCE_UNRESOLVED"
                            else:
                                unit["media_plan"] = brief
                                plan_capacity -= 1
                                status, reason_code = MEDIA_DECISION_PROPOSED, (
                                    "PROCEDURE_VISUAL_CANDIDATE" if media_type == "video"
                                    else "RELATIONSHIP_OR_WARNING_VISUAL_CANDIDATE"
                                )
                decisions.append({"unit_path": unit_path, "status": status, "reason_code": reason_code})
                metrics[status] += 1
    enriched["media_review"] = {"version": MEDIA_REVIEW_VERSION, "decisions": decisions}
    return enriched, {f"media_{status.lower()}_count": count for status, count in metrics.items()}
