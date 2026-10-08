from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.instructional_density import (
    DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET,
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    evaluate_instructional_density,
    instructional_output_budget,
    profile_instructional_texts,
)
from app.instructional_quality import (
    build_source_grounded_single_choice,
    clean_source_facts,
    ordered_source_steps,
    source_clarification_signals,
    source_relationship_pairs,
    source_term_definitions,
)
from app.lesson_author_orchestration_v2 import (
    ArchitectureComponentAuthorReviewV2,
    ArchitectureComponentPlanV2,
    ChapterBlueprintShardV2,
    CourseSkeletonChapterV2,
    CourseSkeletonV2,
    LessonArchitectureV2,
    OrchestrationContractError,
    UnitArchitectureV2,
    canonical_hash,
)
from app.prompt_safety import UNTRUSTED_JSON_CONTEXT_RULE
from app.source_evidence_bundle import (
    build_degraded_source_evidence_bundle,
    build_source_evidence_bundle,
)

MAX_SOURCE_PAGE_FACTS = 500
MAX_SOURCE_SCOPES = 4096
MAX_SHARD_SOURCE_CHARS = 400_000
MAX_V3_ARCHITECTURE_SHARD_SOURCE_CHARS = 60_000
MAX_UNIT_SOURCE_FACTS = 32_768
COMPONENT_PLAN_ID_PATTERN = r"^cp2_[a-f0-9]{32}$"
ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION = "orchestration-v2-google-wire-2"
ORCHESTRATION_V2_PROVIDER_UNSUPPORTED_BOUNDS = frozenset({
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
})


def _project_orchestration_v2_provider_schema(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build the SDK wire schema without weakening server validation.

    Google structured output accepts only a documented JSON Schema subset.
    Pydantic's strict server models emit ``additionalProperties`` plus value
    bounds which have already been observed to trigger provider-side
    ``INVALID_ARGUMENT`` responses for this contract. Remove those keywords
    from the provider wire only. The authoritative Pydantic models remain
    unchanged and enforce every bound after generation.
    """

    result = deepcopy(schema)
    definitions = result.get("$defs", {})
    removed: Counter[str] = Counter()

    def visit(node: dict[str, Any]) -> None:
        if "additionalProperties" in node:
            removed["additionalProperties"] += 1
            del node["additionalProperties"]
        for keyword in ORCHESTRATION_V2_PROVIDER_UNSUPPORTED_BOUNDS:
            if keyword in node:
                removed[keyword] += 1
                del node[keyword]
        # Keep nullable nested objects explicit on the provider wire. Pydantic
        # emits ``anyOf([$ref, null])`` for this shape; inlining only that exact
        # reference avoids SDK/version-specific conversion differences while
        # preserving the authoritative local model and nullable semantics.
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            null_members = [item for item in any_of if isinstance(item, dict) and item.get("type") == "null"]
            value_members = [item for item in any_of if item not in null_members]
            if len(null_members) == 1 and len(value_members) == 1 and isinstance(value_members[0], dict):
                ref = value_members[0].get("$ref")
                prefix = "#/$defs/"
                target = definitions.get(ref[len(prefix):]) if isinstance(ref, str) and ref.startswith(prefix) else None
                if isinstance(target, dict):
                    siblings = {key: value for key, value in node.items() if key != "anyOf"}
                    node.clear()
                    node.update(deepcopy(target))
                    node.update(siblings)
                    node["nullable"] = True
                    removed["nullableRefAnyOf"] += 1
        # Walk schema values, never property names or arbitrary examples.
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    visit(child)
        if isinstance(node.get("items"), dict):
            visit(node["items"])
        for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
            for child in node.get(key, []):
                if isinstance(child, dict):
                    visit(child)

    visit(result)
    return result, dict(sorted(removed.items()))


def orchestration_v2_provider_response_model(
    server_model: type[BaseModel],
) -> tuple[type[BaseModel], dict[str, Any]]:
    """Return a provider-only schema projection backed by strict validation."""

    _, removed = _project_orchestration_v2_provider_schema(server_model.model_json_schema())

    class OrchestrationV2ProviderWire(server_model):
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            projected = _project_orchestration_v2_provider_schema(
                super().model_json_schema(*args, **kwargs),
            )[0]
            # A dynamic subclass changes Pydantic's root title. Keep the
            # server model title so projection metadata is the only wire delta.
            if "title" in projected:
                projected["title"] = server_model.__name__
            return projected

    OrchestrationV2ProviderWire.__name__ = f"{server_model.__name__}ProviderWire"
    OrchestrationV2ProviderWire.__qualname__ = OrchestrationV2ProviderWire.__name__
    return OrchestrationV2ProviderWire, {
        "schema_projection_version": ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION,
        "server_validation_unchanged": True,
        "projected_unsupported_keyword_counts": removed,
    }


class SourceSnapshotFactV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str = Field(min_length=1, max_length=64)
    fact_key: str = Field(min_length=1, max_length=255)
    scope_key: str = Field(min_length=1, max_length=255)
    fact_text: str = Field(min_length=1, max_length=32768)
    source_ref: str | None = Field(default=None, max_length=255)
    source_page: int | None = Field(default=None, ge=1)
    source_chunk: int | None = Field(default=None, ge=0)
    locator: dict[str, Any] = Field(default_factory=dict)


class SourceScopeCatalogEntryV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope_key: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)
    source_ref: str | None = Field(default=None, max_length=255)
    fact_count: int = Field(ge=1, le=1_000_000)
    content_chars: int = Field(ge=1, le=100_000_000)


class SourceOutlineChapterV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    order: int = Field(ge=0, le=511)
    document_id: str = Field(min_length=1, max_length=64)
    source_ref: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=500)


class SourceOutlineAuthorityV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["locked", "model_designed", "needs_review"]
    source: Literal["toc", "headings", "none", "ambiguous"]
    complete: bool
    confidence: float = Field(ge=0, le=1)
    structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: list[str] = Field(max_length=32)
    chapters: list[SourceOutlineChapterV2] = Field(max_length=512)

    @model_validator(mode="after")
    def validate_authority(self) -> "SourceOutlineAuthorityV2":
        if self.mode == "locked":
            if not self.complete or not self.chapters or self.source not in {"toc", "headings"}:
                raise ValueError("locked source authority is incomplete")
            if [chapter.order for chapter in self.chapters] != list(range(len(self.chapters))):
                raise ValueError("source authority order is invalid")
            identities = [(chapter.document_id, chapter.source_ref) for chapter in self.chapters]
            if len(identities) != len(set(identities)):
                raise ValueError("source authority identity is duplicated")
        elif self.chapters:
            raise ValueError("non-locked source authority cannot bind chapters")
        return self


class SourceSnapshotPayloadV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    facts: list[SourceSnapshotFactV2] = Field(min_length=1, max_length=10_000_000)
    scopes: list[SourceScopeCatalogEntryV2] = Field(min_length=1, max_length=MAX_SOURCE_SCOPES)


class CourseSkeletonDraftV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=5000)
    target_audience: str = Field(min_length=1, max_length=2000)
    prerequisites: list[str] = Field(max_length=1000)
    learning_outcomes: list[str] = Field(min_length=1, max_length=4096)
    assessment_strategy: str = Field(min_length=1, max_length=5000)
    assumptions: list[str] = Field(max_length=1000)
    chapters: list[CourseSkeletonChapterV2] = Field(min_length=1, max_length=512)


class ChapterShardDraftV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lessons: list[LessonArchitectureV2] = Field(min_length=1, max_length=512)


def fallback_course_skeleton_draft_v2(
    locale: str,
    scopes: list[SourceScopeCatalogEntryV2],
    source_authority: SourceOutlineAuthorityV2,
) -> CourseSkeletonDraftV2:
    """Build a conservative complete skeleton after malformed model output.

    Source authority and scope ownership remain server-owned. The fallback is
    intentionally generic: chapter shards and unit writers still produce the
    learner-facing instructional design from the immutable source ledger.
    """

    if not scopes or source_authority.mode == "needs_review":
        raise OrchestrationContractError("SOURCE_STRUCTURE_REVIEW_REQUIRED")
    vi = locale == "vi"
    if source_authority.mode == "locked":
        by_ref: dict[str, list[str]] = {}
        unbound: list[str] = []
        for scope in scopes:
            (by_ref.setdefault(scope.source_ref, []) if scope.source_ref else unbound).append(scope.scope_key)
        chapter_groups = []
        for index, chapter in enumerate(source_authority.chapters):
            owned = list(by_ref.get(chapter.source_ref, []))
            if index == 0:
                owned = [*unbound, *owned]
            if not owned:
                raise OrchestrationContractError("ARCHITECTURE_SOURCE_CHAPTER_SCOPE_MISSING")
            chapter_groups.append((owned, chapter.title))
    else:
        chunk_size = max(1, (len(scopes) + 511) // 512)
        chapter_groups = []
        for group_index, start in enumerate(range(0, len(scopes), chunk_size), start=1):
            group = scopes[start:start + chunk_size]
            title = (group[0].title if len(group) == 1 else
                     (f"Phần {group_index}: {group[0].title}" if vi else
                      f"Part {group_index}: {group[0].title}"))
            chapter_groups.append(([scope.scope_key for scope in group], title))
    chapters = [CourseSkeletonChapterV2(
        chapter_key=f"chapter-{index + 1}", order=index, title=title[:500],
        objective=((f"Hiểu và áp dụng nội dung của {title}." if vi else
                    f"Understand and apply the content of {title}.")[:2000]),
        learning_outcomes=[
            (f"Vận dụng được các nội dung trọng tâm của {title}." if vi else
             f"Apply the key concepts from {title}.")[:500]
        ],
        source_scope_ids=scope_ids,
    ) for index, (scope_ids, title) in enumerate(chapter_groups)]
    return CourseSkeletonDraftV2(
        title="Khóa học từ tài liệu nguồn" if vi else "Course from source materials",
        summary=("Lộ trình học tập được xây dựng đầy đủ từ tài liệu nguồn đã chọn."
                 if vi else "A complete learning path built from the selected source materials."),
        target_audience="Người học của khóa học" if vi else "Course learners",
        prerequisites=[],
        learning_outcomes=[
            "Hiểu và vận dụng các nội dung trọng tâm trong tài liệu nguồn."
            if vi else "Understand and apply the key content from the source materials."
        ],
        assessment_strategy=("Đánh giá khả năng hiểu và vận dụng nội dung theo từng chương."
                             if vi else "Assess understanding and application chapter by chapter."),
        assumptions=[], chapters=chapters,
    )


def parse_chapter_shard_draft_v2(text: str) -> ChapterShardDraftV2:
    """Accept the strict contract plus harmless omissions seen on provider wires.

    Server validation remains authoritative. This only restores nullable review
    fields and canonical component aliases before the strict model is applied.
    """

    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("lessons"), list):
        raise ValueError("chapter shard payload is not an object with lessons")
    aliases = {
        "faq": "la_faq", "sortable": "la_sortable", "crossword": "la_crossword",
        "diagram": "la_diagram",
    }
    for lesson in payload["lessons"]:
        if not isinstance(lesson, dict) or not isinstance(lesson.get("units"), list):
            continue
        for unit in lesson["units"]:
            if not isinstance(unit, dict):
                continue
            unit.setdefault("media_brief", None)
            components = unit.get("component_plan")
            if not isinstance(components, list):
                continue
            for component in components:
                if not isinstance(component, dict):
                    continue
                if isinstance(component.get("type"), str):
                    component["type"] = aliases.get(component["type"], component["type"])
                review = component.setdefault("author_review", {})
                if isinstance(review, dict):
                    for field in ("purpose", "example_scenario", "visual_asset", "user_behavior_navigation"):
                        review.setdefault(field, None)
    return ChapterShardDraftV2.model_validate(payload)


def _safe_architecture_failure_code(error: Exception) -> str:
    if isinstance(error, OrchestrationContractError):
        return error.code
    if isinstance(error, ValidationError):
        first = error.errors(include_url=False, include_input=False)[:1]
        return str(first[0].get("type") or "VALIDATION_ERROR") if first else "VALIDATION_ERROR"
    if isinstance(error, json.JSONDecodeError):
        return "JSON_INVALID"
    return type(error).__name__[:80]


def _subplan_for_scopes(
    plan: "ChapterShardPlanV2",
    scope_ids: list[str],
    facts: list[SourceSnapshotFactV2],
) -> "ChapterShardPlanV2":
    selected = [fact for fact in facts if fact.scope_key in set(scope_ids)]
    return ChapterShardPlanV2(
        chapter_key=plan.chapter_key,
        order=plan.order,
        shard_index=plan.shard_index,
        shard_count=plan.shard_count,
        source_scope_ids=scope_ids,
        source_fact_count=len(selected),
        source_content_chars=sum(len(fact.fact_text) for fact in selected),
    )


def salvage_chapter_shard_draft_v2(
    text: str,
    *,
    skeleton: CourseSkeletonV2,
    plan: "ChapterShardPlanV2",
    source_facts: list[SourceSnapshotFactV2],
) -> tuple[ChapterBlueprintShardV2, dict[str, Any]] | None:
    """Keep only independently proven provider units and fill their missing scopes.

    A provider lesson or unit is never repaired by guessing fields. Each unit is
    parsed with its original lesson metadata and then passed through the exact
    same evidence/density compiler as a complete shard. Units that fail any
    boundary are discarded as a whole; deterministic fallback owns only their
    still-unclaimed immutable scopes.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    raw_lessons = payload.get("lessons") if isinstance(payload, dict) else None
    if not isinstance(raw_lessons, list):
        return None

    allowed_scopes = set(plan.source_scope_ids)
    claimed_scopes: set[str] = set()
    provider_lessons: list[tuple[int, LessonArchitectureV2]] = []
    rejection_codes: Counter[str] = Counter()
    candidate_unit_count = 0

    for lesson_index, raw_lesson in enumerate(raw_lessons):
        raw_units = raw_lesson.get("units") if isinstance(raw_lesson, dict) else None
        if not isinstance(raw_lesson, dict) or not isinstance(raw_units, list):
            rejection_codes["LESSON_SHAPE_INVALID"] += 1
            continue
        accepted_units: list[UnitArchitectureV2] = []
        lesson_first_scope_index = len(plan.source_scope_ids)
        for raw_unit in raw_units:
            candidate_unit_count += 1
            try:
                single = parse_chapter_shard_draft_v2(json.dumps({
                    "lessons": [{**raw_lesson, "units": [raw_unit]}],
                }, ensure_ascii=False))
                lesson = single.lessons[0]
                unit = lesson.units[0]
                unit_scopes = list(unit.source_scope_ids)
                unit_scope_set = set(unit_scopes)
                if (not unit_scope_set or not unit_scope_set.issubset(allowed_scopes)
                        or unit_scope_set & claimed_scopes):
                    raise OrchestrationContractError("ARCHITECTURE_SALVAGE_SCOPE_CONFLICT")
                unit_plan = _subplan_for_scopes(plan, unit_scopes, source_facts)
                unit_facts = [fact for fact in source_facts if fact.scope_key in unit_scope_set]
                if {fact.scope_key for fact in unit_facts} != unit_scope_set:
                    raise OrchestrationContractError("ARCHITECTURE_SHARD_CONTEXT_MISMATCH")
                # This call is the authority check. Keep the parsed provider
                # unit only when its normalized result is independently valid;
                # the final full-shard bind repeats the same deterministic pass
                # and materializes obligations with final lesson/unit indices.
                bind_chapter_shard_v2(
                    single,
                    skeleton=skeleton,
                    plan=unit_plan,
                    source_facts=unit_facts,
                )
                accepted_units.append(unit)
                claimed_scopes.update(unit_scope_set)
                lesson_first_scope_index = min(
                    lesson_first_scope_index,
                    *(plan.source_scope_ids.index(scope_id) for scope_id in unit_scopes),
                )
            except (OrchestrationContractError, ValidationError, ValueError, json.JSONDecodeError) as error:
                rejection_codes[_safe_architecture_failure_code(error)] += 1
        if accepted_units:
            # Re-validate the original lesson metadata with exactly the proven
            # provider units. No title/objective/activity field is synthesized.
            try:
                normalized = LessonArchitectureV2.model_validate({
                    **{key: value for key, value in raw_lesson.items() if key != "units"},
                    "units": [unit.model_dump(mode="python") for unit in accepted_units],
                })
                provider_lessons.append((lesson_first_scope_index, normalized))
            except ValidationError as error:
                rejection_codes[_safe_architecture_failure_code(error)] += len(accepted_units)
                claimed_scopes.difference_update(
                    scope_id for unit in accepted_units for scope_id in unit.source_scope_ids
                )

    if not provider_lessons:
        return None

    missing_scopes = [scope_id for scope_id in plan.source_scope_ids if scope_id not in claimed_scopes]
    lesson_segments: list[tuple[int, LessonArchitectureV2]] = list(provider_lessons)
    if missing_scopes:
        missing_plan = _subplan_for_scopes(plan, missing_scopes, source_facts)
        missing_facts = [fact for fact in source_facts if fact.scope_key in set(missing_scopes)]
        fallback = fallback_chapter_shard_draft_v2(skeleton, missing_plan, missing_facts)
        for fallback_lesson in fallback.lessons:
            position = min(
                plan.source_scope_ids.index(scope_id)
                for unit in fallback_lesson.units
                for scope_id in unit.source_scope_ids
            )
            lesson_segments.append((position, fallback_lesson))

    combined = ChapterShardDraftV2(
        lessons=[lesson for _position, lesson in sorted(lesson_segments, key=lambda item: item[0])],
    )
    compiler_diagnostics: list[dict[str, Any]] = []
    shard = bind_chapter_shard_v2(
        combined,
        skeleton=skeleton,
        plan=plan,
        source_facts=source_facts,
        compiler_diagnostics=compiler_diagnostics,
    )
    return shard, {
        "candidate_unit_count": candidate_unit_count,
        "accepted_provider_unit_count": sum(len(lesson.units) for _position, lesson in provider_lessons),
        "fallback_scope_count": len(missing_scopes),
        "rejection_codes": dict(sorted(rejection_codes.items())),
        "component_decisions": compiler_diagnostics,
    }


def _fallback_scope_title(scope_key: str, facts: list[SourceSnapshotFactV2]) -> str:
    scoped = [fact for fact in facts if fact.scope_key == scope_key]
    source_ref = next((fact.source_ref for fact in scoped if fact.source_ref), None)
    candidate = source_ref if source_ref and not source_ref.lower().startswith("src-") else ""
    if not candidate and scoped:
        candidate = scoped[0].fact_text
    normalized = " ".join((candidate or scope_key).split()).strip()
    if len(normalized) > 140:
        normalized = normalized[:137].rstrip() + "..."
    return normalized or scope_key


def fallback_chapter_shard_draft_v2(
    skeleton: CourseSkeletonV2,
    plan: "ChapterShardPlanV2",
    facts: list[SourceSnapshotFactV2],
) -> ChapterShardDraftV2:
    """Build a source-complete conservative architecture without another LLM call.

    The downstream unit writer still produces the learner-facing content. This
    fallback guarantees that one malformed architecture response cannot strand
    the whole course or discard any immutable source scope.
    """

    chapter = next((item for item in skeleton.chapters if item.chapter_key == plan.chapter_key), None)
    if chapter is None or chapter.order != plan.order:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_IDENTITY_MISMATCH")
    lessons: list[LessonArchitectureV2] = []
    vi = skeleton.locale == "vi"
    scope_groups = [plan.source_scope_ids[index:index + 8]
                    for index in range(0, len(plan.source_scope_ids), 8)]
    for lesson_index, scopes in enumerate(scope_groups, start=1):
        labels = [_fallback_scope_title(scope_key, facts) for scope_key in scopes]
        learning_objectives = [
            ((f"Phân tích và vận dụng nội dung trọng tâm: {label}" if vi else
              f"Analyze and apply the core content: {label}")[:500])
            for label in labels
        ]
        units = [UnitArchitectureV2(
            title=label[:180],
            purpose=((f"Giúp người học đạt mục tiêu '{learning_objectives[index - 1]}' "
                      f"và đóng góp vào kết quả đầu ra của chương '{chapter.title}'." if vi else
                     f"Help learners achieve '{learning_objectives[index - 1]}' and contribute "
                     f"to the expected outcomes of chapter '{chapter.title}'.")[:500]),
            learning_objective_refs=[f"lo_{index}"],
            source_scope_ids=[scope_key],
            component_plan=[ArchitectureComponentPlanV2(
                type="html",
                title=f"Nội dung trọng tâm: {label}"[:180],
                rationale=("Trình bày đầy đủ nội dung nguồn để trực tiếp hỗ trợ mục tiêu của bài học."
                           if vi else "Present the source content completely to directly support the lesson objective."),
                author_review=ArchitectureComponentAuthorReviewV2(
                    purpose=("Giải thích và hệ thống hóa nội dung nguồn theo mục tiêu của bài học."
                             if vi else "Explain and organize the source content around the lesson objective."),
                    example_scenario=None,
                    visual_asset=None,
                    user_behavior_navigation=None,
                ),
                source_scope_ids=[scope_key],
            )],
            media_brief=None,
        ) for index, (scope_key, label) in enumerate(zip(scopes, labels), start=1)]
        lesson_title = chapter.title if len(scope_groups) == 1 else (
            f"{chapter.title} - Phần {lesson_index}" if vi else f"{chapter.title} - Part {lesson_index}")
        lesson_objective = ((f"Vận dụng các nội dung {', '.join(labels)} để góp phần đạt kết quả đầu ra của "
                             f"chương '{chapter.title}'.") if vi else
                            (f"Apply {', '.join(labels)} to contribute to the expected outcomes of "
                             f"chapter '{chapter.title}'."))
        lessons.append(LessonArchitectureV2(
            title=lesson_title[:180],
            # LessonArchitectureV2 owns this boundary (max_length=500). Keep
            # deterministic fallback inside the same contract as provider
            # output so a provider outage can never turn fallback into ASGI 500.
            objective=lesson_objective[:500],
            learning_objectives=learning_objectives,
            learning_activities=[("Đọc nội dung, đối chiếu tình huống và thực hành áp dụng."
                                  if vi else "Study the content, compare scenarios, and practice applying it.")],
            assessment=("Đánh giá mức độ hiểu và khả năng áp dụng nội dung của phần học."
                        if vi else "Assess understanding and the ability to apply the section content."),
            units=units,
        ))
    return ChapterShardDraftV2(lessons=lessons)


CourseSkeletonProviderWireV2, COURSE_SKELETON_PROVIDER_SCHEMA_METADATA = (
    orchestration_v2_provider_response_model(CourseSkeletonDraftV2)
)
ChapterShardProviderWireV2, CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA = (
    orchestration_v2_provider_response_model(ChapterShardDraftV2)
)


class ChapterShardPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chapter_key: str
    order: int
    shard_index: int
    shard_count: int
    source_scope_ids: list[str]
    source_fact_count: int
    source_content_chars: int


class UnitComponentPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    component_plan_id: str = Field(pattern=COMPONENT_PLAN_ID_PATTERN)
    type: Literal["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]
    title: str = Field(min_length=1, max_length=180)
    rationale: str = Field(min_length=1, max_length=500)
    purpose: Literal["explain", "assess", "sequence", "relationship", "terminology", "clarify"]
    source_fact_ids: list[str] = Field(max_length=MAX_UNIT_SOURCE_FACTS)
    supporting_evidence_fact_ids: list[str] = Field(max_length=MAX_UNIT_SOURCE_FACTS)
    learning_objective_refs: list[str] = Field(min_length=1, max_length=24)
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)
    content_requirements: list[str] = Field(max_length=8)
    learning_block_ids: list[str] = Field(max_length=12)
    required_artifacts: list[dict[str, Any]] = Field(max_length=6)


class UnitGenerationContractV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal[2]
    unit_content_policy_version: Literal["unit-content-v4-alignment-1"] | None = None
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assembly_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapter_key: str
    unit_path: str = Field(pattern=r"^chapter_[1-9][0-9]*\.lesson_[1-9][0-9]*\.unit_[1-9][0-9]*$")
    chapter_title: str = Field(min_length=1, max_length=500)
    lesson_title: str = Field(min_length=1, max_length=180)
    lesson_learning_objectives: list[str] = Field(min_length=1, max_length=8)
    unit_title: str = Field(min_length=1, max_length=180)
    unit_purpose: str = Field(min_length=1, max_length=500)
    unit_learning_objective_refs: list[str] = Field(min_length=1, max_length=24)
    unit_source_scope_ids: list[str] = Field(min_length=1, max_length=4096)
    unit_source_fact_ids: list[str] = Field(min_length=1, max_length=MAX_UNIT_SOURCE_FACTS)
    component_plan: list[UnitComponentPlanV2] = Field(min_length=1, max_length=4)
    source_facts: list[SourceSnapshotFactV2] = Field(min_length=1, max_length=MAX_UNIT_SOURCE_FACTS)
    # IDM runs only (spec §8.3). Kept as raw JSON so the contract hash matches the
    # Node builder byte for byte; ``app.idm.storyboard`` validates it strictly.
    idm_unit_brief: dict[str, Any] | None = None
    contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_unit_generation_contract(self) -> "UnitGenerationContractV2":
        scopes = set(self.unit_source_scope_ids)
        fact_ids = [fact.fact_key for fact in self.source_facts]
        if (len(scopes) != len(self.unit_source_scope_ids) or len(fact_ids) != len(set(fact_ids))
                or fact_ids != self.unit_source_fact_ids
                or {fact.scope_key for fact in self.source_facts} != scopes
                or sum(len(fact.fact_text) for fact in self.source_facts) > MAX_SHARD_SOURCE_CHARS):
            raise ValueError("ORCHESTRATION_V2_UNIT_CONTEXT_INVALID")
        plan_ids = [plan.component_plan_id for plan in self.component_plan]
        component_types = [plan.type for plan in self.component_plan]
        if (len(plan_ids) != len(set(plan_ids))
                or ("html" in component_types and component_types[0] != "html")
                or ("la_faq" in component_types and component_types[-1] != "la_faq")):
            raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        fact_set = set(fact_ids)
        v3_markers = [
            fact.locator.get("instructional_density_policy_version") == INSTRUCTIONAL_DENSITY_POLICY_VERSION
            for fact in self.source_facts
        ]
        if any(v3_markers) and not all(v3_markers):
            raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        uses_unit_content_v3 = bool(v3_markers) and all(v3_markers)
        owned_fact_ids: set[str] = set()
        uses_alignment_v4 = self.unit_content_policy_version == "unit-content-v4-alignment-1"
        for index, plan in enumerate(self.component_plan):
            expected_fact_ids = [fact.fact_key for fact in self.source_facts
                                 if fact.scope_key in set(plan.source_scope_ids)]
            if uses_alignment_v4:
                expected_owned_fact_ids = [fact_id for fact_id in expected_fact_ids if fact_id not in owned_fact_ids]
                expected_supporting_fact_ids = [fact_id for fact_id in expected_fact_ids if fact_id in owned_fact_ids]
                owned_fact_ids.update(expected_owned_fact_ids)
            else:
                expected_owned_fact_ids = expected_fact_ids if not uses_unit_content_v3 or index == 0 else []
                expected_supporting_fact_ids = [] if not uses_unit_content_v3 or index == 0 else expected_fact_ids
            if (len(plan.source_fact_ids) != len(set(plan.source_fact_ids))
                    or len(plan.supporting_evidence_fact_ids) != len(set(plan.supporting_evidence_fact_ids))
                    or len(plan.source_scope_ids) != len(set(plan.source_scope_ids))
                    or not set(plan.source_fact_ids).issubset(fact_set)
                    or not set(plan.supporting_evidence_fact_ids).issubset(fact_set)
                    or set(plan.source_fact_ids) & set(plan.supporting_evidence_fact_ids)
                    or not set(plan.source_scope_ids).issubset(scopes)
                    or plan.source_fact_ids != expected_owned_fact_ids
                    or plan.supporting_evidence_fact_ids != expected_supporting_fact_ids
                    or plan.learning_objective_refs != self.unit_learning_objective_refs):
                raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        if uses_alignment_v4:
            if owned_fact_ids != fact_set:
                raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        elif (set(self.component_plan[0].source_fact_ids) != fact_set
              or self.component_plan[0].supporting_evidence_fact_ids):
            raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        # Keep nested ``None`` values because they were already part of the
        # canonical V2 wire hash.  Only omit the newly introduced, optional
        # top-level policy marker for legacy contracts.
        base = self.model_dump(exclude={"contract_hash"})
        if self.unit_content_policy_version is None:
            base.pop("unit_content_policy_version", None)
        if self.idm_unit_brief is None:
            base.pop("idm_unit_brief", None)
        if canonical_hash(base) != self.contract_hash:
            raise ValueError("ORCHESTRATION_V2_UNIT_CONTRACT_HASH_INVALID")
        return self


def unit_contract_manifest_v2(
    contract: UnitGenerationContractV2,
    *,
    locale: Literal["vi", "en"] = "vi",
) -> dict[str, Any]:
    facts = [{"fact_id": fact.fact_key, "text": fact.fact_text, "document_id": fact.document_id,
              "source_ref": fact.source_ref, "source_page": fact.source_page,
              "source_chunk": fact.source_chunk, "locator": fact.locator} for fact in contract.source_facts]
    supporting_fact_ids = {
        fact_id
        for plan in contract.component_plan
        for fact_id in plan.supporting_evidence_fact_ids
    }
    supporting_facts = [fact for fact in facts if fact["fact_id"] in supporting_fact_ids]
    try:
        evidence_bundle = build_source_evidence_bundle(
            source_snapshot_hash=contract.source_snapshot_hash,
            source_facts=contract.source_facts,
            locale=locale,
        )
    except (TypeError, ValueError):
        # Structured enrichment is additive. If malformed optional locator
        # metadata slips through an older snapshot, preserve the generated
        # draft and force review instead of failing the entire writer request.
        evidence_bundle = build_degraded_source_evidence_bundle(
            source_snapshot_hash=contract.source_snapshot_hash,
            source_facts=contract.source_facts,
            locale=locale,
        )
    return {"facts": facts, "supporting_evidence_facts": supporting_facts, "total_fact_count": len(facts),
            "represented_fact_count": len(facts), "fact_scope_complete": True, "truncated": False,
            "source_evidence_bundle": evidence_bundle.model_dump(mode="json")}


def unit_contract_v5_architecture_v2(contract: UnitGenerationContractV2) -> dict[str, Any]:
    source_refs = list(dict.fromkeys(fact.source_ref for fact in contract.source_facts if fact.source_ref))
    profile = profile_instructional_texts(
        [fact.fact_text for fact in contract.source_facts],
        learning_objective_count=len(contract.unit_learning_objective_refs),
        independent_topic_count=len(contract.unit_source_scope_ids),
    )
    output_budget = instructional_output_budget(profile)
    uses_density_v3 = all(
        fact.locator.get("instructional_density_policy_version") == INSTRUCTIONAL_DENSITY_POLICY_VERSION
        for fact in contract.source_facts
    )
    plans = [{"component_plan_id": plan.component_plan_id, "learning_objective_refs": plan.learning_objective_refs,
              "type": plan.type, "title": plan.title, "rationale": plan.rationale, "purpose": plan.purpose,
              "source_fact_ids": plan.source_fact_ids,
              "supporting_evidence_fact_ids": plan.supporting_evidence_fact_ids,
              "content_requirements": plan.content_requirements, "reason_code": "orchestration_v2_approved",
              "learning_block_ids": plan.learning_block_ids, "required_artifacts": plan.required_artifacts}
             for plan in contract.component_plan]
    unit_supporting_evidence_fact_ids = list(dict.fromkeys(
        fact_id
        for plan in contract.component_plan
        for fact_id in plan.supporting_evidence_fact_ids
    ))
    return {"component_capabilities": {"version": 2, "max_components_per_unit": 4,
                                        "max_assessments_per_unit": 3,
                                        "assessment_enabled": any(plan.type == "problem" for plan in contract.component_plan)},
            "architecture_contract_version": 5, "chapter_title": contract.chapter_title,
            "source_refs": source_refs, "lessons": [{"title": contract.lesson_title,
                "learning_objectives": contract.lesson_learning_objectives,
                "primary_concept_ids": [], "supporting_concept_ids": [],
                "assessment_required": any(plan.type == "problem" for plan in contract.component_plan),
                "assessment_objective_refs": contract.unit_learning_objective_refs if any(
                    plan.type == "problem" for plan in contract.component_plan) else [],
                "source_refs": source_refs, "units": [{"title": contract.unit_title,
                    "purpose": contract.unit_purpose, "concept_ids": [], "primary_concept_ids": [],
                    "primary_evidence_scope_ids": contract.unit_source_scope_ids,
                    "supporting_evidence_scope_ids": [],
                    "learning_objective_refs": contract.unit_learning_objective_refs,
                    "source_refs": source_refs, "source_fact_ids": contract.unit_source_fact_ids,
                    "supporting_evidence_fact_ids": unit_supporting_evidence_fact_ids,
                    "learning_blocks": [], "component_plan": plans,
                    "instructional_density_policy_version": (
                        INSTRUCTIONAL_DENSITY_POLICY_VERSION if uses_density_v3 else None
                    ),
                    "source_content_chars": profile.source_chars,
                    "source_estimated_words": profile.estimated_words,
                    "max_generated_visible_chars": output_budget.max_visible_chars,
                    "max_generated_words": output_budget.max_words}]}]}


def build_source_snapshot_payload_v2(
    source_map: dict[str, Any], source_manifest: dict[str, Any] | None,
) -> SourceSnapshotPayloadV2:
    manifest_facts = (source_manifest or {}).get("facts")
    scopes = source_map.get("source_evidence_scopes")
    if not isinstance(manifest_facts, list) or not manifest_facts or not isinstance(scopes, list) or not scopes:
        raise OrchestrationContractError("SOURCE_SNAPSHOT_INCOMPLETE")
    by_id = {str(fact.get("fact_id") or "").strip(): fact for fact in manifest_facts if isinstance(fact, dict)}
    if "" in by_id or len(by_id) != len(manifest_facts):
        raise OrchestrationContractError("SOURCE_SNAPSHOT_FACT_IDENTITY_INVALID")
    owner: dict[str, str] = {}
    catalog: list[SourceScopeCatalogEntryV2] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            raise OrchestrationContractError("SOURCE_SNAPSHOT_SCOPE_INVALID")
        scope_key = str(scope.get("id") or "").strip()
        fact_ids = [str(value).strip() for value in scope.get("source_fact_ids", [])]
        if not scope_key or not fact_ids or len(fact_ids) != len(set(fact_ids)):
            raise OrchestrationContractError("SOURCE_SNAPSHOT_SCOPE_INVALID")
        content_chars = 0
        for fact_id in fact_ids:
            if fact_id not in by_id or fact_id in owner:
                raise OrchestrationContractError("SOURCE_SNAPSHOT_FACT_OWNERSHIP_INVALID")
            owner[fact_id] = scope_key
            content_chars += len(str(by_id[fact_id].get("text") or ""))
        catalog.append(SourceScopeCatalogEntryV2(
            scope_key=scope_key,
            title=str(scope.get("title") or scope.get("source_ref") or scope_key),
            source_ref=str(scope.get("source_ref") or "").strip() or None,
            fact_count=len(fact_ids), content_chars=content_chars,
        ))
    if set(owner) != set(by_id):
        raise OrchestrationContractError("SOURCE_SNAPSHOT_FACT_OWNERSHIP_INVALID")
    facts = [SourceSnapshotFactV2(
        document_id=str(fact.get("document_id") or ""), fact_key=fact_id,
        scope_key=owner[fact_id], fact_text=str(fact.get("text") or ""),
        source_ref=str(fact.get("source_ref") or "").strip() or None,
        source_page=fact.get("source_page"), source_chunk=fact.get("source_chunk"),
        locator={"parser_version": source_manifest.get("parser_version")} if source_manifest else {},
    ) for fact_id, fact in by_id.items()]
    return SourceSnapshotPayloadV2(facts=facts, scopes=catalog)


def bind_course_skeleton_v2(
    draft: CourseSkeletonDraftV2, *, source_snapshot_hash: str, locale: str,
    scope_catalog: list[SourceScopeCatalogEntryV2], source_authority: SourceOutlineAuthorityV2 | None = None,
) -> CourseSkeletonV2:
    source_authority = source_authority or SourceOutlineAuthorityV2(
        mode="model_designed", source="none", complete=True, confidence=0,
        structure_hash=canonical_hash({"mode": "model_designed", "chapters": []}),
        reason_codes=[], chapters=[],
    )
    known = {scope.scope_key for scope in scope_catalog}
    assigned = [scope for chapter in draft.chapters for scope in chapter.source_scope_ids]
    if source_authority.mode != "locked" and (
            len(assigned) != len(set(assigned)) or set(assigned) != known):
        raise OrchestrationContractError("ARCHITECTURE_SCOPE_ALLOCATION_INCOMPLETE")
    # Chapter identity and ordering are server-owned. Structured generation is
    # allowed to propose pedagogical content and scope allocation, but a valid
    # response must not depend on the model remembering zero-based sequencing
    # or inventing globally stable identifiers.
    payload = draft.model_dump(exclude={"chapters"})
    if source_authority.mode == "needs_review":
        raise OrchestrationContractError("SOURCE_STRUCTURE_REVIEW_REQUIRED")
    draft_chapters = draft.chapters
    if source_authority.mode == "locked":
        if len(draft_chapters) != len(source_authority.chapters):
            raise OrchestrationContractError("ARCHITECTURE_SOURCE_CHAPTER_COUNT_MISMATCH")
        scopes_by_ref: dict[str, list[str]] = {}
        unbound: list[str] = []
        for scope in scope_catalog:
            if scope.source_ref:
                scopes_by_ref.setdefault(scope.source_ref, []).append(scope.scope_key)
            else:
                unbound.append(scope.scope_key)
        expected_allocations: list[list[str]] = []
        for index, binding in enumerate(source_authority.chapters):
            owned = list(scopes_by_ref.get(binding.source_ref, []))
            if index == 0:
                owned = [*unbound, *owned]
            if not owned:
                raise OrchestrationContractError("ARCHITECTURE_SOURCE_CHAPTER_SCOPE_MISSING")
            expected_allocations.append(owned)
        if set(scope for group in expected_allocations for scope in group) != known:
            raise OrchestrationContractError("ARCHITECTURE_SOURCE_CHAPTER_SCOPE_MISMATCH")
    chapters = [CourseSkeletonChapterV2(
        chapter_key=f"chapter-{index + 1}",
        order=index,
        title=(source_authority.chapters[index].title
               if source_authority.mode == "locked" else chapter.title),
        objective=chapter.objective,
        learning_outcomes=chapter.learning_outcomes,
        source_scope_ids=(expected_allocations[index]
                          if source_authority.mode == "locked" else chapter.source_scope_ids),
    ) for index, chapter in enumerate(draft_chapters)]
    return CourseSkeletonV2(contract_version=2, source_snapshot_hash=source_snapshot_hash,
                            locale=locale, chapters=chapters, **payload)


def plan_chapter_shards_v2(
    skeleton: CourseSkeletonV2, scope_catalog: list[SourceScopeCatalogEntryV2],
    *, max_source_chars: int | None = None,
) -> list[ChapterShardPlanV2]:
    if max_source_chars is not None and (max_source_chars < 1 or max_source_chars > MAX_SHARD_SOURCE_CHARS):
        raise OrchestrationContractError("ARCHITECTURE_SHARD_BUDGET_INVALID")
    by_scope = {scope.scope_key: scope for scope in scope_catalog}
    plans: list[ChapterShardPlanV2] = []
    for chapter in sorted(skeleton.chapters, key=lambda value: value.order):
        chapter_limit = max_source_chars or (
            MAX_V3_ARCHITECTURE_SHARD_SOURCE_CHARS
            if all(scope_id.startswith("scope3_") for scope_id in chapter.source_scope_ids)
            else MAX_SHARD_SOURCE_CHARS
        )
        groups: list[list[SourceScopeCatalogEntryV2]] = []
        current: list[SourceScopeCatalogEntryV2] = []
        current_chars = 0
        for scope_id in chapter.source_scope_ids:
            scope = by_scope.get(scope_id)
            if scope is None or scope.content_chars > chapter_limit:
                raise OrchestrationContractError("ARCHITECTURE_SCOPE_EXCEEDS_SHARD_CAPACITY")
            if current and current_chars + scope.content_chars > chapter_limit:
                groups.append(current)
                current, current_chars = [], 0
            current.append(scope)
            current_chars += scope.content_chars
        if current:
            groups.append(current)
        for index, group in enumerate(groups):
            plans.append(ChapterShardPlanV2(
                chapter_key=chapter.chapter_key, order=chapter.order, shard_index=index,
                shard_count=len(groups), source_scope_ids=[scope.scope_key for scope in group],
                source_fact_count=sum(scope.fact_count for scope in group),
                source_content_chars=sum(scope.content_chars for scope in group),
            ))
    return plans


def _component_review_v2(component_type: str, locale: str) -> ArchitectureComponentAuthorReviewV2:
    vi = locale == "vi"
    purpose = {
        "problem": "Kiểm tra khả năng phân biệt và vận dụng thông tin đã học." if vi else
                   "Check the learner's ability to distinguish and apply the taught information.",
        "la_sortable": "Luyện tập tái lập đúng trình tự đã được nguồn mô tả." if vi else
                       "Practice reconstructing the source-defined sequence.",
        "la_crossword": "Củng cố các thuật ngữ có định nghĩa rõ trong tài liệu." if vi else
                        "Reinforce terms that have explicit source definitions.",
        "la_diagram": "Trực quan hóa quan hệ, cấu trúc hoặc luồng đã có trong nguồn." if vi else
                      "Visualize a relationship, structure, or flow present in the source.",
        "la_faq": "Làm rõ điều kiện, ngoại lệ và cảnh báo dễ gây hiểu sai." if vi else
                  "Clarify source conditions, exceptions, and cautions that may be misunderstood.",
    }[component_type]
    navigation = {
        "problem": "Chọn một đáp án rồi xem phản hồi." if vi else "Choose one answer and review feedback.",
        "la_sortable": "Kéo thả các bước về đúng thứ tự." if vi else "Drag the steps into the correct order.",
        "la_crossword": "Điền thuật ngữ dựa trên định nghĩa." if vi else "Enter terms from their definitions.",
        "la_diagram": "Theo dõi các node và đường nối để hiểu quan hệ." if vi else
                      "Follow nodes and edges to understand the relationship.",
        "la_faq": "Mở từng câu hỏi để xem phần giải thích." if vi else
                  "Open each question to review the clarification.",
    }[component_type]
    return ArchitectureComponentAuthorReviewV2(
        purpose=purpose,
        example_scenario=None,
        visual_asset=("Sơ đồ dùng node và edge tách biệt, dễ đọc." if vi else
                      "A readable diagram with separated nodes and edges.") if component_type == "la_diagram" else None,
        user_behavior_navigation=navigation,
    )


def _compiled_component_v2(
    component_type: str,
    *,
    locale: str,
    unit_title: str,
    scope_ids: list[str],
) -> ArchitectureComponentPlanV2:
    vi = locale == "vi"
    title = {
        "problem": "Kiểm tra nhanh" if vi else "Knowledge check",
        "la_sortable": "Sắp xếp đúng trình tự" if vi else "Order the sequence",
        "la_crossword": "Ôn tập thuật ngữ" if vi else "Terminology review",
        "la_diagram": "Sơ đồ nội dung" if vi else "Content diagram",
        "la_faq": "Câu hỏi thường gặp" if vi else "Frequently asked questions",
    }[component_type]
    rationale = {
        "problem": "Evidence nguồn cho phép tạo một câu hỏi trắc nghiệm một đáp án có thể kiểm chứng." if vi else
                   "Source evidence supports a verifiable one-answer multiple-choice question.",
        "la_sortable": "Nguồn chứa ít nhất ba bước có thứ tự rõ ràng và mục tiêu yêu cầu vận dụng." if vi else
                       "The source contains at least three ordered steps and the objective requires application.",
        "la_crossword": "Nguồn chứa ít nhất ba cặp thuật ngữ và định nghĩa rõ ràng." if vi else
                        "The source contains at least three explicit term-definition pairs.",
        "la_diagram": "Nguồn chứa quan hệ, bảng, phân cấp hoặc luồng cần trực quan hóa." if vi else
                      "The source contains a relationship, table, hierarchy, or flow worth visualizing.",
        "la_faq": "Nguồn chứa nhiều điều kiện, ngoại lệ hoặc cảnh báo cần làm rõ." if vi else
                  "The source contains multiple conditions, exceptions, or cautions that need clarification.",
    }[component_type]
    return ArchitectureComponentPlanV2(
        type=component_type,
        title=f"{title}: {unit_title}"[:180],
        rationale=rationale,
        author_review=_component_review_v2(component_type, locale),
        source_scope_ids=scope_ids,
    )


def _compile_unit_components_v2(
    unit: UnitArchitectureV2,
    lesson: LessonArchitectureV2,
    *,
    facts_by_scope: dict[str, list[SourceSnapshotFactV2]],
    skeleton: CourseSkeletonV2,
    chapter: CourseSkeletonChapterV2,
    plan: ChapterShardPlanV2,
    lesson_index: int,
    unit_index: int,
) -> tuple[UnitArchitectureV2, list[dict[str, Any]], dict[str, Any]]:
    unit_facts = [
        fact
        for scope_id in unit.source_scope_ids
        for fact in facts_by_scope.get(scope_id, [])
    ]
    v3_markers = [
        fact.locator.get("instructional_density_policy_version") == INSTRUCTIONAL_DENSITY_POLICY_VERSION
        for fact in unit_facts
    ]
    if any(v3_markers) and not all(v3_markers):
        raise OrchestrationContractError("ARCHITECTURE_DENSITY_CONTRACT_MIXED")
    if v3_markers and all(v3_markers):
        decision = evaluate_instructional_density(profile_instructional_texts(
            [fact.fact_text for fact in unit_facts],
            learning_objective_count=len(unit.learning_objective_refs),
            independent_topic_count=len(unit.source_scope_ids),
        ))
        if decision.split_required:
            raise OrchestrationContractError("ARCHITECTURE_UNIT_DENSITY_EXCEEDED")

    def scope_facts(scope_ids: list[str]) -> list[SourceSnapshotFactV2]:
        selected = set(scope_ids)
        return [fact for fact in unit_facts if fact.scope_key in selected]

    def source_profile(scope_ids: list[str]) -> dict[str, Any]:
        selected = scope_facts(scope_ids)
        texts = clean_source_facts([fact.fact_text for fact in selected], preserve_table_numeric=True)
        relationships = source_relationship_pairs(texts)
        return {
            "texts": texts,
            "sortable": len(ordered_source_steps(texts, locale=skeleton.locale)) >= 3,
            "crossword": len(source_term_definitions(texts)) >= 3,
            "faq": len(source_clarification_signals(texts)) >= 2,
            "diagram": bool(relationships),
            "problem": build_source_grounded_single_choice(unit.title, texts, locale=skeleton.locale) is not None,
        }

    def interaction_supported(component: ArchitectureComponentPlanV2) -> bool:
        if component.type == "html":
            return True
        profile = source_profile(list(component.source_scope_ids))
        return bool(profile[{
            "problem": "problem",
            "la_sortable": "sortable",
            "la_faq": "faq",
            "la_crossword": "crossword",
            "la_diagram": "diagram",
        }.get(component.type, "")]) if component.type != "html" else True

    admitted_plan: list[ArchitectureComponentPlanV2] = []
    obligations: list[dict[str, Any]] = []
    removed_types: list[str] = []
    for component_index, component in enumerate(unit.component_plan, start=1):
        if interaction_supported(component):
            admitted_plan.append(component)
            continue
        removed_types.append(component.type)
        if component.type == "problem" and component_index <= 3:
            component_scope_ids = list(component.source_scope_ids)
            relevant_fact_ids = [
                fact.fact_key for fact in unit_facts if fact.scope_key in set(component_scope_ids)
            ]
            slot_key = "ao2_" + canonical_hash({
                "source_snapshot_hash": skeleton.source_snapshot_hash,
                "chapter_key": chapter.chapter_key,
                "shard_index": plan.shard_index,
                "lesson_index": lesson_index,
                "unit_index": unit_index,
                "component_index": component_index,
            })[:32]
            obligations.append({
                "planned_slot_key": slot_key,
                "lesson_index": lesson_index,
                "unit_index": unit_index,
                "component_index": component_index,
                "learning_objective_refs": list(unit.learning_objective_refs),
                "required_assessment_kind": "single_choice",
                "relevant_scope_ids": component_scope_ids,
                "relevant_evidence_fact_ids": relevant_fact_ids,
                "unresolved_reason": "ASSESSMENT_SOURCE_CHECK_REQUIRED",
                "status": "open",
            })

    # Only CP8-ready evidence may cause the server to add a new treatment.
    # Legacy evidence can validate a provider proposal conservatively, but it
    # cannot silently claim table/relation fidelity it never persisted.
    evidence_ready = bool(unit_facts) and all(
        fact.locator.get("source_evidence_status") == "ready"
        and isinstance(fact.locator.get("source_evidence_revision"), str)
        and re.fullmatch(r"[0-9a-f]{64}", fact.locator["source_evidence_revision"]) is not None
        for fact in unit_facts
    )
    present_types = {component.type for component in admitted_plan}
    opportunities: dict[str, list[str]] = {}
    if evidence_ready:
        for scope_id in unit.source_scope_ids:
            profile = source_profile([scope_id])
            for component_type, key in (
                ("la_sortable", "sortable"),
                ("la_diagram", "diagram"),
                ("la_crossword", "crossword"),
                ("problem", "problem"),
                ("la_faq", "faq"),
            ):
                if profile[key]:
                    opportunities.setdefault(component_type, []).append(scope_id)

    objective_text = " ".join(
        lesson.learning_objectives[int(reference[3:]) - 1]
        for reference in unit.learning_objective_refs
        if reference.startswith("lo_") and int(reference[3:]) <= len(lesson.learning_objectives)
    ).casefold()
    ordering_objective = any(marker in objective_text for marker in (
        "áp dụng", "thực hiện", "sắp xếp", "trình tự", "quy trình", "vận hành",
        "apply", "perform", "execute", "order", "sequence", "implement",
    ))
    if not ordering_objective:
        opportunities.pop("la_sortable", None)

    # Preserve the provider's FAQ as the final component while compiling the
    # bounded evidence-backed treatments before it.
    faq = next((component for component in admitted_plan if component.type == "la_faq"), None)
    if faq is not None:
        admitted_plan = [component for component in admitted_plan if component.type != "la_faq"]
    added_types: list[str] = []
    omitted_types: list[str] = []
    non_faq_candidates = ["la_sortable", "la_diagram", "la_crossword"]
    # A derived MCQ is selected only when no richer practice/visual/vocabulary
    # treatment exists for the same unit. Provider-requested grounded problems
    # were already admitted above.
    if not any(component_type in opportunities for component_type in non_faq_candidates):
        non_faq_candidates.append("problem")
    for component_type in non_faq_candidates:
        scope_ids = opportunities.get(component_type, [])
        if component_type in present_types or not scope_ids:
            continue
        if len(admitted_plan) + (1 if faq else 0) >= 4:
            omitted_types.append(component_type)
            continue
        admitted_plan.append(_compiled_component_v2(
            component_type,
            locale=skeleton.locale,
            unit_title=unit.title,
            scope_ids=scope_ids,
        ))
        present_types.add(component_type)
        added_types.append(component_type)
    if faq is None and opportunities.get("la_faq"):
        if len(admitted_plan) < 4:
            faq = _compiled_component_v2(
                "la_faq",
                locale=skeleton.locale,
                unit_title=unit.title,
                scope_ids=opportunities["la_faq"],
            )
            added_types.append("la_faq")
        else:
            omitted_types.append("la_faq")
    if faq is not None:
        admitted_plan.append(faq)

    represented_scopes = {
        scope_id for component in admitted_plan for scope_id in component.source_scope_ids
    }
    if not admitted_plan or represented_scopes != set(unit.source_scope_ids):
        raise OrchestrationContractError("ARCHITECTURE_COMPONENT_EVIDENCE_INVALID")
    normalized = UnitArchitectureV2.model_validate({
        **unit.model_dump(mode="python"),
        "component_plan": [component.model_dump(mode="python") for component in admitted_plan],
    })
    return normalized, obligations, {
        "lesson_index": lesson_index,
        "unit_index": unit_index,
        "evidence_status": "ready" if evidence_ready else "legacy_review_required",
        "added_types": added_types,
        "removed_types": removed_types,
        "omitted_types": omitted_types,
        "obligation_omitted_component_indices": [
            component_index
            for component_index, component in enumerate(unit.component_plan, start=1)
            if component.type == "problem" and component_index > 3 and not interaction_supported(component)
        ],
        "opportunity_types": sorted(opportunities),
    }


def bind_chapter_shard_v2(
    draft: ChapterShardDraftV2, *, skeleton: CourseSkeletonV2, plan: ChapterShardPlanV2,
    source_facts: list[SourceSnapshotFactV2] | None = None,
    compiler_diagnostics: list[dict[str, Any]] | None = None,
) -> ChapterBlueprintShardV2:
    chapter = next((item for item in skeleton.chapters if item.chapter_key == plan.chapter_key), None)
    if chapter is None or chapter.order != plan.order:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_IDENTITY_MISMATCH")
    allocated_scopes = [scope for lesson in draft.lessons for unit in lesson.units for scope in unit.source_scope_ids]
    if len(allocated_scopes) != len(set(allocated_scopes)) or set(allocated_scopes) != set(plan.source_scope_ids):
        raise OrchestrationContractError("ARCHITECTURE_SHARD_UNIT_ALLOCATION_INCOMPLETE")
    if sum(len(lesson.units) for lesson in draft.lessons) > 4096:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_UNIT_LIMIT_EXCEEDED")
    if source_facts is not None:
        fact_scopes = {fact.scope_key for fact in source_facts}
        if fact_scopes != set(plan.source_scope_ids):
            raise OrchestrationContractError("ARCHITECTURE_SHARD_CONTEXT_MISMATCH")
        v3_markers = [
            fact.locator.get("instructional_density_policy_version") == INSTRUCTIONAL_DENSITY_POLICY_VERSION
            for fact in source_facts
        ]
        # A mixed snapshot is invalid. V3 snapshots additionally receive the
        # density gate, while every snapshot receives the component evidence
        # gate so a legacy/unmarked run cannot revive fabricated assessments.
        if any(v3_markers) and not all(v3_markers):
            raise OrchestrationContractError("ARCHITECTURE_DENSITY_CONTRACT_MIXED")
        facts_by_scope: dict[str, list[SourceSnapshotFactV2]] = {}
        for fact in source_facts:
            facts_by_scope.setdefault(fact.scope_key, []).append(fact)
        normalized_lessons: list[LessonArchitectureV2] = []
        assessment_obligations: list[dict[str, Any]] = []
        for lesson_index, lesson in enumerate(draft.lessons, start=1):
            normalized_units: list[UnitArchitectureV2] = []
            for unit_index, unit in enumerate(lesson.units, start=1):
                normalized, obligations, diagnostics = _compile_unit_components_v2(
                    unit,
                    lesson,
                    facts_by_scope=facts_by_scope,
                    skeleton=skeleton,
                    chapter=chapter,
                    plan=plan,
                    lesson_index=lesson_index,
                    unit_index=unit_index,
                )
                normalized_units.append(normalized)
                assessment_obligations.extend(obligations)
                if compiler_diagnostics is not None:
                    compiler_diagnostics.append(diagnostics)
            normalized_lessons.append(LessonArchitectureV2.model_validate({
                **lesson.model_dump(mode="python"),
                "units": [unit.model_dump(mode="python") for unit in normalized_units],
            }))
        draft = ChapterShardDraftV2.model_validate({
            "lessons": [lesson.model_dump(mode="python") for lesson in normalized_lessons],
        })
    else:
        assessment_obligations = []
    return ChapterBlueprintShardV2(
        contract_version=2, source_snapshot_hash=skeleton.source_snapshot_hash,
        chapter_key=chapter.chapter_key, order=chapter.order, shard_index=plan.shard_index,
        shard_count=plan.shard_count, source_scope_ids=plan.source_scope_ids,
        title=chapter.title, objective=chapter.objective, lessons=draft.lessons,
        assessment_obligations=assessment_obligations,
    )


def skeleton_prompt_v2(
    locale: str, scopes: list[SourceScopeCatalogEntryV2], source_authority: SourceOutlineAuthorityV2 | None = None,
) -> str:
    source_authority = source_authority or SourceOutlineAuthorityV2(
        mode="model_designed", source="none", complete=True, confidence=0,
        structure_hash=canonical_hash({"mode": "model_designed", "chapters": []}),
        reason_codes=[], chapters=[],
    )
    scope_wire = json.dumps([scope.model_dump() for scope in scopes], ensure_ascii=False, separators=(",", ":"))
    return (
        "Design only the global course skeleton. Allocate every source scope exactly once. "
        "Each chapter must include concrete learner-facing learning_outcomes. "
        "When SOURCE_OUTLINE_AUTHORITY.mode is locked, emit exactly the listed chapters in exact order, "
        "with exact titles and deterministic scope ownership by matching source_ref; never merge, split, rename, or reorder. "
        "Do not emit source fact IDs or lesson content. Output valid JSON matching the response schema. "
        f"{UNTRUSTED_JSON_CONTEXT_RULE} "
        f"Output language: {locale}. SOURCE_OUTLINE_AUTHORITY="
        f"{source_authority.model_dump_json()}. SOURCE_SCOPE_CATALOG={scope_wire}"
    )


def chapter_shard_prompt_v2(
    locale: str, skeleton: CourseSkeletonV2, plan: ChapterShardPlanV2,
    facts: list[SourceSnapshotFactV2],
) -> str:
    if sum(len(fact.fact_text) for fact in facts) > MAX_SHARD_SOURCE_CHARS:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_CONTEXT_TOO_LARGE")
    if {fact.scope_key for fact in facts} != set(plan.source_scope_ids):
        raise OrchestrationContractError("ARCHITECTURE_SHARD_CONTEXT_MISMATCH")
    chapter = next(item for item in skeleton.chapters if item.chapter_key == plan.chapter_key)
    payload = {"chapter": chapter.model_dump(), "shard": plan.model_dump(),
               "facts": [fact.model_dump() for fact in facts]}
    density_budget = {
        "policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
        "max_canonical_facts": DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_canonical_facts,
        "max_source_chars": DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_source_chars,
        "max_estimated_words": DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_estimated_words,
        "max_learning_objectives": DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_learning_objectives,
        "max_source_scopes": DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_independent_topics,
    }
    return (
        "Design strict lesson, unit, and component-plan architecture only for this immutable chapter shard. "
        "Maintain an explicit pedagogical chain: every lesson objective must operationalize one or more chapter learning_outcomes; "
        "the lesson learning_objectives must be observable, specific steps toward that lesson objective; every unit purpose must state "
        "what the learner will achieve and must align with its local learning_objective_refs; every component rationale and author_review "
        "must explain how that block supports the containing unit rather than repeat generic chapter copy. "
        "Allocate every supplied source_scope_id to exactly one unit. Every unit must reference valid local learning objectives "
        "Keep every unit within UNIT_DENSITY_BUDGET; split oversized or multi-topic material into consecutive coherent units "
        "instead of producing one long HTML block. "
        "(lo_1, lo_2, ...). Select the smallest useful set of up to four components. HTML is optional; when selected it MUST be "
        "the first component. FAQ is optional; when selected it MUST be the final component. Order the remaining treatments as "
        "relationship/sequence/terminology practice, then problem assessment, then FAQ. Together the selected components "
        "must cover every unit scope, and each component scope must remain inside its unit. "
        "Select problem only when one correct answer and grounded distractors can be verified from an explicit source-authored "
        "question, term-definition set, keyed table, or ordered procedure; it is rendered exclusively as one-answer multiple choice. "
        "Select la_sortable only when the source explicitly contains at least three ordered procedure steps. Select la_crossword "
        "only when at least three term-definition pairs are explicit. Select la_diagram only for an explicit relationship, table, "
        "hierarchy, or flow. Select la_faq only when at least two explicit source conditions, exceptions, cautions, or misconception boundaries support useful anticipated questions. "
        "Never select an interaction merely to increase component count. "
        "For every component return author_review with optional purpose, example_scenario, visual_asset, and "
        "user_behavior_navigation. These explain the proposed block to the author and are not editable component data. "
        "For every unit, explicitly return media_brief as either null or one concrete video/static_infographic brief with "
        "a title, bullet-ready content_points, context_description, and rationale; it is a production brief, not generated media. "
        "Do not change chapter identity/title/objective and do not emit raw source text outside learner-facing plans. Never place "
        "source filenames, citations, page/slide/chunk locators, internal IDs, or source-attribution phrases in learner-facing content. "
        "Output valid JSON matching the response schema. "
        f"{UNTRUSTED_JSON_CONTEXT_RULE} "
        f"Output language: {locale}. UNIT_DENSITY_BUDGET="
        f"{json.dumps(density_budget, ensure_ascii=False, separators=(',', ':'))}. "
        f"SHARD_CONTEXT={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
