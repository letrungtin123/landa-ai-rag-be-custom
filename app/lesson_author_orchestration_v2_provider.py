from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


MAX_SOURCE_PAGE_FACTS = 500
MAX_SOURCE_SCOPES = 4096
MAX_SHARD_SOURCE_CHARS = 400_000
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
    scope_groups = [plan.source_scope_ids[index:index + 8]
                    for index in range(0, len(plan.source_scope_ids), 8)]
    for lesson_index, scopes in enumerate(scope_groups, start=1):
        labels = [_fallback_scope_title(scope_key, facts) for scope_key in scopes]
        learning_objectives = [f"Hiểu và áp dụng nội dung: {label}"[:500] for label in labels]
        units = [UnitArchitectureV2(
            title=label[:180],
            purpose=f"Giúp người học đạt mục tiêu của chương: {chapter.objective}"[:500],
            learning_objective_refs=[f"lo_{index}"],
            source_scope_ids=[scope_key],
            component_plan=[ArchitectureComponentPlanV2(
                type="html",
                title=f"Nội dung trọng tâm: {label}"[:180],
                rationale="Trình bày đầy đủ nội dung nguồn trước khi bổ sung hoạt động tương tác.",
                author_review=ArchitectureComponentAuthorReviewV2(
                    purpose="Giải thích và hệ thống hóa nội dung nguồn cho người học.",
                    example_scenario=None,
                    visual_asset=None,
                    user_behavior_navigation=None,
                ),
                source_scope_ids=[scope_key],
            )],
            media_brief=None,
        ) for index, (scope_key, label) in enumerate(zip(scopes, labels), start=1)]
        lesson_title = chapter.title if len(scope_groups) == 1 else f"{chapter.title} - Phần {lesson_index}"
        lessons.append(LessonArchitectureV2(
            title=lesson_title[:180],
            objective=chapter.objective,
            learning_objectives=learning_objectives,
            learning_activities=["Đọc nội dung, đối chiếu tình huống và thực hành áp dụng."],
            assessment="Đánh giá mức độ hiểu và khả năng áp dụng nội dung của phần học.",
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
    source_fact_ids: list[str] = Field(min_length=1, max_length=MAX_UNIT_SOURCE_FACTS)
    supporting_evidence_fact_ids: list[str] = Field(max_length=MAX_UNIT_SOURCE_FACTS)
    learning_objective_refs: list[str] = Field(min_length=1, max_length=24)
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)
    content_requirements: list[str] = Field(max_length=8)
    learning_block_ids: list[str] = Field(max_length=12)
    required_artifacts: list[dict[str, Any]] = Field(max_length=6)


class UnitGenerationContractV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal[2]
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
        if len(plan_ids) != len(set(plan_ids)) or self.component_plan[0].type != "html":
            raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        fact_set = set(fact_ids)
        for plan in self.component_plan:
            expected_fact_ids = [fact.fact_key for fact in self.source_facts
                                 if fact.scope_key in set(plan.source_scope_ids)]
            if (len(plan.source_fact_ids) != len(set(plan.source_fact_ids))
                    or len(plan.source_scope_ids) != len(set(plan.source_scope_ids))
                    or not set(plan.source_fact_ids).issubset(fact_set)
                    or plan.supporting_evidence_fact_ids
                    or not set(plan.source_scope_ids).issubset(scopes)
                    or plan.source_fact_ids != expected_fact_ids
                    or plan.learning_objective_refs != self.unit_learning_objective_refs):
                raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        if set(self.component_plan[0].source_fact_ids) != fact_set:
            raise ValueError("ORCHESTRATION_V2_UNIT_PLAN_INVALID")
        base = self.model_dump(exclude={"contract_hash"})
        if canonical_hash(base) != self.contract_hash:
            raise ValueError("ORCHESTRATION_V2_UNIT_CONTRACT_HASH_INVALID")
        return self


def unit_contract_manifest_v2(contract: UnitGenerationContractV2) -> dict[str, Any]:
    facts = [{"fact_id": fact.fact_key, "text": fact.fact_text, "document_id": fact.document_id,
              "source_ref": fact.source_ref, "source_page": fact.source_page,
              "source_chunk": fact.source_chunk} for fact in contract.source_facts]
    return {"facts": facts, "supporting_evidence_facts": [], "total_fact_count": len(facts),
            "represented_fact_count": len(facts), "fact_scope_complete": True, "truncated": False}


def unit_contract_v5_architecture_v2(contract: UnitGenerationContractV2) -> dict[str, Any]:
    source_refs = list(dict.fromkeys(fact.source_ref for fact in contract.source_facts if fact.source_ref))
    plans = [{"component_plan_id": plan.component_plan_id, "learning_objective_refs": plan.learning_objective_refs,
              "type": plan.type, "title": plan.title, "rationale": plan.rationale, "purpose": plan.purpose,
              "source_fact_ids": plan.source_fact_ids,
              "supporting_evidence_fact_ids": plan.supporting_evidence_fact_ids,
              "content_requirements": plan.content_requirements, "reason_code": "orchestration_v2_approved",
              "learning_block_ids": plan.learning_block_ids, "required_artifacts": plan.required_artifacts}
             for plan in contract.component_plan]
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
                    "supporting_evidence_fact_ids": [], "learning_blocks": [], "component_plan": plans}]}]}


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
    *, max_source_chars: int = MAX_SHARD_SOURCE_CHARS,
) -> list[ChapterShardPlanV2]:
    if max_source_chars < 1 or max_source_chars > MAX_SHARD_SOURCE_CHARS:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_BUDGET_INVALID")
    by_scope = {scope.scope_key: scope for scope in scope_catalog}
    plans: list[ChapterShardPlanV2] = []
    for chapter in sorted(skeleton.chapters, key=lambda value: value.order):
        groups: list[list[SourceScopeCatalogEntryV2]] = []
        current: list[SourceScopeCatalogEntryV2] = []
        current_chars = 0
        for scope_id in chapter.source_scope_ids:
            scope = by_scope.get(scope_id)
            if scope is None or scope.content_chars > max_source_chars:
                raise OrchestrationContractError("ARCHITECTURE_SCOPE_EXCEEDS_SHARD_CAPACITY")
            if current and current_chars + scope.content_chars > max_source_chars:
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


def bind_chapter_shard_v2(
    draft: ChapterShardDraftV2, *, skeleton: CourseSkeletonV2, plan: ChapterShardPlanV2,
) -> ChapterBlueprintShardV2:
    chapter = next((item for item in skeleton.chapters if item.chapter_key == plan.chapter_key), None)
    if chapter is None or chapter.order != plan.order:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_IDENTITY_MISMATCH")
    allocated_scopes = [scope for lesson in draft.lessons for unit in lesson.units for scope in unit.source_scope_ids]
    if len(allocated_scopes) != len(set(allocated_scopes)) or set(allocated_scopes) != set(plan.source_scope_ids):
        raise OrchestrationContractError("ARCHITECTURE_SHARD_UNIT_ALLOCATION_INCOMPLETE")
    if sum(len(lesson.units) for lesson in draft.lessons) > 4096:
        raise OrchestrationContractError("ARCHITECTURE_SHARD_UNIT_LIMIT_EXCEEDED")
    return ChapterBlueprintShardV2(
        contract_version=2, source_snapshot_hash=skeleton.source_snapshot_hash,
        chapter_key=chapter.chapter_key, order=chapter.order, shard_index=plan.shard_index,
        shard_count=plan.shard_count, source_scope_ids=plan.source_scope_ids,
        title=chapter.title, objective=chapter.objective, lessons=draft.lessons,
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
    return (
        "Design strict lesson, unit, and component-plan architecture only for this immutable chapter shard. "
        "Allocate every supplied source_scope_id to exactly one unit. Every unit must reference valid local learning objectives "
        "(lo_1, lo_2, ...), start component_plan with one html component covering all unit scopes, use at most one additional "
        "interactive component, and place la_faq last when present. Component scopes must remain inside their unit. "
        "For every component return author_review with optional purpose, example_scenario, visual_asset, and "
        "user_behavior_navigation. These explain the proposed block to the author and are not editable component data. "
        "For every unit, explicitly return media_brief as either null or one concrete video/static_infographic brief with "
        "a title, bullet-ready content_points, context_description, and rationale; it is a production brief, not generated media. "
        "Do not change chapter identity/title/objective and do not emit raw source text outside learner-facing plans. "
        "Output valid JSON matching the response schema. "
        f"Output language: {locale}. SHARD_CONTEXT={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
