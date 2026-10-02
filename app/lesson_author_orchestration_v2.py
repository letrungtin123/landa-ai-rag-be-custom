from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ORCHESTRATION_CONTRACT_VERSION = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,159}$")
OBJECTIVE_REF_PATTERN = re.compile(r"^lo_([1-9][0-9]*)$")
COMPONENT_TYPES = {"html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"}


class OrchestrationContractError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CourseSkeletonChapterV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chapter_key: str
    order: int = Field(ge=0, le=511)
    title: str = Field(min_length=1, max_length=500)
    objective: str = Field(min_length=1, max_length=2000)
    learning_outcomes: list[str] = Field(min_length=1, max_length=24)
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)

    @field_validator("chapter_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not KEY_PATTERN.fullmatch(value):
            raise ValueError("invalid chapter key")
        return value

    @field_validator("source_scope_ids")
    @classmethod
    def validate_unique_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 255 for item in value) or len(set(value)) != len(value):
            raise ValueError("source identifiers must be non-empty and unique")
        return value


class CourseSkeletonV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal[2]
    source_snapshot_hash: str
    locale: Literal["vi", "en"]
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=5000)
    target_audience: str = Field(min_length=1, max_length=2000)
    prerequisites: list[str] = Field(max_length=1000)
    learning_outcomes: list[str] = Field(min_length=1, max_length=4096)
    assessment_strategy: str = Field(min_length=1, max_length=5000)
    assumptions: list[str] = Field(max_length=1000)
    chapters: list[CourseSkeletonChapterV2] = Field(min_length=1, max_length=512)

    @field_validator("source_snapshot_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("invalid source snapshot hash")
        return value

    @model_validator(mode="after")
    def validate_identity_and_ownership(self) -> "CourseSkeletonV2":
        keys = [chapter.chapter_key for chapter in self.chapters]
        orders = [chapter.order for chapter in self.chapters]
        if len(set(keys)) != len(keys) or sorted(orders) != list(range(len(self.chapters))):
            raise ValueError("chapter identity/order invalid")
        scopes = [scope for chapter in self.chapters for scope in chapter.source_scope_ids]
        if len(scopes) != len(set(scopes)):
            raise ValueError("canonical scopes must have exactly one owning chapter")
        return self


class ArchitectureComponentPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]
    title: str = Field(min_length=1, max_length=180)
    rationale: str = Field(min_length=1, max_length=500)
    author_review: "ArchitectureComponentAuthorReviewV2"
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)

    @field_validator("source_scope_ids")
    @classmethod
    def validate_scope_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 255 for item in value) or len(set(value)) != len(value):
            raise ValueError("component scope identifiers must be non-empty and unique")
        return value


class ArchitectureComponentAuthorReviewV2(BaseModel):
    """Read-only author context. It is not editable component payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    purpose: str | None = Field(default=None, max_length=1200)
    example_scenario: str | None = Field(default=None, max_length=2000)
    visual_asset: str | None = Field(default=None, max_length=1200)
    user_behavior_navigation: str | None = Field(default=None, max_length=1200)

    @field_validator("purpose", "example_scenario", "visual_asset", "user_behavior_navigation")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ArchitectureMediaBriefV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["video", "static_infographic"]
    title: str = Field(min_length=1, max_length=180)
    content_points: list[str] = Field(min_length=1, max_length=6)
    context_description: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("content_points")
    @classmethod
    def validate_content_points(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 600 for item in value):
            raise ValueError("media brief content points are invalid")
        return value


class UnitArchitectureV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1, max_length=180)
    purpose: str = Field(min_length=1, max_length=500)
    learning_objective_refs: list[str] = Field(min_length=1, max_length=24)
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)
    component_plan: list[ArchitectureComponentPlanV2] = Field(min_length=1, max_length=3)
    media_brief: ArchitectureMediaBriefV2 | None

    @model_validator(mode="after")
    def validate_unit_contract(self) -> "UnitArchitectureV2":
        if len(set(self.source_scope_ids)) != len(self.source_scope_ids) or any(
                not item or len(item) > 255 for item in self.source_scope_ids):
            raise ValueError("unit scope identifiers must be non-empty and unique")
        if len(set(self.learning_objective_refs)) != len(self.learning_objective_refs) or any(
                not OBJECTIVE_REF_PATTERN.fullmatch(item) for item in self.learning_objective_refs):
            raise ValueError("unit learning objective references are invalid")
        component_types = [component.type for component in self.component_plan]
        if component_types[0] != "html" or len(set(component_types)) != len(component_types):
            raise ValueError("component plan requires one leading html and unique component types")
        if "la_faq" in component_types and component_types[-1] != "la_faq":
            raise ValueError("FAQ must be the final component")
        allowed_scopes = set(self.source_scope_ids)
        if any(not set(component.source_scope_ids).issubset(allowed_scopes) for component in self.component_plan):
            raise ValueError("component scope exceeds its unit")
        if set(self.component_plan[0].source_scope_ids) != allowed_scopes:
            raise ValueError("the leading html component must teach every unit scope")
        return self


class LessonArchitectureV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(min_length=1, max_length=180)
    objective: str = Field(min_length=1, max_length=500)
    learning_objectives: list[str] = Field(min_length=1, max_length=8)
    learning_activities: list[str] = Field(min_length=1, max_length=3)
    assessment: str = Field(min_length=1, max_length=500)
    units: list[UnitArchitectureV2] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_objective_bindings(self) -> "LessonArchitectureV2":
        if any(not value.strip() or len(value) > 500 for value in self.learning_objectives):
            raise ValueError("lesson learning objectives are invalid")
        if any(not value.strip() or len(value) > 280 for value in self.learning_activities):
            raise ValueError("lesson activities are invalid")
        maximum = len(self.learning_objectives)
        for unit in self.units:
            for reference in unit.learning_objective_refs:
                match = OBJECTIVE_REF_PATTERN.fullmatch(reference)
                if match is None or int(match.group(1)) > maximum:
                    raise ValueError("unit references an unknown lesson objective")
        return self


class ChapterBlueprintShardV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal[2]
    source_snapshot_hash: str
    chapter_key: str
    order: int = Field(ge=0, le=511)
    shard_index: int = Field(ge=0, le=4095)
    shard_count: int = Field(ge=1, le=4096)
    source_scope_ids: list[str] = Field(min_length=1, max_length=4096)
    title: str = Field(min_length=1, max_length=500)
    objective: str = Field(min_length=1, max_length=2000)
    lessons: list[LessonArchitectureV2] = Field(min_length=1, max_length=512)

    @field_validator("source_snapshot_hash")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("invalid source snapshot hash")
        return value

    @field_validator("chapter_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not KEY_PATTERN.fullmatch(value):
            raise ValueError("invalid chapter key")
        return value

    @field_validator("source_scope_ids")
    @classmethod
    def validate_coverage_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 255 for item in value) or len(set(value)) != len(value):
            raise ValueError("scope identifiers must be non-empty and unique")
        return value

    @model_validator(mode="after")
    def validate_shard_identity(self) -> "ChapterBlueprintShardV2":
        if self.shard_index >= self.shard_count:
            raise ValueError("shard index is outside shard count")
        owned_scopes = [scope for lesson in self.lessons for unit in lesson.units for scope in unit.source_scope_ids]
        if len(owned_scopes) != len(set(owned_scopes)) or set(owned_scopes) != set(self.source_scope_ids):
            raise ValueError("shard unit scope ownership is incomplete")
        return self


class AssembledBlueprintV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal[2]
    source_snapshot_hash: str
    skeleton_hash: str
    shard_hashes: list[str]
    admitted_fact_count: int = Field(ge=1)
    covered_fact_count: int = Field(ge=1)
    blueprint: dict[str, Any]
    assembly_hash: str


def assemble_blueprint_v2(
    skeleton: CourseSkeletonV2,
    shards: list[ChapterBlueprintShardV2],
    source_scope_facts: dict[str, list[str]],
) -> AssembledBlueprintV2:
    """Deterministically assemble bounded chapter responses.

    The assembler does not ask a provider to join JSON and does not infer
    completeness from successful calls. Every canonical fact admitted by the
    skeleton must be claimed by exactly the matching chapter shard.
    """

    required_scopes = {scope for chapter in skeleton.chapters for scope in chapter.source_scope_ids}
    if set(source_scope_facts) != required_scopes or any(not facts or len(facts) != len(set(facts))
                                                        for facts in source_scope_facts.values()):
        raise OrchestrationContractError("ARCHITECTURE_SOURCE_SCOPE_FACTS_INVALID")
    admitted = [fact for scope in sorted(required_scopes) for fact in source_scope_facts[scope]]
    if len(admitted) != len(set(admitted)):
        raise OrchestrationContractError("ARCHITECTURE_GLOBAL_FACT_OWNERSHIP_INVALID")
    by_key: dict[str, list[ChapterBlueprintShardV2]] = {}
    for shard in shards:
        if shard.source_snapshot_hash != skeleton.source_snapshot_hash:
            raise OrchestrationContractError("ARCHITECTURE_SOURCE_SNAPSHOT_CHANGED")
        by_key.setdefault(shard.chapter_key, []).append(shard)

    chapters: list[dict[str, Any]] = []
    shard_hashes: list[str] = []
    all_covered_scopes: list[str] = []
    for expected in sorted(skeleton.chapters, key=lambda chapter: chapter.order):
        chapter_shards = sorted(by_key.get(expected.chapter_key, []), key=lambda shard: shard.shard_index)
        if not chapter_shards or any(shard.order != expected.order for shard in chapter_shards):
            raise OrchestrationContractError("ARCHITECTURE_SHARD_IDENTITY_MISMATCH")
        if (any(shard.shard_count != len(chapter_shards) for shard in chapter_shards)
                or [shard.shard_index for shard in chapter_shards] != list(range(len(chapter_shards)))):
            raise OrchestrationContractError("ARCHITECTURE_SHARD_SET_INCOMPLETE")
        covered_scopes = [scope for shard in chapter_shards for scope in shard.source_scope_ids]
        if len(covered_scopes) != len(set(covered_scopes)) or set(covered_scopes) != set(expected.source_scope_ids):
            raise OrchestrationContractError("ARCHITECTURE_SHARD_COVERAGE_INCOMPLETE")
        if any(shard.title.strip() != expected.title.strip() or shard.objective.strip() != expected.objective.strip()
               for shard in chapter_shards):
            raise OrchestrationContractError("ARCHITECTURE_SHARD_TITLE_CHANGED")
        chapters.append({"chapter_key": expected.chapter_key, "title": expected.title, "objective": expected.objective,
                         "lessons": [lesson.model_dump() for shard in chapter_shards for lesson in shard.lessons]})
        all_covered_scopes.extend(covered_scopes)
        shard_hashes.extend(canonical_hash(shard.model_dump()) for shard in chapter_shards)

    if (set(by_key) != {chapter.chapter_key for chapter in skeleton.chapters}
            or len(all_covered_scopes) != len(set(all_covered_scopes)) or set(all_covered_scopes) != required_scopes):
        raise OrchestrationContractError("ARCHITECTURE_GLOBAL_COVERAGE_INCOMPLETE")

    blueprint = {
        "orchestration_contract_version": 2,
        "content_contract_version": 1,
        "title": skeleton.title,
        "summary": skeleton.summary,
        "target_audience": skeleton.target_audience,
        "prerequisites": skeleton.prerequisites,
        "learning_outcomes": skeleton.learning_outcomes,
        "assessment_strategy": skeleton.assessment_strategy,
        "assumptions": skeleton.assumptions,
        "chapters": chapters,
    }
    skeleton_hash = canonical_hash(skeleton.model_dump())
    base = {
        "contract_version": ORCHESTRATION_CONTRACT_VERSION,
        "source_snapshot_hash": skeleton.source_snapshot_hash,
        "skeleton_hash": skeleton_hash,
        "shard_hashes": shard_hashes,
        "admitted_fact_count": len(admitted),
        "covered_fact_count": len(admitted),
        "blueprint": blueprint,
    }
    return AssembledBlueprintV2(**base, assembly_hash=canonical_hash(base))
