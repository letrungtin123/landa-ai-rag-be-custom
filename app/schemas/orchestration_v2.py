"""Request and response models of the orchestration-v2 routes: source snapshot, skeleton, chapter shard, unit."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic.json_schema import SkipJsonSchema

from app.contracts.orchestration_v2.blueprint import ChapterBlueprintShardV2, CourseSkeletonV2
from app.contracts.orchestration_v2.provider import (
    ChapterShardPlanV2,
    SourceOutlineAuthorityV2,
    SourceScopeCatalogEntryV2,
    SourceSnapshotFactV2,
    UnitGenerationContractV2,
)
from app.idm.contracts import IdmCourseDesignV1, IdmCourseSkeletonRequestV1, IdmModuleContextV1, IdmShardDesignV1
from app.schemas.chat import RagRequestBase


class RagLessonAuthorSourceSnapshotV2Request(RagRequestBase):
    model_config = ConfigDict(extra="forbid")
    # Source paging is deterministic database work. It deliberately carries no
    # tenant provider secret because this endpoint cannot call Gemini.
    # The default stays the plain (unvalidated) string, exactly as before typing was enforced.
    api_key: SecretStr = Field(default="source-snapshot-no-provider", repr=False)  # type: ignore[assignment]
    contract_version: Literal[2] = 2
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cursor: dict[str, Any] | None = None
    expected_source_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page_max_facts: int = Field(default=500, ge=1, le=500, strict=True)
    page_max_bytes: int = Field(default=4_194_304, ge=1, le=4_194_304, strict=True)

    @model_validator(mode="after")
    def validate_source_snapshot_request(self) -> RagLessonAuthorSourceSnapshotV2Request:
        if self.target != "lesson_author" or not self.kb_id or not self.correlation_id or not self.source_documents:
            raise ValueError("ORCHESTRATION_V2_SOURCE_REQUEST_INVALID")
        if self.cursor is not None:
            if set(self.cursor) != {"document_id", "chunk_no"}:
                raise ValueError("ORCHESTRATION_V2_SOURCE_CURSOR_INVALID")
            try:
                UUID(str(self.cursor["document_id"]))
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError("ORCHESTRATION_V2_SOURCE_CURSOR_INVALID") from error
            chunk_no = self.cursor["chunk_no"]
            if isinstance(chunk_no, bool) or not isinstance(chunk_no, int) or chunk_no < 0:
                raise ValueError("ORCHESTRATION_V2_SOURCE_CURSOR_INVALID")
        return self


class RagLessonAuthorCourseSkeletonV2Request(RagRequestBase):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal[2] = 2
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_catalog: list[SourceScopeCatalogEntryV2] = Field(min_length=1, max_length=4096)
    source_authority: SourceOutlineAuthorityV2
    max_attempts: int = Field(default=2, ge=1, le=2)
    # Present only for runs admitted to the IDM pipeline (spec §11.2).
    idm: IdmCourseSkeletonRequestV1 | None = None

    @model_validator(mode="after")
    def validate_course_skeleton_request(self) -> RagLessonAuthorCourseSkeletonV2Request:
        if self.target != "lesson_author" or not self.correlation_id:
            raise ValueError("ORCHESTRATION_V2_SKELETON_REQUEST_INVALID")
        if self.idm is not None and self.idm.project_context.locale != self.locale:
            raise ValueError("ORCHESTRATION_V2_SKELETON_REQUEST_INVALID")
        scope_ids = [scope.scope_key for scope in self.scope_catalog]
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("ORCHESTRATION_V2_SCOPE_CATALOG_INVALID")
        return self


class RagLessonAuthorChapterShardV2Request(RagRequestBase):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal[2] = 2
    skeleton: CourseSkeletonV2
    shard_plan: ChapterShardPlanV2
    source_facts: list[SourceSnapshotFactV2] = Field(min_length=1, max_length=100_000)
    max_attempts: int = Field(default=2, ge=1, le=2)
    # Present only for IDM runs; ``source_facts`` then carry block-scope keys (spec §11.2).
    idm_module_context: IdmModuleContextV1 | None = None

    @model_validator(mode="after")
    def validate_chapter_shard_request(self) -> RagLessonAuthorChapterShardV2Request:
        if self.target != "lesson_author" or not self.correlation_id or self.locale != self.skeleton.locale:
            raise ValueError("ORCHESTRATION_V2_CHAPTER_REQUEST_INVALID")
        if self.idm_module_context is not None and (
                self.idm_module_context.project_context.locale != self.locale
                or {fact.scope_key for fact in self.source_facts}
                - {scope.scope_key for scope in self.idm_module_context.block_scopes}):
            raise ValueError("ORCHESTRATION_V2_CHAPTER_REQUEST_INVALID")
        if self.shard_plan.chapter_key not in {chapter.chapter_key for chapter in self.skeleton.chapters}:
            raise ValueError("ORCHESTRATION_V2_CHAPTER_REQUEST_INVALID")
        return self


class RagLessonAuthorUnitV2Request(RagRequestBase):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal[2] = 2
    unit_contract: UnitGenerationContractV2
    max_attempts: int = Field(default=2, ge=1, le=2)
    remaining_workflow_budget_ms: int = Field(ge=1, le=480_000, strict=True)
    fallback_only: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_unit_request(self) -> RagLessonAuthorUnitV2Request:
        document_ids = {document.document_id for document in self.source_documents}
        if (self.target != "lesson_author" or not self.correlation_id or not self.source_documents
                or self.locale not in {"vi", "en"}
                or self.contract_version != self.unit_contract.contract_version
                or any(fact.document_id not in document_ids for fact in self.unit_contract.source_facts)):
            raise ValueError("ORCHESTRATION_V2_UNIT_REQUEST_INVALID")
        return self


class SourceSnapshotV2Response(BaseModel):
    """One deterministic source page (field order is the wire order)."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2]
    source_snapshot_hash: str
    source_revision: str
    source_authority: dict[str, Any]
    facts: list[dict[str, Any]]
    next_cursor: dict[str, Any] | None
    has_more: bool
    page_content_bytes: int
    usage: dict[str, Any]


# ---- generation responses (step 2.6) --------------------------------------------------------------------------------
# These models are the published contract of the dicts the services already return (field order is the wire order;
# an optional add-on field is omitted when absent, never sent as null, exactly as the services leave the key out).
# The routes still answer the service dict unvalidated; app.services.response_contracts checks it against these
# models without changing it and only counts and logs a mismatch.
UsageSource = Literal["provider", "reserved_upper_bound", "deterministic_fallback"]
ContentOrigin = Literal["provider_validated", "structured_fallback"]
QualityState = Literal["validated", "review_required"]


def _absent(value: object) -> bool:
    return value is None


class CourseSkeletonV2Response(BaseModel):
    """``/course-skeleton`` answer: the course skeleton and, on IDM runs, the IDM course design."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2]
    skeleton: CourseSkeletonV2
    idm: IdmCourseDesignV1 | SkipJsonSchema[None] = Field(default=None, exclude_if=_absent)
    usage: dict[str, int]
    usage_complete: bool
    usage_source: UsageSource
    content_origin: ContentOrigin
    quality_state: QualityState


class ChapterShardV2Payload(ChapterBlueprintShardV2):
    """One chapter shard; IDM runs append their module design as ``idm_design`` (spec §8.2)."""

    idm_design: IdmShardDesignV1 | SkipJsonSchema[None] = Field(default=None, exclude_if=_absent)


class ChapterShardV2Response(BaseModel):
    """``/chapter-shard`` answer."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2]
    shard: ChapterShardV2Payload
    usage: dict[str, int]
    usage_complete: bool
    usage_source: UsageSource
    content_origin: ContentOrigin
    quality_state: QualityState


class UnitV2Response(BaseModel):
    """``/unit`` answer.

    ``unit`` stays an object: the IDM writer, the legacy writer and the fallbacks each emit its keys in their own
    order, and Node validates the components itself. ``attempt_trace`` events come from several writers as well.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2]
    source_snapshot_hash: str
    unit_path: str
    unit: dict[str, Any]
    usage: dict[str, int]
    usage_complete: bool
    usage_source: UsageSource
    content_origin: ContentOrigin
    quality_state: QualityState
    attempt_trace: list[dict[str, Any]]
    # Present only when semantic review is on (``AI_RAG_SEMANTIC_REVIEW_MODE``).
    semantic_review: dict[str, Any] | SkipJsonSchema[None] = Field(default=None, exclude_if=_absent)
