"""Request models of the legacy lesson-author routes (proposal, chapter checkpoint, blueprint)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.component_capabilities import ComponentCapabilities
from app.lesson_author_checkpoint import ChapterCheckpointUnit
from app.schemas.chat import RagChatRequest

# A chapter checkpoint runs on V5 course architectures only.
CHECKPOINT_ARCHITECTURE_CONTRACT_VERSION = 5
# Upper bound of units in one checkpointed chapter (index bound and inventory bound).
MAX_CHECKPOINT_UNITS = 512


class RagLessonAuthorBlueprintComponentPlan(BaseModel):
    component_plan_id: str | None = None
    learning_objective_refs: list[str] = Field(default_factory=list)
    type: str
    title: str
    rationale: str
    purpose: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    # Resolved V5 reinforcement evidence is read-only grounding, not
    # canonical ownership and must never be copied into source_fact_ids.
    supporting_evidence_fact_ids: list[str] = Field(default_factory=list)
    content_requirements: list[str] = Field(default_factory=list)
    reason_code: str | None = None
    learning_block_ids: list[str] = Field(default_factory=list)
    required_artifacts: list[dict[str, Any]] = Field(default_factory=list)


class RagLessonAuthorBlueprintUnit(BaseModel):
    title: str
    purpose: str = ""
    concept_ids: list[str] = Field(default_factory=list)
    primary_concept_ids: list[str] = Field(default_factory=list)
    # V5 evidence-scope semantics are part of the approved Blueprint draft
    # contract. They are provenance context only: canonical source_fact_ids
    # remain the downstream source-coverage authority.
    primary_evidence_scope_ids: list[str] = Field(default_factory=list)
    supporting_evidence_scope_ids: list[str] = Field(default_factory=list)
    learning_objective_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    supporting_evidence_fact_ids: list[str] = Field(default_factory=list)
    learning_blocks: list[dict[str, Any]] = Field(default_factory=list)
    component_plan: list[RagLessonAuthorBlueprintComponentPlan] = Field(default_factory=list)
    instructional_density_policy_version: str | None = None
    source_content_chars: int | None = Field(default=None, ge=0)
    source_estimated_words: int | None = Field(default=None, ge=0)
    max_generated_visible_chars: int | None = Field(default=None, ge=1, le=18_000)
    max_generated_words: int | None = Field(default=None, ge=1, le=2_400)


class RagLessonAuthorBlueprintLesson(BaseModel):
    title: str
    learning_objectives: list[str] = Field(default_factory=list)
    primary_concept_ids: list[str] = Field(default_factory=list)
    supporting_concept_ids: list[str] = Field(default_factory=list)
    assessment_required: bool = False
    assessment_objective_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    units: list[RagLessonAuthorBlueprintUnit] = Field(default_factory=list)


class RagLessonAuthorDraftArchitecture(BaseModel):
    component_capabilities: ComponentCapabilities | None = None
    architecture_contract_version: int | None = None
    chapter_title: str
    source_refs: list[str] = Field(default_factory=list)
    lessons: list[RagLessonAuthorBlueprintLesson] = Field(default_factory=list)


class RagLessonAuthorRequest(RagChatRequest):
    outline_context: str = ""
    target_scope_instruction: str = ""
    output_schema_hint: str
    operation: Literal[
        "answer",
        "course_blueprint",
        "create",
        "rename",
        "update_content",
        "delete",
        "move",
        "clarify",
    ] = "answer"
    target_type: Literal["course", "chapter", "lesson", "unit", "component"] | None = None
    generation_mode: Literal["auto", "staged", "single"] = "auto"
    max_attempts: int = Field(default=2, ge=1, le=2)
    blueprint_architecture: RagLessonAuthorDraftArchitecture | None = None


class RagLessonAuthorCheckpointRequest(RagLessonAuthorRequest):
    """Internal-token-only path. Regular chat cannot opt into it via extra fields."""
    model_config = ConfigDict(extra="forbid")
    checkpoint_version: Literal[1] = 1
    checkpoint_action: Literal["generate_unit", "validate_chapter"]
    checkpoint_unit_index: int | None = Field(default=None, ge=0, lt=MAX_CHECKPOINT_UNITS, strict=True)
    checkpoint_units: list[ChapterCheckpointUnit] = Field(default_factory=list, max_length=MAX_CHECKPOINT_UNITS)
    remaining_workflow_budget_ms: int = Field(ge=1, le=480_000, strict=True)

    @model_validator(mode="after")
    def validate_checkpoint_contract(self) -> RagLessonAuthorCheckpointRequest:
        if (self.target != "lesson_author" or self.operation != "create" or self.target_type != "chapter"
                or self.generation_mode != "staged" or not self.correlation_id or not self.source_documents
                or self.blueprint_architecture is None
                or self.blueprint_architecture.architecture_contract_version
                != CHECKPOINT_ARCHITECTURE_CONTRACT_VERSION):
            raise ValueError("CHAPTER_CHECKPOINT_CONTRACT_INVALID")
        total = sum(len(lesson.units) for lesson in self.blueprint_architecture.lessons)
        if not 1 <= total <= MAX_CHECKPOINT_UNITS:
            raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
        if self.checkpoint_action == "generate_unit":
            if self.checkpoint_unit_index is None or self.checkpoint_unit_index >= total or self.checkpoint_units:
                raise ValueError("CHAPTER_CHECKPOINT_UNIT_OUT_OF_SCOPE")
        elif (self.checkpoint_unit_index is not None
              or sorted(unit.unit_index for unit in self.checkpoint_units) != list(range(total))):
            raise ValueError("CHAPTER_CHECKPOINT_INCOMPLETE")
        return self


class RagLessonAuthorBlueprintRequest(RagChatRequest):
    component_capabilities: ComponentCapabilities | None = None
    # Gemini 3.5 Flash supports up to 65,536 generated tokens. Keep the wider
    # contract scoped to Blueprints; regular RAG chat remains bounded by its
    # parent request model.
    max_output_tokens: int = Field(default=65_536, ge=1, le=65_536)
    outline_context: str = ""
    blueprint_schema_hint: str
    max_attempts: int = Field(default=2, ge=1, le=2)
    # Node-resolved CMS ID, retained only for cross-service observability.
    # It is not used for authorisation or any database mutation in Python.
    course_id: str | None = Field(default=None, max_length=255)
