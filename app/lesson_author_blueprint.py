from __future__ import annotations

import json
import re
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from app.source_structure import strip_source_range_suffix


# Keep the generation contract aligned with the backend. A source without a
# trustworthy table of contents can contain many independent procedures in one
# inferred topic; rejecting the fifth procedure loses a valid Blueprint rather
# than making it easier to review.
MAX_BLUEPRINT_CHAPTERS = 12
MAX_BLUEPRINT_LESSONS_PER_CHAPTER = 6
MAX_BLUEPRINT_LEARNING_OUTCOMES = 12
MAX_BLUEPRINT_PREREQUISITES = 10
MAX_BLUEPRINT_ASSUMPTIONS = 8
MAX_BLUEPRINT_ACTIVITIES_PER_LESSON = 3
MAX_BLUEPRINT_UNITS_PER_LESSON = 3
MAX_BLUEPRINT_COMPONENTS_PER_UNIT = 3
MAX_BLUEPRINT_TOTAL_LESSONS = 24
MAX_BLUEPRINT_TOTAL_UNITS = 24
MAX_BLUEPRINT_TOTAL_COMPONENTS = 72
MAX_BLUEPRINT_MEDIA_PLANS = 12
MAX_SOURCE_REFS_PER_SCOPE = 8
MAX_CANONICAL_IDENTIFIER_LENGTH = 96
MAX_LOCAL_LEARNING_OBJECTIVE_REF_LENGTH = 16
_LOCAL_LEARNING_OBJECTIVE_REF = re.compile(r"^lo_([1-9][0-9]*)$")
# This is a serialization safety limit for server-owned provenance, not an
# instructional-design target.  It is deliberately higher than the former
# 160-item normalizer so one valid, source-grounded scope is never silently
# truncated before Node can validate the server allocation.  Architecture
# quality is evaluated separately from this transport boundary.
MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE = 512
BLUEPRINT_COMPONENT_TYPES = {
    "html",
    "problem",
    "la_faq",
    "la_sortable",
    "la_crossword",
    "la_diagram",
}
BLUEPRINT_MEDIA_TYPES = {"video", "static_infographic"}
SEMANTIC_LEARNING_BLOCK_INTENTS = {
    "introduction", "concept_explanation", "definition", "example", "worked_example",
    "procedure", "comparison", "warning", "tip", "scenario", "reflection", "practice",
    "knowledge_check", "terminology_reinforcement", "faq", "relationship_visualization",
    "summary", "media_reference",
}
# A bounded action-objective repair may change an existing generic teaching
# block only to one of these evidence-compatible treatments.  The wider
# semantic enum remains valid for Architect output and other repair
# operations, but would let a scoped intent repair turn an assessment anchor
# into a non-teaching block.
ACTION_OBJECTIVE_REPAIR_INTENTS = frozenset({"procedure", "worked_example"})
# One operation contract for provider schema, prompt and server acceptance.
# The broader Architect vocabulary is NOT the permission set of a depth repair.
INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS = frozenset({
    "concept_explanation", "definition", "example", "worked_example",
    "procedure", "comparison", "warning", "tip", "practice",
    "relationship_visualization",
})
SEMANTIC_REPAIR_CONTRACT_VERSION = "v5-semantic-repair-2"
INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS = 480
SEMANTIC_LEARNING_BLOCK_IMPORTANCE = {"supporting", "core", "critical", "assessment"}


class LessonAuthorBlueprintComponentPlanResponse(BaseModel):
    type: str
    title: str
    rationale: str
    purpose: str | None = None
    source_fact_ids: list[str] = Field(default_factory=list)
    content_requirements: list[str] = Field(default_factory=list)
    required_artifacts: list[dict[str, Any]] = Field(default_factory=list)


class LessonAuthorBlueprintMediaPlanResponse(BaseModel):
    type: str
    title: str
    content_outline: str
    rationale: str


class LessonAuthorBlueprintUnitResponse(BaseModel):
    title: str
    purpose: str | None = None
    concept_ids: list[str] = Field(default_factory=list)
    primary_concept_ids: list[str] = Field(default_factory=list)
    primary_evidence_scope_ids: list[str] = Field(default_factory=list)
    supporting_evidence_scope_ids: list[str] = Field(default_factory=list)
    learning_objective_refs: list[str] = Field(default_factory=list)
    learning_blocks: list[dict[str, Any]] = Field(default_factory=list)
    # Retained solely to parse Blueprints created before Phase 3.
    component_plan: list[LessonAuthorBlueprintComponentPlanResponse] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    media_plan: LessonAuthorBlueprintMediaPlanResponse | None = None


class LessonAuthorBlueprintLessonResponse(BaseModel):
    title: str
    objective: str
    learning_activities: list[str]
    assessment: str
    units: list[LessonAuthorBlueprintUnitResponse]
    source_refs: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    primary_concept_ids: list[str] = Field(default_factory=list)
    supporting_concept_ids: list[str] = Field(default_factory=list)
    prerequisite_concept_ids: list[str] = Field(default_factory=list)
    estimated_minutes: int | None = None
    assessment_required: bool = False
    assessment_objective_refs: list[str] = Field(default_factory=list)


class LessonAuthorBlueprintChapterResponse(BaseModel):
    title: str
    objective: str
    lessons: list[LessonAuthorBlueprintLessonResponse]
    source_refs: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)


class LessonAuthorBlueprintResponse(BaseModel):
    architecture_contract_version: int | None = None
    content_contract_version: int | None = None
    title: str
    summary: str
    target_audience: str
    prerequisites: list[str]
    learning_outcomes: list[str]
    course_outcomes: list[str] = Field(default_factory=list)
    assessment_strategy: str
    assumptions: list[str]
    chapters: list[LessonAuthorBlueprintChapterResponse]


class LessonAuthorBlueprintValidationError(ValueError):
    """A structural Blueprint failure with content-safe diagnostic metadata."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "course",
        constraint: str = "BLUEPRINT_CONTRACT",
        expected_type: str = "valid schema value",
        actual_type: str = "invalid schema value",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.constraint = constraint
        self.expected_type = expected_type
        self.actual_type = actual_type

    def safe_diagnostic(self) -> dict[str, str]:
        """Only structural metadata; never values, source text, or model output."""

        return {
            "validator": "lesson_author_blueprint.validate_lesson_author_blueprint",
            "error_code": self.code,
            "path": self.path,
            "constraint": self.constraint,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
        }


def _safe_json_shape(value: Any) -> str:
    if isinstance(value, list):
        return f"array[length={len(value)}]"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _string_schema(description: str) -> types.Schema:
    return types.Schema(
        type=types.Type.STRING,
        description=description,
    )


def _string_array_schema(description: str) -> types.Schema:
    return types.Schema(
        type=types.Type.ARRAY,
        description=description,
        items=_string_schema("A concise text item."),
    )


def _enum_string_schema(description: str, values: set[str]) -> types.Schema:
    """Express a canonical finite string domain to the structured-output API."""

    return types.Schema(
        type=types.Type.STRING,
        description=description,
        enum=sorted(values),
    )


def _semantic_learning_block_response_schema() -> types.Schema:
    """One provider-facing representation of the server semantic-block contract.

    The server validator below remains the authority.  Keeping the provider
    schema derived from the same module-level domains prevents initial
    Architect and scoped-repair responses from drifting apart.
    """

    treatment_descriptor_schema = types.Schema(
        type=types.Type.OBJECT,
        description=(
            "Compact server-validated instructional-treatment descriptors only. "
            "Set a descriptor only when the selected evidence scope actually supports it; never include lesson prose or CMS payloads."
        ),
        properties={
            "anticipated_questions": types.Schema(type=types.Type.BOOLEAN, description="True only for genuine anticipated learner questions supported by the scope."),
            "question_count": types.Schema(type=types.Type.INTEGER, description="Count of distinct supported FAQ questions; use only with faq intent."),
            "relationship_evidence": types.Schema(type=types.Type.BOOLEAN, description="True only when the scope contains an explicit relationship, flow, hierarchy, system, or conceptual connection."),
            "relationship_count": types.Schema(type=types.Type.INTEGER, description="Count of explicit supported relationships; use only with relationship_visualization intent."),
            "requires_ordering_practice": types.Schema(type=types.Type.BOOLEAN, description="True only when the learner must practise reconstructing an evidence-backed order."),
            "ordered_sequence": types.Schema(type=types.Type.BOOLEAN, description="True only when the source scope contains an explicit ordered sequence."),
            "sequence_item_count": types.Schema(type=types.Type.INTEGER, description="Count of explicit sequence items; use only for ordering practice."),
            "definitions_supported": types.Schema(type=types.Type.BOOLEAN, description="True only when the scope provides clear source-grounded terminology definitions."),
            "terminology_count": types.Schema(type=types.Type.INTEGER, description="Count of distinct supported terms and definitions; use only for terminology reinforcement."),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        description="A source-grounded semantic learning block; never a CMS component or payload.",
        required=["id", "intent", "importance", "concept_ids", "primary_concept_ids", "primary_evidence_scope_ids", "supporting_evidence_scope_ids", "content"],
        properties={
            "id": _string_schema("Stable local learning-block identifier."),
            "intent": _enum_string_schema(
                "One supported pedagogical semantic learning-block intent.",
                SEMANTIC_LEARNING_BLOCK_INTENTS,
            ),
            "importance": _enum_string_schema(
                "One supported semantic learning-block importance value.",
                SEMANTIC_LEARNING_BLOCK_IMPORTANCE,
            ),
            "concept_ids": _string_array_schema("Exact Source Map concept IDs in this block's semantic/source scope."),
            "primary_concept_ids": _string_array_schema("Exact Source Map concepts emphasized by this block. Concepts may be associated with more than one lesson; evidence scopes, not concepts, have unique primary ownership."),
            "primary_evidence_scope_ids": _string_array_schema("Exact server-provided evidence scope IDs owned primarily by this block. Every scope has exactly one primary block owner globally."),
            "supporting_evidence_scope_ids": _string_array_schema("Exact server-provided evidence scope IDs referenced for grounded reinforcement/practice only. These never own canonical source facts."),
            "source_refs": _string_array_schema("Optional exact Source Map source references that define this block's semantic/source scope."),
            "learning_objective_refs": _string_array_schema("Optional exact local lesson objective IDs only: lo_1, lo_2, and so on. Never repeat objective prose."),
            "content": treatment_descriptor_schema,
        },
    )


def _lesson_author_media_plan_response_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        description="An optional proposed visual asset placed before this unit's learning components. It is a recommendation only, never media payload data.",
        required=["type", "title", "content_outline", "rationale"],
        properties={
            "type": _string_schema("One supported media type: video or static_infographic."),
            "title": _string_schema("Short learner-facing media title."),
            "content_outline": _string_schema("What the proposed video or static infographic should show, using source-supported facts only."),
            "rationale": _string_schema("Why the visual asset improves comprehension for this unit."),
        },
    )
def _lesson_author_unit_response_schema(learning_block_schema: types.Schema) -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        description="A coherent, draftable learning unit inside a lesson.",
        required=["title", "purpose", "concept_ids", "primary_concept_ids", "learning_objective_refs", "learning_blocks"],
        properties={
            "title": _string_schema("Semantic unit title without numbering."),
            "purpose": _string_schema("The unit's instructional purpose, not a component choice."),
            "concept_ids": _string_array_schema("One or more exact Source Map concept IDs owned or reinforced by this unit."),
            "primary_concept_ids": _string_array_schema("Exact Source Map concepts taught primarily by this unit. Keep empty for a reinforcement/practice unit."),
            "learning_objective_refs": _string_array_schema("Exact local lesson objective IDs only: lo_1, lo_2, and so on."),
            "source_refs": _string_array_schema(
                "Optional source outline references supporting this unit.",
            ),
            "learning_blocks": types.Schema(type=types.Type.ARRAY, items=learning_block_schema),
            "media_plan": _lesson_author_media_plan_response_schema(),
        },
    )


def _lesson_author_lesson_response_schema(unit_schema: types.Schema) -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        description="A concise lesson inside one course chapter.",
        required=["title", "objective", "learning_objectives", "learning_activities", "assessment", "primary_concept_ids", "supporting_concept_ids", "prerequisite_concept_ids", "assessment_required", "assessment_objective_refs", "units"],
        properties={
            "title": _string_schema("Lesson title."),
            "objective": _string_schema("Measurable lesson objective."),
            "learning_activities": _string_array_schema(
                "One to three learner activities that support the objective.",
            ),
            "assessment": _string_schema("How the lesson objective is checked."),
            "learning_objectives": _string_array_schema("Observable, source-aligned lesson learning objectives."),
            "primary_concept_ids": _string_array_schema("Exact Source Map concepts taught primarily by this lesson."),
            "supporting_concept_ids": _string_array_schema("Exact Source Map concepts reinforced without re-teaching them as primary."),
            "prerequisite_concept_ids": _string_array_schema("Exact Source Map concepts learners must encounter before this lesson."),
            "estimated_minutes": types.Schema(type=types.Type.INTEGER, description="Optional estimated learning time in minutes when known."),
            "assessment_required": types.Schema(type=types.Type.BOOLEAN, description="Whether this lesson needs a knowledge check or another assessment later."),
            "assessment_objective_refs": _string_array_schema("Exact local lesson objective IDs assessed by this lesson: lo_1, lo_2, and so on."),
            "units": types.Schema(
                type=types.Type.ARRAY,
                # The current Gemini Developer API compatibility contract used
                # by this service rejects array min/max fields in structured
                # output schemas. The provider receives the explicit 1..3
                # instruction; this server validator remains authoritative
                # and routes an over-limit lesson to a local patch repair.
                description="One to three ordered learning units, each with a reviewable component plan.",
                items=unit_schema,
            ),
            "source_refs": _string_array_schema(
                "Optional source outline references such as src-001. Use only references supplied by the server.",
            ),
        },
    )


def _lesson_author_chapter_response_schema(lesson_schema: types.Schema) -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        description="A coherent chapter in the course Blueprint.",
        required=["title", "objective", "learning_objectives", "concept_ids", "lessons"],
        properties={
            "title": _string_schema("Chapter title."),
            "objective": _string_schema("Measurable chapter objective."),
            "source_refs": _string_array_schema(
                "Optional source outline references supporting this chapter.",
            ),
            "learning_objectives": _string_array_schema("Observable chapter learning objectives."),
            "concept_ids": _string_array_schema("Exact Source Map concepts covered by this chapter."),
            "lessons": types.Schema(
                type=types.Type.ARRAY,
                description="Lessons ordered from foundation to application.",
                items=lesson_schema,
            ),
        },
    )


def build_lesson_author_blueprint_response_schema() -> types.Schema:
    # Phase 3 Course Architect outputs learning intent, not CMS component
    # choices.  The component registry/planner on the Node boundary remains
    # the only authority that maps these blocks to a CMS component.
    learning_block_schema = _semantic_learning_block_response_schema()
    unit_schema = _lesson_author_unit_response_schema(learning_block_schema)
    lesson_schema = _lesson_author_lesson_response_schema(unit_schema)
    chapter_schema = _lesson_author_chapter_response_schema(lesson_schema)
    return types.Schema(
        type=types.Type.OBJECT,
        description="A review-only enterprise course Blueprint based on supplied source material.",
        required=[
            "architecture_contract_version",
            "content_contract_version",
            "title",
            "summary",
            "target_audience",
            "prerequisites",
            "learning_outcomes",
            "course_outcomes",
            "assessment_strategy",
            "assumptions",
            "chapters",
        ],
        properties={
            "architecture_contract_version": types.Schema(type=types.Type.INTEGER, description="Must be 5 for the server-owned evidence-scope Course Architect contract."),
            "content_contract_version": types.Schema(type=types.Type.INTEGER, description="Must be 1."),
            "title": _string_schema("Course title."),
            "summary": _string_schema("Concise course design summary."),
            "target_audience": _string_schema("Primary intended learners."),
            "prerequisites": _string_array_schema(
                "Learner prerequisites. Use an empty array when none are needed.",
            ),
            "learning_outcomes": _string_array_schema(
                "Three to twelve measurable learning outcomes.",
            ),
            "course_outcomes": _string_array_schema("Course-level observable outcomes; normally mirrors or refines learning_outcomes."),
            "assessment_strategy": _string_schema("Course-level assessment strategy."),
            "assumptions": _string_array_schema(
                "Open assumptions requiring administrator confirmation.",
            ),
            "chapters": types.Schema(
                type=types.Type.ARRAY,
                description="One to twelve course chapters.",
                items=chapter_schema,
            ),
        },
    )


def build_course_architecture_repair_response_schema(
    allowed_fields: set[str] | None = None,
) -> types.Schema:
    """Structured provider contract for bounded Course Architect repairs.

    The target-specific field whitelist and the server-side pre-apply validator
    remain authoritative. This schema narrows the high-risk nested
    ``learning_blocks`` value so a repair cannot treat it as an arbitrary JSON
    object merely because the outer patch envelope is syntactically valid.
    """

    learning_block_schema = _semantic_learning_block_response_schema()
    unit_schema = _lesson_author_unit_response_schema(learning_block_schema)
    lesson_schema = _lesson_author_lesson_response_schema(unit_schema)
    chapter_schema = _lesson_author_chapter_response_schema(lesson_schema)
    all_properties: dict[str, types.Schema] = {
        "course_title": _string_schema("Course title only when explicitly authorized."),
        "course_outcomes": _string_array_schema("Course outcomes only when explicitly authorized."),
        "chapters": types.Schema(
            type=types.Type.ARRAY,
            description="Full chapter replacements only when explicitly authorized.",
            items=chapter_schema,
        ),
        "title": _string_schema("Target title only when explicitly authorized."),
        "purpose": _string_schema("Target unit purpose only when explicitly authorized."),
        "learning_objectives": _string_array_schema("Exact target-local learning objective prose only when explicitly authorized."),
        "assessment_objective_refs": _string_array_schema("Exact local objective IDs such as lo_1 only when explicitly authorized."),
        "assessment_required": types.Schema(type=types.Type.BOOLEAN, description="Assessment flag only when explicitly authorized."),
        "estimated_minutes": types.Schema(type=types.Type.INTEGER, description="Estimated minutes only when explicitly authorized."),
        "concept_ids": _string_array_schema("Exact Source Map concept IDs only when explicitly authorized."),
        "primary_concept_ids": _string_array_schema("Exact primary concept IDs only when explicitly authorized."),
        "supporting_concept_ids": _string_array_schema("Exact supporting concept IDs only when explicitly authorized."),
        "prerequisite_concept_ids": _string_array_schema("Exact prerequisite concept IDs only when explicitly authorized."),
        "source_refs": _string_array_schema("Exact server-provided source references only when explicitly authorized."),
        "learning_objective_refs": _string_array_schema("Exact local objective IDs such as lo_1 only when explicitly authorized."),
        "learning_blocks": types.Schema(
            type=types.Type.ARRAY,
            description="A complete replacement list of validated semantic learning blocks.",
            items=learning_block_schema,
        ),
        "units": types.Schema(
            type=types.Type.ARRAY,
            description="Full unit replacements only when explicitly authorized. Return one to three complete units.",
            items=unit_schema,
        ),
        "lessons": types.Schema(
            type=types.Type.ARRAY,
            description="Full lesson replacements only when explicitly authorized.",
            items=lesson_schema,
        ),
    }
    properties = (
        {field: schema for field, schema in all_properties.items() if field in allowed_fields}
        if allowed_fields is not None
        else all_properties
    )
    replacement_schema = types.Schema(
        type=types.Type.OBJECT,
        description="Only the target-specific fields explicitly allowed in REPAIR TARGETS.",
        properties=properties,
    )
    patch_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["path", "operation"],
        properties={
            "path": _string_schema("One exact server-approved repair target path."),
            "operation": types.Schema(
                type=types.Type.STRING,
                enum=["replace", "remove_unit"],
                description="One server-approved operation for the target.",
            ),
            "replacement": replacement_schema,
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["patches"],
        properties={
            "patches": types.Schema(
                type=types.Type.ARRAY,
                description="One patch for every server-approved bounded repair target.",
                items=patch_schema,
            ),
        },
    )


V5_SEMANTIC_DELTA_REPAIR_OPERATIONS = frozenset({
    "align_concepts_to_evidence",
    "set_block_intent",
    "repair_knowledge_check",
    "repair_assessment_alignment",
    "add_knowledge_check",
    "select_assessment_teaching_alignment",
    "add_instructional_support_block",
})


def _v5_semantic_delta_patch_schema(operation: str) -> types.Schema:
    """Return one narrow, provider-facing V5 coherence-repair operation.

    These schemas deliberately contain no source references, evidence scopes,
    concept IDs, canonical fact IDs, or full learning-block objects.  Those
    fields are immutable server-owned provenance and are preserved when a
    delta is applied to the baseline Blueprint.
    """

    common = {
        "path": _string_schema("One exact server-approved repair target path."),
        "operation": types.Schema(
            type=types.Type.STRING,
            enum=[operation],
            description="One exact server-approved semantic-delta operation.",
        ),
    }
    if operation == "align_concepts_to_evidence":
        return types.Schema(
            type=types.Type.OBJECT,
            required=["path", "operation", "concept_ids", "primary_concept_ids"],
            properties={
                **common,
                "concept_ids": _string_array_schema(
                    "The exact server-approved concept alignment for this evidence-backed unit."
                ),
                "primary_concept_ids": _string_array_schema(
                    "One exact server-approved primary-concept alignment for this unit."
                ),
            },
        )
    if operation == "set_block_intent":
        return types.Schema(
            type=types.Type.OBJECT,
            required=["path", "operation", "block_id", "intent"],
            properties={
                **common,
                "block_id": _string_schema("An existing semantic block ID in the target unit."),
                "intent": _enum_string_schema(
                    "One server-approved procedural or worked-example intent for an action-oriented unit.",
                    ACTION_OBJECTIVE_REPAIR_INTENTS,
                ),
            },
        )
    if operation == "repair_knowledge_check":
        return types.Schema(
            type=types.Type.OBJECT,
            required=["path", "operation", "block_id", "learning_objective_refs"],
            properties={
                **common,
                "block_id": _string_schema("An existing knowledge_check block ID in the target unit."),
                "learning_objective_refs": _string_array_schema(
                    "Exact existing local lesson objective IDs only."
                ),
            },
        )
    if operation == "repair_assessment_alignment":
        return types.Schema(
            type=types.Type.OBJECT,
            required=[
                "path", "operation", "knowledge_check_block_id", "teaching_selections",
            ],
            properties={
                **common,
                "knowledge_check_block_id": _string_schema(
                    "One exact server-approved existing knowledge_check block ID."
                ),
                "teaching_selections": types.Schema(
                    type=types.Type.ARRAY,
                    description=(
                        "One exact server-approved teaching selection for every listed local assessment objective. "
                        "The server owns all evidence, provenance and block paths."
                    ),
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        required=["teaching_block_id", "learning_objective_refs"],
                        properties={
                            "teaching_block_id": _string_schema(
                                "One exact server-approved existing evidence-owning teaching block ID."
                            ),
                            "learning_objective_refs": _string_array_schema(
                                "One or more exact existing local lesson objective IDs assigned to this teaching block."
                            ),
                        },
                    ),
                ),
            },
        )
    if operation == "add_knowledge_check":
        return types.Schema(
            type=types.Type.OBJECT,
            required=["path", "operation", "after_block_id", "learning_objective_refs"],
            properties={
                **common,
                "after_block_id": _string_schema(
                    "The exact existing eligible teaching block ID in the target unit."
                ),
                "learning_objective_refs": _string_array_schema(
                    "Exact existing local lesson objective IDs only."
                ),
            },
        )
    if operation == "select_assessment_teaching_alignment":
        return types.Schema(
            type=types.Type.OBJECT,
            required=["path", "operation", "selections"],
            properties={
                **common,
                "selections": types.Schema(
                    type=types.Type.ARRAY,
                    description=(
                        "One selection for every listed local assessment objective. "
                        "Return SELECT only when the listed teaching block semantically teaches the objective; "
                        "otherwise return NO_MATCH without a block path or ID."
                    ),
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        required=["objective_ref", "decision"],
                        properties={
                            "objective_ref": _string_schema("One exact listed local assessment objective ID."),
                            "decision": types.Schema(
                                type=types.Type.STRING,
                                enum=["SELECT", "NO_MATCH"],
                                description="SELECT an exact approved anchor or fail closed with NO_MATCH.",
                            ),
                            "unit_path": _string_schema("The exact server-approved candidate unit path when decision is SELECT."),
                            "teaching_block_id": _string_schema("The exact server-approved candidate teaching block ID when decision is SELECT."),
                        },
                    ),
                ),
            },
        )
    if operation == "add_instructional_support_block":
        return types.Schema(
            type=types.Type.OBJECT,
            required=[
                "path", "operation", "unit_path", "after_block_id", "intent",
                "learning_objective_refs", "content",
            ],
            properties={
                **common,
                "unit_path": _string_schema(
                    "One exact server-approved unit path inside the target lesson."
                ),
                "after_block_id": _string_schema(
                    "The exact existing eligible teaching block ID in that unit."
                ),
                "intent": _enum_string_schema(
                    "One canonical semantic intent for an evidence-grounded instructional support block.",
                    INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS,
                ),
                "learning_objective_refs": _string_array_schema(
                    "Exact existing local lesson objective IDs only."
                ),
                "content": types.Schema(
                    type=types.Type.OBJECT,
                    required=["purpose"],
                    properties={
                        "purpose": types.Schema(
                            type=types.Type.STRING, min_length=1, max_length=INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS,
                            description="Non-empty compact instructional purpose; never source text, HTML, CSS, URLs, or component payload data.",
                        ),
                        "learner_action": types.Schema(
                            type=types.Type.STRING, min_length=1, max_length=INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS,
                            description="Optional non-empty compact learner action when the selected intent requires one; omit if absent.",
                        ),
                    },
                ),
            },
        )
    raise ValueError(f"Unsupported V5 semantic-delta repair operation: {operation}")


def semantic_delta_required_fields(operation: str) -> frozenset[str]:
    """The server mutation guard consumes the same typed outer-field contract."""
    return frozenset(_v5_semantic_delta_patch_schema(operation).required or [])


def build_v5_semantic_delta_repair_response_schema(
    operations: set[str] | frozenset[str],
) -> types.Schema:
    """Build one operation-specific provider schema for one bounded call.

    The installed Google SDK cannot serialize ``any_of`` schemas that omit a
    top-level JSON type. Rather than weaken this contract into a broad patch
    object, mixed coherence targets are partitioned into bounded calls by the
    caller. Each provider call therefore has one exact operation shape.
    """

    requested = set(operations)
    unsupported = requested - set(V5_SEMANTIC_DELTA_REPAIR_OPERATIONS)
    if len(requested) != 1 or unsupported:
        raise ValueError("V5 semantic-delta repair requires exactly one supported operation per provider call")
    patch_schema = _v5_semantic_delta_patch_schema(next(iter(requested)))
    return types.Schema(
        type=types.Type.OBJECT,
        required=["patches"],
        properties={
            "patches": types.Schema(
                type=types.Type.ARRAY,
                description="Exactly one typed semantic delta for every listed repair target.",
                items=patch_schema,
            ),
        },
    )


LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA = build_lesson_author_blueprint_response_schema()
COURSE_ARCHITECTURE_REPAIR_RESPONSE_SCHEMA = build_course_architecture_repair_response_schema()
LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL = LessonAuthorBlueprintResponse


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must be a JSON object.",
            path=label,
            constraint="TYPE_OBJECT",
            expected_type="object",
            actual_type=_safe_json_shape(value),
        )
    return value


def _require_text(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must be text.",
            path=label,
            constraint="TYPE_STRING",
            expected_type="string",
            actual_type=_safe_json_shape(value),
        )
    text = value.strip()
    if not text or len(text) > max_length:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must contain concise text.",
            path=label,
            constraint="TEXT_LENGTH",
            expected_type=f"string[length=1..{max_length}]",
            actual_type=(f"string[length={len(value.strip())}]" if isinstance(value, str) else _safe_json_shape(value)),
        )
    return text


def _require_text_array(
    value: Any,
    label: str,
    *,
    min_items: int,
    max_items: int,
    item_max_length: int,
) -> list[str]:
    if not isinstance(value, list) or not min_items <= len(value) <= max_items:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must contain {min_items} to {max_items} items.",
            path=label,
            constraint="ARRAY_LENGTH",
            expected_type=f"array[length={min_items}..{max_items}]",
            actual_type=_safe_json_shape(value),
        )
    return [
        _require_text(item, f"{label}[{index}]", item_max_length)
        for index, item in enumerate(value)
    ]


def _optional_source_refs(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    return _require_text_array(
        value,
        label,
        min_items=0,
        max_items=MAX_SOURCE_REFS_PER_SCOPE,
        # Source references are canonical provenance identifiers.  They must
        # be preserved verbatim across the Python -> Node contract, never
        # display-truncated.
        item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH,
    )


def _validate_local_learning_objective_refs(
    refs: list[str],
    label: str,
    objective_count: int,
) -> None:
    """Require exact, lesson-local objective IDs for the v4 contract.

    Learning objective prose is deliberately not accepted as an alias.  The
    provider creates the ordered objective list; references are machine IDs
    that must resolve inside that one lesson.
    """

    for index, ref in enumerate(refs):
        match = _LOCAL_LEARNING_OBJECTIVE_REF.fullmatch(ref)
        if match is None or int(match.group(1)) > objective_count:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"{label}[{index}] must reference an objective in this lesson.",
                path=f"{label}[{index}]",
                constraint="LOCAL_LEARNING_OBJECTIVE_REFERENCE",
                expected_type=f"local lesson objective ID lo_1..lo_{objective_count}",
                actual_type="unresolved or non-local objective reference",
            )


def _copy_first_alias(record: dict[str, Any], canonical: str, *aliases: str) -> None:
    if canonical in record:
        return
    for alias in aliases:
        if alias in record:
            record[canonical] = record[alias]
            return


def _coerce_json_array(value: Any, *, empty_when_none: bool = False) -> Any:
    """Normalize harmless Gemini shape drift without manufacturing content."""
    if value is None and empty_when_none:
        return []
    if isinstance(value, str):
        return [value]
    return value


def normalize_lesson_author_blueprint_candidate(value: Any) -> Any:
    """Accept only lossless structural variants before strict validation.

    Gemini occasionally emits a scalar array, a numeric string, or a common
    synonym even when JSON mode is enabled. This adapter only changes shape;
    it never invents missing learning content or drops extra items.
    """
    if not isinstance(value, dict):
        return value

    raw = dict(value)
    _copy_first_alias(raw, "title", "course_title", "course_name")
    _copy_first_alias(raw, "summary", "course_summary", "description")
    _copy_first_alias(raw, "target_audience", "audience", "learners")
    _copy_first_alias(raw, "learning_outcomes", "outcomes")
    _copy_first_alias(raw, "assessment_strategy", "assessment")

    for field in ("prerequisites", "learning_outcomes", "assumptions"):
        if field in raw:
            raw[field] = _coerce_json_array(raw[field], empty_when_none=field in {"prerequisites", "assumptions"})

    chapters = raw.get("chapters")
    if isinstance(chapters, dict):
        chapters = [chapters]
        raw["chapters"] = chapters
    if not isinstance(chapters, list):
        return raw

    normalized_chapters: list[Any] = []
    for chapter_value in chapters:
        if not isinstance(chapter_value, dict):
            normalized_chapters.append(chapter_value)
            continue
        chapter = dict(chapter_value)
        _copy_first_alias(chapter, "title", "chapter_title", "name")
        _copy_first_alias(chapter, "objective", "goal")
        chapter.pop("duration_minutes", None)
        chapter.pop("duration", None)
        if "source_refs" in chapter:
            chapter["source_refs"] = _coerce_json_array(chapter["source_refs"], empty_when_none=True)

        lessons = chapter.get("lessons")
        if isinstance(lessons, dict):
            lessons = [lessons]
            chapter["lessons"] = lessons
        if isinstance(lessons, list):
            normalized_lessons: list[Any] = []
            for lesson_value in lessons:
                if not isinstance(lesson_value, dict):
                    normalized_lessons.append(lesson_value)
                    continue
                lesson = dict(lesson_value)
                _copy_first_alias(lesson, "title", "lesson_title", "name")
                _copy_first_alias(lesson, "objective", "goal")
                _copy_first_alias(lesson, "learning_activities", "activities")
                _copy_first_alias(lesson, "assessment", "evaluation", "assessment_method")
                _copy_first_alias(lesson, "units", "learning_units", "content_units")
                lesson.pop("duration_minutes", None)
                lesson.pop("duration", None)
                for field in ("learning_activities", "source_refs"):
                    if field in lesson:
                        lesson[field] = _coerce_json_array(lesson[field], empty_when_none=field == "source_refs")
                units = lesson.get("units")
                if isinstance(units, dict):
                    units = [units]
                    lesson["units"] = units
                if isinstance(units, list):
                    normalized_units: list[Any] = []
                    for unit_value in units:
                        if not isinstance(unit_value, dict):
                            normalized_units.append(unit_value)
                            continue
                        unit = dict(unit_value)
                        _copy_first_alias(unit, "title", "unit_title", "name")
                        _copy_first_alias(unit, "component_plan", "components", "planned_components")
                        _copy_first_alias(unit, "media_plan", "media_suggestion", "media")
                        if "source_refs" in unit:
                            unit["source_refs"] = _coerce_json_array(unit["source_refs"], empty_when_none=True)
                        if "source_fact_ids" in unit:
                            unit["source_fact_ids"] = _coerce_json_array(unit["source_fact_ids"], empty_when_none=True)
                        media_plan = unit.get("media_plan")
                        if isinstance(media_plan, list) and len(media_plan) == 1:
                            media_plan = media_plan[0]
                            unit["media_plan"] = media_plan
                        if isinstance(media_plan, dict):
                            normalized_media_plan = dict(media_plan)
                            _copy_first_alias(normalized_media_plan, "type", "media_type", "format")
                            _copy_first_alias(normalized_media_plan, "title", "label", "name")
                            _copy_first_alias(normalized_media_plan, "content_outline", "content", "description", "outline")
                            _copy_first_alias(normalized_media_plan, "rationale", "reason", "selection_rationale")
                            unit["media_plan"] = normalized_media_plan
                        component_plan = unit.get("component_plan")
                        if isinstance(component_plan, dict):
                            component_plan = [component_plan]
                            unit["component_plan"] = component_plan
                        if isinstance(component_plan, list):
                            normalized_plan: list[Any] = []
                            for plan_value in component_plan:
                                if not isinstance(plan_value, dict):
                                    normalized_plan.append(plan_value)
                                    continue
                                plan = dict(plan_value)
                                _copy_first_alias(plan, "type", "component_type", "block_type")
                                _copy_first_alias(plan, "title", "label", "name")
                                _copy_first_alias(plan, "rationale", "selection_rationale", "reason")
                                if "source_refs" in plan:
                                    plan["source_refs"] = _coerce_json_array(plan["source_refs"], empty_when_none=True)
                                normalized_plan.append(plan)
                            unit["component_plan"] = normalized_plan
                        normalized_units.append(unit)
                    lesson["units"] = normalized_units
                normalized_lessons.append(lesson)
            chapter["lessons"] = normalized_lessons
        normalized_chapters.append(chapter)
    raw["chapters"] = normalized_chapters
    return raw


def _find_balanced_json_objects(value: str) -> list[str]:
    objects: list[str] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(value):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}":
            if not stack:
                continue
            start = stack.pop()
            if not stack:
                objects.append(value[start : index + 1])
    return objects


def describe_lesson_author_blueprint_response(text: str) -> dict[str, Any]:
    candidate = text.strip().lstrip("\ufeff")
    balanced_objects = _find_balanced_json_objects(candidate)
    return {
        "response_chars": len(text),
        "first_non_whitespace": candidate[:1] or None,
        "last_non_whitespace": candidate[-1:] or None,
        "has_code_fence": candidate.startswith(chr(96) * 3),
        "first_object_index": candidate.find("{"),
        "last_object_index": candidate.rfind("}"),
        "balanced_object_count": len(balanced_objects),
        "balanced_object_lengths": [len(item) for item in balanced_objects[-3:]],
    }


def _escape_control_chars_in_json_strings(value: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in value:
        if in_string:
            if escaped:
                output.append(char)
                escaped = False
            elif char == "\\":
                output.append(char)
                escaped = True
            elif char == '"':
                output.append(char)
                in_string = False
            elif char == "\n":
                output.append("\\n")
            elif char == "\r":
                output.append("\\r")
            elif char == "\t":
                output.append("\\t")
            elif ord(char) < 32:
                output.append(f"\\u{ord(char):04x}")
            else:
                output.append(char)
            continue
        output.append(char)
        if char == '"':
            in_string = True
    return "".join(output)


def _remove_trailing_json_commas(value: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(value) and value[lookahead].isspace():
                lookahead += 1
            if lookahead < len(value) and value[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _decode_blueprint_json(text: str) -> list[Any]:
    candidate = text.strip().lstrip("\ufeff")
    if candidate.startswith("```"):
        first_line_end = candidate.find("\n")
        if first_line_end >= 0 and candidate.rstrip().endswith("```"):
            candidate = candidate[first_line_end + 1 : candidate.rstrip().rfind("```")].strip()
    elif not candidate.startswith("{"):
        object_start = candidate.find("{")
        if object_start >= 0:
            candidate = candidate[object_start:]
    if not candidate:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_JSON",
            "Blueprint response is empty.",
        )

    parsed_candidates: list[Any] = []
    try:
        parsed, end_index = json.JSONDecoder().raw_decode(candidate)
        if not candidate[end_index:].strip():
            parsed_candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    # Bounded recovery for malformed provider formatting. We intentionally do
    # not repair missing fields, quote arbitrary text, or close a truncated
    # object because those cases could change the meaning of the blueprint.
    for balanced in reversed(_find_balanced_json_objects(candidate)):
        repaired = _remove_trailing_json_commas(_escape_control_chars_in_json_strings(balanced))
        try:
            parsed = json.loads(repaired)
            if parsed not in parsed_candidates:
                parsed_candidates.append(parsed)
        except json.JSONDecodeError:
            continue

    if parsed_candidates:
        return parsed_candidates

    raise LessonAuthorBlueprintValidationError(
        "BLUEPRINT_INVALID_JSON",
        "Blueprint response is not valid JSON.",
    )


def _validate_semantic_learning_blocks(
    value: Any,
    label: str,
    *,
    require_source_fact_ownership: bool,
    require_primary_source_fact_ownership: bool = False,
    include_source_fact_ids: bool = True,
    require_primary_semantic_ownership: bool = False,
    require_evidence_scope_ownership: bool = False,
    include_evidence_scope_ids: bool = False,
    source_fact_ownership_by_evidence_scope: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 12:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must contain 1 to 12 semantic learning blocks.",
            path=label,
            constraint="ARRAY_LENGTH",
            expected_type="array[length=1..12]",
            actual_type=_safe_json_shape(value),
        )
    blocks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw_block in enumerate(value):
        block = _require_object(raw_block, f"{label}[{index}]")
        block_id = _require_text(block.get("id"), f"{label}[{index}].id", 96)
        if block_id in ids:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"{label} must not repeat learning-block IDs.",
                path=f"{label}[{index}].id",
                constraint="UNIQUE_LEARNING_BLOCK_ID",
                expected_type="unique string identifier",
                actual_type="duplicate string identifier",
            )
        ids.add(block_id)
        intent = _require_text(block.get("intent"), f"{label}[{index}].intent", 64).casefold()
        if intent not in SEMANTIC_LEARNING_BLOCK_INTENTS:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"{label}[{index}].intent is not supported.",
                path=f"{label}[{index}].intent",
                constraint="ENUM_SEMANTIC_LEARNING_BLOCK_INTENT",
                expected_type="supported semantic learning-block intent",
                actual_type="unsupported enum value",
            )
        importance = _require_text(block.get("importance"), f"{label}[{index}].importance", 32).casefold()
        if importance not in SEMANTIC_LEARNING_BLOCK_IMPORTANCE:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"{label}[{index}].importance is not supported.",
                path=f"{label}[{index}].importance",
                constraint="ENUM_SEMANTIC_LEARNING_BLOCK_IMPORTANCE",
                expected_type="supported semantic learning-block importance",
                actual_type="unsupported enum value",
            )
        content = _require_object(block.get("content"), f"{label}[{index}].content")
        concept_ids = _require_text_array(
            block.get("concept_ids") if require_primary_semantic_ownership else block.get("concept_ids") or [],
            f"{label}[{index}].concept_ids",
            min_items=1 if require_primary_semantic_ownership else 0,
            max_items=24,
            item_max_length=96,
        )
        primary_concept_ids = _require_text_array(
            block.get("primary_concept_ids") if require_primary_semantic_ownership else block.get("primary_concept_ids") or [],
            f"{label}[{index}].primary_concept_ids",
            min_items=0,
            max_items=24,
            item_max_length=96,
        )
        primary_evidence_scope_ids = _require_text_array(
            block.get("primary_evidence_scope_ids") if require_evidence_scope_ownership else block.get("primary_evidence_scope_ids") or [],
            f"{label}[{index}].primary_evidence_scope_ids",
            min_items=0,
            max_items=12,
            item_max_length=96,
        )
        supporting_evidence_scope_ids = _require_text_array(
            block.get("supporting_evidence_scope_ids") if require_evidence_scope_ownership else block.get("supporting_evidence_scope_ids") or [],
            f"{label}[{index}].supporting_evidence_scope_ids",
            min_items=0,
            max_items=12,
            item_max_length=96,
        )
        if set(primary_evidence_scope_ids) & set(supporting_evidence_scope_ids):
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                "A semantic learning block cannot primary-own and supporting-reference the same evidence scope.",
                path=f"{label}[{index}].supporting_evidence_scope_ids",
                constraint="DISJOINT_EVIDENCE_SCOPE_OWNERSHIP",
                expected_type="disjoint primary/supporting evidence-scope arrays",
                actual_type="overlapping arrays",
            )
        if require_evidence_scope_ownership and not primary_evidence_scope_ids and not supporting_evidence_scope_ids:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                "A v5 semantic learning block must declare primary or supporting evidence scope references.",
                path=f"{label}[{index}].primary_evidence_scope_ids",
                constraint="EVIDENCE_SCOPE_REFERENCE_REQUIRED",
                expected_type="at least one server-provided evidence scope ID",
                actual_type="array[length=0]",
            )
        source_fact_ids = _require_text_array(
            block.get("source_fact_ids") or [],
            f"{label}[{index}].source_fact_ids",
            min_items=1 if require_source_fact_ownership else 0,
            max_items=MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE,
            item_max_length=96,
        )
        primary_fact_owner = (
            bool(primary_evidence_scope_ids)
            if source_fact_ownership_by_evidence_scope
            else bool(primary_concept_ids or primary_evidence_scope_ids)
        )
        if require_primary_source_fact_ownership and primary_fact_owner and not source_fact_ids:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_PRIMARY_BLOCK_SOURCE_FACT_OWNERSHIP_REQUIRED",
                "A primary semantic learning block must own at least one server-allocated Source Fact.",
                path=f"{label}[{index}].source_fact_ids",
                constraint="PRIMARY_SEMANTIC_BLOCK_SOURCE_FACT_OWNERSHIP",
                expected_type="array[minItems=1] of server-allocated canonical fact IDs",
                actual_type="array[length=0]",
            )
        normalized_block = {
            "id": block_id,
            "intent": intent,
            "importance": importance,
            "concept_ids": concept_ids,
            "primary_concept_ids": primary_concept_ids,
            "source_refs": _require_text_array(block.get("source_refs") or [], f"{label}[{index}].source_refs", min_items=0, max_items=8, item_max_length=96),
            "learning_objective_refs": _require_text_array(
                block.get("learning_objective_refs") or [],
                f"{label}[{index}].learning_objective_refs",
                min_items=0,
                max_items=24,
                item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH,
            ),
            "content": content,
        }
        if include_evidence_scope_ids:
            normalized_block["primary_evidence_scope_ids"] = primary_evidence_scope_ids
            normalized_block["supporting_evidence_scope_ids"] = supporting_evidence_scope_ids
        if include_source_fact_ids:
            normalized_block["source_fact_ids"] = source_fact_ids
        blocks.append(normalized_block)
    return blocks


_PROVIDER_OWNED_FACT_FIELDS = {
    "source_fact_ids",
    "covered_source_fact_ids",
    "source_fact_allocation",
    "_allocation_declared_source_fact_ids",
    "source_evidence_scope_allocation",
}


def _assert_no_provider_owned_canonical_facts(value: Any) -> None:
    """Reject a server-owned Architect response that tries to own canonical facts.

    Canonical IDs identify server evidence.  Ignoring a provider-supplied list
    would be silent data loss; accepting it would make Gemini an authority.
    A v4/v5 response must therefore contain neither IDs nor allocation metadata.
    """
    if isinstance(value, dict):
        forbidden = _PROVIDER_OWNED_FACT_FIELDS.intersection(value)
        if forbidden:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
                "Course Architect output must not contain canonical source fact IDs or allocation metadata.",
            )
        for child in value.values():
            _assert_no_provider_owned_canonical_facts(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_provider_owned_canonical_facts(child)


def validate_lesson_author_blueprint(
    value: Any,
    *,
    require_source_fact_ownership: bool = True,
    forbid_provider_fact_ownership: bool = False,
) -> dict[str, Any]:
    raw = _require_object(value, "Blueprint")
    architecture_version = raw.get("architecture_contract_version")
    is_source_map_architecture = architecture_version in {3, 4, 5}
    is_server_owned_fact_architecture = architecture_version in {4, 5}
    is_evidence_scope_architecture = architecture_version == 5
    if architecture_version is not None and not isinstance(architecture_version, int):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            "architecture_contract_version must be an integer.",
            path="architecture_contract_version",
            constraint="TYPE_INTEGER",
            expected_type="integer",
            actual_type=_safe_json_shape(architecture_version),
        )
    if forbid_provider_fact_ownership:
        if architecture_version not in {4, 5}:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                "Course Architect output must use architecture_contract_version 4 or 5.",
                path="architecture_contract_version",
                constraint="ARCHITECTURE_CONTRACT_VERSION",
                expected_type="integer enum 4 or 5",
                actual_type=_safe_json_shape(architecture_version),
            )
        _assert_no_provider_owned_canonical_facts(raw)
    chapters_raw = raw.get("chapters")
    if not isinstance(chapters_raw, list) or not 1 <= len(chapters_raw) <= MAX_BLUEPRINT_CHAPTERS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"chapters must contain 1 to {MAX_BLUEPRINT_CHAPTERS} items.",
            path="chapters",
            constraint="ARRAY_LENGTH",
            expected_type=f"array[length=1..{MAX_BLUEPRINT_CHAPTERS}]",
            actual_type=_safe_json_shape(chapters_raw),
        )

    chapters: list[dict[str, Any]] = []
    total_lessons = total_units = total_components = total_media_plans = 0
    for chapter_index, chapter_value in enumerate(chapters_raw):
        chapter_path = f"chapters[{chapter_index}]"
        chapter = _require_object(chapter_value, chapter_path)
        lessons_raw = chapter.get("lessons")
        if not isinstance(lessons_raw, list) or not 1 <= len(lessons_raw) <= MAX_BLUEPRINT_LESSONS_PER_CHAPTER:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"chapters[{chapter_index}].lessons must contain 1 to {MAX_BLUEPRINT_LESSONS_PER_CHAPTER} items.",
                path=f"chapters[{chapter_index}].lessons",
                constraint="ARRAY_LENGTH",
                expected_type=f"array[length=1..{MAX_BLUEPRINT_LESSONS_PER_CHAPTER}]",
                actual_type=_safe_json_shape(lessons_raw),
            )
        lessons: list[dict[str, Any]] = []
        total_lessons += len(lessons_raw)
        for lesson_index, lesson_value in enumerate(lessons_raw):
            lesson_path = f"{chapter_path}.lessons[{lesson_index}]"
            lesson = _require_object(lesson_value, lesson_path)
            units_raw = lesson.get("units")
            if not isinstance(units_raw, list) or not 1 <= len(units_raw) <= MAX_BLUEPRINT_UNITS_PER_LESSON:
                raise LessonAuthorBlueprintValidationError(
                    "BLUEPRINT_INVALID_SCHEMA",
                    f"{lesson_path}.units must contain 1 to {MAX_BLUEPRINT_UNITS_PER_LESSON} items.",
                    path=f"{lesson_path}.units",
                    constraint="ARRAY_LENGTH",
                    expected_type=f"array[length=1..{MAX_BLUEPRINT_UNITS_PER_LESSON}]",
                    actual_type=_safe_json_shape(units_raw),
                )
            units: list[dict[str, Any]] = []
            total_units += len(units_raw)
            for unit_index, unit_value in enumerate(units_raw):
                unit_path = f"{lesson_path}.units[{unit_index}]"
                unit = _require_object(unit_value, unit_path)
                unit_label = unit_path if is_source_map_architecture else f"lesson.units[{unit_index}]"
                component_plan_raw = unit.get("component_plan")
                component_plan: list[dict[str, Any]] = []
                if not is_source_map_architecture:
                    if not isinstance(component_plan_raw, list) or not 1 <= len(component_plan_raw) <= MAX_BLUEPRINT_COMPONENTS_PER_UNIT:
                        raise LessonAuthorBlueprintValidationError("BLUEPRINT_INVALID_SCHEMA", f"lesson.units[{unit_index}].component_plan must contain 1 to {MAX_BLUEPRINT_COMPONENTS_PER_UNIT} items.")
                    total_components += len(component_plan_raw)
                    seen_component_types: set[str] = set()
                    for plan_index, plan_value in enumerate(component_plan_raw):
                        plan = _require_object(plan_value, f"lesson.units[{unit_index}].component_plan[{plan_index}]")
                        component_type = _require_text(plan.get("type"), "component_plan.type", 40).casefold()
                        if component_type not in BLUEPRINT_COMPONENT_TYPES or component_type in seen_component_types:
                            raise LessonAuthorBlueprintValidationError("BLUEPRINT_INVALID_SCHEMA", "component_plan contains an unsupported or duplicate component type.")
                        seen_component_types.add(component_type)
                        purpose = str(plan.get("purpose") or "").strip().casefold()
                        if purpose not in {"explain", "assess", "clarify", "sequence", "relationship", "terminology"}:
                            purpose = ""
                        artifacts: list[dict[str, Any]] = []
                        for artifact_value in plan.get("required_artifacts") or []:
                            if not isinstance(artifact_value, dict):
                                continue
                            artifact_type = str(artifact_value.get("type") or "").strip().casefold()
                            if artifact_type not in {"ordered_list", "checklist", "table", "warning", "requirement", "exception", "comparison"}:
                                continue
                            minimum = artifact_value.get("minimum_items")
                            artifacts.append({
                                "type": artifact_type,
                                **({"minimum_items": min(minimum, 100)} if isinstance(minimum, int) and minimum > 0 else {}),
                            })
                        component_plan.append({
                            "type": component_type,
                            "title": _require_text(plan.get("title"), "component_plan.title", 180),
                            "rationale": _require_text(plan.get("rationale"), "component_plan.rationale", 240),
                            **({"purpose": purpose} if purpose else {}),
                            "source_fact_ids": _require_text_array(plan.get("source_fact_ids") or [], "component_plan.source_fact_ids", min_items=0, max_items=160, item_max_length=96),
                            "content_requirements": _require_text_array(plan.get("content_requirements") or [], "component_plan.content_requirements", min_items=0, max_items=8, item_max_length=500),
                            **({"required_artifacts": artifacts[:6]} if artifacts else {}),
                        })
                    if "html" not in seen_component_types:
                        raise LessonAuthorBlueprintValidationError("BLUEPRINT_INVALID_SCHEMA", "lesson.units.component_plan must include one html explanation component.")
                    if sum(component_type != "la_faq" for component_type in seen_component_types) > 2:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            "lesson.units.component_plan may contain html and at most one additional interactive component before FAQ.",
                        )
                learning_blocks = _validate_semantic_learning_blocks(
                    unit.get("learning_blocks"),
                    f"{unit_path}.learning_blocks",
                    # The allocator intentionally assigns canonical facts to
                    # the one primary block that teaches each concept. A
                    # supporting/reinforcement block may therefore be factless
                    # without losing unit-level source ownership.
                    require_source_fact_ownership=(
                        require_source_fact_ownership and not is_server_owned_fact_architecture
                    ),
                    require_primary_source_fact_ownership=(
                        require_source_fact_ownership and is_server_owned_fact_architecture
                    ),
                    include_source_fact_ids=not (forbid_provider_fact_ownership and is_server_owned_fact_architecture),
                    require_primary_semantic_ownership=forbid_provider_fact_ownership and is_server_owned_fact_architecture,
                    require_evidence_scope_ownership=forbid_provider_fact_ownership and is_evidence_scope_architecture,
                    include_evidence_scope_ids=is_evidence_scope_architecture,
                    source_fact_ownership_by_evidence_scope=is_evidence_scope_architecture,
                ) if is_source_map_architecture else []
                raw_media_plan = unit.get("media_plan")
                media_plan: dict[str, str] | None = None
                if raw_media_plan is not None:
                    media_record = _require_object(raw_media_plan, f"lesson.units[{unit_index}].media_plan")
                    media_type = _require_text(media_record.get("type"), "media_plan.type", 40).casefold()
                    if media_type not in BLUEPRINT_MEDIA_TYPES:
                        raise LessonAuthorBlueprintValidationError("BLUEPRINT_INVALID_SCHEMA", f"media_plan.type must be one of: {', '.join(sorted(BLUEPRINT_MEDIA_TYPES))}.")
                    media_plan = {
                        "type": media_type,
                        "title": _require_text(media_record.get("title"), "media_plan.title", 180),
                        "content_outline": _require_text(media_record.get("content_outline"), "media_plan.content_outline", 600),
                        "rationale": _require_text(media_record.get("rationale"), "media_plan.rationale", 240),
                    }
                    total_media_plans += 1
                unit_primary_concept_ids = _require_text_array(
                    unit.get("primary_concept_ids") if forbid_provider_fact_ownership and is_server_owned_fact_architecture else unit.get("primary_concept_ids") or [],
                    f"{unit_label}.primary_concept_ids",
                    min_items=0,
                    max_items=24,
                    item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH,
                ) if is_source_map_architecture else []
                unit_has_primary_semantic_scope = (
                    any(bool(block.get("primary_evidence_scope_ids")) for block in learning_blocks)
                    if is_evidence_scope_architecture
                    else bool(unit_primary_concept_ids) or any(bool(block.get("primary_concept_ids")) for block in learning_blocks)
                )
                unit_source_fact_ids = _require_text_array(
                    unit.get("source_fact_ids") or [],
                    f"{unit_label}.source_fact_ids",
                    min_items=(
                        1
                        if is_source_map_architecture
                        and require_source_fact_ownership
                        and (not is_server_owned_fact_architecture or unit_has_primary_semantic_scope)
                        else 0
                    ),
                    max_items=MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE,
                    item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH,
                )
                if is_source_map_architecture and require_source_fact_ownership:
                    owned = {fact_id for block in learning_blocks for fact_id in block["source_fact_ids"]}
                    if not set(unit_source_fact_ids).issubset(owned):
                        raise LessonAuthorBlueprintValidationError("BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE", "Every unit source fact must be owned by a semantic learning block.")
                units.append({
                    "title": _require_text(strip_source_range_suffix(str(unit.get("title") or "")), "unit.title", 180),
                    **({"purpose": _require_text(unit.get("purpose"), f"{unit_label}.purpose", 300)} if is_source_map_architecture else {}),
                    **({"concept_ids": _require_text_array(unit.get("concept_ids"), f"{unit_label}.concept_ids", min_items=1, max_items=24, item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH)} if is_source_map_architecture else {}),
                    **({"primary_concept_ids": unit_primary_concept_ids} if is_source_map_architecture else {}),
                    **({"primary_evidence_scope_ids": _require_text_array(unit.get("primary_evidence_scope_ids") or [], f"{unit_label}.primary_evidence_scope_ids", min_items=0, max_items=12, item_max_length=96)} if is_evidence_scope_architecture else {}),
                    **({"supporting_evidence_scope_ids": _require_text_array(unit.get("supporting_evidence_scope_ids") or [], f"{unit_label}.supporting_evidence_scope_ids", min_items=0, max_items=12, item_max_length=96)} if is_evidence_scope_architecture else {}),
                    **({"learning_objective_refs": _require_text_array(unit.get("learning_objective_refs"), f"{unit_label}.learning_objective_refs", min_items=1, max_items=24, item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH)} if is_source_map_architecture else {}),
                    "component_plan": component_plan,
                    "source_refs": _optional_source_refs(unit.get("source_refs"), f"{unit_label}.source_refs"),
                    **({"source_fact_ids": unit_source_fact_ids} if not (forbid_provider_fact_ownership and is_server_owned_fact_architecture) else {}),
                    **({"learning_blocks": learning_blocks} if learning_blocks else {}),
                    **({"media_plan": media_plan} if media_plan is not None else {}),
                })
            lesson_label = lesson_path if is_source_map_architecture else "lesson"
            lesson_objectives = _require_text_array(lesson.get("learning_objectives"), f"{lesson_label}.learning_objectives", min_items=1, max_items=8, item_max_length=500) if is_source_map_architecture else []
            assessment_objective_refs = _require_text_array(
                lesson.get("assessment_objective_refs") or [],
                f"{lesson_label}.assessment_objective_refs",
                min_items=0,
                max_items=8,
                item_max_length=MAX_CANONICAL_IDENTIFIER_LENGTH,
            ) if is_source_map_architecture else []
            if is_server_owned_fact_architecture:
                for unit_index, normalized_unit in enumerate(units):
                    _validate_local_learning_objective_refs(
                        normalized_unit.get("learning_objective_refs", []),
                        f"{lesson_label}.units[{unit_index}].learning_objective_refs",
                        len(lesson_objectives),
                    )
                    for block_index, block in enumerate(normalized_unit.get("learning_blocks", [])):
                        _validate_local_learning_objective_refs(
                            block.get("learning_objective_refs", []),
                            f"{lesson_label}.units[{unit_index}].learning_blocks[{block_index}].learning_objective_refs",
                            len(lesson_objectives),
                        )
                _validate_local_learning_objective_refs(
                    assessment_objective_refs,
                    f"{lesson_label}.assessment_objective_refs",
                    len(lesson_objectives),
                )
            lessons.append({
                "title": _require_text(strip_source_range_suffix(str(lesson.get("title") or "")), "lesson.title", 180),
                "objective": _require_text(lesson.get("objective"), "lesson.objective", 500),
                "learning_activities": _require_text_array(lesson.get("learning_activities"), "lesson.learning_activities", min_items=1, max_items=MAX_BLUEPRINT_ACTIVITIES_PER_LESSON, item_max_length=280),
                "assessment": _require_text(lesson.get("assessment"), "lesson.assessment", 500),
                "units": units,
                "source_refs": _optional_source_refs(lesson.get("source_refs"), "lesson.source_refs"),
                **({"learning_objectives": lesson_objectives} if lesson_objectives else {}),
                **({"primary_concept_ids": _require_text_array(lesson.get("primary_concept_ids"), f"{lesson_label}.primary_concept_ids", min_items=1, max_items=24, item_max_length=96)} if is_source_map_architecture else {}),
                **({"supporting_concept_ids": _require_text_array(lesson.get("supporting_concept_ids") or [], f"{lesson_label}.supporting_concept_ids", min_items=0, max_items=24, item_max_length=96)} if is_source_map_architecture else {}),
                **({"prerequisite_concept_ids": _require_text_array(lesson.get("prerequisite_concept_ids") or [], f"{lesson_label}.prerequisite_concept_ids", min_items=0, max_items=24, item_max_length=96)} if is_source_map_architecture else {}),
                **({"estimated_minutes": lesson.get("estimated_minutes")} if is_source_map_architecture and isinstance(lesson.get("estimated_minutes"), int) and 1 <= lesson.get("estimated_minutes") <= 600 else {}),
                **({"assessment_required": bool(lesson.get("assessment_required"))} if is_source_map_architecture else {}),
                **({"assessment_objective_refs": assessment_objective_refs} if is_source_map_architecture else {}),
            })
        chapters.append({
            "title": _require_text(strip_source_range_suffix(str(chapter.get("title") or "")), "chapter.title", 220),
            "objective": _require_text(chapter.get("objective"), "chapter.objective", 500),
            "lessons": lessons,
            "source_refs": _optional_source_refs(chapter.get("source_refs"), "chapter.source_refs"),
            **({"learning_objectives": _require_text_array(chapter.get("learning_objectives"), f"{chapter_path}.learning_objectives", min_items=1, max_items=12, item_max_length=500)} if is_source_map_architecture else {}),
            **({"concept_ids": _require_text_array(chapter.get("concept_ids"), f"{chapter_path}.concept_ids", min_items=1, max_items=48, item_max_length=96)} if is_source_map_architecture else {}),
        })

    if total_lessons > MAX_BLUEPRINT_TOTAL_LESSONS or total_units > MAX_BLUEPRINT_TOTAL_UNITS or total_components > MAX_BLUEPRINT_TOTAL_COMPONENTS or total_media_plans > MAX_BLUEPRINT_MEDIA_PLANS:
        raise LessonAuthorBlueprintValidationError("BLUEPRINT_INVALID_SCHEMA", "Blueprint exceeds configured structural limits.")
    learning_outcomes = _require_text_array(raw.get("learning_outcomes"), "learning_outcomes", min_items=3, max_items=MAX_BLUEPRINT_LEARNING_OUTCOMES, item_max_length=500)
    return {
        "architecture_contract_version": architecture_version if is_source_map_architecture else None,
        "content_contract_version": 1,
        "title": _require_text(raw.get("title"), "title", 220),
        "summary": _require_text(raw.get("summary"), "summary", 1400),
        "target_audience": _require_text(raw.get("target_audience"), "target_audience", 500),
        "prerequisites": _require_text_array(raw.get("prerequisites"), "prerequisites", min_items=0, max_items=MAX_BLUEPRINT_PREREQUISITES, item_max_length=280),
        "learning_outcomes": learning_outcomes,
        "course_outcomes": _require_text_array(raw.get("course_outcomes"), "course_outcomes", min_items=1, max_items=MAX_BLUEPRINT_LEARNING_OUTCOMES, item_max_length=500) if is_source_map_architecture else learning_outcomes,
        "assessment_strategy": _require_text(raw.get("assessment_strategy"), "assessment_strategy", 900),
        "assumptions": _require_text_array(raw.get("assumptions"), "assumptions", min_items=0, max_items=MAX_BLUEPRINT_ASSUMPTIONS, item_max_length=400),
        "chapters": chapters,
    }


def ensure_lesson_author_blueprint_faqs(
    blueprint: dict[str, Any],
    locale: str,
) -> dict[str, Any]:
    """Place one source-grounded FAQ plan at the end of every lesson.

    The Blueprint is the content contract for later staged authoring. Keeping
    the FAQ in the final unit makes its placement deterministic after all
    explanation and practice components, including when the model originally
    put it on an earlier unit.
    """
    # Phase 3 Course Architect output intentionally has no component plan.
    # FAQ selection belongs to the Phase-2 semantic-block planner, so do not
    # manufacture an FAQ CMS choice for a semantic blueprint.
    if blueprint.get("architecture_contract_version") in {3, 4, 5}:
        return blueprint

    is_english = locale == "en"
    fallback_faq = {
        "type": "la_faq",
        "title": "Frequently asked questions" if is_english else "Câu hỏi thường gặp",
        "rationale": (
            "Consolidates source-grounded clarifications after the lesson content."
            if is_english
            else "Củng cố các điểm cần làm rõ dựa trên tài liệu nguồn sau nội dung bài học."
        ),
    }

    for chapter in blueprint.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        for lesson in chapter.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            units = lesson.get("units")
            if not isinstance(units, list) or not units:
                continue

            faq_plan: dict[str, str] | None = None
            for unit in units:
                if not isinstance(unit, dict):
                    continue
                plans = unit.get("component_plan")
                if not isinstance(plans, list):
                    continue
                retained_plans: list[dict[str, Any]] = []
                for plan in plans:
                    if not isinstance(plan, dict):
                        continue
                    if plan.get("type") == "la_faq":
                        if faq_plan is None:
                            faq_plan = {
                                "type": "la_faq",
                                "title": str(plan.get("title") or fallback_faq["title"]),
                                "rationale": str(plan.get("rationale") or fallback_faq["rationale"]),
                            }
                        continue
                    retained_plans.append(plan)
                unit["component_plan"] = retained_plans

            final_unit = units[-1]
            if not isinstance(final_unit, dict):
                continue
            final_plan = final_unit.get("component_plan")
            if not isinstance(final_plan, list):
                continue
            if len(final_plan) > MAX_BLUEPRINT_COMPONENTS_PER_UNIT - 1:
                raise LessonAuthorBlueprintValidationError(
                    "BLUEPRINT_INVALID_SCHEMA",
                    "The final unit must reserve one component slot for the required lesson FAQ.",
                )
            final_plan.append(faq_plan or dict(fallback_faq))

    return blueprint


def parse_and_validate_lesson_author_blueprint(
    text: str,
    *,
    require_source_fact_ownership: bool = True,
    forbid_provider_fact_ownership: bool = False,
) -> dict[str, Any]:
    last_validation_error: LessonAuthorBlueprintValidationError | None = None
    for parsed in _decode_blueprint_json(text):
        try:
            return validate_lesson_author_blueprint(
                normalize_lesson_author_blueprint_candidate(parsed),
                require_source_fact_ownership=require_source_fact_ownership,
                forbid_provider_fact_ownership=forbid_provider_fact_ownership,
            )
        except LessonAuthorBlueprintValidationError as error:
            last_validation_error = error
    if last_validation_error is not None:
        raise last_validation_error
    raise LessonAuthorBlueprintValidationError(
        "BLUEPRINT_INVALID_JSON",
        "Blueprint response is not valid JSON.",
    )


def parse_lesson_author_blueprint_candidate(
    text: str,
    *,
    forbid_provider_fact_ownership: bool = False,
) -> dict[str, Any]:
    """Decode a complete provider object without accepting its schema yet.

    The Course Architecture graph uses this narrow helper only when a V5
    response is valid JSON but has one repairable *local* structural issue.
    It deliberately preserves the candidate for a bounded patch repair while
    keeping the provider-fact ownership guard at the first decoding boundary.
    It never recovers truncated JSON or turns a malformed/global candidate
    into a repair target.
    """

    for parsed in _decode_blueprint_json(text):
        candidate = normalize_lesson_author_blueprint_candidate(parsed)
        if not isinstance(candidate, dict):
            continue
        if forbid_provider_fact_ownership:
            _assert_no_provider_owned_canonical_facts(candidate)
        return candidate
    raise LessonAuthorBlueprintValidationError(
        "BLUEPRINT_INVALID_JSON",
        "Blueprint response is not a JSON object.",
    )
