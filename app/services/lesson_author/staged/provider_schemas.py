"""Provider response schemas and models of the staged writer (skeleton, unit, instance, repair)."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

from google.genai import types
from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

from app.component_capabilities import validate_instance_plan
from app.lesson_prompt_policy import lesson_instructional_quality_policy
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import semantic_learning_visible_text
from app.services.lesson_author.staged.plan import normalize_staged_component_type
from app.services.provider import safe_provider_error_diagnostics
from app.workflows.contracts import WorkflowFailure


def build_lesson_author_skeleton_response_schema() -> types.Schema:
    """Constrain the planning pass to a small, parseable structure."""
    string_schema = lambda description: types.Schema(
        type=types.Type.STRING,
        description=description,
    )
    string_array_schema = lambda description: types.Schema(
        type=types.Type.ARRAY,
        description=description,
        items=string_schema("A source fact identifier."),
    )
    component_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["type", "rationale", "purpose", "source_fact_ids", "content_requirements"],
        properties={
            "type": string_schema("Learning component type selected from the source evidence."),
            "rationale": string_schema("One sentence explaining why this format fits the source facts."),
            "purpose": string_schema("One instructional purpose: explain, assess, clarify, sequence, relationship, or terminology."),
            "source_fact_ids": string_array_schema("Source fact identifiers supporting this format."),
            "content_requirements": string_array_schema("Specific source facts, steps, or fidelity requirements this component must convey."),
            "required_artifacts": types.Schema(type=types.Type.ARRAY, items=types.Schema(
                type=types.Type.OBJECT,
                required=["type"],
                properties={
                    "type": string_schema("ordered_list, checklist, table, warning, requirement, exception, or comparison"),
                    "minimum_items": types.Schema(type=types.Type.INTEGER, description="Minimum source item count to preserve."),
                },
            )),
        },
    )
    unit_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "component_plan", "source_fact_ids"],
        properties={
            "title": string_schema("Semantic unit title without numbering."),
            "source_fact_ids": string_array_schema("All mandatory source fact IDs assigned to this unit."),
            "components": types.Schema(
                type=types.Type.ARRAY,
                items=component_schema,
            ),
            "component_plan": types.Schema(
                type=types.Type.ARRAY,
                description="Exact learning formats to generate for this unit.",
                items=component_schema,
            ),
        },
    )
    lesson_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "units"],
        properties={
            "title": string_schema("Semantic lesson title without numbering."),
            "units": types.Schema(
                type=types.Type.ARRAY,
                items=unit_schema,
            ),
        },
    )
    chapter_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "lessons"],
        properties={
            "title": string_schema("Semantic chapter title without numbering."),
            "lessons": types.Schema(
                type=types.Type.ARRAY,
                items=lesson_schema,
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["chapters"],
        properties={
            "chapters": types.Schema(
                type=types.Type.ARRAY,
                items=chapter_schema,
            ),
        },
    )


def build_lesson_author_proposal_response_schema() -> types.Schema:
    """Constrain proposal generation to the tree persisted by the backend.

    Prompt-only JSON contracts are not reliable for large lesson proposals:
    the model can emit markdown, prose, or a partial object even when JSON
    mode is enabled.  Keep the schema permissive inside each component so the
    content validators remain the source of truth for quality rules.
    """
    string_schema = lambda description: types.Schema(
        type=types.Type.STRING,
        description=description,
    )
    string_array_schema = lambda description: types.Schema(
        type=types.Type.ARRAY,
        description=description,
        items=string_schema("A text item."),
    )
    item_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "text": string_schema("A sortable item or choice text."),
            "label": string_schema("A display label."),
            "question": string_schema("A FAQ question."),
            "answer": string_schema("A FAQ answer."),
            "term": string_schema("A crossword answer term."),
            "clue": string_schema("A crossword clue."),
            "hint": string_schema("A crossword hint."),
            "correct": types.Schema(type=types.Type.BOOLEAN, description="Whether a choice is correct."),
        },
    )
    choice_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "text": string_schema("Choice text."),
            "correct": types.Schema(type=types.Type.BOOLEAN, description="Whether the choice is correct."),
        },
    )
    node_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "label": string_schema("Diagram node label."),
            "shape": string_schema("Diagram node shape."),
            "tooltip": string_schema("Diagram node explanation."),
        },
    )
    edge_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "source": types.Schema(type=types.Type.INTEGER, description="Source node index."),
            "target": types.Schema(type=types.Type.INTEGER, description="Target node index."),
            "label": string_schema("Edge label."),
        },
    )
    component_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["type", "source_fact_ids", "covered_source_fact_ids"],
        properties={
            "type": string_schema("Learning component type."),
            "source_fact_ids": string_array_schema("Source fact identifiers supporting this component."),
            "covered_source_fact_ids": string_array_schema("Source fact IDs that this generated component explicitly covers."),
            "selection_rationale": string_schema("Why this component format fits the source facts."),
            "title": string_schema("Component title."),
            "html": string_schema("Safe HTML learning content."),
            "data": string_schema("Serialized component content when applicable."),
            "problem_type": string_schema("Question type."),
            "question": string_schema("Question text."),
            "question_text": string_schema("Interactive question text."),
            "choices": types.Schema(type=types.Type.ARRAY, items=choice_schema),
            "options": string_array_schema("Question options."),
            "answer": string_schema("Short or numerical answer."),
            "tolerance": string_schema("Answer tolerance."),
            "explanation": string_schema("Answer explanation."),
            # Object items support FAQ, crossword and sortable entries. The
            # normalizer also accepts legacy string items where the provider
            # omits object metadata.
            "items": types.Schema(type=types.Type.ARRAY, items=item_schema),
            "ordered_items": string_array_schema("Ordered sortable items."),
            "steps": string_array_schema("Ordered steps."),
            "words": types.Schema(type=types.Type.ARRAY, items=item_schema),
            "name": string_schema("Diagram name."),
            "nodes": types.Schema(type=types.Type.ARRAY, items=node_schema),
            "edges": types.Schema(type=types.Type.ARRAY, items=edge_schema),
        },
    )
    unit_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "components"],
        properties={
            "title": string_schema("Semantic unit title without numbering."),
            "source_refs": string_array_schema("Source references."),
            "source_fact_ids": string_array_schema("Covered source fact identifiers."),
            "components": types.Schema(type=types.Type.ARRAY, items=component_schema),
            "blocks": types.Schema(type=types.Type.ARRAY, items=component_schema),
            "html": string_schema("Legacy unit HTML content."),
            "content": string_schema("Legacy unit text content."),
        },
    )
    lesson_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "units"],
        properties={
            "title": string_schema("Semantic lesson title without numbering."),
            "source_refs": string_array_schema("Source references."),
            "units": types.Schema(type=types.Type.ARRAY, items=unit_schema),
        },
    )
    chapter_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["title", "lessons"],
        properties={
            "title": string_schema("Semantic chapter title without numbering."),
            "source_refs": string_array_schema("Source references."),
            "lessons": types.Schema(type=types.Type.ARRAY, items=lesson_schema),
        },
    )
    change_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "operation": string_schema("Requested structural operation."),
            "target": string_schema("Target node type."),
            "target_block_id": string_schema("Target outline node id."),
            "title": string_schema("Replacement title when applicable."),
            "content": string_schema("Replacement content when applicable."),
            "reason": string_schema("Reason for the change."),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "summary": string_schema("Concise proposal summary."),
            "chapters": types.Schema(type=types.Type.ARRAY, items=chapter_schema),
            "changes": types.Schema(type=types.Type.ARRAY, items=change_schema),
            "operation_plan": types.Schema(type=types.Type.OBJECT, properties={
                "operation": string_schema("Requested structural operation."),
                "target": string_schema("Target node type."),
                "target_block_id": string_schema("Target outline node id."),
            }),
        },
    )


def build_lesson_author_unit_response_schema(
    component_types: list[str] | None = None,
) -> types.Schema:
    """Schema for one bounded Stage-B call, limited to its selected types."""
    string_schema = lambda description: types.Schema(
        type=types.Type.STRING,
        description=description,
    )
    string_array_schema = lambda description: types.Schema(
        type=types.Type.ARRAY,
        description=description,
        items=string_schema("A text item."),
    )
    item_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "text": string_schema("A sortable item or choice text."),
            "label": string_schema("A display label."),
            "question": string_schema("A FAQ question."),
            "answer": string_schema("A FAQ answer."),
            "term": string_schema("A crossword answer term."),
            "clue": string_schema("A crossword clue."),
            "hint": string_schema("A crossword hint."),
            "correct": types.Schema(type=types.Type.BOOLEAN, description="Whether a choice is correct."),
        },
    )
    choice_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "text": string_schema("Choice text."),
            "correct": types.Schema(type=types.Type.BOOLEAN, description="Whether the choice is correct."),
        },
    )
    selected_types = {
        normalized
        for value in (component_types or [])
        for normalized in [normalize_staged_component_type(value)]
        if normalized
    }
    if not selected_types:
        # Backward-compatible default for callers that have not supplied a
        # Stage-A plan. All staged Lesson Author generation paths now pass the
        # selected plan explicitly.
        selected_types = {"html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"}
    component_properties: dict[str, types.Schema] = {
        "type": string_schema(
            "One selected learning component type: " + ", ".join(sorted(selected_types)) + ".",
        ),
        "source_fact_ids": string_array_schema("Source fact identifiers supporting this component."),
        "covered_source_fact_ids": string_array_schema("Source fact IDs that this generated component explicitly covers."),
        "selection_rationale": string_schema("Why this already-selected component fits the source facts."),
        "title": string_schema("Component title."),
    }
    if "html" in selected_types:
        component_properties["semantic_content"] = types.Schema(
            type=types.Type.OBJECT,
            description=(
                "Preferred semantic explanatory content. The Node backend renders this "
                "deterministically to sanitized HTML."
            ),
            properties={
                "heading": string_schema("Optional explanatory heading."),
                "version": types.Schema(type=types.Type.INTEGER, description="2 for ordered semantic sections."),
                "sections": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT, properties={
                    "heading": string_schema("Local section heading."),
                    "learning_block_ids": string_array_schema("Exact approved teaching group references."),
                    "blocks": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT, properties={
                        "kind": string_schema("paragraph, task, warning, bullets, steps or table."),
                        "text": string_schema("Paragraph, task or warning."),
                        "items": string_array_schema("Bullet or step items."),
                        "rows": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT, properties={
                            "label": string_schema("Row label."), "value": string_schema("Associated explanation."),
                        })),
                    })),
                })),
                "paragraphs": string_array_schema("Explanatory paragraphs."),
                "bullet_points": string_array_schema("Key points."),
                "ordered_steps": string_array_schema("Read-only procedure steps."),
                "warnings": string_array_schema("Warnings or exceptions."),
                "comparison_rows": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "label": string_schema("Comparison label."),
                            "value": string_schema("Comparison value."),
                        },
                    ),
                ),
            },
        )
        component_properties["html"] = string_schema("Legacy safe HTML fallback only.")
    if "problem" in selected_types:
        component_properties.update({
            "problem_type": string_schema("Question type."),
            "question": string_schema("Question text."),
            "choices": types.Schema(type=types.Type.ARRAY, items=choice_schema),
            "options": string_array_schema("Question options."),
            "answer": string_schema("Short or numerical answer."),
            "tolerance": string_schema("Answer tolerance."),
            "explanation": string_schema("Answer explanation."),
        })
    if "la_faq" in selected_types:
        component_properties["items"] = types.Schema(type=types.Type.ARRAY, items=item_schema)
    if "la_sortable" in selected_types:
        component_properties.update({
            "question_text": string_schema("Interactive question text."),
            "items": types.Schema(type=types.Type.ARRAY, items=item_schema),
            "ordered_items": string_array_schema("Ordered sortable items."),
            "steps": string_array_schema("Ordered steps."),
        })
    if "la_crossword" in selected_types:
        component_properties["words"] = types.Schema(type=types.Type.ARRAY, items=item_schema)
    if "la_diagram" in selected_types:
        component_properties.update({
            "name": string_schema("Diagram name."),
            "nodes": types.Schema(type=types.Type.ARRAY, items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "label": string_schema("Diagram node label."),
                    "shape": string_schema("Diagram node shape."),
                    "tooltip": string_schema("Diagram node explanation."),
                },
            )),
            "edges": types.Schema(type=types.Type.ARRAY, items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "source": types.Schema(type=types.Type.INTEGER, description="Source node index."),
                    "target": types.Schema(type=types.Type.INTEGER, description="Target node index."),
                    "label": string_schema("Edge label."),
                },
            )),
        })
    component_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["type", "source_fact_ids", "covered_source_fact_ids"],
        properties=component_properties,
    )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["title", "components", "source_fact_ids"],
        properties={
            "title": string_schema("Exact semantic unit title without numbering."),
            "source_refs": string_array_schema("Source references."),
            "source_fact_ids": string_array_schema("Covered source fact identifiers."),
            "components": types.Schema(type=types.Type.ARRAY, items=component_schema),
            "blocks": types.Schema(type=types.Type.ARRAY, items=component_schema),
            "html": string_schema("Legacy unit HTML content."),
            "content": string_schema("Legacy unit text content."),
        },
    )


STAGED_COMPONENT_CONTRACT_VERSION = "component-payload-4"
STAGED_COMPONENT_REPAIR_CONTRACT_VERSION = "component-payload-delta-1"
STAGED_COMPONENT_COVERAGE_REPAIR_CONTRACT_VERSION = "component-content-coverage-delta-2"
STAGED_MULTI_REPAIR_CONTRACT_VERSION = "component-repair-slots-1"
STAGED_INSTANCE_OUTPUT_CONTRACT_VERSION = "component-instance-payload-1"


STAGED_COMPONENT_PAYLOAD_FIELDS = {
    "html": {"semantic_content", "html"},
    "problem": {"problem_type", "question", "choices", "options", "answer", "tolerance", "explanation"},
    "la_faq": {"items"},
    "la_sortable": {"question_text", "items", "ordered_items", "steps"},
    "la_crossword": {"words"},
    "la_diagram": {"name", "nodes", "edges"},
}


class StagedChoice(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    correct: bool = Field(strict=True)


class StagedSemanticComparisonRow(BaseModel):
    label: str = Field(min_length=1, max_length=500)
    value: str = Field(min_length=1, max_length=1000)


class StagedOrderedBlock(BaseModel):
    kind: Literal["paragraph", "bullets", "steps", "warning", "table", "task"]
    text: str | None = None
    items: list[str] = Field(default_factory=list)
    rows: list[StagedSemanticComparisonRow] = Field(default_factory=list)


class StagedOrderedSection(BaseModel):
    heading: str = Field(min_length=1, max_length=240)
    learning_block_ids: list[str] = Field(default_factory=list, max_length=24)
    blocks: list[StagedOrderedBlock] = Field(min_length=1, max_length=12)


class StagedSemanticContent(BaseModel):
    """Concrete SDK wire vocabulary; the existing renderer validator is authoritative.

    This object is optional on non-HTML components, not a nullable model $ref
    (unsupported by google-genai 1.0.0). Its arrays always retain typed items.
    """
    heading: str | None = Field(default=None, max_length=240)
    paragraphs: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(default_factory=list, max_length=12)
    version: int | None = None
    sections: list[StagedOrderedSection] = Field(default_factory=list, max_length=12)
    bullet_points: list[Annotated[str, Field(min_length=1, max_length=800)]] = Field(default_factory=list, max_length=20)
    ordered_steps: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(default_factory=list, max_length=20)
    warnings: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(default_factory=list, max_length=8)
    comparison_rows: list[StagedSemanticComparisonRow] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_renderable_shape(self):
        _, failure = semantic_learning_visible_text(self.model_dump(exclude_none=True))
        if failure:
            raise ValueError(failure)
        return self


class StagedOrderedSemanticOutput(BaseModel):
    """New-write-only model. Legacy readers deliberately use the old contract."""
    version: int = 2
    sections: list[StagedOrderedSection] = Field(min_length=1, max_length=12)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler.resolve_ref_schema(handler(core_schema))
        # Server-owned serialization marker; Gemini only designs sections.
        schema.get("properties", {}).pop("version", None)
        if "required" in schema:
            schema["required"] = [key for key in schema["required"] if key != "version"]
        return schema

    @model_validator(mode="before")
    @classmethod
    def validate_writer_shape(cls, value):
        if not isinstance(value, dict) or set(value) - {"version", "sections"}:
            raise ValueError("Ordered writer accepts sections only.")
        if "version" in value and (type(value["version"]) is not int or value["version"] != 2):
            raise ValueError("Ordered writer version is server-owned.")
        stamped = {**value, "version": 2}
        _, failure = semantic_learning_visible_text(stamped)
        if failure:
            raise ValueError(failure)
        # Before-validation injection survives exclude_unset=True serialization.
        return stamped


class StagedFaqItem(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=2000)


class StagedSortableItem(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class StagedCrosswordWord(BaseModel):
    answer: str = Field(min_length=2, max_length=80)
    clue: str = Field(min_length=1, max_length=500)
    hint: str | None = None


class StagedDiagramNode(BaseModel):
    label: str = Field(min_length=1, max_length=500)
    shape: Literal["rectangle", "rounded", "ellipse"]
    tooltip: str | None = None


class StagedDiagramEdge(BaseModel):
    source: int = Field(ge=0, strict=True)
    target: int = Field(ge=0, strict=True)
    label: str | None = None


def staged_component_payload_code(component: dict[str, Any]) -> str | None:
    """Strict new-generation acceptance; never guess an answer or repair evidence.

    The installed SDK cannot encode discriminated unions. Its wire schema has
    typed leaves; this conditional contract enforces the selected component.
    Codes contain no provider text and are safe for logs and scoped feedback.
    """
    kind = component.get("type")
    def nonempty(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def rows(key: str, model: type[BaseModel], minimum: int, maximum: int) -> list[dict[str, Any]]:
        values = component.get(key)
        if not isinstance(values, list) or not minimum <= len(values) <= maximum:
            raise ValueError("CARDINALITY")
        return [model.model_validate(value).model_dump() for value in values]

    try:
        if kind == "html":
            semantic = component.get("semantic_content")
            if semantic is not None:
                if re.search(r"<\s*/?\s*(?:script|iframe|style|div|span)\b|\b(?:style|class|onerror|onclick)\s*=", json.dumps(semantic), re.I):
                    return "HTML_PRESENTATION_FORBIDDEN"
                _text, failure = semantic_learning_visible_text(semantic)
                return "HTML_SEMANTIC_INVALID" if failure else None
            return None if nonempty(component.get("html")) else "HTML_CONTENT_REQUIRED"
        if kind == "problem":
            subtype = component.get("problem_type")
            if subtype != "multiple_choice":
                return "PROBLEM_SINGLE_CHOICE_REQUIRED"
            if not nonempty(component.get("question")):
                return "PROBLEM_QUESTION_REQUIRED"
            choices = rows("choices", StagedChoice, 3, 6)
            if not all(nonempty(x["text"]) for x in choices):
                return "PROBLEM_CHOICES_INVALID"
            texts = [x["text"].strip().casefold() for x in choices]
            correct_count = sum(x["correct"] for x in choices)
            if len(set(texts)) != len(texts):
                return "PROBLEM_DUPLICATE_CHOICES"
            if correct_count != 1:
                return "PROBLEM_CORRECT_ANSWER_INVALID"
        elif kind == "la_faq":
            items = rows("items", StagedFaqItem, 2, 8)
            if not all(nonempty(x["question"]) and nonempty(x["answer"]) for x in items):
                return "FAQ_ITEM_INVALID"
            if len({x["question"].strip().casefold() for x in items}) != len(items):
                return "FAQ_DUPLICATE_QUESTION"
        elif kind == "la_sortable":
            items = rows("items", StagedSortableItem, 3, 10)
            if not nonempty(component.get("question_text")) or not all(nonempty(x["text"]) for x in items):
                return "SORTABLE_ITEM_INVALID"
            if len({x["text"].strip().casefold() for x in items}) != len(items):
                return "SORTABLE_DUPLICATE_ITEM"
        elif kind == "la_crossword":
            import unicodedata
            words = rows("words", StagedCrosswordWord, 3, 10)
            normalized = [re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFD", x["answer"].replace("đ", "D").replace("Đ", "D")).upper()) for x in words]
            if any(not 2 <= len(x) <= 24 for x in normalized) or not all(nonempty(x["clue"]) for x in words):
                return "CROSSWORD_TERM_INVALID"
            if len(set(normalized)) != len(normalized):
                return "CROSSWORD_DUPLICATE_TERM"
        elif kind == "la_diagram":
            nodes = rows("nodes", StagedDiagramNode, 2, 20)
            edges = rows("edges", StagedDiagramEdge, 1, 40)
            used: set[int] = set()
            for edge in edges:
                if edge["source"] >= len(nodes) or edge["target"] >= len(nodes) or edge["source"] == edge["target"]:
                    return "DIAGRAM_EDGE_INVALID"
                used.update([edge["source"], edge["target"]])
            if len(used) != len(nodes) or not all(nonempty(x["label"]) for x in nodes):
                return "DIAGRAM_DISCONNECTED_NODE"
        else:
            return "COMPONENT_TYPE_UNSUPPORTED"
    except (ValidationError, ValueError, TypeError):
        return "COMPONENT_PAYLOAD_SCHEMA_INVALID"
    return None


def staged_component_contract_prompt(component_types: list[str]) -> str:
    """Only selected contracts; storage/XML/layout/IDs remain server-owned."""
    contracts = {
        "html": 'html: use semantic_content={"sections":[{"heading":"Local topic","learning_block_ids":["exact approved learning block ID"],"blocks":[{"kind":"paragraph","text":"Substantive explanation"},{"kind":"table","rows":[{"label":"Source category","value":"Correct associated explanation"}]}]}]}. Preserve section and block order. Build clear visual hierarchy with concise section headings, short paragraphs, lists for parallel ideas or steps, semantic tables for comparisons/data, task blocks for learner action, and warning blocks for cautions; avoid walls of text. Use kind paragraph/task/warning with text; bullets/steps with items (nonempty strings); table with rows ({label,value}). The server supplies version 2; do not emit version or any top-level heading, paragraphs, bullet_points, ordered_steps, warnings or comparison_rows. Only sections is permitted inside semantic_content. Omit inactive fields or leave arrays empty/text null. 1-12 sections, 1-12 blocks per section. Aggregate limits across ALL sections unchanged: paragraph+task 12/2000 chars, bullets 20/800, steps 20/1000, warnings 8/1000, table rows 30 (label500/value1000); headings240. Bind every approved teaching learning block to a section with substantive explanation, not only a title or list of names. Explain ALL members of named frameworks, including the last member. HTML must NEVER contain an FAQ / Frequently Asked Questions / Câu hỏi thường gặp / Hỏi đáp thường gặp section, repeated Q:/A: pairs, or details/summary FAQ markup. Convert source questions into ordinary explanatory teaching; only a selected la_faq component may contain FAQ content. Keep tables/checklists directly under their own heading. For source-backed Canvas/action plans add a task specifying what learners must produce and evidence-based completion criteria; do not invent company targets or pretend submissions are stored. No CSS/classes/scripts/assets, internal IDs in visible text, unsupported facts or fabricated examples. Server renders HTML; other component types omit semantic_content.',
        "problem": 'problem: problem_type MUST be multiple_choice. Return one complete question and choices=[{"text":"source-grounded correct answer","correct":true},{"text":"plausible distinct distractor","correct":false}], with 3-6 distinct choices and EXACTLY one correct. Include a substantive explanation grounded in the taught evidence. Never emit multiple_select, dropdown, numerical, short_text, free-text answers, or raw problem XML.',
        "la_faq": 'la_faq: items=[{"question":"anticipated question","answer":"source-grounded clarification"}], 2-8 distinct Q&A. Clarify conditions/exceptions/misconceptions, not repeat paragraphs. Place FAQ last in the unit.',
        "la_sortable": 'la_sortable: question_text plus items=[{"text":"first step"},{"text":"second step"},{"text":"third step"}], 3-10 distinct items in SOURCE-CORRECT order. Only approved ordering practice. Do not fabricate dependencies or turn an unordered list into a sequence.',
        "la_crossword": 'la_crossword: words=[{"answer":"TERM","clue":"source-backed definition","hint":"optional"}], 3-10 distinct terms. Normalized spelling 2-24 letters/digits. Preserve meaning across EN/VI. Do not generate coordinates or invent terminology.',
        "la_diagram": 'la_diagram: name, nodes=[{"label":"source concept","shape":"rounded","tooltip":"optional"}], edges=[{"source":0,"target":1,"label":"source-supported relation"}]. Prefer 2-10 concise nodes over a crowded graph; 20 is the hard maximum. Use one short single-line label per node and edge: never emit literal \\n, /n, carriage returns or control characters. 1-40 edges; indices reference existing nodes; every node participates in a relationship. Shapes: rectangle/rounded/ellipse. Server supplies IDs, icons, spacing and orthogonal routing. Do not invent causal edges.',
    }
    return lesson_instructional_quality_policy() + "\nCOMPONENT CONTRACT " + STAGED_COMPONENT_CONTRACT_VERSION + "\nOnly populate fields for the current component type. In the SDK's combined selected-types envelope, use null/empty arrays for inapplicable required fields.\n" + "\n".join(
        contracts[t] for t in dict.fromkeys(component_types) if t in contracts
    )


def build_staged_lesson_content_response_model(
    component_types: list[str] | None = None,
    *, payload_only: bool = False, expected_unit_title: str | None = None,
    coverage_repair: bool = False,
    coverage_allowed_ids: list[str] | None = None,
) -> type[BaseModel]:
    """Build the Stage-2 typed response model for exactly the selected types.

    ``google-genai==1.0.0`` parses ``types.Schema`` responses by calling
    ``json.loads(response.text)`` without guarding a missing text value.  The
    Pydantic branch catches its own validation error and leaves ``parsed``
    empty, so the existing request-local validator can return a controlled
    failure rather than leaking the SDK ``TypeError`` as an HTTP 500.

    This model is deliberately used only by Stage-2 lesson content. Course
    Architect, chat, embeddings and legacy proposal generation retain their
    established schema paths.
    """
    selected_types = {
        normalized
        for value in (component_types or [])
        for normalized in [normalize_staged_component_type(value)]
        if normalized
    }
    if not selected_types:
        selected_types = {"html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"}

    component_fields: dict[str, Any] = {
        "component_plan_id": (str | None, None),
        "type": (Enum("StagedComponentType_" + "_".join(sorted(selected_types)), {t: t for t in sorted(selected_types)}, type=str), ...),
        "source_fact_ids": (list[str], ...),
        "covered_source_fact_ids": (list[str], ...),
        "supporting_evidence_fact_ids": (list[str], ...),
        "selection_rationale": (str | None, ...),
        "title": (str | None, ...),
    }
    if "html" in selected_types:
        component_fields.update({
            # Omission is permitted for other selected component types. A
            # non-nullable model reference survives SDK 1.0.0 serialization;
            # dict[str, Any] previously became an unconstrained OBJECT and
            # encouraged empty HTML in both generation and repair.
            "semantic_content": (StagedOrderedSemanticOutput, ... if selected_types == {"html"} else None),
            "html": (str | None, ...),
        })
    # google-genai 1.0.0 drops ``items`` when it lowers a nullable array
    # (``list[T] | None``) to its Gemini schema.  The provider then rejects
    # the request before generation.  Array fields therefore default to an
    # empty list when omitted instead of being nullable.  This keeps the
    # generated response compact while preserving a concrete ARRAY.items
    # contract on the wire; downstream component validators remain strict
    # about arrays that are actually required by a selected component.
    optional_array = lambda item_type: (list[item_type], Field(default_factory=list))
    if "problem" in selected_types:
        component_fields.update({
            "problem_type": (Literal["multiple_choice"] | None, ...),
            "question": (str | None, ...),
            "choices": optional_array(StagedChoice),
            "options": optional_array(str),
            "answer": (str | None, ...),
            "tolerance": (str | None, ...),
            "explanation": (str | None, ...),
        })
    if "la_faq" in selected_types:
        component_fields["items"] = optional_array(StagedFaqItem)
    if "la_sortable" in selected_types:
        component_fields.update({
            "question_text": (str | None, ...),
            "items": optional_array(StagedSortableItem),
            "ordered_items": optional_array(str),
            "steps": optional_array(str),
        })
    if "la_crossword" in selected_types:
        component_fields["words"] = optional_array(StagedCrosswordWord)
    if "la_diagram" in selected_types:
        component_fields.update({
            "name": (str | None, ...),
            # Mixed-type envelopes must allow empty arrays on other types.
            # A diagram-only repair can express its full cardinality contract.
            "nodes": (list[StagedDiagramNode], Field(min_length=2, max_length=20)) if selected_types == {"la_diagram"} else optional_array(StagedDiagramNode),
            "edges": (list[StagedDiagramEdge], Field(min_length=1, max_length=40)) if selected_types == {"la_diagram"} else optional_array(StagedDiagramEdge),
        })

    if {"la_faq", "la_sortable"}.issubset(selected_types):
        # SDK 1.0.0 rejects anyOf. Explicit nullable fields are safe on the
        # wire; the per-type validator above requires the relevant fields.
        item_model = create_model("StagedFaqOrSortableItem", question=(str | None, ...), answer=(str | None, ...), text=(str | None, ...))
        component_fields["items"] = optional_array(item_model)

    if payload_only:
        # A repair selects an already-authorized array address, not provenance.
        # Never ask Gemini to echo canonical fact IDs, type or unit metadata.
        for key in ("type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids"):
            component_fields.pop(key, None)
        component_fields["component_index"] = (int, ...)
        if coverage_repair:
            # A checked content claim, never canonical ownership. Only explicitly
            # authorized coverage targets may emit this optional field.
            component_fields["covered_source_fact_ids"] = (list[str], Field(default_factory=list))
            if coverage_allowed_ids:
                allowed = Enum("StagedRepairCoverageReference", {f"F{i}": value for i, value in enumerate(dict.fromkeys(coverage_allowed_ids))}, type=str)
                component_fields["covered_source_fact_ids"] = (list[allowed], Field(default_factory=list))
    component_name = ("StagedRepairPayload_" if payload_only else "StagedLessonComponent_") + "_".join(sorted(selected_types))
    component_model = create_model(component_name, **component_fields)
    if payload_only:
        return create_model("StagedComponentRepairDelta", components=(list[component_model], ...))
    return create_model(
        "StagedLessonUnit_" + "_".join(sorted(selected_types)),
        title=(Enum("StagedUnitTitle", {"APPROVED": expected_unit_title}, type=str) if expected_unit_title else str, ...),
        source_fact_ids=(list[str], ...),
        supporting_evidence_fact_ids=(list[str], ...),
        components=(list[component_model], ...),
    )


class StagedMultiRepairResponse(BaseModel):
    # SDK/Pydantic may silently ignore extra fields or collapse duplicate keys.
    # This contract checks the original text locally; never log that text.
    retain_raw_provider_text: ClassVar[bool] = True


def build_staged_multi_repair_model(baseline: dict[str, Any], targets: list[int], coverage_targets: list[int]) -> type[BaseModel]:
    """Required, separately typed addresses survive the SDK bounds projection.

    Single-target/legacy repair contracts remain unchanged. No provider-owned
    indices or union-of-types fields are needed for multi-component repair.
    """
    originals = baseline.get("components", [])
    if (not targets or len(set(targets)) != len(targets)
            or any(type(i) is not int or not 0 <= i < len(originals) for i in targets)
            or len(set(coverage_targets)) != len(coverage_targets) or not set(coverage_targets) <= set(targets)):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_INVALID")
    slots = {}
    for index in targets:
        original = originals[index]
        kind = original["type"]
        if kind not in STAGED_COMPONENT_PAYLOAD_FIELDS:
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_INVALID")
        envelope = build_staged_lesson_content_response_model(
            [kind], payload_only=True, coverage_repair=index in coverage_targets,
            coverage_allowed_ids=original.get("source_fact_ids", []))
        component = envelope.model_fields["components"].annotation.__args__[0]
        fields = {k: (f.annotation, deepcopy(f)) for k, f in component.model_fields.items() if k != "component_index"}
        if index in coverage_targets:
            fields["covered_source_fact_ids"] = (component.model_fields["covered_source_fact_ids"].annotation, ...)
        slots[f"c{index}"] = (create_model(f"StagedRepairSlot{index}_{kind}", **fields), ...)
    return create_model("StagedMultiRepair", __base__=StagedMultiRepairResponse,
                        components=(create_model("StagedRepairSlots", **slots), ...))


def decode_staged_multi_repair(text: str, baseline: dict[str, Any], targets: list[int],
                               coverage_targets: list[int], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Decode exact slots before atomic merge; diagnostics never expose values."""
    expected = {f"c{i}" for i in targets}
    diagnostics.update(expected_slot_count=len(expected), received_slot_count=None,
                       missing_slot_count=None, unexpected_slot_count=None, duplicate_key_count=0)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                diagnostics["duplicate_key_count"] = 1
                raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, TypeError):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_JSON_INVALID") from None
    if not isinstance(value, dict) or set(value) != {"components"}:
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_ENVELOPE_FORBIDDEN")
    slots = value["components"]
    diagnostics["response_container"] = "object" if isinstance(slots, dict) else "array" if isinstance(slots, list) else "other"
    diagnostics["received_slot_count"] = len(slots) if isinstance(slots, (dict, list)) else None
    if not isinstance(slots, dict):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_SLOTS_REQUIRED")
    diagnostics.update(missing_slot_count=len(expected - set(slots)), unexpected_slot_count=len(set(slots) - expected))
    if set(slots) != expected:
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_SLOT_INVENTORY_INVALID")
    changes = []
    for index in targets:
        payload = slots[f"c{index}"]
        if not isinstance(payload, dict):
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_NOT_OBJECT")
        kind = baseline["components"][index]["type"]
        allowed = STAGED_COMPONENT_PAYLOAD_FIELDS[kind] | {"title", "selection_rationale"}
        if index in coverage_targets:
            allowed |= {"covered_source_fact_ids"}
        if set(payload) - allowed:
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_SLOT_FIELD_FORBIDDEN")
        changes.append({"component_index": index, **deepcopy(payload)})
    # Use the established strict merge/ownership/coverage/quality checks next.
    return {"components": changes}


def build_staged_instance_response_model(plans: list[dict[str, Any]]) -> type[BaseModel]:
    """Each server-addressed slot has one concrete payload schema, never a union.

    Ownership/identity stay server-owned. Coverage remains a provider claim and
    is NOT filled by the server; normal evidence/quality acceptance still runs.
    """
    validate_instance_plan(plans)
    slots = {}
    for index, plan in enumerate(plans):
        kind = plan["type"]
        envelope = build_staged_lesson_content_response_model([kind])
        component = envelope.model_fields["components"].annotation.__args__[0]
        allowed = STAGED_COMPONENT_PAYLOAD_FIELDS[kind] | {"title", "selection_rationale", "covered_source_fact_ids"}
        fields = {key: (field.annotation, deepcopy(field)) for key, field in component.model_fields.items() if key in allowed}
        if kind == "la_faq":
            fields["items"] = (list[StagedFaqItem], Field(min_length=2, max_length=8))
        elif kind == "la_sortable":
            fields["items"] = (list[StagedSortableItem], Field(min_length=3, max_length=10))
            fields["question_text"] = (str, Field(min_length=1))
        elif kind == "la_crossword":
            fields["words"] = (list[StagedCrosswordWord], Field(min_length=3, max_length=10))
        elif kind == "problem":
            fields["question"] = (str, Field(min_length=1))
            fields["problem_type"] = (Literal["multiple_choice"], ...)
            fields["choices"] = (list[StagedChoice], Field(min_length=3, max_length=6))
            fields["explanation"] = (str, Field(min_length=20, max_length=1500))
        if not plan.get("source_fact_ids"):
            fields["covered_source_fact_ids"] = (list[str], Field(max_length=0))
        else:
            # References are model claims, not ownership. Exact approved values
            # constrain the wire; raw fallback and final validators stay strict.
            allowed = Enum(f"StagedCoverageReference{index}", {f"F{i}": value for i, value in enumerate(plan["source_fact_ids"])}, type=str)
            fields["covered_source_fact_ids"] = (list[allowed], ...)
        slots[f"c{index}"] = (create_model(f"StagedInstance{index}_{kind}", **fields), ...)
    return create_model("StagedInstancePayloadUnit", components=(create_model("StagedInstanceSlots", **slots), ...))


def bind_staged_instance_payload(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    """Bind exact slots to the approved plan, without repairing content or claims."""
    def reject(reason: str, path: str) -> None:
        raise WorkflowFailure("LESSON_VALIDATION_FAILED", "Invalid component instance response.",
                              internal_code="CHAPTER_UNIT_CONTRACT_REJECTED", failure_stage="chapter_component_binding",
                              diagnostics={"validation_finding": {"code": reason, "path": path, "repairable": False}})
    if not isinstance(value, dict) or set(value) != {"components"}:
        reject("INSTANCE_ENVELOPE_INVALID", "unit")
    slots = value["components"]
    plans = expected["component_plan"]
    if not isinstance(slots, dict) or set(slots) != {f"c{i}" for i in range(len(plans))}:
        reject("INSTANCE_SLOT_INVENTORY_INVALID", "unit.components")
    components = []
    for index, plan in enumerate(plans):
        payload = slots[f"c{index}"]
        path = f"components[{index}]"
        if not isinstance(payload, dict):
            reject("INSTANCE_PAYLOAD_NOT_OBJECT", path)
        allowed = STAGED_COMPONENT_PAYLOAD_FIELDS[plan["type"]] | {"title", "selection_rationale", "covered_source_fact_ids"}
        if set(payload) - allowed:
            reject("INSTANCE_FIELD_NOT_ALLOWED", path)
        components.append({**deepcopy(payload), "type": plan["type"], "component_plan_id": plan["component_plan_id"],
                           "source_fact_ids": list(plan.get("source_fact_ids", [])),
                           "supporting_evidence_fact_ids": list(plan.get("supporting_evidence_fact_ids", []))})
    return {"title": expected["unit_title"], "source_fact_ids": list(expected.get("source_fact_ids", [])),
            "supporting_evidence_fact_ids": list(expected.get("supporting_evidence_fact_ids", [])), "components": components}


def staged_response_schema_diagnostics(response_schema: types.Schema | type[BaseModel]) -> dict[str, Any]:
    """Return safe shape metadata for a Stage-2 provider contract.

    This is a local guard against malformed schemas only. Source, component,
    and pedagogical validators remain authoritative after generation.
    """

    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        schema = response_schema.model_json_schema()
        adapter = "pydantic-v2"
    elif isinstance(response_schema, types.Schema):
        schema = response_schema.model_dump(exclude_none=True)
        adapter = "google-schema"
    else:
        return {
            "schema_adapter": "unknown",
            "schema_valid": False,
            "array_field_count": 0,
            "array_missing_items_count": 0,
            "nullable_array_count": 0,
            "safe_invalid_schema_paths": ["response_schema"],
            "provider_schema_fingerprint": "unavailable",
        }

    missing_items: list[str] = []
    nullable_arrays: list[str] = []
    array_count = 0

    def visit(value: Any, path: str) -> None:
        nonlocal array_count
        if isinstance(value, dict):
            declared_type = str(value.get("type") or "").lower()
            variants = value.get("anyOf")
            nullable_array = (
                isinstance(variants, list)
                and any(isinstance(item, dict) and str(item.get("type") or "").lower() == "array" for item in variants)
                and any(isinstance(item, dict) and str(item.get("type") or "").lower() == "null" for item in variants)
            )
            if declared_type == "array":
                array_count += 1
                if "items" not in value:
                    missing_items.append(path)
            if nullable_array:
                nullable_arrays.append(path)
            for key, child in value.items():
                if key in {"$defs", "properties"} and isinstance(child, dict):
                    for child_key, child_value in child.items():
                        visit(child_value, f"{path}.{child_key}" if path else str(child_key))
                elif key in {"items", "anyOf"}:
                    if isinstance(child, list):
                        for index, child_value in enumerate(child):
                            visit(child_value, f"{path}.{key}[{index}]")
                    else:
                        visit(child, f"{path}.{key}")

    visit(schema, "response_schema")
    canonical = json.dumps(schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_adapter": adapter,
        "schema_valid": not missing_items and not nullable_arrays,
        "array_field_count": array_count,
        "array_missing_items_count": len(missing_items),
        "nullable_array_count": len(nullable_arrays),
        "safe_invalid_schema_paths": sorted((missing_items + nullable_arrays))[:12],
        "provider_schema_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def is_stage_two_provider_schema_error(error: Exception) -> bool:
    """Recognize a provider-side response-schema rejection without logging it."""

    return safe_provider_error_diagnostics(error)["provider_error_category"] == "RESPONSE_SCHEMA_INVALID"
