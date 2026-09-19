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
BLUEPRINT_COMPONENT_TYPES = {
    "html",
    "problem",
    "la_faq",
    "la_sortable",
    "la_crossword",
    "la_diagram",
}
BLUEPRINT_MEDIA_TYPES = {"video", "static_infographic"}


class LessonAuthorBlueprintComponentPlanResponse(BaseModel):
    type: str
    title: str
    rationale: str


class LessonAuthorBlueprintMediaPlanResponse(BaseModel):
    type: str
    title: str
    content_outline: str
    rationale: str


class LessonAuthorBlueprintUnitResponse(BaseModel):
    title: str
    component_plan: list[LessonAuthorBlueprintComponentPlanResponse]
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


class LessonAuthorBlueprintChapterResponse(BaseModel):
    title: str
    objective: str
    lessons: list[LessonAuthorBlueprintLessonResponse]
    source_refs: list[str] = Field(default_factory=list)


class LessonAuthorBlueprintResponse(BaseModel):
    title: str
    summary: str
    target_audience: str
    prerequisites: list[str]
    learning_outcomes: list[str]
    assessment_strategy: str
    assumptions: list[str]
    chapters: list[LessonAuthorBlueprintChapterResponse]


class LessonAuthorBlueprintValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def build_lesson_author_blueprint_response_schema() -> types.Schema:
    component_plan_schema = types.Schema(
        type=types.Type.OBJECT,
        description="A planned learning component. This is architecture only, never component payload data.",
        required=["type", "title", "rationale"],
        properties={
            "type": _string_schema("One supported type: html, problem, la_faq, la_sortable, la_crossword, or la_diagram."),
            "title": _string_schema("Short learner-facing component title."),
            "rationale": _string_schema("A concise reason this format fits the lesson objective and source evidence."),
        },
    )
    media_plan_schema = types.Schema(
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
    unit_schema = types.Schema(
        type=types.Type.OBJECT,
        description="A draftable learning unit inside a lesson.",
        required=["title", "component_plan"],
        properties={
            "title": _string_schema("Semantic unit title without numbering."),
            "component_plan": types.Schema(
                type=types.Type.ARRAY,
                description="One to three planned component types, always including one html explanation.",
                items=component_plan_schema,
            ),
            "source_refs": _string_array_schema(
                "Optional source outline references supporting this unit.",
            ),
            "source_fact_ids": _string_array_schema(
                "Optional internal source fact IDs allocated by the server for complete detailed authoring.",
            ),
            "media_plan": media_plan_schema,
        },
    )
    lesson_schema = types.Schema(
        type=types.Type.OBJECT,
        description="A concise lesson inside one course chapter.",
        required=["title", "objective", "learning_activities", "assessment", "units"],
        properties={
            "title": _string_schema("Lesson title."),
            "objective": _string_schema("Measurable lesson objective."),
            "learning_activities": _string_array_schema(
                "One to three learner activities that support the objective.",
            ),
            "assessment": _string_schema("How the lesson objective is checked."),
            "units": types.Schema(
                type=types.Type.ARRAY,
                description="One to three ordered learning units, each with a reviewable component plan.",
                items=unit_schema,
            ),
            "source_refs": _string_array_schema(
                "Optional source outline references such as src-001. Use only references supplied by the server.",
            ),
        },
    )
    chapter_schema = types.Schema(
        type=types.Type.OBJECT,
        description="A coherent chapter in the course Blueprint.",
        required=["title", "objective", "lessons"],
        properties={
            "title": _string_schema("Chapter title."),
            "objective": _string_schema("Measurable chapter objective."),
            "source_refs": _string_array_schema(
                "Optional source outline references supporting this chapter.",
            ),
            "lessons": types.Schema(
                type=types.Type.ARRAY,
                description="Lessons ordered from foundation to application.",
                items=lesson_schema,
            ),
        },
    )
    return types.Schema(
        type=types.Type.OBJECT,
        description="A review-only enterprise course Blueprint based on supplied source material.",
        required=[
            "title",
            "summary",
            "target_audience",
            "prerequisites",
            "learning_outcomes",
            "assessment_strategy",
            "assumptions",
            "chapters",
        ],
        properties={
            "title": _string_schema("Course title."),
            "summary": _string_schema("Concise course design summary."),
            "target_audience": _string_schema("Primary intended learners."),
            "prerequisites": _string_array_schema(
                "Learner prerequisites. Use an empty array when none are needed.",
            ),
            "learning_outcomes": _string_array_schema(
                "Three to twelve measurable learning outcomes.",
            ),
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


LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA = build_lesson_author_blueprint_response_schema()
LESSON_AUTHOR_BLUEPRINT_RESPONSE_MODEL = LessonAuthorBlueprintResponse


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must be a JSON object.",
        )
    return value


def _require_text(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must be text.",
        )
    text = value.strip()
    if not text or len(text) > max_length:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"{label} must contain concise text.",
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
        item_max_length=32,
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


def validate_lesson_author_blueprint(value: Any) -> dict[str, Any]:
    raw = _require_object(value, "Blueprint")
    chapters_raw = raw.get("chapters")
    if not isinstance(chapters_raw, list) or not 1 <= len(chapters_raw) <= MAX_BLUEPRINT_CHAPTERS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"chapters must contain 1 to {MAX_BLUEPRINT_CHAPTERS} items.",
        )

    chapters: list[dict[str, Any]] = []
    total_lessons = 0
    total_units = 0
    total_components = 0
    total_media_plans = 0
    for chapter_index, chapter_value in enumerate(chapters_raw):
        chapter = _require_object(chapter_value, f"chapters[{chapter_index}]")
        lessons_raw = chapter.get("lessons")
        if not isinstance(lessons_raw, list) or not 1 <= len(lessons_raw) <= MAX_BLUEPRINT_LESSONS_PER_CHAPTER:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_INVALID_SCHEMA",
                f"chapters[{chapter_index}].lessons must contain 1 to {MAX_BLUEPRINT_LESSONS_PER_CHAPTER} items.",
            )
        lessons: list[dict[str, Any]] = []
        total_lessons += len(lessons_raw)
        for lesson_index, lesson_value in enumerate(lessons_raw):
            lesson = _require_object(lesson_value, f"chapters[{chapter_index}].lessons[{lesson_index}]")
            units_raw = lesson.get("units")
            if not isinstance(units_raw, list) or not 1 <= len(units_raw) <= MAX_BLUEPRINT_UNITS_PER_LESSON:
                raise LessonAuthorBlueprintValidationError(
                    "BLUEPRINT_INVALID_SCHEMA",
                    f"lesson.units must contain 1 to {MAX_BLUEPRINT_UNITS_PER_LESSON} items.",
                )
            units: list[dict[str, Any]] = []
            total_units += len(units_raw)
            for unit_index, unit_value in enumerate(units_raw):
                unit = _require_object(unit_value, f"lesson.units[{unit_index}]")
                component_plan_raw = unit.get("component_plan")
                if (
                    not isinstance(component_plan_raw, list)
                    or not 1 <= len(component_plan_raw) <= MAX_BLUEPRINT_COMPONENTS_PER_UNIT
                ):
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_INVALID_SCHEMA",
                        f"lesson.units[{unit_index}].component_plan must contain 1 to {MAX_BLUEPRINT_COMPONENTS_PER_UNIT} items.",
                    )
                component_plan: list[dict[str, Any]] = []
                total_components += len(component_plan_raw)
                seen_component_types: set[str] = set()
                for plan_index, plan_value in enumerate(component_plan_raw):
                    plan = _require_object(plan_value, f"lesson.units[{unit_index}].component_plan[{plan_index}]")
                    component_type = _require_text(
                        plan.get("type"),
                        "component_plan.type",
                        40,
                    ).casefold()
                    if component_type not in BLUEPRINT_COMPONENT_TYPES:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            f"component_plan.type must be one of: {', '.join(sorted(BLUEPRINT_COMPONENT_TYPES))}.",
                        )
                    if component_type in seen_component_types:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            "component_plan must not repeat a component type within one unit.",
                        )
                    seen_component_types.add(component_type)
                    component_plan.append(
                        {
                            "type": component_type,
                            "title": _require_text(plan.get("title"), "component_plan.title", 180),
                            "rationale": _require_text(plan.get("rationale"), "component_plan.rationale", 240),
                        }
                    )
                if "html" not in seen_component_types:
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_INVALID_SCHEMA",
                        "lesson.units.component_plan must include one html explanation component.",
                    )
                non_faq_component_count = sum(
                    component_type != "la_faq"
                    for component_type in seen_component_types
                )
                if non_faq_component_count > 2:
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_INVALID_SCHEMA",
                        "lesson.units.component_plan may contain html and at most one additional interactive component before FAQ.",
                    )
                raw_media_plan = unit.get("media_plan")
                media_plan: dict[str, str] | None = None
                if raw_media_plan is not None:
                    media_record = _require_object(raw_media_plan, f"lesson.units[{unit_index}].media_plan")
                    media_type = _require_text(media_record.get("type"), "media_plan.type", 40).casefold()
                    if media_type not in BLUEPRINT_MEDIA_TYPES:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            f"media_plan.type must be one of: {', '.join(sorted(BLUEPRINT_MEDIA_TYPES))}.",
                        )
                    media_plan = {
                        "type": media_type,
                        "title": _require_text(media_record.get("title"), "media_plan.title", 180),
                        "content_outline": _require_text(media_record.get("content_outline"), "media_plan.content_outline", 600),
                        "rationale": _require_text(media_record.get("rationale"), "media_plan.rationale", 240),
                    }
                    total_media_plans += 1
                units.append(
                    {
                        "title": _require_text(
                            strip_source_range_suffix(str(unit.get("title") or "")),
                            "unit.title",
                            180,
                        ),
                        "component_plan": component_plan,
                        "source_refs": _optional_source_refs(unit.get("source_refs"), "unit.source_refs"),
                        "source_fact_ids": _require_text_array(
                            unit.get("source_fact_ids") or [],
                            "unit.source_fact_ids",
                            min_items=0,
                            max_items=160,
                            item_max_length=96,
                        ),
                        **({"media_plan": media_plan} if media_plan is not None else {}),
                    }
                )
            lessons.append(
                {
                    "title": _require_text(
                        strip_source_range_suffix(str(lesson.get("title") or "")),
                        "lesson.title",
                        180,
                    ),
                    "objective": _require_text(lesson.get("objective"), "lesson.objective", 500),
                    "learning_activities": _require_text_array(
                        lesson.get("learning_activities"),
                        "lesson.learning_activities",
                        min_items=1,
                        max_items=MAX_BLUEPRINT_ACTIVITIES_PER_LESSON,
                        item_max_length=280,
                    ),
                    "assessment": _require_text(lesson.get("assessment"), "lesson.assessment", 500),
                    "units": units,
                    "source_refs": _optional_source_refs(
                        lesson.get("source_refs"),
                        "lesson.source_refs",
                    ),
                }
            )
        chapters.append(
            {
                "title": _require_text(
                    strip_source_range_suffix(str(chapter.get("title") or "")),
                    "chapter.title",
                    220,
                ),
                "objective": _require_text(chapter.get("objective"), "chapter.objective", 500),
                "lessons": lessons,
                "source_refs": _optional_source_refs(chapter.get("source_refs"), "chapter.source_refs"),
            }
        )

    if total_lessons > MAX_BLUEPRINT_TOTAL_LESSONS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"Blueprint must contain at most {MAX_BLUEPRINT_TOTAL_LESSONS} lessons in total.",
        )
    if total_units > MAX_BLUEPRINT_TOTAL_UNITS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"Blueprint must contain at most {MAX_BLUEPRINT_TOTAL_UNITS} units in total.",
        )
    if total_components > MAX_BLUEPRINT_TOTAL_COMPONENTS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"Blueprint must contain at most {MAX_BLUEPRINT_TOTAL_COMPONENTS} component plans in total.",
        )
    if total_media_plans > MAX_BLUEPRINT_MEDIA_PLANS:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_INVALID_SCHEMA",
            f"Blueprint must contain at most {MAX_BLUEPRINT_MEDIA_PLANS} media plans in total.",
        )

    return {
        "title": _require_text(raw.get("title"), "title", 220),
        "summary": _require_text(raw.get("summary"), "summary", 1400),
        "target_audience": _require_text(raw.get("target_audience"), "target_audience", 500),
        "prerequisites": _require_text_array(
            raw.get("prerequisites"),
            "prerequisites",
            min_items=0,
            max_items=MAX_BLUEPRINT_PREREQUISITES,
            item_max_length=280,
        ),
        "learning_outcomes": _require_text_array(
            raw.get("learning_outcomes"),
            "learning_outcomes",
            min_items=3,
            max_items=MAX_BLUEPRINT_LEARNING_OUTCOMES,
            item_max_length=500,
        ),
        "assessment_strategy": _require_text(raw.get("assessment_strategy"), "assessment_strategy", 900),
        "assumptions": _require_text_array(
            raw.get("assumptions"),
            "assumptions",
            min_items=0,
            max_items=MAX_BLUEPRINT_ASSUMPTIONS,
            item_max_length=400,
        ),
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


def parse_and_validate_lesson_author_blueprint(text: str) -> dict[str, Any]:
    last_validation_error: LessonAuthorBlueprintValidationError | None = None
    for parsed in _decode_blueprint_json(text):
        try:
            return validate_lesson_author_blueprint(normalize_lesson_author_blueprint_candidate(parsed))
        except LessonAuthorBlueprintValidationError as error:
            last_validation_error = error
    if last_validation_error is not None:
        raise last_validation_error
    raise LessonAuthorBlueprintValidationError(
        "BLUEPRINT_INVALID_JSON",
        "Blueprint response is not valid JSON.",
    )
