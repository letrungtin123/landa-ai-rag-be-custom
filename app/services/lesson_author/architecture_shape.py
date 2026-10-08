"""Course-architecture paths, workflow issues and V5 shape helpers shared by validation and repair."""

from __future__ import annotations

import re
from typing import Any, Literal

from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError
from app.workflows.contracts import WorkflowIssue


def _workflow_issue(
    code: str,
    message: str,
    *,
    severity: Literal["error", "warning", "info"] = "error",
    path: str = "course",
    related_paths: list[str] | None = None,
    constraint: str | None = None,
    expected_type: str | None = None,
    actual_type: str | None = None,
    validator: str | None = None,
    schema_error_code: str | None = None,
    schema_path: str | None = None,
) -> WorkflowIssue:
    issue: WorkflowIssue = {"code": code, "severity": severity, "message": message, "path": path}
    if related_paths:
        issue["related_paths"] = related_paths
    if constraint:
        issue["constraint"] = constraint
    if expected_type:
        issue["expected_type"] = expected_type
    if actual_type:
        issue["actual_type"] = actual_type
    if validator:
        issue["validator"] = validator
    if schema_error_code:
        issue["schema_error_code"] = schema_error_code
    if schema_path:
        issue["schema_path"] = schema_path
    return issue


def _workflow_path_from_blueprint_schema_path(schema_path: str) -> str:
    """Map a safe JSON-style schema path to the smallest repairable node."""

    match = re.match(
        r"^chapters\[(\d+)\](?:\.lessons\[(\d+)\](?:\.units\[(\d+)\])?)?",
        schema_path,
    )
    if match is None:
        return "course"
    path = f"chapter_{int(match.group(1)) + 1}"
    if match.group(2) is not None:
        path += f".lesson_{int(match.group(2)) + 1}"
    if match.group(3) is not None:
        path += f".unit_{int(match.group(3)) + 1}"
    return path


def _workflow_issue_from_blueprint_validation_error(
    error: LessonAuthorBlueprintValidationError,
) -> WorkflowIssue:
    """Preserve content-safe parser diagnostics for scoped repair and logs."""

    diagnostic = error.safe_diagnostic()
    return _workflow_issue(
        error.code,
        "Blueprint structural validation failed.",
        path=_workflow_path_from_blueprint_schema_path(diagnostic["path"]),
        schema_path=diagnostic["path"],
        constraint=diagnostic["constraint"],
        expected_type=diagnostic["expected_type"],
        actual_type=diagnostic["actual_type"],
        validator=diagnostic["validator"],
        schema_error_code=diagnostic["error_code"],
    )


def _blueprint_path_object(blueprint: dict[str, Any], path: str) -> dict[str, Any] | None:
    """Resolve only deterministic chapter/lesson/unit repair paths."""
    match = re.fullmatch(r"course|chapter_(\d+)(?:\.lesson_(\d+)(?:\.unit_(\d+))?)?", path)
    if match is None:
        return None
    if path == "course":
        return blueprint
    chapters = blueprint.get("chapters") if isinstance(blueprint.get("chapters"), list) else []
    chapter_index = int(match.group(1)) - 1
    if not 0 <= chapter_index < len(chapters) or not isinstance(chapters[chapter_index], dict):
        return None
    node: dict[str, Any] = chapters[chapter_index]
    if match.group(2) is None:
        return node
    lessons = node.get("lessons") if isinstance(node.get("lessons"), list) else []
    lesson_index = int(match.group(2)) - 1
    if not 0 <= lesson_index < len(lessons) or not isinstance(lessons[lesson_index], dict):
        return None
    node = lessons[lesson_index]
    if match.group(3) is None:
        return node
    units = node.get("units") if isinstance(node.get("units"), list) else []
    unit_index = int(match.group(3)) - 1
    return units[unit_index] if 0 <= unit_index < len(units) and isinstance(units[unit_index], dict) else None


def _safe_json_shape(value: Any) -> str:
    """Return a non-sensitive JSON shape for structured validation logs."""

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


def _is_v4_supporting_factless_unit(unit: dict[str, Any]) -> bool:
    """Return whether a factless v4 unit is proven to be non-primary.

    This is intentionally narrow.  A unit may be factless only after the
    server allocator has completed globally and only when neither the unit
    nor any retained semantic block claims primary instructional ownership.
    It is not a fallback for an incomplete primary allocation.
    """

    blocks = unit.get("learning_blocks")
    return (
        isinstance(blocks, list)
        and bool(blocks)
        and not bool(unit.get("primary_concept_ids"))
        and all(
            isinstance(block, dict) and not bool(block.get("primary_concept_ids"))
            for block in blocks
        )
    )


def _is_v5_supporting_factless_unit(unit: dict[str, Any]) -> bool:
    """A v5 support unit has no primary evidence-scope ownership."""
    blocks = unit.get("learning_blocks")
    return (
        isinstance(blocks, list)
        and bool(blocks)
        and not bool(unit.get("source_fact_ids"))
        and all(
            isinstance(block, dict) and not bool(block.get("primary_evidence_scope_ids"))
            for block in blocks
        )
    )


_V5_TEACHING_INTENTS = {
    "concept_explanation", "definition", "example", "worked_example",
    "procedure", "comparison", "warning", "tip",
}


_V5_GENERIC_EXPLANATION_INTENTS = {"concept_explanation", "definition", "introduction"}


_V5_ACTION_OR_PROCEDURE_OBJECTIVE = re.compile(
    r"\b(?:apply|perform|demonstrate|execute|practice|procedure|process|"
    r"áp\s+dụng|thực\s+hiện|thực\s+hành|quy\s+trình|vận\s+hành)\b",
    flags=re.IGNORECASE,
)


def _v5_text_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    }


def _repair_parent_lesson_path(path: str) -> str:
    return path.rsplit(".unit_", 1)[0] if ".unit_" in path else path


def _semantic_delta_blocks(unit: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = unit.get("learning_blocks")
    return [block for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []
