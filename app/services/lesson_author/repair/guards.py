"""Repair guards: V5 provenance, source scope, unaffected-snapshot and patch-domain checks."""

from __future__ import annotations

import re
from typing import Any

from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError, validate_lesson_author_blueprint
from app.services.lesson_author.architecture_shape import _blueprint_path_object, _v5_text_ids
from app.services.lesson_author.repair.prompt import _collect_nested_source_values, _semantic_architecture_snapshot
from app.workflows.contracts import WorkflowFailure, safe_workflow_path


def _v5_provenance_guard_failure(
    path: tuple[str | int, ...], reason: str, *, patch_count: int,
    outer_field: str = "learning_blocks",
) -> WorkflowFailure:
    # Derive only a structural address. Tuple keys may contain private block
    # IDs; neither keys nor values are interpolated into diagnostics/messages.
    parts: list[str] = []
    for key, label in (("chapters", "chapter"), ("lessons", "lesson"), ("units", "unit")):
        offset = len(parts) * 2
        if len(path) > offset + 1 and path[offset] == key and isinstance(path[offset + 1], int):
            parts.append(f"{label}_{path[offset + 1] + 1}")
        else:
            break
    return _repair_scope_violation(
        "Semantic repair did not preserve existing block identity, order or provenance.",
        path=".".join(parts) or "course", patch_count=patch_count,
        internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
        failure_stage="architecture_repair_semantic_delta_guard",
        guard_reason=reason, outer_field=outer_field,
    )


def _v5_primary_provenance_snapshot(value: Any, *, patch_count: int = 0) -> tuple[
    dict[tuple[str | int, ...], tuple[tuple[str, ...], tuple[str, ...]]],
    dict[tuple[str | int, ...], tuple[str, ...]],
]:
    """Unit-scoped block identity, plus original order; never array-index identity.

    Other hierarchy/metadata arrays retain positional protection. Only direct
    unit.learning_blocks receive ID addressing; no provider ID can escape its
    parent tuple, and duplicate/missing IDs never silently collapse a snapshot.
    """
    snapshot: dict[tuple[str | int, ...], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    orders: dict[tuple[str | int, ...], tuple[str, ...]] = {}

    def walk(node: Any, path: tuple[str | int, ...], *, block: bool = False) -> None:
        if isinstance(node, dict):
            if block or "primary_evidence_scope_ids" in node or "source_refs" in node:
                snapshot[path] = (
                    tuple(sorted(_v5_text_ids(node.get("primary_evidence_scope_ids")))),
                    tuple(sorted(_v5_text_ids(node.get("source_refs")))),
                )
            for key, child in node.items():
                child_path = (*path, key)
                if key == "learning_blocks" and len(path) == 6 and path[0::2] == ("chapters", "lessons", "units"):
                    if not isinstance(child, list):
                        raise _v5_provenance_guard_failure(child_path, "INVALID_BLOCK_COLLECTION", patch_count=patch_count)
                    identifiers: list[str] = []
                    seen: set[str] = set()
                    for item in child:
                        identifier = item.get("id") if isinstance(item, dict) else None
                        if not isinstance(identifier, str) or not identifier.strip():
                            raise _v5_provenance_guard_failure(child_path, "MISSING_BLOCK_ID", patch_count=patch_count)
                        # Existing target resolution uses stripped IDs. Reject
                        # aliases here, but retain the exact ID in the snapshot.
                        if identifier.strip() in seen:
                            raise _v5_provenance_guard_failure(child_path, "DUPLICATE_BLOCK_ID", patch_count=patch_count)
                        seen.add(identifier.strip())
                        identifiers.append(identifier)
                        walk(item, (*child_path, identifier), block=True)
                    orders[child_path] = tuple(identifiers)
                else:
                    walk(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, (*path, index))

    walk(value, ())
    return snapshot, orders


def _assert_v5_primary_provenance_preserved(
    baseline: dict[str, Any], candidate: dict[str, Any], *, patch_count: int = 0,
) -> None:
    before, before_orders = _v5_primary_provenance_snapshot(baseline, patch_count=patch_count)
    after, after_orders = _v5_primary_provenance_snapshot(candidate, patch_count=patch_count)
    for path, identifiers in before_orders.items():
        current = after_orders.get(path)
        if current is None or not set(identifiers).issubset(current):
            raise _v5_provenance_guard_failure(path, "EXISTING_BLOCK_MISSING_OR_MOVED", patch_count=patch_count)
        original_ids = set(identifiers)
        if tuple(identifier for identifier in current if identifier in original_ids) != identifiers:
            raise _v5_provenance_guard_failure(path, "EXISTING_BLOCK_ORDER_CHANGED", patch_count=patch_count)
    for path, provenance in before.items():
        if after.get(path) != provenance:
            field = "primary_evidence_scope_ids" if path not in after or after[path][0] != provenance[0] else "source_refs"
            raise _v5_provenance_guard_failure(path, "PRIMARY_OWNERSHIP_MUTATION", patch_count=patch_count, outer_field=field)
    for path, identifiers in after_orders.items():
        original_ids = set(before_orders.get(path, ()))
        for identifier in identifiers:
            if identifier not in original_ids and after[(*path, identifier)][0]:
                raise _v5_provenance_guard_failure(path, "NEW_BLOCK_PRIMARY_OWNERSHIP", patch_count=patch_count, outer_field="primary_evidence_scope_ids")


def _contains_provider_fact_ownership(value: Any) -> bool:
    if isinstance(value, dict):
        if {"source_fact_ids", "covered_source_fact_ids", "source_fact_allocation", "source_evidence_scope_allocation"}.intersection(value):
            return True
        return any(_contains_provider_fact_ownership(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_provider_fact_ownership(child) for child in value)
    return False


def _repair_keeps_existing_source_scope(
    original: dict[str, Any],
    replacement: dict[str, Any],
    *,
    allowed_evidence_scope_ids: set[str] | None = None,
) -> bool:
    """A repair can refine only the server-approved semantic source scope."""
    for key in ("source_fact_ids", "covered_source_fact_ids", "source_refs"):
        allowed = _collect_nested_source_values(original, key)
        proposed = _collect_nested_source_values(replacement, key)
        if proposed and not proposed.issubset(allowed):
            return False
    # A missing V5 owner is the one intentional local expansion: the server
    # names exact immutable scope IDs for that exact repair target. Scope IDs
    # still do not confer fact ownership until deterministic allocation.
    existing_scope_ids = (
        _collect_nested_source_values(original, "primary_evidence_scope_ids")
        | _collect_nested_source_values(original, "supporting_evidence_scope_ids")
    )
    proposed_scope_ids = (
        _collect_nested_source_values(replacement, "primary_evidence_scope_ids")
        | _collect_nested_source_values(replacement, "supporting_evidence_scope_ids")
    )
    permitted_scope_ids = existing_scope_ids | set(allowed_evidence_scope_ids or set())
    return not proposed_scope_ids or proposed_scope_ids.issubset(permitted_scope_ids)
    return True


def _repair_scope_violation(
    message: str,
    *,
    path: str = "course",
    patch_count: int | None = None,
    internal_code: str = "ARCH_REPAIR_SCOPE_VIOLATION",
    failure_stage: str = "architecture_repair_mutation_guard",
    guard_reason: str | None = None,
    semantic_operation: str | None = None,
    block_id: str | None = None,
    outer_field: str | None = None,
) -> WorkflowFailure:
    diagnostics: dict[str, Any] = {"repair_target_path": safe_workflow_path(path)}
    if patch_count is not None:
        diagnostics["patch_count"] = patch_count
    if guard_reason:
        diagnostics["guard_reason"] = guard_reason
    if semantic_operation:
        diagnostics["semantic_operation"] = semantic_operation
    if outer_field:
        diagnostics["outer_field"] = outer_field
    if block_id and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", block_id):
        diagnostics["block_id"] = block_id
    return WorkflowFailure(
        "ARCHITECTURE_REPAIR_SCOPE_VIOLATION",
        message,
        internal_code=internal_code,
        failure_stage=failure_stage,
        diagnostics=diagnostics,
    )


def _repair_unit_parent_and_index(blueprint: dict[str, Any], path: str) -> tuple[list[Any], int] | None:
    match = re.fullmatch(r"chapter_(\d+)\.lesson_(\d+)\.unit_(\d+)", path)
    if match is None:
        return None
    chapters = blueprint.get("chapters") if isinstance(blueprint.get("chapters"), list) else []
    chapter_index, lesson_index, unit_index = (int(value) - 1 for value in match.groups())
    if not 0 <= chapter_index < len(chapters) or not isinstance(chapters[chapter_index], dict):
        return None
    lessons = chapters[chapter_index].get("lessons") if isinstance(chapters[chapter_index].get("lessons"), list) else []
    if not 0 <= lesson_index < len(lessons) or not isinstance(lessons[lesson_index], dict):
        return None
    units = lessons[lesson_index].get("units") if isinstance(lessons[lesson_index].get("units"), list) else []
    return (units, unit_index) if 0 <= unit_index < len(units) else None


def _repair_lesson_parent_and_index(blueprint: dict[str, Any], path: str) -> tuple[list[Any], int] | None:
    match = re.fullmatch(r"chapter_(\d+)\.lesson_(\d+)", path)
    if match is None:
        return None
    chapters = blueprint.get("chapters") if isinstance(blueprint.get("chapters"), list) else []
    chapter_index, lesson_index = (int(value) - 1 for value in match.groups())
    if not 0 <= chapter_index < len(chapters) or not isinstance(chapters[chapter_index], dict):
        return None
    lessons = chapters[chapter_index].get("lessons") if isinstance(chapters[chapter_index].get("lessons"), list) else []
    return (lessons, lesson_index) if 0 <= lesson_index < len(lessons) else None


def _is_removable_factless_reinforcement_unit(unit: dict[str, Any]) -> bool:
    """Allow deletion only for a unit that cannot own a canonical fact.

    ``remove_unit`` is the narrow fallback for a generated supporting unit
    which has no server-allocated facts and declares no primary concept at
    either unit or block level.  A provider therefore cannot delete a
    fact-bearing instructional unit merely because it is an approved repair
    target.
    """

    if unit.get("source_fact_ids"):
        return False
    if unit.get("primary_concept_ids"):
        return False
    blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
    return all(
        isinstance(block, dict) and not block.get("primary_concept_ids")
        for block in blocks
    )


def _redundant_factless_lesson_removal_reason(lesson: dict[str, Any]) -> str | None:
    """Return the safe server-only reason to delete an empty parent lesson."""

    if lesson.get("primary_concept_ids"):
        return None
    if lesson.get("assessment_required"):
        return None
    assessment_objective_refs = lesson.get("assessment_objective_refs")
    if isinstance(assessment_objective_refs, list) and assessment_objective_refs:
        return None
    return "ALL_UNITS_FACTLESS_NON_PRIMARY_NO_LESSON_PRIMARY_OR_ASSESSMENT"


def _remove_unit_from_snapshot(snapshot: dict[str, Any], path: str) -> bool:
    target = _repair_unit_parent_and_index(snapshot, path)
    if target is None:
        return False
    units, index = target
    units.pop(index)
    return True


def _architecture_repair_preserves_unaffected_snapshot(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    replacement_paths: set[str],
    removal_paths: set[str],
) -> bool:
    """Compare semantic structure with only approved targets excluded.

    This is deliberately content-free: server allocation, component plans, and
    fact arrays are removed before comparison. Any structural change outside a
    patch target is rejected before candidate allocation can be committed.
    """

    baseline_snapshot = _semantic_architecture_snapshot(baseline)
    candidate_snapshot = _semantic_architecture_snapshot(candidate)
    if not isinstance(baseline_snapshot, dict) or not isinstance(candidate_snapshot, dict):
        return False
    for path in sorted(removal_paths, reverse=True):
        if not _remove_unit_from_snapshot(baseline_snapshot, path):
            return False
    # Validate/mask descendants before an explicitly authorized ancestor.
    # Masking a lesson first destroys an approved child's lookup path and
    # made mixed lesson/unit repair acceptance depend on set iteration order.
    for path in sorted(replacement_paths, key=lambda value: (value.count("."), value), reverse=True):
        baseline_node = _blueprint_path_object(baseline_snapshot, path)
        candidate_node = _blueprint_path_object(candidate_snapshot, path)
        if baseline_node is None or candidate_node is None:
            return False
        baseline_node.clear()
        candidate_node.clear()
        baseline_node["_repair_target"] = safe_workflow_path(path)
        candidate_node["_repair_target"] = safe_workflow_path(path)
    return baseline_snapshot == candidate_snapshot


def _repair_path_from_blueprint_schema_path(value: str) -> str:
    """Convert a server validation path to a safe bounded repair path."""

    match = re.match(
        r"^chapters\[(\d+)\](?:\.lessons\[(\d+)\](?:\.units\[(\d+)\])?)?",
        value,
    )
    if match is None:
        return "course"
    path = f"chapter_{int(match.group(1)) + 1}"
    if match.group(2) is not None:
        path += f".lesson_{int(match.group(2)) + 1}"
    if match.group(3) is not None:
        path += f".unit_{int(match.group(3)) + 1}"
    return safe_workflow_path(path)


def _repair_patch_domain_diagnostics(
    error: LessonAuthorBlueprintValidationError,
    target_paths: set[str],
    replacement_fields: dict[str, set[str]],
    *,
    patch_count: int,
    baseline: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return structural-only metadata for a rejected pre-apply patch set."""

    schema_path = str(error.path or "")
    node_path = _repair_path_from_blueprint_schema_path(schema_path)
    matching_paths = [
        path for path in target_paths
        if node_path == path or node_path.startswith(f"{path}.")
    ]
    target_path = max(matching_paths, key=len) if matching_paths else "course"
    remaining = schema_path
    # Remove the normalized target prefix from the validator path to report
    # the outer field being validated, never provider content.
    target_match = re.match(
        r"^chapter_(\d+)(?:\.lesson_(\d+)(?:\.unit_(\d+))?)?$",
        target_path,
    )
    if target_match is not None:
        segments = [f"chapters[{int(target_match.group(1)) - 1}]"]
        if target_match.group(2) is not None:
            segments.append(f"lessons[{int(target_match.group(2)) - 1}]")
        if target_match.group(3) is not None:
            segments.append(f"units[{int(target_match.group(3)) - 1}]")
        target_schema_prefix = ".".join(segments)
        if remaining.startswith(target_schema_prefix):
            remaining = remaining[len(target_schema_prefix):].lstrip(".")
    field_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", remaining)
    field = field_match.group(1) if field_match else "replacement"
    if field not in replacement_fields.get(target_path, set()):
        # A full Blueprint assertion can identify a nested field in a parent
        # replacement (for example lesson.units[0].learning_blocks). Surface
        # the authorized outer field rather than inventing a provider value.
        candidates = replacement_fields.get(target_path, set())
        field = next(iter(sorted(candidates)), "replacement")
    def array_count(value: Any) -> int | None:
        return len(value) if isinstance(value, list) else None

    baseline_value = None
    candidate_value = None
    if field != "replacement":
        baseline_node = _blueprint_path_object(baseline or {}, target_path)
        candidate_node = _blueprint_path_object(candidate or {}, target_path)
        baseline_value = baseline_node.get(field) if isinstance(baseline_node, dict) else None
        candidate_value = candidate_node.get(field) if isinstance(candidate_node, dict) else None
    expected_range = re.search(r"length=(\d+)\.\.(\d+)", str(error.expected_type or ""))
    safe_schema_path = (
        schema_path
        if re.fullmatch(
            r"(?:course|chapters\[\d+\](?:\.lessons\[\d+\](?:\.units\[\d+\](?:\.learning_blocks\[\d+\])?)?)?)(?:\.[A-Za-z_][A-Za-z0-9_]*)?",
            schema_path,
        )
        else "course"
    )
    return {
        "repair_target_path": safe_workflow_path(target_path),
        "repair_patch_field": field,
        "safe_json_path": safe_schema_path,
        "validation_category": "BLUEPRINT_FIELD_DOMAIN",
        "validation_constraint": str(error.constraint or "BLUEPRINT_INVALID_SCHEMA"),
        "validation_error_code": str(error.code or "BLUEPRINT_INVALID_SCHEMA"),
        "expected_min_items": int(expected_range.group(1)) if expected_range else None,
        "expected_max_items": int(expected_range.group(2)) if expected_range else None,
        "actual_count": array_count(candidate_value),
        "baseline_count": array_count(baseline_value),
        "domain_validation_passed": False,
        "transactional_apply_passed": False,
        "patch_count": patch_count,
    }


def _validate_v5_repair_patch_set_before_apply(
    candidate: dict[str, Any],
    *,
    target_paths: set[str],
    replacement_fields: dict[str, set[str]],
    patch_count: int,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Reject invalid provider field values before a V5 candidate is returned.

    This deliberately validates the complete transactional patch set against
    the same canonical V5 Blueprint contract used for the initial Architect
    response. It does not perform Source Map/allocator work; those layered
    validators remain downstream and unchanged.
    """

    try:
        validate_lesson_author_blueprint(
            candidate,
            require_source_fact_ownership=False,
            forbid_provider_fact_ownership=True,
        )
    except LessonAuthorBlueprintValidationError as error:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_INVALID",
            "Course architecture repair contains a field outside the authoritative Blueprint contract.",
            internal_code="ARCH_REPAIR_PATCH_SCHEMA_INVALID",
            failure_stage="architecture_repair_patch_domain_validation",
            diagnostics=_repair_patch_domain_diagnostics(
                error,
                target_paths,
                replacement_fields,
                patch_count=patch_count,
                baseline=baseline,
                candidate=candidate,
            ),
        ) from error


def _allocation_target_map(blueprint: dict[str, Any]) -> dict[str, tuple[str, str]]:
    allocation = blueprint.get("source_fact_allocation")
    if not isinstance(allocation, dict) or not isinstance(allocation.get("allocations"), list):
        return {}
    return {
        str(item.get("fact_id") or ""): (
            str(item.get("unit_path") or ""),
            str(item.get("learning_block_id") or ""),
        )
        for item in allocation["allocations"]
        if isinstance(item, dict) and str(item.get("fact_id") or "").strip()
    }
