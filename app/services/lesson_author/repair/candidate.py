"""Repair candidates: deterministic factless-unit removal, evidence alignment and candidate validation."""

from __future__ import annotations

from typing import Any

from app.services.lesson_author.architecture_shape import _blueprint_path_object, _repair_parent_lesson_path
from app.services.lesson_author.architecture_validation import validate_course_architecture_workflow
from app.services.lesson_author.evidence_scope import allocate_source_map_architecture_facts
from app.services.lesson_author.granularity import allocate_blueprint_source_fact_ids
from app.services.lesson_author.repair.guards import (
    _allocation_target_map,
    _is_removable_factless_reinforcement_unit,
    _redundant_factless_lesson_removal_reason,
    _repair_lesson_parent_and_index,
    _repair_unit_parent_and_index,
)
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.workflows.contracts import RepairTarget, WorkflowFailure, safe_workflow_path


def _deterministic_factless_unit_removal_candidate(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    *,
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
    source_structure_nodes: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Remove server-proven redundant units and, only when safe, their lesson.

    A target with no canonical facts and no primary ownership has no semantic
    material that a provider can safely repair. The server therefore removes it
    deterministically. If every unit in a lesson is such a target, the parent
    can be removed only when it also declares no primary or assessment scope.
    Allocation, full canonical validation, and the transactional candidate guard
    still decide whether the cloned candidate is acceptable.
    """

    allocation = blueprint.get("source_fact_allocation")
    if not isinstance(allocation, dict) or not allocation.get("complete"):
        # A unit can only be proven redundant after the server has already
        # established complete global canonical ownership for this baseline.
        return None

    eligible: list[RepairTarget] = []
    non_eligible: list[RepairTarget] = []
    for target in targets:
        is_ownership_target = (
            target.get("scope") == "unit"
            and "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED" in target.get("codes", [])
        )
        unit = _blueprint_path_object(blueprint, target.get("path", "")) if is_ownership_target else None
        if isinstance(unit, dict) and _is_removable_factless_reinforcement_unit(unit):
            eligible.append(target)
        else:
            non_eligible.append(target)

    if not eligible:
        return None
    if non_eligible:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_MIXED_DETERMINISTIC_TARGETS",
            "A factless unit can be removed deterministically, but another target requires separate scoped review.",
            internal_code="ARCH_REPAIR_MIXED_DETERMINISTIC_TARGETS",
            failure_stage="architecture_repair_deterministic_classification",
            diagnostics={
                "deterministic_target_count": len(eligible),
                "non_deterministic_target_count": len(non_eligible),
            },
        )

    removals_by_lesson: dict[str, set[int]] = {}
    removable_lesson_paths: set[str] = set()
    for target in eligible:
        location = _repair_unit_parent_and_index(blueprint, target["path"])
        if location is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "A deterministic repair target no longer exists.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_deterministic_classification",
                diagnostics={"repair_target_path": safe_workflow_path(target["path"])},
            )
        units, unit_index = location
        lesson_path = _repair_parent_lesson_path(target["path"])
        removals_by_lesson.setdefault(lesson_path, set()).add(unit_index)
        if len(units) - len(removals_by_lesson[lesson_path]) <= 0:
            lesson = _blueprint_path_object(blueprint, lesson_path)
            reason = _redundant_factless_lesson_removal_reason(lesson) if isinstance(lesson, dict) else None
            lesson_location = _repair_lesson_parent_and_index(blueprint, lesson_path)
            if reason is not None and lesson_location is not None and len(lesson_location[0]) > 1:
                removable_lesson_paths.add(lesson_path)
                continue
            raise WorkflowFailure(
                "ARCHITECTURE_LESSON_PRIMARY_SCOPE_UNRESOLVED",
                "An empty parent lesson cannot be proven redundant for deterministic removal.",
                internal_code="ARCH_REPAIR_LESSON_PRIMARY_SCOPE_UNRESOLVED",
                failure_stage="architecture_repair_deterministic_parent_classification",
                diagnostics={
                    "repair_target_count": len(eligible),
                    "lesson_path": safe_workflow_path(lesson_path),
                    "parent_classification": (
                        "LAST_LESSON_IN_CHAPTER" if lesson_location is not None and len(lesson_location[0]) <= 1
                        else "LESSON_PRIMARY_OR_ASSESSMENT_SCOPE_UNRESOLVED"
                    ),
                    "lesson_primary_concept_count": len(lesson.get("primary_concept_ids") or []) if isinstance(lesson, dict) else 0,
                    "lesson_assessment_required": bool(lesson.get("assessment_required")) if isinstance(lesson, dict) else False,
                },
            )

    candidate = apply_course_architecture_repair_patches(
        blueprint,
        eligible,
        {"patches": [
            {"path": target["path"], "operation": "remove_unit"}
            for target in eligible
        ]},
    )
    for lesson_path in sorted(removable_lesson_paths, reverse=True):
        lesson_location = _repair_lesson_parent_and_index(candidate, lesson_path)
        if lesson_location is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "A deterministic parent lesson target no longer exists.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_deterministic_parent_classification",
                diagnostics={"lesson_path": safe_workflow_path(lesson_path)},
            )
        lessons, lesson_index = lesson_location
        lessons.pop(lesson_index)
    candidate = allocate_source_map_architecture_facts(
        candidate,
        source_map,
        source_coverage_manifest,
    )
    if (candidate.get("source_fact_allocation") or {}).get("complete"):
        candidate = allocate_blueprint_source_fact_ids(
            candidate,
            source_coverage_manifest,
            source_structure_nodes,
        )
    validate_course_architecture_repair_candidate(
        baseline=blueprint,
        candidate=candidate,
        targets=eligible,
        source_map=source_map,
        source_coverage_manifest=source_coverage_manifest,
        known_source_refs=known_source_refs,
    )
    return candidate, {
        "operation": "remove_lesson" if removable_lesson_paths else "remove_unit",
        "removed_lesson_paths": [safe_workflow_path(path) for path in sorted(removable_lesson_paths)],
        "parent_classification": (
            "ALL_UNITS_FACTLESS_NON_PRIMARY_NO_LESSON_PRIMARY_OR_ASSESSMENT"
            if removable_lesson_paths else "UNIT_ONLY"
        ),
    }


def deterministic_factless_unit_removal_repair(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    *,
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
    source_structure_nodes: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Compatibility wrapper for deterministic repair tests and callers."""

    outcome = _deterministic_factless_unit_removal_candidate(
        blueprint,
        targets,
        source_map=source_map,
        source_coverage_manifest=source_coverage_manifest,
        known_source_refs=known_source_refs,
        source_structure_nodes=source_structure_nodes,
    )
    return outcome[0] if outcome is not None else None


def validate_course_architecture_repair_candidate(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    targets: list[RepairTarget],
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
) -> None:
    """Commit a repaired Blueprint only when it improves without regression.

    The authoritative workflow state remains the baseline until this function
    accepts the cloned candidate. No provider output, fact IDs, or source text
    is logged or retained as an acceptance diagnostic.
    """

    ownership_targets = {
        target["path"]
        for target in targets
        if "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED" in target.get("codes", [])
    }
    if not ownership_targets:
        return
    baseline_allocation = baseline.get("source_fact_allocation")
    candidate_allocation = candidate.get("source_fact_allocation")
    baseline_count = int(baseline_allocation.get("allocated_count") or 0) if isinstance(baseline_allocation, dict) else 0
    candidate_count = int(candidate_allocation.get("allocated_count") or 0) if isinstance(candidate_allocation, dict) else 0
    baseline_complete = bool(isinstance(baseline_allocation, dict) and baseline_allocation.get("complete"))
    candidate_complete = bool(isinstance(candidate_allocation, dict) and candidate_allocation.get("complete"))
    diagnostics = {
        "baseline_allocated_fact_count": baseline_count,
        "candidate_allocated_fact_count": candidate_count,
        "baseline_allocation_complete": baseline_complete,
        "candidate_allocation_complete": candidate_complete,
        "repair_target_count": len(ownership_targets),
    }
    if baseline_complete and (not candidate_complete or candidate_count < baseline_count):
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_REGRESSION",
            "Scoped repair regressed complete canonical Source Fact allocation.",
            internal_code="ARCH_REPAIR_REGRESSION",
            failure_stage="architecture_repair_candidate_validation",
            diagnostics={**diagnostics, "regression_reason": "CANONICAL_ALLOCATION_REGRESSED"},
        )

    baseline_targets = _allocation_target_map(baseline)
    candidate_targets = _allocation_target_map(candidate)
    for fact_id, target in baseline_targets.items():
        candidate_target = candidate_targets.get(fact_id)
        if candidate_target is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_REGRESSION",
                "Scoped repair left a previously allocated canonical Source Fact without an owner.",
                internal_code="ARCH_REPAIR_REGRESSION",
                failure_stage="architecture_repair_candidate_validation",
                diagnostics={**diagnostics, "regression_reason": "PREVIOUSLY_ALLOCATED_FACT_UNALLOCATED"},
            )
        if target[0] in ownership_targets:
            continue
        if candidate_target != target:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_REGRESSION",
                "Scoped repair changed canonical ownership outside its approved target.",
                internal_code="ARCH_REPAIR_REGRESSION",
                failure_stage="architecture_repair_candidate_validation",
                diagnostics={**diagnostics, "regression_reason": "UNRELATED_ALLOCATION_CHANGED"},
            )

    baseline_validation = validate_course_architecture_workflow(
        baseline, source_map, source_coverage_manifest, known_source_refs,
    )
    candidate_validation = validate_course_architecture_workflow(
        candidate, source_map, source_coverage_manifest, known_source_refs,
    )
    baseline_target_findings = sum(
        1 for issue in baseline_validation.errors
        if issue.get("code") == "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED"
        and str(issue.get("path") or "") in ownership_targets
    )
    candidate_target_findings = sum(
        1 for issue in candidate_validation.errors
        if issue.get("code") == "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED"
        and str(issue.get("path") or "") in ownership_targets
    )
    diagnostics.update({
        "baseline_target_finding_count": baseline_target_findings,
        "candidate_target_finding_count": candidate_target_findings,
        "candidate_blocking_codes": sorted({str(issue.get("code") or "") for issue in candidate_validation.errors}),
    })
    if candidate_validation.errors:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_REGRESSION",
            "Scoped repair introduced or retained invalid canonical architecture state.",
            internal_code="ARCH_REPAIR_REGRESSION",
            failure_stage="architecture_repair_candidate_validation",
            diagnostics={**diagnostics, "regression_reason": "CANONICAL_VALIDATION_FAILED"},
        )
    if candidate_target_findings >= baseline_target_findings:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_NO_PROGRESS",
            "Scoped repair did not reduce its targeted canonical ownership findings.",
            internal_code="ARCH_REPAIR_NO_PROGRESS",
            failure_stage="architecture_repair_candidate_validation",
            diagnostics={**diagnostics, "regression_reason": "TARGET_FINDINGS_UNCHANGED"},
        )


def _v5_deterministic_evidence_alignment_candidate(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
) -> dict[str, Any] | None:
    """Apply only an unambiguous all-target concept alignment server-side."""

    if not targets or not all(
        set(target.get("semantic_operations") or []) == {"align_concepts_to_evidence"}
        for target in targets
    ):
        return None
    if any(len(target.get("primary_concept_options") or []) != 1 for target in targets):
        return None
    payload = {
        "patches": [
            {
                "path": target["path"],
                "operation": "align_concepts_to_evidence",
                "concept_ids": list(target.get("required_concept_ids") or []),
                "primary_concept_ids": list((target.get("primary_concept_options") or [[]])[0]),
            }
            for target in targets
        ],
    }
    return apply_course_architecture_repair_patches(blueprint, targets, payload)
