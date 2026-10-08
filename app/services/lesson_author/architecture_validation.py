"""Course-architecture workflow validation (V4/V5 ownership, V5 instructional coherence and depth)."""

from __future__ import annotations

import re
from typing import Any

from app.assessment_planner import evaluate_assessment_teaching_anchor
from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError, validate_lesson_author_blueprint
from app.services.lesson_author.architecture_shape import (
    _V5_ACTION_OR_PROCEDURE_OBJECTIVE,
    _V5_GENERIC_EXPLANATION_INTENTS,
    _V5_TEACHING_INTENTS,
    _is_v4_supporting_factless_unit,
    _safe_json_shape,
    _semantic_delta_blocks,
    _v5_text_ids,
    _workflow_issue,
    _workflow_issue_from_blueprint_validation_error,
)
from app.services.lesson_author.blueprint import validate_lesson_author_source_refs
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.evidence_scope import _SEMANTIC_SCOPE_DIAGNOSTIC_FIELDS
from app.services.retrieval.source_coverage import source_coverage_metrics, validate_lesson_author_source_coverage
from app.workflows.contracts import WorkflowIssue, WorkflowValidationResult


def _validate_v4_unit_source_fact_ownership(blueprint: dict[str, Any]) -> list[WorkflowIssue]:
    """Enforce v4 primary evidence ownership without inventing support facts."""

    if blueprint.get("architecture_contract_version") != 4:
        return []
    allocation = blueprint.get("source_fact_allocation")
    if not isinstance(allocation, dict) or allocation.get("complete") is not True:
        return []

    issues: list[WorkflowIssue] = []
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                fact_ids = unit.get("source_fact_ids")
                if isinstance(fact_ids, list) and fact_ids:
                    continue
                if _is_v4_supporting_factless_unit(unit):
                    continue
                issues.append(_workflow_issue(
                    "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED",
                    "A primary Source-Map-backed unit has no server-allocated Source Fact ownership.",
                    path=f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}",
                    constraint="v4 primary units require server allocation to assign at least one canonical fact",
                    expected_type="array[minItems=1] of server-allocated canonical fact IDs",
                    actual_type=_safe_json_shape(fact_ids),
                ))
    return issues


def _v5_lesson_block_records(
    lesson: dict[str, Any],
    lesson_path: str,
) -> list[dict[str, Any]]:
    """Return authoritative pedagogical order without exposing source text."""

    records: list[dict[str, Any]] = []
    position = 0
    for unit_index, unit in enumerate(lesson.get("units", []), start=1):
        if not isinstance(unit, dict):
            continue
        unit_path = f"{lesson_path}.unit_{unit_index}"
        unit_objective_refs = _v5_text_ids(unit.get("learning_objective_refs"))
        for block in _semantic_delta_blocks(unit):
            position += 1
            block_id = str(block.get("id") or "").strip()
            records.append({
                "position": position,
                "unit_path": unit_path,
                "block": block,
                "block_id": block_id,
                "unit_objective_refs": unit_objective_refs,
            })
    return records


def _v5_objective_ids_are_local(lesson: dict[str, Any], objective_refs: set[str]) -> bool:
    valid = {
        f"lo_{index}"
        for index, value in enumerate(lesson.get("learning_objectives", []), start=1)
        if isinstance(value, str) and value.strip()
    }
    return bool(objective_refs) and objective_refs.issubset(valid)


def _v5_unit_action_objective_texts(
    lesson: dict[str, Any],
    unit: dict[str, Any],
) -> list[str]:
    """Return the objective prose actually assigned to this V5 unit.

    V5 units have required local objective refs.  A lesson can contain both
    informational and procedural objectives, so using every lesson objective
    for each unit falsely turns an explanatory unit into an action mismatch.
    Invalid direct callers retain the conservative legacy fallback; normal
    provider candidates have already passed the local-reference schema gate.
    """

    objectives = [
        value.strip()
        for value in lesson.get("learning_objectives", [])
        if isinstance(value, str) and value.strip()
    ]
    refs = _v5_text_ids(unit.get("learning_objective_refs"))
    by_ref = {f"lo_{index}": objective for index, objective in enumerate(objectives, start=1)}
    if refs and refs.issubset(by_ref):
        return [by_ref[ref] for ref in sorted(refs, key=lambda ref: int(ref.removeprefix("lo_")))]
    return [
        str(lesson.get("objective") or "").strip(),
        *objectives,
    ]


def _v5_base_teaching_anchor_candidates(
    lesson: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    knowledge_check: dict[str, Any],
    objective_refs: set[str],
) -> list[dict[str, Any]]:
    """Return anchors eligible to become aligned through local objective links.

    This is deliberately weaker than full alignment only in one dimension:
    the selected local objectives need not already be present on the teaching
    block.  Every provenance, scope, ordering, teaching-intent and lesson
    boundary requirement remains deterministic and identical to the final
    coherence rule.
    """

    if not _v5_objective_ids_are_local(lesson, objective_refs):
        return []
    check_position = int(knowledge_check.get("position") or 0)
    check_block = knowledge_check.get("block")
    if not isinstance(check_block, dict) or check_position <= 0:
        return []
    candidates: list[dict[str, Any]] = []
    for record in records:
        block = record.get("block")
        if not isinstance(block, dict) or int(record.get("position") or 0) >= check_position:
            continue
        eligibility = evaluate_assessment_teaching_anchor(
            teaching_block=block,
            knowledge_check_block=check_block,
            objective_refs=objective_refs,
            unit_objective_refs=set(record.get("unit_objective_refs") or set()),
            precedes_check=int(record.get("position") or 0) < check_position,
        )
        if not eligibility.base_eligible:
            continue
        candidates.append(record)
    return candidates


def _v5_fully_aligned_teaching_anchor_candidates(
    lesson: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    knowledge_check: dict[str, Any],
    objective_refs: set[str],
) -> list[dict[str, Any]]:
    """Return base-eligible anchors which already teach every selected objective."""

    return [
        record
        for record in _v5_base_teaching_anchor_candidates(
            lesson,
            records,
            knowledge_check=knowledge_check,
            objective_refs=objective_refs,
        )
        if objective_refs.issubset(_v5_text_ids(record["block"].get("learning_objective_refs")))
    ]


def validate_v5_instructional_coherence(blueprint: dict[str, Any]) -> WorkflowValidationResult:
    """Check V5 semantic coherence before canonical allocation.

    This phase deliberately excludes depth checks that depend on final
    server-owned fact allocation. It does not make component choices, alter
    canonical ownership, or inspect source text. The Node planner remains the
    sole mapping of ``knowledge_check`` to ``problem``.
    """

    if blueprint.get("architecture_contract_version") != 5:
        return WorkflowValidationResult()

    issues: list[WorkflowIssue] = []
    assessment_total = assessment_covered = 0

    def add(
        code: str,
        message: str,
        *,
        path: str,
        objective_ids: set[str] | None = None,
        learning_block_ids: list[str] | None = None,
        unit_paths: list[str] | None = None,
        repairable: bool | None = None,
    ) -> None:
        issue = _workflow_issue(code, message, path=path)
        if objective_ids:
            issue["objective_ids"] = sorted(objective_ids)[:12]
        if learning_block_ids:
            issue["learning_block_ids"] = [block_id for block_id in learning_block_ids if block_id][:12]
        if unit_paths:
            issue["unit_paths"] = [unit_path for unit_path in unit_paths if unit_path][:12]
        if repairable is not None:
            issue["repairable"] = repairable
        issues.append(issue)

    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_path = f"chapter_{chapter_index}.lesson_{lesson_index}"
            assessment_refs = _v5_text_ids(lesson.get("assessment_objective_refs"))
            action_units: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"{lesson_path}.unit_{unit_index}"
                blocks = [block for block in unit.get("learning_blocks", []) if isinstance(block, dict)]
                action_units.append((unit_path, blocks, unit))
            flat_blocks = _v5_lesson_block_records(lesson, lesson_path)

            if lesson.get("assessment_required") is True:
                assessment_total += 1
                checks = [
                    item for item in flat_blocks
                    if str(item["block"].get("intent") or "").strip() == "knowledge_check"
                ]
                if not checks:
                    # The V5 assessment-plan compiler normally runs before
                    # this validator and creates server-owned factless checks
                    # per objective. Keep this as a hard invariant for direct
                    # callers: do not revive the old one-anchor-for-the-whole
                    # lesson heuristic here.
                    fully_aligned = [
                        item for item in flat_blocks
                        if str(item["block"].get("intent") or "").strip() in _V5_TEACHING_INTENTS
                        and assessment_refs.issubset(_v5_text_ids(item["block"].get("learning_objective_refs")))
                        and bool(_v5_text_ids(item["block"].get("primary_evidence_scope_ids")))
                    ]
                    if len(fully_aligned) == 1:
                        # Compatibility for callers that invoke coherence
                        # directly. The normal V5 workflow reaches this only
                        # after the per-objective compiler has run.
                        add(
                            "ASSESSMENT_BLOCK_REQUIRED",
                            "An assessment-required V5 lesson needs a knowledge_check semantic block before draft planning.",
                            path=str(fully_aligned[0]["unit_path"]),
                            objective_ids=assessment_refs,
                            learning_block_ids=[str(fully_aligned[0]["block_id"])],
                        )
                    else:
                        add(
                            "ASSESSMENT_BLOCK_REQUIRED",
                            "An assessment-required V5 lesson has no compiled knowledge_check semantic block.",
                            path=lesson_path,
                            objective_ids=assessment_refs,
                            repairable=False,
                        )
                else:
                    check_objectives = set().union(*(
                        _v5_text_ids(item["block"].get("learning_objective_refs"))
                        for item in checks
                    ))
                    missing_objectives = assessment_refs - check_objectives
                    if missing_objectives:
                        target_unit_path = str(checks[-1]["unit_path"])
                        target_block_id = str(checks[-1]["block_id"])
                        add(
                            "ASSESSMENT_OBJECTIVE_NOT_COVERED",
                            "Knowledge-check semantic blocks do not cover every declared assessment objective.",
                            # The exact assessment block is the smallest
                            # provider-editable repair scope. The lesson's
                            # declared objectives remain immutable here.
                            path=target_unit_path,
                            objective_ids=missing_objectives,
                            learning_block_ids=[target_block_id],
                        )

                    missing_prior_teaching: list[tuple[str, str, set[str]]] = []
                    missing_grounding: list[tuple[str, str, set[str]]] = []
                    for check in checks:
                        check_block = check["block"]
                        check_refs = _v5_text_ids(check_block.get("learning_objective_refs")) & assessment_refs
                        prior_teaching_by_objective: dict[str, list[dict[str, Any]]] = {}
                        for objective_ref in check_refs:
                            prior_teaching_by_objective[objective_ref] = _v5_fully_aligned_teaching_anchor_candidates(
                                lesson,
                                flat_blocks,
                                knowledge_check=check,
                                objective_refs={objective_ref},
                            )
                        missing_objective_refs = {
                            objective_ref
                            for objective_ref, anchors in prior_teaching_by_objective.items()
                            if not anchors
                        }
                        if missing_objective_refs:
                            missing_prior_teaching.append((
                                str(check["unit_path"]), str(check["block_id"]), missing_objective_refs,
                            ))
                        supporting = _v5_text_ids(check_block.get("supporting_evidence_scope_ids"))
                        prior_primary = set().union(*(
                            _v5_text_ids(record["block"].get("primary_evidence_scope_ids"))
                            for anchors in prior_teaching_by_objective.values()
                            for record in anchors
                        )) if prior_teaching_by_objective else set()
                        if not supporting.intersection(prior_primary):
                            missing_grounding.append((str(check["unit_path"]), str(check["block_id"]), check_refs))

                    for check_unit_path, check_id, objective_ids in missing_prior_teaching:
                        add(
                            "ASSESSMENT_OBJECTIVE_NOT_COVERED",
                            "Assessment objectives must be taught by an earlier explanatory or procedural semantic block.",
                            path=check_unit_path,
                            objective_ids=objective_ids,
                            learning_block_ids=[check_id],
                        )
                    for check_unit_path, check_id, objective_ids in missing_grounding:
                        add(
                            "ASSESSMENT_EVIDENCE_NOT_GROUNDED",
                            "A knowledge check must supporting-reference evidence already primary-owned by earlier teaching in the lesson.",
                            path=check_unit_path,
                            objective_ids=objective_ids,
                            learning_block_ids=[check_id],
                        )
                    if not missing_objectives and not missing_prior_teaching and not missing_grounding:
                        assessment_covered += 1

            for unit_path, blocks, unit in action_units:
                unit_text = " ".join([
                    *_v5_unit_action_objective_texts(lesson, unit),
                    str(unit.get("purpose") or ""),
                ])
                intents = {str(block.get("intent") or "").strip() for block in blocks}
                if _V5_ACTION_OR_PROCEDURE_OBJECTIVE.search(unit_text) and intents and intents <= _V5_GENERIC_EXPLANATION_INTENTS:
                    add(
                        "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH",
                        "An action-oriented unit needs procedural or supported learner-action treatment, not generic explanation alone.",
                        path=unit_path,
                        learning_block_ids=[str(block.get("id") or "") for block in blocks],
                    )

    return WorkflowValidationResult(issues, {
        "assessment_alignment": round(assessment_covered / assessment_total, 4) if assessment_total else None,
    })


def validate_v5_post_allocation_instructional_depth(
    blueprint: dict[str, Any],
) -> WorkflowValidationResult:
    """Check V5 instructional depth only after server allocation is complete.

    The server-owned ``source_fact_ids`` are intentionally unavailable to the
    Architect and pre-allocation coherence validator. This phase therefore
    runs only on a complete server allocation and can offer one narrow lesson
    target for a supporting instructional block without granting provenance
    ownership to the provider.
    """

    if blueprint.get("architecture_contract_version") != 5:
        return WorkflowValidationResult()
    allocation = blueprint.get("source_fact_allocation")
    if not isinstance(allocation, dict) or allocation.get("authority") != "server" or allocation.get("complete") is not True:
        return WorkflowValidationResult()

    issues: list[WorkflowIssue] = []
    depth_total = depth_covered = 0
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_path = f"chapter_{chapter_index}.lesson_{lesson_index}"
            objectives = _v5_text_ids(lesson.get("learning_objectives"))
            local_objective_refs = {
                f"lo_{index}"
                for index, value in enumerate(lesson.get("learning_objectives", []), start=1)
                if isinstance(value, str) and value.strip()
            }
            lesson_fact_ids: set[str] = set()
            primary_scope_ids: set[str] = set()
            non_assessment: list[tuple[str, dict[str, Any]]] = []
            eligible_anchors: list[tuple[str, str]] = []

            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"{lesson_path}.unit_{unit_index}"
                lesson_fact_ids.update(_v5_text_ids(unit.get("source_fact_ids")))
                for block in _semantic_delta_blocks(unit):
                    intent = str(block.get("intent") or "").strip()
                    block_id = str(block.get("id") or "").strip()
                    primary_scope_ids.update(_v5_text_ids(block.get("primary_evidence_scope_ids")))
                    if intent != "knowledge_check":
                        non_assessment.append((unit_path, block))
                    if (
                        intent in _V5_TEACHING_INTENTS
                        and block_id
                        and _v5_text_ids(block.get("primary_evidence_scope_ids"))
                        and _v5_text_ids(block.get("learning_objective_refs"))
                        and _v5_text_ids(block.get("learning_objective_refs")).issubset(local_objective_refs)
                    ):
                        eligible_anchors.append((unit_path, block_id))

            substantial_evidence = len(lesson_fact_ids) >= 8 or len(primary_scope_ids) >= 2
            non_assessment_intents = {
                str(block.get("intent") or "").strip()
                for _unit_path, block in non_assessment
            }
            if len(objectives) < 2 or not substantial_evidence:
                continue
            depth_total += 1
            if len(non_assessment) != 1 or not non_assessment_intents <= _V5_GENERIC_EXPLANATION_INTENTS:
                depth_covered += 1
                continue

            # A provider may choose a server-approved unit/block anchor, but
            # cannot resolve absent or ungrounded support by inventing a scope.
            if not eligible_anchors:
                issue = _workflow_issue(
                    "INSTRUCTIONAL_DEPTH_INSUFFICIENT",
                    "A depth-insufficient lesson has no source-grounded teaching block for a safe supporting treatment.",
                    path=lesson_path,
                )
                issue["repairable"] = False
                issues.append(issue)
                continue
            issue = _workflow_issue(
                "INSTRUCTIONAL_DEPTH_INSUFFICIENT",
                "A multi-objective lesson with substantial server-owned evidence needs an additional grounded instructional treatment.",
                path=lesson_path,
            )
            issue["objective_ids"] = sorted(local_objective_refs)[:12]
            issue["learning_block_ids"] = [block_id for _unit_path, block_id in eligible_anchors][:12]
            issue["unit_paths"] = list(dict.fromkeys(unit_path for unit_path, _block_id in eligible_anchors))[:12]
            issues.append(issue)

    return WorkflowValidationResult(issues, {
        "instructional_depth": round(depth_covered / depth_total, 4) if depth_total else None,
    })


def _validate_v5_course_architecture_workflow(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
) -> WorkflowValidationResult:
    """Validate v5 server scope/fact ownership without reintroducing v4 rules."""
    issues: list[WorkflowIssue] = []
    scope_allocation = blueprint.get("source_evidence_scope_allocation")
    fact_allocation = blueprint.get("source_fact_allocation")
    required_scope_ids = {
        str(scope.get("id") or "").strip()
        for scope in source_map.get("source_evidence_scopes", [])
        if isinstance(scope, dict) and str(scope.get("id") or "").strip()
    }
    required_fact_ids = {
        str(fact.get("fact_id") or "").strip()
        for fact in (source_coverage_manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    }
    if not isinstance(scope_allocation, dict) or (
        scope_allocation.get("version") != "source-evidence-scope-allocation-v1"
        or scope_allocation.get("authority") != "server"
        or scope_allocation.get("architecture_contract_version") != 5
    ):
        return WorkflowValidationResult([_workflow_issue("EVIDENCE_SCOPE_ALLOCATION_AUTHORITY_INVALID", "Evidence-scope allocation was not issued by the server allocator.", path="course")])
    if not isinstance(fact_allocation, dict) or (
        fact_allocation.get("version") != "source-fact-allocation-v3"
        or fact_allocation.get("authority") != "server"
        or fact_allocation.get("architecture_contract_version") != 5
    ):
        return WorkflowValidationResult([_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "Source Fact allocation was not issued by the v5 server allocator.", path="course")])
    preallocation = scope_allocation.get("preallocation_validation") or fact_allocation.get("preallocation_validation")
    if isinstance(preallocation, list) and preallocation:
        for item in preallocation[:80]:
            if isinstance(item, dict):
                issues.append(_workflow_issue(
                    str(item.get("code") or "ARCHITECTURE_SCOPE_INCOMPLETE"),
                    "Course architecture semantic evidence-scope ownership is insufficient for deterministic allocation.",
                    path=str(item.get("path") or "course"),
                ))
        return WorkflowValidationResult(issues or [_workflow_issue("ARCHITECTURE_SCOPE_INCOMPLETE", "Evidence-scope ownership is incomplete.", path="course")], {
            "source_coverage": 0.0, "concept_coverage": None,
        })

    scope_entries = scope_allocation.get("allocations") if isinstance(scope_allocation.get("allocations"), list) else []
    fact_entries = fact_allocation.get("allocations") if isinstance(fact_allocation.get("allocations"), list) else []
    allocated_scope_ids = [str(item.get("evidence_scope_id") or "").strip() for item in scope_entries if isinstance(item, dict)]
    allocated_fact_ids = [str(item.get("fact_id") or "").strip() for item in fact_entries if isinstance(item, dict)]
    if (
        scope_allocation.get("complete") is not True
        or scope_allocation.get("required_count") != len(required_scope_ids)
        or scope_allocation.get("allocated_count") != len(required_scope_ids)
        or set(allocated_scope_ids) != required_scope_ids
        or len(allocated_scope_ids) != len(set(allocated_scope_ids))
        or scope_allocation.get("unallocated")
    ):
        issues.append(_workflow_issue("UNALLOCATED_EVIDENCE_SCOPE", "Canonical evidence-scope allocation is incomplete or inconsistent.", path="course"))
    if (
        fact_allocation.get("complete") is not True
        or fact_allocation.get("required_count") != len(required_fact_ids)
        or fact_allocation.get("allocated_count") != len(required_fact_ids)
        or set(allocated_fact_ids) != required_fact_ids
        or len(allocated_fact_ids) != len(set(allocated_fact_ids))
        or fact_allocation.get("unallocated")
    ):
        issues.append(_workflow_issue("UNALLOCATED_SOURCE_FACT", "Canonical Source Fact allocation is incomplete or inconsistent.", path="course"))

    actual_scope_targets: dict[str, tuple[str, str]] = {}
    actual_fact_targets: dict[str, tuple[str, str]] = {}
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                block_fact_targets: dict[str, str] = {}
                for block in unit.get("learning_blocks", []) if isinstance(unit.get("learning_blocks"), list) else []:
                    if not isinstance(block, dict):
                        continue
                    block_id = str(block.get("id") or "").strip()
                    for scope_id in block.get("primary_evidence_scope_ids", []) if isinstance(block.get("primary_evidence_scope_ids"), list) else []:
                        if isinstance(scope_id, str) and scope_id.strip():
                            if scope_id in actual_scope_targets:
                                issues.append(_workflow_issue("DUPLICATE_PRIMARY_EVIDENCE_SCOPE_OWNER", "A scope is primary-owned by more than one block.", path=unit_path))
                            actual_scope_targets[scope_id] = (unit_path, block_id)
                    for fact_id in block.get("source_fact_ids", []) if isinstance(block.get("source_fact_ids"), list) else []:
                        if isinstance(fact_id, str) and fact_id.strip():
                            if fact_id in block_fact_targets:
                                issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "A Fact is assigned to more than one block in its unit.", path=unit_path))
                            block_fact_targets[fact_id] = block_id
                for fact_id in unit.get("source_fact_ids", []) if isinstance(unit.get("source_fact_ids"), list) else []:
                    if isinstance(fact_id, str) and fact_id.strip():
                        if fact_id in actual_fact_targets:
                            issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "A Fact is assigned to more than one unit.", path=unit_path))
                        actual_fact_targets[fact_id] = (unit_path, block_fact_targets.get(fact_id, ""))

    expected_scope_targets = {
        str(item.get("evidence_scope_id") or "").strip(): (str(item.get("unit_path") or "").strip(), str(item.get("learning_block_id") or "").strip())
        for item in scope_entries if isinstance(item, dict)
    }
    expected_fact_targets = {
        str(item.get("fact_id") or "").strip(): (str(item.get("unit_path") or "").strip(), str(item.get("learning_block_id") or "").strip())
        for item in fact_entries if isinstance(item, dict)
    }
    if expected_scope_targets != actual_scope_targets:
        issues.append(_workflow_issue("EVIDENCE_SCOPE_ALLOCATION_AUTHORITY_INVALID", "Final primary scope ownership does not match server allocation metadata.", path="course"))
    if expected_fact_targets != actual_fact_targets:
        issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "Final Fact ownership does not match server allocation metadata.", path="course"))

    try:
        normalized = validate_lesson_author_blueprint(blueprint)
        validate_lesson_author_source_refs(normalized, known_source_refs)
        source_metrics = validate_lesson_author_source_coverage({"chapters": normalized.get("chapters", [])}, source_coverage_manifest)
    except LessonAuthorBlueprintValidationError as error:
        issues.append(_workflow_issue_from_blueprint_validation_error(error))
        source_metrics = source_coverage_metrics({"chapters": blueprint.get("chapters", [])}, source_coverage_manifest)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("SOURCE_COVERAGE_INCOMPLETE", str(error), path="course"))
        source_metrics = source_coverage_metrics({"chapters": blueprint.get("chapters", [])}, source_coverage_manifest)
    # Preserve pre-allocation semantic checks in the final authoritative
    # validation report. The endpoint invokes these before allocation as a
    # separate graph boundary; running them here makes the completed artifact
    # fail closed if a later repair regresses action/assessment coherence.
    pre_coherence = validate_v5_instructional_coherence(blueprint)
    issues.extend(pre_coherence.issues)
    depth = validate_v5_post_allocation_instructional_depth(blueprint)
    issues.extend(depth.issues)
    return WorkflowValidationResult(issues, {
        "source_coverage": source_metrics.get("coverage_ratio"),
        "concept_coverage": None,
        "evidence_scope_coverage": round(len(set(allocated_scope_ids) & required_scope_ids) / max(1, len(required_scope_ids)), 4),
        **pre_coherence.metrics,
        **depth.metrics,
    })


def validate_course_architecture_workflow(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
) -> WorkflowValidationResult:
    """Generation-quality checks only; Node remains the authoritative gate.

    These checks deliberately use the Phase-3 Source Map and manifest. They
    do not perform tenant, RBAC, editor-target, or component-permission work.
    """
    if blueprint.get("architecture_contract_version") == 5:
        return _validate_v5_course_architecture_workflow(
            blueprint, source_map, source_coverage_manifest, known_source_refs,
        )
    issues: list[WorkflowIssue] = []
    allocation = blueprint.get("source_fact_allocation")
    if blueprint.get("architecture_contract_version") == 4:
        if not isinstance(allocation, dict):
            return WorkflowValidationResult([
                _workflow_issue("UNALLOCATED_SOURCE_FACT", "Server Source Fact allocation metadata is missing.", path="course"),
            ])
        if (
            allocation.get("version") != "source-fact-allocation-v2"
            or allocation.get("authority") != "server"
            or allocation.get("architecture_contract_version") != 4
        ):
            return WorkflowValidationResult([
                _workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "Source Fact allocation was not issued by the server allocator.", path="course"),
            ])
        preallocation = allocation.get("preallocation_validation")
        if isinstance(preallocation, list) and preallocation:
            for item in preallocation[:80]:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or "ARCHITECTURE_SCOPE_INCOMPLETE")
                issue = _workflow_issue(
                    code,
                    "Course architecture semantic scope is not sufficient for deterministic canonical Source Fact allocation.",
                    path=str(item.get("path") or "course"),
                )
                for key in _SEMANTIC_SCOPE_DIAGNOSTIC_FIELDS:
                    value = item.get(key)
                    if value is not None:
                        issue[key] = value
                if isinstance(item.get("learning_block_ids"), list):
                    issue["learning_block_ids"] = list(item["learning_block_ids"])[:12]
                issues.append(issue)
            metrics = {
                "source_coverage": 0.0,
                "concept_coverage": 0.0,
                **(allocation.get("semantic_scope_metrics") if isinstance(allocation.get("semantic_scope_metrics"), dict) else {}),
            }
            return WorkflowValidationResult(issues or [
                _workflow_issue("ARCHITECTURE_SCOPE_INCOMPLETE", "Course architecture semantic scope is incomplete.", path="course"),
            ], metrics)
        required_ids = {
            str(fact.get("fact_id") or "").strip()
            for fact in (source_coverage_manifest or {}).get("facts", [])
            if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
        }
        allocations = allocation.get("allocations") if isinstance(allocation.get("allocations"), list) else []
        allocated_ids: list[str] = []
        assigned_targets: dict[str, tuple[str, str]] = {}
        for item in allocations:
            if not isinstance(item, dict):
                issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "Source Fact allocation contains an invalid entry.", path="course"))
                continue
            fact_id = str(item.get("fact_id") or "").strip()
            unit_path = str(item.get("unit_path") or "").strip()
            block_id = str(item.get("learning_block_id") or "").strip()
            basis = str(item.get("basis") or "").strip()
            if not fact_id or not unit_path or not block_id or basis not in {"SECTION_MATCH", "CONCEPT_MATCH", "SOURCE_REF_MATCH", "OWNERSHIP_MATCH"}:
                issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "Source Fact allocation entry is incomplete.", path="course"))
                continue
            allocated_ids.append(fact_id)
            assigned_targets[fact_id] = (unit_path, block_id)
        actual_targets: dict[str, tuple[str, str]] = {}
        for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
            if not isinstance(chapter, dict):
                continue
            for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
                if not isinstance(lesson, dict):
                    continue
                for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                    if not isinstance(unit, dict):
                        continue
                    unit_path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                    blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
                    block_by_fact: dict[str, str] = {}
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        block_id = str(block.get("id") or "").strip()
                        for fact_id in block.get("source_fact_ids", []) if isinstance(block.get("source_fact_ids"), list) else []:
                            if isinstance(fact_id, str) and fact_id.strip():
                                if fact_id in block_by_fact:
                                    issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "A Source Fact is assigned to more than one semantic block.", path=unit_path))
                                block_by_fact[fact_id] = block_id
                    for fact_id in unit.get("source_fact_ids", []) if isinstance(unit.get("source_fact_ids"), list) else []:
                        if isinstance(fact_id, str) and fact_id.strip():
                            if fact_id in actual_targets:
                                issues.append(_workflow_issue("SOURCE_FACT_ALLOCATION_AUTHORITY_INVALID", "A Source Fact is assigned to more than one unit.", path=unit_path))
                            actual_targets[fact_id] = (unit_path, block_by_fact.get(fact_id, ""))
        if (
            not allocation.get("complete")
            or allocation.get("required_count") != len(required_ids)
            or allocation.get("allocated_count") != len(required_ids)
            or len(allocated_ids) != len(set(allocated_ids))
            or set(allocated_ids) != required_ids
            or assigned_targets != actual_targets
        ):
            issues.append(_workflow_issue("UNALLOCATED_SOURCE_FACT", "Canonical Source Fact allocation is incomplete, inconsistent, or not server-owned.", path="course"))
        for item in allocation.get("unallocated", [])[:80] if isinstance(allocation.get("unallocated"), list) else []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "UNALLOCATED_SOURCE_FACT")
            issues.append(_workflow_issue(code, "A canonical Source Fact has no deterministic section/concept/source/block ownership.", path=str(item.get("path") or "course")))
        if issues:
            metrics = {"source_coverage": round(len(set(allocated_ids) & required_ids) / max(1, len(required_ids)), 4), "concept_coverage": None}
            return WorkflowValidationResult(issues, metrics)
        for item in allocation.get("quality_findings", []) if isinstance(allocation.get("quality_findings"), list) else []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "INSTRUCTIONAL_SCOPE_COARSE")
            if code != "INSTRUCTIONAL_SCOPE_COARSE":
                continue
            issues.append(_workflow_issue(
                code,
                "A dominant source section is represented by a single semantic block; review instructional granularity.",
                severity="warning",
                path=str(item.get("path") or "course"),
            ))
    elif blueprint.get("architecture_contract_version") == 3:
        if not isinstance(allocation, dict):
            return WorkflowValidationResult([
                _workflow_issue("UNALLOCATED_SOURCE_FACT", "Source Fact allocation metadata is missing.", path="course"),
            ])
        for invalid_fact_id in allocation.get("invalid_claimed_fact_ids", [])[:40]:
            issues.append(_workflow_issue(
                "SOURCE_FACT_UNAVAILABLE",
                f"The architecture claimed an unknown Source Fact '{invalid_fact_id}'.",
                path="course",
            ))
        for item in allocation.get("unallocated", [])[:80]:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "UNALLOCATED_SOURCE_FACT")
            issues.append(_workflow_issue(
                code,
                "A canonical Source Fact has no deterministic section/concept/source/block ownership.",
                path=str(item.get("path") or "course"),
            ))
        if not allocation.get("complete"):
            metrics = {
                "source_coverage": round(
                    int(allocation.get("allocated_count") or 0) / max(1, int(allocation.get("required_count") or 0)),
                    4,
                ),
                "concept_coverage": None,
            }
            return WorkflowValidationResult(issues or [
                _workflow_issue("UNALLOCATED_SOURCE_FACT", "Canonical Source Fact allocation is incomplete.", path="course"),
            ], metrics)
    unit_fact_ownership_issues = _validate_v4_unit_source_fact_ownership(blueprint)
    if unit_fact_ownership_issues:
        metrics = {
            "source_coverage": 1.0,
            "concept_coverage": None,
        }
        return WorkflowValidationResult([*issues, *unit_fact_ownership_issues], metrics)

    try:
        normalized = validate_lesson_author_blueprint(blueprint)
    except LessonAuthorBlueprintValidationError as error:
        return WorkflowValidationResult([
            _workflow_issue_from_blueprint_validation_error(error),
        ])
    try:
        validate_lesson_author_source_refs(normalized, known_source_refs)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("INVALID_SOURCE_REF", str(error), path="course"))
    try:
        source_metrics = validate_lesson_author_source_coverage(
            {"chapters": normalized.get("chapters", [])}, source_coverage_manifest,
        )
    except LessonAuthorProposalValidationError as error:
        source_metrics = source_coverage_metrics({"chapters": normalized.get("chapters", [])}, source_coverage_manifest)
        issues.append(_workflow_issue("SOURCE_COVERAGE_INCOMPLETE", str(error), path="course"))

    source_concepts = {
        str(concept.get("id"))
        for concept in source_map.get("concepts", []) if isinstance(concept, dict) and str(concept.get("id") or "").strip()
    }
    owners: dict[str, str] = {}
    objective_paths: dict[str, str] = {}
    positions: dict[str, int] = {}
    concept_paths: list[tuple[str, str]] = []
    lesson_position = 0
    for chapter_index, chapter in enumerate(normalized.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_path = f"chapter_{chapter_index}"
        for concept_id in chapter.get("concept_ids", []) or []:
            concept_paths.append((str(concept_id), chapter_path))
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_position += 1
            lesson_path = f"{chapter_path}.lesson_{lesson_index}"
            objectives = [str(value).strip() for value in lesson.get("learning_objectives", []) or [] if str(value).strip()]
            if not objectives:
                issues.append(_workflow_issue("EMPTY_LESSON", "Lesson has no learning objective.", path=lesson_path))
            for objective in objectives:
                key = re.sub(r"[^0-9a-zà-ỹ]+", " ", objective.casefold()).strip()
                if key in objective_paths:
                    issues.append(_workflow_issue("DUPLICATE_OBJECTIVE", "Learning objective is duplicated.", path=lesson_path, related_paths=[objective_paths[key]]))
                else:
                    objective_paths[key] = lesson_path
                if re.match(r"^(?:understand|know|hiểu|biết)\b", key):
                    issues.append(_workflow_issue("OBJECTIVE_TOO_GENERIC", "Learning objective starts with a non-observable verb.", severity="warning", path=lesson_path))
            primary_ids = [str(value) for value in lesson.get("primary_concept_ids", []) or []]
            for concept_id in primary_ids:
                if concept_id in owners:
                    issues.append(_workflow_issue("DUPLICATE_PRIMARY_CONCEPT_OWNERSHIP", "A core concept has more than one primary lesson owner.", path=lesson_path, related_paths=[owners[concept_id]]))
                else:
                    owners[concept_id] = lesson_path
                    positions[concept_id] = lesson_position
                concept_paths.append((concept_id, lesson_path))
            for concept_id in [
                *(lesson.get("supporting_concept_ids", []) or []),
                *(lesson.get("prerequisite_concept_ids", []) or []),
            ]:
                concept_paths.append((str(concept_id), lesson_path))
            if lesson.get("assessment_required") and not lesson.get("assessment_objective_refs"):
                issues.append(_workflow_issue("ASSESSMENT_ALIGNMENT_MISSING", "Assessment-required lesson does not name an objective.", path=lesson_path))
            units = lesson.get("units") if isinstance(lesson.get("units"), list) else []
            if not units:
                issues.append(_workflow_issue("EMPTY_LESSON", "Lesson has no units.", path=lesson_path))
            for unit_index, unit in enumerate(units, start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"{lesson_path}.unit_{unit_index}"
                for concept_id in unit.get("concept_ids", []) or []:
                    concept_paths.append((str(concept_id), unit_path))
                if not unit.get("source_fact_ids") and not (
                    blueprint.get("architecture_contract_version") == 4
                    and _is_v4_supporting_factless_unit(unit)
                ):
                    issues.append(_workflow_issue("LESSON_SOURCE_FACT_MISSING", "Unit has no Source Fact ownership.", path=unit_path))
                blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
                if len(blocks) == 1 and str(blocks[0].get("intent") if isinstance(blocks[0], dict) else "") == "introduction":
                    issues.append(_workflow_issue("LESSON_TOO_THIN", "Unit only has an introductory learning block.", path=unit_path))
    for concept_id, path in concept_paths:
        if concept_id and concept_id not in source_concepts:
            issues.append(_workflow_issue("UNSUPPORTED_CONCEPT", f"Concept '{concept_id}' is not in the Source Map.", path=path))
    for concept in source_map.get("concepts", []):
        if not isinstance(concept, dict):
            continue
        concept_id = str(concept.get("id") or "")
        dependent_position = positions.get(concept_id)
        for prerequisite_id in concept.get("prerequisite_concept_ids", []) or []:
            prerequisite_position = positions.get(str(prerequisite_id))
            if prerequisite_position is not None and dependent_position is not None and prerequisite_position > dependent_position:
                issues.append(_workflow_issue("PREREQUISITE_ORDER_INVALID", "A prerequisite concept is introduced after its dependent concept.", path=owners.get(concept_id, "course"), related_paths=[owners.get(str(prerequisite_id), "course")]))
    required_concepts = {
        str(concept.get("id"))
        for concept in source_map.get("concepts", [])
        if isinstance(concept, dict) and concept.get("importance") == "core" and str(concept.get("id") or "").strip()
    }
    covered_concepts = required_concepts & set(owners)
    if required_concepts - covered_concepts:
        issues.append(_workflow_issue("CONCEPT_COVERAGE_INCOMPLETE", "One or more core Source Map concepts have no primary lesson owner.", path="course"))
    metrics = {
        "source_coverage": source_metrics.get("coverage_ratio"),
        "concept_coverage": round(len(covered_concepts) / len(required_concepts), 4) if required_concepts else None,
    }
    return WorkflowValidationResult(issues, metrics)
