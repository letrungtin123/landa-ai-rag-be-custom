"""V5 semantic-delta repair targets (evidence, assessment, action intent)."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.assessment_planner import (
    assessment_intent_repair_options,
    assessment_plan_fingerprint,
    assessment_teaching_semantic_descriptor,
    compile_v5_assessment_plan,
)
from app.lesson_author_blueprint import (
    ACTION_OBJECTIVE_REPAIR_INTENTS,
    INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS,
    SEMANTIC_LEARNING_BLOCK_INTENTS,
    SEMANTIC_REPAIR_CONTRACT_VERSION,
    V5_SEMANTIC_DELTA_REPAIR_OPERATIONS,
)
from app.services.lesson_author.architecture_shape import (
    _V5_GENERIC_EXPLANATION_INTENTS,
    _V5_TEACHING_INTENTS,
    _blueprint_path_object,
    _repair_parent_lesson_path,
    _semantic_delta_blocks,
    _v5_text_ids,
)
from app.services.lesson_author.architecture_validation import (
    _v5_base_teaching_anchor_candidates,
    _v5_fully_aligned_teaching_anchor_candidates,
    _v5_lesson_block_records,
    _v5_objective_ids_are_local,
)
from app.workflows.contracts import RepairTarget, WorkflowFailure, safe_workflow_path

_V5_SEMANTIC_DELTA_OPERATIONS = V5_SEMANTIC_DELTA_REPAIR_OPERATIONS


def _is_v5_semantic_delta_repair(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
) -> bool:
    return (
        blueprint.get("architecture_contract_version") == 5
        and bool(targets)
        and all(
            isinstance(target.get("semantic_operations"), list)
            and bool(target.get("semantic_operations"))
            for target in targets
        )
    )


def _semantic_delta_failure(
    message: str,
    *,
    path: str,
    patch_count: int,
    internal_code: str,
    guard_reason: str,
    semantic_operation: str | None = None,
    block_id: str | None = None,
    outer_field: str | None = None,
    selected_intent: str | None = None,
) -> WorkflowFailure:
    diagnostics: dict[str, Any] = {
        "repair_target_path": safe_workflow_path(path),
        "patch_count": patch_count,
        "guard_reason": guard_reason,
    }
    if semantic_operation:
        diagnostics["semantic_operation"] = semantic_operation
    if outer_field:
        diagnostics["outer_field"] = outer_field
    if selected_intent is not None:
        diagnostics["semantic_repair_contract_version"] = SEMANTIC_REPAIR_CONTRACT_VERSION
        # Log only a known enum, never an arbitrary provider value.
        diagnostics["selected_intent"] = (
            selected_intent if selected_intent in SEMANTIC_LEARNING_BLOCK_INTENTS else "UNKNOWN"
        )
        diagnostics["intent_allowlist_id"] = (
            "INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS"
            if semantic_operation == "add_instructional_support_block"
            else "ACTION_OBJECTIVE_REPAIR_INTENTS"
        )
    if block_id and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", block_id):
        diagnostics["block_id"] = block_id
    return WorkflowFailure(
        "ARCHITECTURE_REPAIR_INVALID",
        message,
        internal_code=internal_code,
        failure_stage="architecture_repair_semantic_delta_guard",
        diagnostics=diagnostics,
    )


def _semantic_delta_unit_and_lesson(
    blueprint: dict[str, Any],
    path: str,
    *,
    patch_count: int,
    operation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    unit = _blueprint_path_object(blueprint, path)
    lesson = _blueprint_path_object(blueprint, _repair_parent_lesson_path(path))
    if not isinstance(unit, dict) or not isinstance(lesson, dict):
        raise _semantic_delta_failure(
            "Semantic repair target no longer exists.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
            guard_reason="TARGET_BOUNDARY_MUTATION",
            semantic_operation=operation,
        )
    return unit, lesson


def _v5_target_evidence_scope_ids(target: RepairTarget) -> set[str]:
    """Return only server-recorded mismatch scope IDs for one repair target."""

    scope_ids: set[str] = set()
    for diagnostic in target.get("diagnostics", []):
        if not isinstance(diagnostic, dict):
            continue
        if str(diagnostic.get("code") or "") != "EVIDENCE_SCOPE_CONCEPT_MISMATCH":
            continue
        scope_id = str(diagnostic.get("evidence_scope_id") or "").strip()
        if scope_id:
            scope_ids.add(scope_id)
    return scope_ids


def _v5_prepare_evidence_alignment_targets(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    source_map: dict[str, Any],
) -> list[RepairTarget]:
    """Attach immutable, concept-only alignment authority to V5 targets.

    A provider never derives these values.  The target is valid only when the
    exact affected unit already references the immutable evidence scopes and
    its parent lesson can safely contain every required concept.
    """

    scopes = {
        str(scope.get("id") or "").strip(): scope
        for scope in source_map.get("source_evidence_scopes", [])
        if isinstance(scope, dict) and str(scope.get("id") or "").strip()
    }
    known_concepts = {
        str(concept.get("id") or "").strip()
        for concept in source_map.get("concepts", [])
        if isinstance(concept, dict) and str(concept.get("id") or "").strip()
    }
    prepared: list[RepairTarget] = []

    for raw_target in targets:
        target = deepcopy(raw_target)
        operations = {
            str(value).strip()
            for value in target.get("semantic_operations", [])
            if isinstance(value, str) and value.strip()
        }
        if "align_concepts_to_evidence" not in operations:
            prepared.append(target)
            continue

        path = str(target.get("path") or "")
        unit, lesson = _semantic_delta_unit_and_lesson(
            blueprint,
            path,
            patch_count=0,
            operation="align_concepts_to_evidence",
        )
        scope_ids = _v5_target_evidence_scope_ids(target)
        if not scope_ids or not scope_ids.issubset(scopes):
            raise _semantic_delta_failure(
                "Evidence-concept repair has no immutable compatible source scope.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                guard_reason="NO_COMPATIBLE_CONCEPT",
                semantic_operation="align_concepts_to_evidence",
            )

        blocks = _semantic_delta_blocks(unit)
        referenced_by_block: dict[str, set[str]] = {}
        for block in blocks:
            block_id = str(block.get("id") or "").strip()
            if not block_id:
                continue
            block_scope_ids = (
                _v5_text_ids(block.get("primary_evidence_scope_ids"))
                | _v5_text_ids(block.get("supporting_evidence_scope_ids"))
            )
            overlap = block_scope_ids & scope_ids
            if overlap:
                referenced_by_block[block_id] = overlap
        if not referenced_by_block:
            raise _semantic_delta_failure(
                "Evidence-concept repair target does not own the immutable scope it is asked to align.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                guard_reason="NO_COMPATIBLE_CONCEPT",
                semantic_operation="align_concepts_to_evidence",
            )

        scope_concepts: set[str] = set()
        for scope_id in scope_ids:
            scope_concepts.update(_v5_text_ids(scopes[scope_id].get("concept_ids")))
        if not scope_concepts or not scope_concepts.issubset(known_concepts):
            raise _semantic_delta_failure(
                "Evidence-concept repair scope has no valid canonical concept alignment.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                guard_reason="NO_COMPATIBLE_CONCEPT",
                semantic_operation="align_concepts_to_evidence",
            )

        lesson_primary = _v5_text_ids(lesson.get("primary_concept_ids"))
        lesson_supporting = _v5_text_ids(lesson.get("supporting_concept_ids"))
        lesson_allowed = lesson_primary | lesson_supporting
        if not lesson_allowed or not (scope_concepts & lesson_allowed):
            raise _semantic_delta_failure(
                "No target-lesson concept is compatible with the immutable evidence scope.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                guard_reason="NO_COMPATIBLE_CONCEPT",
                semantic_operation="align_concepts_to_evidence",
            )
        if not scope_concepts.issubset(lesson_allowed):
            raise _semantic_delta_failure(
                "Evidence concepts are outside the target lesson's immutable concept scope.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                semantic_operation="align_concepts_to_evidence",
            )

        unit_concepts = _v5_text_ids(unit.get("concept_ids"))
        unit_primary = _v5_text_ids(unit.get("primary_concept_ids"))
        block_primary: set[str] = set()
        block_concepts: dict[str, list[str]] = {}
        for block in blocks:
            block_id = str(block.get("id") or "").strip()
            matched_scope_ids = referenced_by_block.get(block_id)
            if not block_id or not matched_scope_ids:
                continue
            block_primary.update(_v5_text_ids(block.get("primary_concept_ids")))
            required_block_concepts = _v5_text_ids(block.get("concept_ids"))
            for scope_id in matched_scope_ids:
                required_block_concepts.update(_v5_text_ids(scopes[scope_id].get("concept_ids")))
            if not required_block_concepts.issubset(lesson_allowed):
                raise _semantic_delta_failure(
                    "A target learning block would require concepts outside its lesson scope.",
                    path=path,
                    patch_count=0,
                    internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    semantic_operation="align_concepts_to_evidence",
                    block_id=block_id,
                )
            block_concepts[block_id] = sorted(required_block_concepts)

        required_concepts = unit_concepts | scope_concepts
        required_primary = unit_primary | block_primary
        if (
            not required_concepts.issubset(lesson_allowed)
            or not required_primary.issubset(lesson_primary)
            or not required_primary.issubset(required_concepts)
        ):
            raise _semantic_delta_failure(
                "The target unit has no concept alignment compatible with its lesson ownership.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                semantic_operation="align_concepts_to_evidence",
            )

        if required_primary:
            primary_options = [sorted(required_primary)]
        else:
            primary_options = [[concept_id] for concept_id in sorted(scope_concepts & lesson_primary)]
        if not primary_options:
            raise _semantic_delta_failure(
                "No primary concept can be selected without widening the target lesson scope.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                guard_reason="NO_COMPATIBLE_CONCEPT",
                semantic_operation="align_concepts_to_evidence",
            )

        target["required_concept_ids"] = sorted(required_concepts)
        target["allowed_concept_ids"] = sorted(required_concepts)
        target["known_concept_ids"] = sorted(known_concepts)
        target["primary_concept_options"] = primary_options
        target["evidence_alignment_block_concepts"] = block_concepts
        prepared.append(target)
    return prepared


def _v5_prepare_assessment_alignment_targets(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
) -> list[RepairTarget]:
    """Classify assessment repair before a provider can be called.

    ``repair_knowledge_check`` is retained only when an earlier teaching
    anchor is *already* fully aligned.  Otherwise the server may authorize a
    narrow objective-link delta only for base-eligible existing teaching
    blocks.  The provider never receives provenance, source facts, concepts,
    or arbitrary paths.
    """

    prepared: list[RepairTarget] = []
    for raw_target in targets:
        target = deepcopy(raw_target)
        operations = set(target.get("semantic_operations") or [])
        assessment_codes = {
            "ASSESSMENT_OBJECTIVE_NOT_COVERED",
            "ASSESSMENT_EVIDENCE_NOT_GROUNDED",
            "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED",
        }
        if operations != {"repair_knowledge_check"} or not assessment_codes.intersection(target.get("codes") or []):
            prepared.append(target)
            continue

        path = str(target.get("path") or "")
        unit, lesson = _semantic_delta_unit_and_lesson(
            blueprint,
            path,
            patch_count=0,
            operation="repair_assessment_alignment",
        )
        lesson_path = _repair_parent_lesson_path(path)
        records = _v5_lesson_block_records(lesson, lesson_path)
        allowed_check_ids = {
            str(value).strip()
            for value in target.get("allowed_block_ids", [])
            if isinstance(value, str) and value.strip()
        }
        checks = [
            record for record in records
            if record["unit_path"] == path
            and record["block_id"] in allowed_check_ids
            and str(record["block"].get("intent") or "").strip() == "knowledge_check"
        ]
        if len(checks) != 1:
            raise _semantic_delta_failure(
                "Assessment repair target must resolve exactly one existing knowledge check.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                guard_reason="INVALID_BLOCK_ID",
                semantic_operation="repair_assessment_alignment",
            )
        check = checks[0]
        assessment_refs = _v5_text_ids(lesson.get("assessment_objective_refs"))
        requested_refs = {
            str(value).strip()
            for value in target.get("allowed_objective_ids", [])
            if isinstance(value, str) and value.strip()
        } or assessment_refs
        if (
            not _v5_objective_ids_are_local(lesson, assessment_refs)
            or not requested_refs.issubset(assessment_refs)
        ):
            raise _semantic_delta_failure(
                "Assessment repair cannot use an invalid local lesson objective.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                guard_reason="INVALID_LOCAL_OBJECTIVE",
                semantic_operation="repair_assessment_alignment",
                block_id=str(check["block_id"]),
            )

        # Preserve the established narrow check-only repair when one fully
        # aligned teaching anchor already exists in the same target unit.  A
        # cross-unit or per-objective case uses the richer typed delta below;
        # that path explicitly authorizes every touched block address.
        fully_aligned = _v5_fully_aligned_teaching_anchor_candidates(
            lesson,
            records,
            knowledge_check=check,
            objective_refs=requested_refs,
        )
        if len(fully_aligned) == 1 and str(fully_aligned[0].get("unit_path") or "") == path:
            target["allowed_objective_ids"] = sorted(requested_refs)
            prepared.append(target)
            continue

        candidates: list[dict[str, Any]] = []
        candidate_counts: dict[str, int] = {}
        for objective_ref in sorted(requested_refs):
            base_candidates = _v5_base_teaching_anchor_candidates(
                lesson,
                records,
                knowledge_check=check,
                objective_refs={objective_ref},
            )
            # Only when ordinary anchors are absent, expose potential intent
            # repairs. This does not make them valid teaching anchors yet.
            intent_options: dict[str, tuple[str, ...]] = {}
            if not base_candidates:
                for record in records:
                    options = assessment_intent_repair_options(
                        teaching_block=record["block"], knowledge_check_block=check["block"],
                        objective_refs={objective_ref}, unit_objective_refs=set(record["unit_objective_refs"]),
                        precedes_check=record["position"] < check["position"],
                        same_unit=record["unit_path"] == path,
                    )
                    if options:
                        base_candidates.append(record)
                        intent_options[record["block_id"]] = options
            candidate_counts[objective_ref] = len(base_candidates)
            if not base_candidates:
                raise _semantic_delta_failure(
                    "No existing source-compatible teaching anchor can be aligned safely for one assessment objective.",
                    path=path,
                    patch_count=0,
                    internal_code="ARCH_REPAIR_NO_VALID_TEACHING_ANCHOR",
                    guard_reason="NO_VALID_TEACHING_ANCHOR",
                    semantic_operation="repair_assessment_alignment",
                    block_id=str(check["block_id"]),
                )
            candidates.extend({
                "knowledge_check_path": path,
                "knowledge_check_block_id": str(check["block_id"]),
                "teaching_block_path": str(candidate["unit_path"]),
                "teaching_block_id": str(candidate["block_id"]),
                "learning_objective_refs": [objective_ref],
                **({
                    "allowed_intents": list(intent_options[candidate["block_id"]]),
                    "semantic_descriptor": assessment_teaching_semantic_descriptor(candidate["block"]),
                    "objective_text": str(lesson["learning_objectives"][int(objective_ref[3:]) - 1])[:480],
                }
                   if candidate["block_id"] in intent_options else {}),
            } for candidate in base_candidates)
        # Exact duplicate candidate records would make provider selection
        # non-deterministic; reject rather than silently choosing a path.
        candidate_keys = {
            (item["knowledge_check_path"], item["knowledge_check_block_id"],
             item["teaching_block_path"], item["teaching_block_id"],
             tuple(item["learning_objective_refs"]))
            for item in candidates
        }
        if len(candidate_keys) != len(candidates):
            raise _semantic_delta_failure(
                "Assessment repair produced ambiguous duplicate anchor authority.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_AMBIGUOUS_SUPPORTING_EVIDENCE",
                guard_reason="AMBIGUOUS_TEACHING_ANCHOR",
                semantic_operation="repair_assessment_alignment",
                block_id=str(check["block_id"]),
            )
        target["semantic_operations"] = ["repair_assessment_alignment"]
        target["allowed_operations"] = ["repair_assessment_alignment"]
        target["allowed_block_ids"] = [str(check["block_id"])]
        target["allowed_objective_ids"] = sorted(requested_refs)
        target["assessment_alignment_candidates"] = candidates
        target["assessment_alignment_candidate_counts"] = candidate_counts
        if all(count == 1 for count in candidate_counts.values()) and not any(c.get("allowed_intents") for c in candidates):
            target["deterministic_semantic_delta"] = True
        prepared.append(target)
    return prepared


def _v5_prepare_assessment_plan_selection_targets(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
) -> list[RepairTarget]:
    """Bind a semantic resolver to compiler-produced candidate authority.

    The repair scheduler may call a provider only after this preflight proves
    the exact lesson/objective/block candidate set. The provider sees opaque
    paths/IDs plus compact instructional semantics; primary/supporting scopes,
    source references and canonical facts remain server-owned.
    """

    prepared: list[RepairTarget] = []
    for raw_target in targets:
        target = deepcopy(raw_target)
        if set(target.get("semantic_operations") or []) != {"select_assessment_teaching_alignment"}:
            prepared.append(target)
            continue
        path = str(target.get("path") or "")
        compilation = compile_v5_assessment_plan(blueprint, lesson_paths={path})
        if compilation.status != "NEEDS_SEMANTIC_RESOLUTION":
            raise _semantic_delta_failure(
                "Assessment semantic selection no longer has a server-approved unresolved candidate set.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_ASSESSMENT_PLAN_STALE",
                guard_reason="TARGET_BOUNDARY_MUTATION",
                semantic_operation="select_assessment_teaching_alignment",
            )
        requested_objectives = {
            str(value).strip()
            for value in target.get("allowed_objective_ids", [])
            if isinstance(value, str) and value.strip()
        }
        candidate_items: list[dict[str, Any]] = []
        available_objectives: set[str] = set()
        for (lesson_path, objective_ref), candidates in sorted(compilation.candidates.items()):
            if lesson_path != path or objective_ref not in requested_objectives:
                continue
            available_objectives.add(objective_ref)
            candidate_items.extend(candidate.safe_provider_value() for candidate in candidates)
        if not requested_objectives or available_objectives != requested_objectives:
            raise _semantic_delta_failure(
                "Assessment semantic selection is missing one approved local objective candidate set.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_VALID_TEACHING_ANCHOR",
                guard_reason="NO_VALID_TEACHING_ANCHOR",
                semantic_operation="select_assessment_teaching_alignment",
            )
        lesson = _semantic_delta_lesson_target(
            blueprint,
            path,
            patch_count=0,
            operation="select_assessment_teaching_alignment",
        )
        objective_values = lesson.get("learning_objectives")
        objective_values = objective_values if isinstance(objective_values, list) else []
        objective_descriptors = {
            objective_ref: str(objective_values[int(objective_ref.removeprefix("lo_")) - 1]).strip()[:480]
            for objective_ref in sorted(requested_objectives)
            if objective_ref.removeprefix("lo_").isdigit()
            and 0 < int(objective_ref.removeprefix("lo_")) <= len(objective_values)
            and isinstance(objective_values[int(objective_ref.removeprefix("lo_")) - 1], str)
            and str(objective_values[int(objective_ref.removeprefix("lo_")) - 1]).strip()
        }
        if set(objective_descriptors) != requested_objectives:
            raise _semantic_delta_failure(
                "Assessment semantic selection cannot expose an invalid local objective descriptor.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                guard_reason="INVALID_OBJECTIVE_REF",
                semantic_operation="select_assessment_teaching_alignment",
            )
        target["assessment_plan_candidates"] = candidate_items
        target["assessment_plan_objectives"] = objective_descriptors
        target["assessment_plan_fingerprint"] = assessment_plan_fingerprint(
            blueprint,
            lesson_path=path,
            candidates=compilation.candidates,
        )
        prepared.append(target)
    return prepared


def _v5_deterministic_assessment_alignment_payload(
    targets: list[RepairTarget],
) -> dict[str, Any] | None:
    deterministic_targets = [
        target for target in targets
        if target.get("deterministic_semantic_delta") is True
        and target.get("semantic_operations") == ["repair_assessment_alignment"]
    ]
    if not deterministic_targets:
        return None
    patches: list[dict[str, Any]] = []
    for target in deterministic_targets:
        candidates = target.get("assessment_alignment_candidates")
        if not isinstance(candidates, list) or not candidates or not all(isinstance(candidate, dict) for candidate in candidates):
            return None
        check_ids = {
            str(candidate.get("knowledge_check_block_id") or "").strip()
            for candidate in candidates
        }
        expected_refs = {
            str(value).strip()
            for value in target.get("allowed_objective_ids", [])
            if isinstance(value, str) and value.strip()
        }
        by_teaching_block: dict[str, set[str]] = {}
        for candidate in candidates:
            block_id = str(candidate.get("teaching_block_id") or "").strip()
            refs = _v5_text_ids(candidate.get("learning_objective_refs"))
            if not block_id or len(refs) != 1:
                return None
            by_teaching_block.setdefault(block_id, set()).update(refs)
        if len(check_ids) != 1 or set().union(*by_teaching_block.values()) != expected_refs:
            return None
        patches.append({
            "path": target["path"],
            "operation": "repair_assessment_alignment",
            "knowledge_check_block_id": next(iter(check_ids)),
            "teaching_selections": [
                {"teaching_block_id": block_id, "learning_objective_refs": sorted(refs)}
                for block_id, refs in sorted(by_teaching_block.items())
            ],
        })
    return {"patches": patches}


def _v5_prepare_action_intent_targets(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
) -> list[RepairTarget]:
    """Bind action-intent repair to existing teaching blocks and safe intents.

    The initial Architect may use the wider semantic intent vocabulary. Once a
    deterministic validator proves that a unit needs action treatment, the
    repair is deliberately narrower: only a procedure or worked example can
    replace the generic teaching intent. Both remain valid assessment anchors,
    so a successful local repair cannot make an already-grounded later check
    lose its teaching predecessor.
    """

    prepared: list[RepairTarget] = []
    for raw_target in targets:
        target = deepcopy(raw_target)
        if (
            set(target.get("semantic_operations") or []) != {"set_block_intent"}
            or "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH" not in set(target.get("codes") or [])
        ):
            prepared.append(target)
            continue

        path = str(target.get("path") or "")
        unit, lesson = _semantic_delta_unit_and_lesson(
            blueprint,
            path,
            patch_count=0,
            operation="set_block_intent",
        )
        unit_objective_refs = _v5_text_ids(unit.get("learning_objective_refs"))
        if not _v5_objective_ids_are_local(lesson, unit_objective_refs):
            raise _semantic_delta_failure(
                "Action-intent repair requires exact local objectives from the target unit.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                guard_reason="INVALID_LOCAL_OBJECTIVE",
                semantic_operation="set_block_intent",
            )
        requested_block_ids = {
            str(value).strip()
            for value in target.get("allowed_block_ids", [])
            if isinstance(value, str) and value.strip()
        }
        repairable_block_ids = {
            str(block.get("id") or "").strip()
            for block in _semantic_delta_blocks(unit)
            if str(block.get("id") or "").strip() in requested_block_ids
            and str(block.get("intent") or "").strip() in _V5_GENERIC_EXPLANATION_INTENTS
        }
        if not repairable_block_ids:
            raise _semantic_delta_failure(
                "Action-intent repair has no generic teaching block in the approved target unit.",
                path=path,
                patch_count=0,
                internal_code="ARCH_REPAIR_NO_ACTION_TEACHING_BLOCK",
                guard_reason="NO_ACTION_TEACHING_BLOCK",
                semantic_operation="set_block_intent",
            )

        lesson_path = _repair_parent_lesson_path(path)
        records = _v5_lesson_block_records(lesson, lesson_path)
        assessment_refs = _v5_text_ids(lesson.get("assessment_objective_refs"))
        dependent_objectives: set[str] = set()
        for record in records:
            block = record.get("block")
            if not isinstance(block, dict) or str(record.get("block_id") or "") not in repairable_block_ids:
                continue
            record_refs = _v5_text_ids(block.get("learning_objective_refs"))
            for check in records:
                check_block = check.get("block")
                if (
                    not isinstance(check_block, dict)
                    or int(check.get("position") or 0) <= int(record.get("position") or 0)
                    or str(check_block.get("intent") or "").strip() != "knowledge_check"
                ):
                    continue
                dependent_objectives.update(
                    record_refs
                    & _v5_text_ids(check_block.get("learning_objective_refs"))
                    & assessment_refs
                )

        target["allowed_block_ids"] = sorted(repairable_block_ids)
        target["allowed_intents_by_block_id"] = {
            block_id: sorted(ACTION_OBJECTIVE_REPAIR_INTENTS)
            for block_id in sorted(repairable_block_ids)
        }
        target["assessment_dependency_objective_count"] = len(dependent_objectives)
        prepared.append(target)
    return prepared


def _v5_safe_action_intent_repair_metadata(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    patches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return operational intent-repair metadata without source or content text."""

    targets_by_path = {str(target.get("path") or ""): target for target in targets}
    repairs: list[dict[str, Any]] = []
    for patch in patches:
        if str(patch.get("operation") or "") != "set_block_intent":
            continue
        path = str(patch.get("path") or "")
        target = targets_by_path.get(path)
        block_id = str(patch.get("block_id") or "").strip()
        next_intent = str(patch.get("intent") or "").strip()
        unit = _blueprint_path_object(blueprint, path)
        blocks = _semantic_delta_blocks(unit) if isinstance(unit, dict) else []
        previous_intent = next(
            (str(block.get("intent") or "").strip() for block in blocks if str(block.get("id") or "").strip() == block_id),
            "",
        )
        raw_allowed = target.get("allowed_intents_by_block_id") if isinstance(target, dict) else None
        allowed = raw_allowed.get(block_id, []) if isinstance(raw_allowed, dict) else ACTION_OBJECTIVE_REPAIR_INTENTS
        repairs.append({
            "target_path": safe_workflow_path(path),
            "block_id": block_id,
            "previous_intent": previous_intent,
            "next_intent": next_intent,
            "allowed_intents": sorted({str(value).strip() for value in allowed if isinstance(value, str) and value.strip()}),
            "assessment_dependency_objective_count": max(
                0,
                int(target.get("assessment_dependency_objective_count") or 0),
            ) if isinstance(target, dict) else 0,
        })
    return {
        "action_intent_repair_count": len(repairs),
        **({"action_intent_repairs": repairs} if repairs else {}),
    }


def _v5_prepare_semantic_repair_targets(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    source_map: dict[str, Any],
) -> list[RepairTarget]:
    """Attach all V5 server-owned repair authority before provider budgeting."""

    return _v5_prepare_action_intent_targets(
        blueprint,
        _v5_prepare_assessment_alignment_targets(
            blueprint,
            _v5_prepare_assessment_plan_selection_targets(
                blueprint,
                _v5_prepare_evidence_alignment_targets(blueprint, targets, source_map),
            ),
        ),
    )


def _semantic_delta_objectives_are_local(
    lesson: dict[str, Any],
    objective_refs: Any,
    *,
    target: RepairTarget,
    path: str,
    patch_count: int,
    operation: str,
    block_id: str | None,
) -> list[str]:
    if not isinstance(objective_refs, list) or not objective_refs or not all(isinstance(value, str) and value.strip() for value in objective_refs):
        raise _semantic_delta_failure(
            "Semantic repair must reference one or more local learning objectives.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
            guard_reason="INVALID_OBJECTIVE_REF",
            semantic_operation=operation,
            block_id=block_id,
        )
    valid_ids = {
        f"lo_{index}"
        for index, value in enumerate(lesson.get("learning_objectives", []), start=1)
        if isinstance(value, str) and value.strip()
    }
    normalized = [str(value).strip() for value in objective_refs]
    allowed_ids = {
        str(value).strip()
        for value in target.get("allowed_objective_ids", [])
        if isinstance(value, str) and value.strip()
    }
    if (
        len(normalized) != len(set(normalized))
        or not set(normalized).issubset(valid_ids)
        or (allowed_ids and not set(normalized).issubset(allowed_ids))
    ):
        raise _semantic_delta_failure(
            "Semantic repair referenced an objective outside its approved local lesson scope.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
            guard_reason="INVALID_OBJECTIVE_REF",
            semantic_operation=operation,
            block_id=block_id,
        )
    return normalized


def _semantic_delta_server_block_id(
    unit: dict[str, Any],
    *,
    prefix: str = "knowledge_check",
) -> str:
    existing = {
        str(block.get("id") or "").strip()
        for block in _semantic_delta_blocks(unit)
        if str(block.get("id") or "").strip()
    }
    index = 1
    while f"lb_server_{prefix}_{index}" in existing:
        index += 1
    return f"lb_server_{prefix}_{index}"


def _semantic_delta_compatible_teaching_blocks(
    lesson: dict[str, Any],
    *,
    target_unit: dict[str, Any],
    target_block_id: str | None,
    objective_refs: list[str],
    require_before_block: bool,
) -> list[dict[str, Any]]:
    """Return eligible evidence-owning teaching blocks in lesson order.

    The operation target has already restricted the selected block ID. This
    helper still validates order, objective coverage, canonical teaching intent,
    and non-empty primary scope server-side.  It never expands scope or lets a
    provider choose provenance.
    """

    records = _v5_lesson_block_records(lesson, "course")
    target: dict[str, Any] | None = None
    # Match by unit identity as block IDs are only guaranteed unique inside
    # their unit. This preserves the existing target-unit authority.
    for record in records:
        if str(record["block_id"]) != str(target_block_id or ""):
            continue
        unit_path = str(record["unit_path"])
        unit_index = int(unit_path.rsplit("unit_", 1)[1]) - 1
        units = lesson.get("units", [])
        if 0 <= unit_index < len(units) and units[unit_index] is target_unit:
            target = record
            break
    if require_before_block:
        if target is None:
            return []
        return [
            record["block"]
            for record in _v5_fully_aligned_teaching_anchor_candidates(
                lesson,
                records,
                knowledge_check=target,
                objective_refs=set(objective_refs),
            )
        ]

    candidates: list[dict[str, Any]] = []
    for record in records:
        block = record["block"]
        if str(block.get("intent") or "").strip() not in _V5_TEACHING_INTENTS:
            continue
        if not _v5_text_ids(block.get("primary_evidence_scope_ids")):
            continue
        if set(objective_refs).issubset(_v5_text_ids(block.get("learning_objective_refs"))):
            candidates.append(block)
    return candidates


def _semantic_delta_lesson_target(
    blueprint: dict[str, Any],
    path: str,
    *,
    patch_count: int,
    operation: str,
) -> dict[str, Any]:
    """Resolve a lesson-scoped semantic delta without widening its target."""

    lesson = _blueprint_path_object(blueprint, path)
    if not isinstance(lesson, dict) or not re.fullmatch(r"chapter_\d+\.lesson_\d+", path):
        raise _semantic_delta_failure(
            "Semantic repair target no longer resolves to its approved lesson.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
            guard_reason="TARGET_BOUNDARY_MUTATION",
            semantic_operation=operation,
        )
    return lesson


def _semantic_delta_instructional_content(
    value: Any,
    *,
    path: str,
    patch_count: int,
    operation: str,
    block_id: str,
) -> dict[str, str]:
    """Accept compact provider-owned semantics, never rendered content/data."""

    if not isinstance(value, dict) or set(value) - {"purpose", "learner_action"}:
        raise _semantic_delta_failure(
            "Instructional support content is outside the typed semantic contract.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_PATCH_INVALID",
            guard_reason="DISALLOWED_OUTER_FIELD",
            semantic_operation=operation,
            block_id=block_id,
            outer_field="content",
        )
    normalized: dict[str, str] = {}
    for key in ("purpose", "learner_action"):
        raw = value.get(key)
        if raw is None and key == "learner_action":
            continue
        if not isinstance(raw, str) or not raw.strip() or len(raw.strip()) > INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS:
            raise _semantic_delta_failure(
                "Instructional support content must contain bounded semantic text.",
                path=path,
                patch_count=patch_count,
                internal_code="ARCH_REPAIR_PATCH_INVALID",
                guard_reason="INVALID_SEMANTIC_CONTENT",
                semantic_operation=operation,
                block_id=block_id,
                outer_field="content",
            )
        normalized[key] = raw.strip()
    if "purpose" not in normalized:
        raise _semantic_delta_failure(
            "Instructional support content requires a semantic purpose.",
            path=path,
            patch_count=patch_count,
            internal_code="ARCH_REPAIR_PATCH_INVALID",
            guard_reason="INVALID_SEMANTIC_CONTENT",
            semantic_operation=operation,
            block_id=block_id,
            outer_field="content",
        )
    return normalized
