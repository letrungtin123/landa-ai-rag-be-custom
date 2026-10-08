"""Application of V5 semantic-delta and V4/V5 course-architecture repair patches."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.assessment_planner import (
    assessment_intent_repair_options,
    assessment_plan_fingerprint,
    compile_v5_assessment_plan,
    preserve_assessment_visual_support,
)
from app.lesson_author_blueprint import (
    ACTION_OBJECTIVE_REPAIR_INTENTS,
    INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS,
    SEMANTIC_LEARNING_BLOCK_INTENTS,
    semantic_delta_required_fields,
)
from app.services.lesson_author.architecture_shape import (
    _V5_TEACHING_INTENTS,
    _blueprint_path_object,
    _repair_parent_lesson_path,
    _semantic_delta_blocks,
    _v5_text_ids,
)
from app.services.lesson_author.architecture_validation import (
    _v5_base_teaching_anchor_candidates,
    _v5_lesson_block_records,
)
from app.services.lesson_author.repair.guards import (
    _architecture_repair_preserves_unaffected_snapshot,
    _assert_v5_primary_provenance_preserved,
    _contains_provider_fact_ownership,
    _is_removable_factless_reinforcement_unit,
    _repair_keeps_existing_source_scope,
    _repair_scope_violation,
    _repair_unit_parent_and_index,
    _v5_primary_provenance_snapshot,
    _validate_v5_repair_patch_set_before_apply,
)
from app.services.lesson_author.repair.prompt import _semantic_architecture_snapshot
from app.services.lesson_author.repair.targets import (
    _V5_SEMANTIC_DELTA_OPERATIONS,
    _is_v5_semantic_delta_repair,
    _semantic_delta_compatible_teaching_blocks,
    _semantic_delta_failure,
    _semantic_delta_instructional_content,
    _semantic_delta_lesson_target,
    _semantic_delta_objectives_are_local,
    _semantic_delta_server_block_id,
    _semantic_delta_unit_and_lesson,
)
from app.workflows.contracts import RepairTarget, WorkflowFailure, safe_workflow_path


def _apply_v5_semantic_delta_repair_patches(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Apply provider-owned instructional deltas without exposing provenance.

    The input baseline remains untouched until every operation has passed. The
    returned candidate intentionally strips server allocation artifacts so the
    existing allocator is still the only canonical-fact authority.
    """

    patches = payload.get("patches") if isinstance(payload.get("patches"), list) else []
    expected = {str(target["path"]): target for target in targets}
    if len(patches) != len(expected):
        raise _semantic_delta_failure(
            "Semantic repair did not provide one operation for every approved target.",
            path="course",
            patch_count=len(patches),
            internal_code="ARCH_REPAIR_TARGET_MISSING",
            guard_reason="TARGET_BOUNDARY_MUTATION",
        )
    result = deepcopy(_semantic_architecture_snapshot(blueprint))
    _v5_primary_provenance_snapshot(result, patch_count=len(patches))
    seen: set[str] = set()
    replacement_paths: set[str] = set()

    for patch in patches:
        if not isinstance(patch, dict):
            raise _semantic_delta_failure(
                "Semantic repair contains an invalid operation.",
                path="course",
                patch_count=len(patches),
                internal_code="ARCH_REPAIR_PATCH_INVALID",
                guard_reason="DISALLOWED_OUTER_FIELD",
            )
        path = str(patch.get("path") or "")
        operation = str(patch.get("operation") or "").strip()
        target = expected.get(path)
        if target is None or path in seen:
            raise _repair_scope_violation(
                "Semantic repair tried to change an unapproved target.",
                path=path,
                patch_count=len(patches),
                internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                failure_stage="architecture_repair_semantic_delta_guard",
                guard_reason="TARGET_BOUNDARY_MUTATION",
                semantic_operation=operation or None,
            )
        allowed_operations = {
            str(value).strip()
            for value in target.get("semantic_operations", [])
            if isinstance(value, str) and value.strip()
        }
        if operation not in _V5_SEMANTIC_DELTA_OPERATIONS or operation not in allowed_operations:
            raise _semantic_delta_failure(
                "Semantic repair used an operation outside its approved target authority.",
                path=path,
                patch_count=len(patches),
                internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
                guard_reason="DISALLOWED_OPERATION",
                semantic_operation=operation or None,
            )

        required_keys = semantic_delta_required_fields(operation)
        if set(patch) != required_keys:
            unexpected = sorted(set(patch) - required_keys)
            raise _semantic_delta_failure(
                "Semantic repair included fields outside its typed operation contract.",
                path=path,
                patch_count=len(patches),
                internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
                guard_reason="DISALLOWED_OUTER_FIELD",
                semantic_operation=operation,
                outer_field=unexpected[0] if unexpected else "missing_required_field",
            )

        allowed_block_ids = {
            str(value).strip()
            for value in target.get("allowed_block_ids", [])
            if isinstance(value, str) and value.strip()
        }

        if operation == "select_assessment_teaching_alignment":
            lesson = _semantic_delta_lesson_target(
                result,
                path,
                patch_count=len(patches),
                operation=operation,
            )
            raw_selections = patch.get("selections")
            if not isinstance(raw_selections, list) or not raw_selections:
                raise _semantic_delta_failure(
                    "Assessment semantic selection must provide one decision for every approved objective.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                    guard_reason="INVALID_OBJECTIVE_REF",
                    semantic_operation=operation,
                )
            compilation = compile_v5_assessment_plan(result, lesson_paths={path})
            if compilation.status != "NEEDS_SEMANTIC_RESOLUTION":
                raise _semantic_delta_failure(
                    "Assessment semantic selection no longer matches a pending compiler state.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_ASSESSMENT_PLAN_STALE",
                    guard_reason="TARGET_BOUNDARY_MUTATION",
                    semantic_operation=operation,
                )
            expected_fingerprint = str(target.get("assessment_plan_fingerprint") or "")
            actual_fingerprint = assessment_plan_fingerprint(
                result,
                lesson_path=path,
                candidates=compilation.candidates,
            )
            if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
                raise _semantic_delta_failure(
                    "Assessment semantic selection candidate authority is stale.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_ASSESSMENT_PLAN_STALE",
                    guard_reason="TARGET_BOUNDARY_MUTATION",
                    semantic_operation=operation,
                )
            expected_objectives = {
                str(value).strip()
                for value in target.get("allowed_objective_ids", [])
                if isinstance(value, str) and value.strip()
            }
            selections: dict[tuple[str, str], tuple[str, str]] = {}
            intent_changes: dict[tuple[str, str], str] = {}
            for selection in raw_selections:
                if not isinstance(selection, dict):
                    raise _semantic_delta_failure(
                        "Assessment semantic selection contains an invalid decision.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_PATCH_INVALID",
                        guard_reason="DISALLOWED_OUTER_FIELD",
                        semantic_operation=operation,
                    )
                objective_ref = str(selection.get("objective_ref") or "").strip()
                decision = str(selection.get("decision") or "").strip()
                if decision == "NO_MATCH":
                    if set(selection) != {"objective_ref", "decision"} or objective_ref not in expected_objectives:
                        raise _semantic_delta_failure(
                            "Assessment semantic selection NO_MATCH is outside the approved objective contract.",
                            path=path,
                            patch_count=len(patches),
                            internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                            guard_reason="INVALID_OBJECTIVE_REF",
                            semantic_operation=operation,
                        )
                    raise _semantic_delta_failure(
                        "No server-approved teaching anchor semantically teaches one required assessment objective.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_ASSESSMENT_NO_MATCH",
                        guard_reason="NO_VALID_TEACHING_ANCHOR",
                        semantic_operation=operation,
                    )
                if decision != "SELECT" or set(selection) - {"intent"} != {
                    "objective_ref", "decision", "unit_path", "teaching_block_id",
                }:
                    raise _semantic_delta_failure(
                        "Assessment semantic selection used an invalid typed decision shape.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_PATCH_INVALID",
                        guard_reason="DISALLOWED_OUTER_FIELD",
                        semantic_operation=operation,
                    )
                unit_path = str(selection.get("unit_path") or "").strip()
                block_id = str(selection.get("teaching_block_id") or "").strip()
                approved = compilation.candidates.get((path, objective_ref), ())
                matching = [candidate for candidate in approved if candidate.unit_path == unit_path and candidate.block_id == block_id]
                if (
                    objective_ref not in expected_objectives
                    or objective_ref in {key[1] for key in selections}
                    or len(matching) != 1
                ):
                    raise _semantic_delta_failure(
                        "Assessment semantic selection chose a teaching block outside its server-approved candidate set.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                        guard_reason="TARGET_BOUNDARY_MUTATION",
                        semantic_operation=operation,
                        block_id=block_id or None,
                    )
                candidate = matching[0]
                selected_intent = selection.get("intent")
                address = (unit_path, block_id)
                if candidate.allowed_intents:
                    if (not isinstance(selected_intent, str) or selected_intent not in candidate.allowed_intents
                            or (address in intent_changes and intent_changes[address] != selected_intent)):
                        raise _semantic_delta_failure(
                            "Assessment selection has a missing, forbidden or conflicting teaching intent.",
                            path=path, patch_count=len(patches),
                            internal_code="ARCH_REPAIR_INVALID_SEMANTIC_INTENT",
                            guard_reason="INVALID_SEMANTIC_INTENT", semantic_operation=operation,
                        )
                    intent_changes[address] = selected_intent
                elif "intent" in selection:
                    raise _semantic_delta_failure(
                        "Assessment selection cannot change an already teaching-capable anchor intent.",
                        path=path, patch_count=len(patches),
                        internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
                        guard_reason="DISALLOWED_OUTER_FIELD", semantic_operation=operation,
                    )
                selections[(path, objective_ref)] = (unit_path, block_id)
            if {key[1] for key in selections} != expected_objectives:
                raise _semantic_delta_failure(
                    "Assessment semantic selection did not resolve every approved objective exactly once.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                    guard_reason="INVALID_OBJECTIVE_REF",
                    semantic_operation=operation,
                )
            # All decisions are validated before touching this private copy.
            # Exact lesson-scoped unit/block addresses are server-owned;
            # provenance/content and every unselected block stay unchanged.
            for (unit_path, block_id), intent in intent_changes.items():
                unit = _blueprint_path_object(result, unit_path)
                block = next(item for item in unit["learning_blocks"] if str(item.get("id") or "").strip() == block_id)
                try:
                    preserve_assessment_visual_support(unit, block, unit_path=unit_path)
                except ValueError:
                    raise _semantic_delta_failure(
                        "Assessment visual support cannot use an occupied server block identity.",
                        path=path, patch_count=len(patches),
                        internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                        guard_reason="SERVER_VISUAL_BLOCK_ID_COLLISION", semantic_operation=operation,
                    ) from None
                block["intent"] = intent
            compiled = compile_v5_assessment_plan(
                result,
                selections=selections,
                lesson_paths={path},
            )
            if compiled.status != "READY":
                raise _semantic_delta_failure(
                    "Assessment semantic selection could not compile to a source-safe assessment plan.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_NO_PROGRESS",
                    guard_reason="OBJECTIVE_ALIGNMENT_NO_PROGRESS",
                    semantic_operation=operation,
                )
            compiled_lesson = _semantic_delta_lesson_target(
                compiled.blueprint,
                path,
                patch_count=len(patches),
                operation=operation,
            )
            lesson.clear()
            lesson.update(deepcopy(compiled_lesson))
            replacement_paths.add(path)
            seen.add(path)
            continue

        if operation == "add_instructional_support_block":
            lesson = _semantic_delta_lesson_target(
                result,
                path,
                patch_count=len(patches),
                operation=operation,
            )
            unit_path = str(patch.get("unit_path") or "").strip()
            allowed_unit_paths = {
                str(value).strip()
                for value in target.get("allowed_unit_paths", [])
                if isinstance(value, str) and value.strip()
            }
            if unit_path not in allowed_unit_paths or _repair_parent_lesson_path(unit_path) != path:
                raise _semantic_delta_failure(
                    "Instructional support selected a unit outside its approved lesson target.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                    guard_reason="TARGET_BOUNDARY_MUTATION",
                    semantic_operation=operation,
                )
            unit = _blueprint_path_object(result, unit_path)
            if not isinstance(unit, dict):
                raise _semantic_delta_failure(
                    "Instructional support selected an unknown approved unit.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                )
            blocks = _semantic_delta_blocks(unit)
            after_block_id = str(patch.get("after_block_id") or "").strip()
            after_index = next(
                (index for index, item in enumerate(blocks)
                 if str(item.get("id") or "").strip() == after_block_id),
                None,
            )
            teaching = blocks[after_index] if after_index is not None else None
            if (
                teaching is None
                or after_block_id not in allowed_block_ids
                or str(teaching.get("intent") or "").strip() not in _V5_TEACHING_INTENTS
            ):
                raise _semantic_delta_failure(
                    "Instructional support selected an invalid source-grounded teaching block.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                    block_id=after_block_id or None,
                )
            if not _v5_text_ids(teaching.get("primary_evidence_scope_ids")):
                raise _semantic_delta_failure(
                    "Instructional support cannot derive evidence from an ungrounded teaching block.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    guard_reason="NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    semantic_operation=operation,
                    block_id=after_block_id,
                )
            refs = _semantic_delta_objectives_are_local(
                lesson,
                patch.get("learning_objective_refs"),
                target=target,
                path=path,
                patch_count=len(patches),
                operation=operation,
                block_id=after_block_id,
            )
            if not set(refs).issubset(_v5_text_ids(teaching.get("learning_objective_refs"))):
                raise _semantic_delta_failure(
                    "Instructional support objectives are not taught by its selected anchor block.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    guard_reason="NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    semantic_operation=operation,
                    block_id=after_block_id,
                )
            compatible_teaching = _semantic_delta_compatible_teaching_blocks(
                lesson,
                target_unit=unit,
                target_block_id=after_block_id,
                objective_refs=refs,
                require_before_block=False,
            )
            if len(compatible_teaching) != 1 or compatible_teaching[0] is not teaching:
                raise _semantic_delta_failure(
                    "Instructional support has no unique compatible teaching evidence anchor.",
                    path=path,
                    patch_count=len(patches),
                    internal_code=(
                        "ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "ARCH_REPAIR_AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    guard_reason=(
                        "NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    semantic_operation=operation,
                    block_id=after_block_id,
                )
            intent = str(patch.get("intent") or "").strip()
            if intent not in INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS:
                raise _semantic_delta_failure(
                    "Instructional support selected an unsupported support intent.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_SEMANTIC_INTENT",
                    guard_reason="INVALID_SEMANTIC_INTENT",
                    semantic_operation=operation,
                    block_id=after_block_id,
                    selected_intent=intent,
                )
            content = _semantic_delta_instructional_content(
                patch.get("content"),
                path=path,
                patch_count=len(patches),
                operation=operation,
                block_id=after_block_id,
            )
            new_block = {
                "id": _semantic_delta_server_block_id(unit, prefix="instructional_support"),
                "intent": intent,
                "importance": "supporting",
                "concept_ids": list(teaching.get("concept_ids") or []),
                "primary_concept_ids": [],
                "primary_evidence_scope_ids": [],
                "supporting_evidence_scope_ids": sorted(
                    _v5_text_ids(teaching.get("primary_evidence_scope_ids"))
                ),
                "source_refs": list(teaching.get("source_refs") or []),
                "learning_objective_refs": refs,
                "content": content,
            }
            blocks.insert(after_index + 1, new_block)
            unit["learning_blocks"] = blocks
            replacement_paths.add(path)
            seen.add(path)
            continue

        unit, lesson = _semantic_delta_unit_and_lesson(
            result,
            path,
            patch_count=len(patches),
            operation=operation,
        )
        blocks = _semantic_delta_blocks(unit)

        if operation == "align_concepts_to_evidence":
            def normalized_concept_ids(value: Any, *, field: str) -> list[str]:
                if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                    raise _semantic_delta_failure(
                        "Evidence-concept repair must contain only non-empty canonical concept IDs.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_INVALID_CONCEPT_ID",
                        guard_reason="INVALID_CONCEPT_ID",
                        semantic_operation=operation,
                        outer_field=field,
                    )
                normalized = [str(item).strip() for item in value]
                if len(normalized) != len(set(normalized)):
                    raise _semantic_delta_failure(
                        "Evidence-concept repair cannot repeat a concept ID.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_INVALID_CONCEPT_ID",
                        guard_reason="INVALID_CONCEPT_ID",
                        semantic_operation=operation,
                        outer_field=field,
                    )
                return sorted(normalized)

            concept_ids = normalized_concept_ids(patch.get("concept_ids"), field="concept_ids")
            primary_concept_ids = normalized_concept_ids(
                patch.get("primary_concept_ids"),
                field="primary_concept_ids",
            )
            required_concept_ids = sorted({
                str(value).strip()
                for value in target.get("required_concept_ids", [])
                if isinstance(value, str) and value.strip()
            })
            allowed_concept_ids = {
                str(value).strip()
                for value in target.get("allowed_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }
            known_concept_ids = {
                str(value).strip()
                for value in target.get("known_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }
            primary_options = {
                tuple(sorted({str(value).strip() for value in option if isinstance(value, str) and value.strip()}))
                for option in target.get("primary_concept_options", [])
                if isinstance(option, list)
            }
            if not set(concept_ids).issubset(known_concept_ids):
                raise _semantic_delta_failure(
                    "Evidence-concept repair selected an unknown Source Map concept ID.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_CONCEPT_ID",
                    guard_reason="INVALID_CONCEPT_ID",
                    semantic_operation=operation,
                    outer_field="concept_ids",
                )
            if concept_ids != required_concept_ids or not set(concept_ids).issubset(allowed_concept_ids):
                raise _semantic_delta_failure(
                    "Evidence-concept repair selected an ID outside the server-approved alignment.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    semantic_operation=operation,
                    outer_field="concept_ids",
                )
            if (
                tuple(primary_concept_ids) not in primary_options
                or not set(primary_concept_ids).issubset(concept_ids)
            ):
                raise _semantic_delta_failure(
                    "Evidence-concept repair selected an invalid primary concept alignment.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    semantic_operation=operation,
                    outer_field="primary_concept_ids",
                )
            block_concepts = target.get("evidence_alignment_block_concepts")
            if not isinstance(block_concepts, dict):
                raise _semantic_delta_failure(
                    "Evidence-concept repair is missing its server-owned block alignment plan.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_NO_COMPATIBLE_CONCEPT",
                    guard_reason="NO_COMPATIBLE_CONCEPT",
                    semantic_operation=operation,
                )
            blocks_by_id = {
                str(block.get("id") or "").strip(): block
                for block in blocks
                if str(block.get("id") or "").strip()
            }
            for block_id, planned_concepts in block_concepts.items():
                block = blocks_by_id.get(str(block_id).strip())
                if block is None or not isinstance(planned_concepts, list):
                    raise _semantic_delta_failure(
                        "Evidence-concept repair cannot resolve one protected target learning block.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                        guard_reason="TARGET_BOUNDARY_MUTATION",
                        semantic_operation=operation,
                        block_id=str(block_id).strip() or None,
                    )
                normalized_block_concepts = sorted({
                    str(value).strip()
                    for value in planned_concepts
                    if isinstance(value, str) and value.strip()
                })
                if not normalized_block_concepts or not set(normalized_block_concepts).issubset(set(concept_ids)):
                    raise _semantic_delta_failure(
                        "Evidence-concept repair block alignment exceeds the approved unit concept scope.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                        guard_reason="CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                        semantic_operation=operation,
                        block_id=str(block_id).strip() or None,
                    )
                # Concept metadata is the only server-side propagation. The
                # provider cannot replace blocks or alter their evidence,
                # source refs, objectives, content, or ownership fields.
                block["concept_ids"] = normalized_block_concepts
            unit["concept_ids"] = concept_ids
            unit["primary_concept_ids"] = primary_concept_ids

        elif operation == "set_block_intent":
            block_id = str(patch.get("block_id") or "").strip()
            intent = str(patch.get("intent") or "").strip()
            block = next((item for item in blocks if str(item.get("id") or "").strip() == block_id), None)
            if block is None or block_id not in allowed_block_ids:
                raise _semantic_delta_failure(
                    "Semantic repair selected a block outside the approved target unit.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                    block_id=block_id or None,
                )
            if intent not in SEMANTIC_LEARNING_BLOCK_INTENTS:
                raise _semantic_delta_failure(
                    "Semantic repair selected an unsupported instructional intent.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_SEMANTIC_INTENT",
                    guard_reason="INVALID_SEMANTIC_INTENT",
                    semantic_operation=operation,
                    block_id=block_id,
                )
            target_codes = {
                str(code).strip()
                for code in target.get("codes", [])
                if isinstance(code, str) and code.strip()
            }
            allowed_intent_map = target.get("allowed_intents_by_block_id")
            declared_intents = {
                str(value).strip()
                for value in (allowed_intent_map.get(block_id, []) if isinstance(allowed_intent_map, dict) else [])
                if isinstance(value, str) and value.strip()
            }
            # Classifier-created targets used by focused callers predate the
            # preflight metadata. The canonical set remains the server
            # authority in that compatibility path; a supplied map can only
            # narrow it, never authorize scenario/reflection/etc.
            allowed_intents = set(ACTION_OBJECTIVE_REPAIR_INTENTS)
            if isinstance(allowed_intent_map, dict):
                allowed_intents &= declared_intents
            if (
                "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH" not in target_codes
                or intent not in allowed_intents
            ):
                raise _semantic_delta_failure(
                    "Action-intent repair selected an intent outside the server-approved teaching treatment.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_SEMANTIC_INTENT",
                    guard_reason="INTENT_NOT_ALLOWED_FOR_ACTION_REPAIR",
                    semantic_operation=operation,
                    block_id=block_id,
                    selected_intent=intent,
                )
            block["intent"] = intent

        elif operation == "repair_assessment_alignment":
            knowledge_check_block_id = str(patch.get("knowledge_check_block_id") or "").strip()
            knowledge_check = next(
                (item for item in blocks if str(item.get("id") or "").strip() == knowledge_check_block_id),
                None,
            )
            if (
                knowledge_check is None
                or knowledge_check_block_id not in allowed_block_ids
                or str(knowledge_check.get("intent") or "").strip() != "knowledge_check"
            ):
                raise _semantic_delta_failure(
                    "Assessment alignment cannot resolve one approved existing block.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                    block_id=knowledge_check_block_id or None,
                )
            lesson_path = _repair_parent_lesson_path(path)
            records = _v5_lesson_block_records(lesson, lesson_path)
            check_record = next(
                (
                    record for record in records
                    if record["unit_path"] == path and record["block"] is knowledge_check
                ),
                None,
            )
            if check_record is None:
                raise _semantic_delta_failure(
                    "Assessment alignment cannot resolve the approved knowledge-check order record.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                    guard_reason="TARGET_BOUNDARY_MUTATION",
                    semantic_operation=operation,
                    block_id=knowledge_check_block_id,
                )
            if _v5_text_ids(knowledge_check.get("primary_evidence_scope_ids")):
                raise _semantic_delta_failure(
                    "Assessment alignment cannot clear or replace existing primary evidence ownership.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
                    guard_reason="PRIMARY_OWNERSHIP_MUTATION",
                    semantic_operation=operation,
                    block_id=knowledge_check_block_id,
                )

            raw_selections = patch.get("teaching_selections")
            if not isinstance(raw_selections, list) or not raw_selections:
                raise _semantic_delta_failure(
                    "Assessment alignment must select one approved teaching anchor for every target objective.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                    guard_reason="INVALID_OBJECTIVE_REF",
                    semantic_operation=operation,
                    block_id=knowledge_check_block_id,
                )
            expected_refs = {
                str(value).strip()
                for value in target.get("allowed_objective_ids", [])
                if isinstance(value, str) and value.strip()
            }
            approved_candidates = [
                candidate for candidate in target.get("assessment_alignment_candidates", [])
                if isinstance(candidate, dict)
                and str(candidate.get("knowledge_check_path") or "") == path
                and str(candidate.get("knowledge_check_block_id") or "") == knowledge_check_block_id
            ]
            selections: list[tuple[dict[str, Any], set[str], str | None]] = []
            selected_intents: dict[str, str] = {}
            selected_refs: set[str] = set()
            for selection in raw_selections:
                if (not isinstance(selection, dict)
                        or not {"teaching_block_id", "learning_objective_refs"}.issubset(selection)
                        or set(selection) - {"teaching_block_id", "learning_objective_refs", "intent"}):
                    raise _semantic_delta_failure(
                        "Assessment alignment contains a typed selection outside its approved contract.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_PATCH_INVALID",
                        guard_reason="DISALLOWED_OUTER_FIELD",
                        semantic_operation=operation,
                        block_id=knowledge_check_block_id,
                    )
                teaching_block_id = str(selection.get("teaching_block_id") or "").strip()
                refs = set(_semantic_delta_objectives_are_local(
                    lesson,
                    selection.get("learning_objective_refs"),
                    target=target,
                    path=path,
                    patch_count=len(patches),
                    operation=operation,
                    block_id=knowledge_check_block_id,
                ))
                if selected_refs.intersection(refs) or not refs.issubset(expected_refs):
                    raise _semantic_delta_failure(
                        "Assessment alignment repeated or expanded an approved objective.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                        guard_reason="INVALID_OBJECTIVE_REF",
                        semantic_operation=operation,
                        block_id=knowledge_check_block_id,
                    )
                matching = [
                    candidate for candidate in approved_candidates
                    if str(candidate.get("teaching_block_id") or "") == teaching_block_id
                    and refs == _v5_text_ids(candidate.get("learning_objective_refs"))
                ]
                if len(matching) != 1:
                    raise _semantic_delta_failure(
                        "Assessment alignment selected a block or objective outside its server-approved candidate set.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                        guard_reason="TARGET_BOUNDARY_MUTATION",
                        semantic_operation=operation,
                        block_id=knowledge_check_block_id,
                    )
                authorized = matching[0]
                intent = selection.get("intent")
                allowed_intents = authorized.get("allowed_intents", [])
                if (allowed_intents and (not isinstance(intent, str) or intent not in allowed_intents)
                        or not allowed_intents and "intent" in selection
                        or teaching_block_id in selected_intents and selected_intents[teaching_block_id] != intent):
                    raise _semantic_delta_failure(
                        "Assessment alignment attempted an unauthorized or inconsistent teaching intent.",
                        path=path, patch_count=len(patches), internal_code="ARCH_REPAIR_INVALID_SEMANTIC_INTENT",
                        guard_reason="INVALID_SEMANTIC_INTENT", semantic_operation=operation, block_id=teaching_block_id,
                    )
                if intent is not None:
                    selected_intents[teaching_block_id] = intent
                selections.append((authorized, refs, intent))
                selected_refs.update(refs)
            if selected_refs != expected_refs:
                raise _semantic_delta_failure(
                    "Assessment alignment did not resolve every approved objective exactly once.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                    guard_reason="INVALID_OBJECTIVE_REF",
                    semantic_operation=operation,
                    block_id=knowledge_check_block_id,
                )

            selected_primary_scopes: set[str] = set()
            # Resolve all changes against the immutable pre-apply lesson;
            # one selection must not make a later unauthorized one eligible.
            baseline_records = _v5_lesson_block_records(deepcopy(lesson), lesson_path)
            baseline_check = next(record for record in baseline_records if record["unit_path"] == path and record["block_id"] == knowledge_check_block_id)
            for candidate, refs, intent in selections:
                teaching_path = str(candidate.get("teaching_block_path") or "")
                teaching_block_id = str(candidate.get("teaching_block_id") or "").strip()
                if _repair_parent_lesson_path(teaching_path) != _repair_parent_lesson_path(path):
                    raise _semantic_delta_failure(
                        "Assessment alignment cannot mutate a teaching block outside the target lesson.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                        guard_reason="CROSS_LESSON_BLOCK",
                        semantic_operation=operation,
                        block_id=teaching_block_id or None,
                    )
                teaching_unit = _blueprint_path_object(result, teaching_path)
                teaching = next(
                    (item for item in _semantic_delta_blocks(teaching_unit)
                     if str(item.get("id") or "").strip() == teaching_block_id),
                    None,
                ) if isinstance(teaching_unit, dict) else None
                if teaching is None:
                    raise _semantic_delta_failure(
                        "Assessment alignment teaching path no longer resolves to its approved unit.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                        guard_reason="TARGET_BOUNDARY_MUTATION",
                        semantic_operation=operation,
                        block_id=teaching_block_id or None,
                    )
                baseline_teaching = next(record for record in baseline_records if record["unit_path"] == teaching_path and record["block_id"] == teaching_block_id)
                if intent is not None:
                    options = assessment_intent_repair_options(
                        teaching_block=baseline_teaching["block"], knowledge_check_block=baseline_check["block"],
                        objective_refs=refs, unit_objective_refs=set(baseline_teaching["unit_objective_refs"]),
                        precedes_check=baseline_teaching["position"] < baseline_check["position"],
                        same_unit=teaching_path == path,
                    )
                    eligible = intent in options
                else:
                    eligible = any(
                        record["unit_path"] == teaching_path and record["block_id"] == teaching_block_id
                        for record in _v5_base_teaching_anchor_candidates(lesson, baseline_records, knowledge_check=baseline_check, objective_refs=refs)
                    )
                if not eligible:
                    raise _semantic_delta_failure(
                        "Assessment alignment teaching block is no longer a base-eligible source-grounded anchor.",
                        path=path,
                        patch_count=len(patches),
                        internal_code="ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                        guard_reason="NO_VALID_TEACHING_ANCHOR",
                        semantic_operation=operation,
                        block_id=teaching_block_id,
                    )
                if intent is not None:
                    teaching["intent"] = intent
                teaching["learning_objective_refs"] = sorted(
                    _v5_text_ids(teaching.get("learning_objective_refs")) | refs
                )
                selected_primary_scopes.update(_v5_text_ids(teaching.get("primary_evidence_scope_ids")))
                replacement_paths.add(teaching_path)
            knowledge_check["learning_objective_refs"] = sorted(
                _v5_text_ids(knowledge_check.get("learning_objective_refs")) | selected_refs
            )
            knowledge_check["supporting_evidence_scope_ids"] = sorted(
                _v5_text_ids(knowledge_check.get("supporting_evidence_scope_ids")) | selected_primary_scopes
            )

        elif operation == "repair_knowledge_check":
            block_id = str(patch.get("block_id") or "").strip()
            block = next((item for item in blocks if str(item.get("id") or "").strip() == block_id), None)
            if (
                block is None
                or block_id not in allowed_block_ids
                or str(block.get("intent") or "").strip() != "knowledge_check"
            ):
                raise _semantic_delta_failure(
                    "Semantic repair selected an invalid knowledge-check block.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                    block_id=block_id or None,
                )
            refs = _semantic_delta_objectives_are_local(
                lesson,
                patch.get("learning_objective_refs"),
                target=target,
                path=path,
                patch_count=len(patches),
                operation=operation,
                block_id=block_id,
            )
            compatible_teaching = _semantic_delta_compatible_teaching_blocks(
                lesson,
                target_unit=unit,
                target_block_id=block_id,
                objective_refs=refs,
                require_before_block=True,
            )
            if len(compatible_teaching) != 1:
                raise _semantic_delta_failure(
                    "No unique earlier teaching block can ground this knowledge check.",
                    path=path,
                    patch_count=len(patches),
                    internal_code=(
                        "ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "ARCH_REPAIR_AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    guard_reason=(
                        "NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    semantic_operation=operation,
                    block_id=block_id,
                )
            teaching = compatible_teaching[0]
            block["learning_objective_refs"] = refs
            if _v5_text_ids(block.get("primary_evidence_scope_ids")):
                raise _semantic_delta_failure(
                    "Knowledge-check repair cannot alter existing primary evidence ownership.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
                    guard_reason="PRIMARY_OWNERSHIP_MUTATION",
                    semantic_operation=operation,
                    block_id=block_id,
                )
            block["supporting_evidence_scope_ids"] = sorted(
                _v5_text_ids(teaching.get("primary_evidence_scope_ids"))
            )

        else:  # add_knowledge_check
            after_block_id = str(patch.get("after_block_id") or "").strip()
            after_index = next(
                (index for index, item in enumerate(blocks) if str(item.get("id") or "").strip() == after_block_id),
                None,
            )
            teaching = blocks[after_index] if after_index is not None else None
            if (
                teaching is None
                or after_block_id not in allowed_block_ids
                or str(teaching.get("intent") or "").strip() not in _V5_TEACHING_INTENTS
            ):
                raise _semantic_delta_failure(
                    "Semantic repair selected an invalid teaching block for knowledge-check insertion.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_INVALID_BLOCK_ID",
                    guard_reason="INVALID_BLOCK_ID",
                    semantic_operation=operation,
                    block_id=after_block_id or None,
                )
            refs = _semantic_delta_objectives_are_local(
                lesson,
                patch.get("learning_objective_refs"),
                target=target,
                path=path,
                patch_count=len(patches),
                operation=operation,
                block_id=after_block_id,
            )
            if (
                not set(refs).issubset(_v5_text_ids(teaching.get("learning_objective_refs")))
                or not _v5_text_ids(teaching.get("primary_evidence_scope_ids"))
            ):
                raise _semantic_delta_failure(
                    "The selected teaching block cannot safely ground the requested knowledge-check objectives.",
                    path=path,
                    patch_count=len(patches),
                    internal_code="ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    guard_reason="NO_COMPATIBLE_SUPPORTING_EVIDENCE",
                    semantic_operation=operation,
                    block_id=after_block_id,
                )
            compatible_teaching = _semantic_delta_compatible_teaching_blocks(
                lesson,
                target_unit=unit,
                target_block_id=after_block_id,
                objective_refs=refs,
                require_before_block=False,
            )
            if len(compatible_teaching) != 1:
                raise _semantic_delta_failure(
                    "No unique teaching block can ground a new knowledge check.",
                    path=path,
                    patch_count=len(patches),
                    internal_code=(
                        "ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "ARCH_REPAIR_AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    guard_reason=(
                        "NO_COMPATIBLE_SUPPORTING_EVIDENCE"
                        if not compatible_teaching else "AMBIGUOUS_SUPPORTING_EVIDENCE"
                    ),
                    semantic_operation=operation,
                    block_id=after_block_id,
                )
            teaching = compatible_teaching[0]
            new_block = {
                "id": _semantic_delta_server_block_id(unit),
                "intent": "knowledge_check",
                "importance": "assessment",
                "concept_ids": list(teaching.get("concept_ids") or []),
                "primary_concept_ids": [],
                "primary_evidence_scope_ids": [],
                "supporting_evidence_scope_ids": sorted(_v5_text_ids(teaching.get("primary_evidence_scope_ids"))),
                "source_refs": list(teaching.get("source_refs") or []),
                "learning_objective_refs": refs,
                "content": {},
            }
            blocks.insert(after_index + 1, new_block)
            unit["learning_blocks"] = blocks

        replacement_paths.add(path)
        seen.add(path)

    if seen != set(expected):
        raise _semantic_delta_failure(
            "Semantic repair did not apply every required target.",
            path="course",
            patch_count=len(patches),
            internal_code="ARCH_REPAIR_TARGET_MISSING",
            guard_reason="TARGET_BOUNDARY_MUTATION",
        )
    if not _architecture_repair_preserves_unaffected_snapshot(
        blueprint,
        result,
        replacement_paths=replacement_paths,
        removal_paths=set(),
    ):
        raise _repair_scope_violation(
            "Semantic repair modified architecture outside approved target boundaries.",
            patch_count=len(patches),
            internal_code="ARCH_REPAIR_SCOPE_VIOLATION",
            failure_stage="architecture_repair_semantic_delta_guard",
            guard_reason="TARGET_BOUNDARY_MUTATION",
        )
    _assert_v5_primary_provenance_preserved(
        _semantic_architecture_snapshot(blueprint), result, patch_count=len(patches),
    )
    _validate_v5_repair_patch_set_before_apply(
        result,
        target_paths=set(expected),
        replacement_fields={path: {"semantic_delta"} for path in expected},
        patch_count=len(patches),
    )
    return _semantic_architecture_snapshot(result)


def apply_course_architecture_repair_patches(
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if _is_v5_semantic_delta_repair(blueprint, targets):
        return _apply_v5_semantic_delta_repair_patches(blueprint, targets, payload)
    patches = payload.get("patches") if isinstance(payload.get("patches"), list) else []
    expected = {target["path"]: target for target in targets}
    seen: set[str] = set()
    replacement_paths: set[str] = set()
    removal_paths: set[str] = set()
    replacements: list[tuple[str, dict[str, Any]]] = []
    replacement_fields: dict[str, set[str]] = {}

    # The first pass accepts only a complete, target-authorized patch set. It
    # intentionally does not mutate the workflow Blueprint or its disposable
    # candidate while inspecting provider values.
    for patch in patches:
        if not isinstance(patch, dict):
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair response contains an invalid patch.",
                internal_code="ARCH_REPAIR_PATCH_INVALID",
                failure_stage="architecture_repair_contract_validation",
                diagnostics={"patch_count": len(patches)},
            )
        path = str(patch.get("path") or "")
        target = expected.get(path)
        operation = str(patch.get("operation") or "replace").strip()
        replacement = patch.get("replacement")
        if target is None or path in seen:
            raise _repair_scope_violation(
                "Repair response tried to change an unapproved scope.",
                path=path,
                patch_count=len(patches),
                internal_code="ARCH_REPAIR_TARGET_OUT_OF_SCOPE",
                failure_stage="architecture_repair_target_whitelist",
            )
        allowed_operations = target.get("allowed_operations", ["replace"])
        if operation not in allowed_operations:
            raise _repair_scope_violation("Repair response used an operation outside its approved target scope.", path=path, patch_count=len(patches))
        if operation == "remove_unit":
            if target["scope"] != "unit" or replacement not in (None, {}):
                raise _repair_scope_violation("Repair response tried to remove a non-unit or supplied a replacement with removal.", path=path, patch_count=len(patches))
            location = _repair_unit_parent_and_index(blueprint, path)
            if location is None:
                raise WorkflowFailure(
                    "ARCHITECTURE_REPAIR_INVALID",
                    "Repair target no longer exists.",
                    internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                    failure_stage="architecture_repair_patch_apply",
                    diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
                )
            units, index = location
            unit = units[index]
            if not isinstance(unit, dict) or not _is_removable_factless_reinforcement_unit(unit):
                raise _repair_scope_violation(
                    "Repair response tried to remove a unit that still has canonical or primary instructional ownership.",
                    path=path,
                    patch_count=len(patches),
                )
            removal_paths.add(path)
            seen.add(path)
            continue
        if not isinstance(replacement, dict):
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair response contains an invalid replacement patch.",
                internal_code="ARCH_REPAIR_PATCH_INVALID",
                failure_stage="architecture_repair_contract_validation",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if set(replacement) - set(target["allowed_fields"]):
            raise _repair_scope_violation("Repair response tried to change fields outside its allowed scope.", path=path, patch_count=len(patches))
        if _contains_provider_fact_ownership(replacement):
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair response tried to own canonical Source Fact allocation.",
                internal_code="ARCH_REPAIR_FACT_OWNERSHIP_VIOLATION",
                failure_stage="architecture_repair_canonical_fact_guard",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node = _blueprint_path_object(blueprint, path)
        if node is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if not _repair_keeps_existing_source_scope(
            node,
            replacement,
            allowed_evidence_scope_ids={
                str(scope_id).strip()
                for scope_id in target.get("allowed_evidence_scope_ids", [])
                if isinstance(scope_id, str) and scope_id.strip()
            },
        ):
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair response tried to expand the approved source scope.",
                internal_code="ARCH_REPAIR_SOURCE_SCOPE_EXPANSION",
                failure_stage="architecture_repair_source_scope_guard",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        replacements.append((path, deepcopy(replacement)))
        replacement_fields[path] = set(replacement)
        replacement_paths.add(path)
        seen.add(path)
    if seen != set(expected):
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_INVALID",
            "Repair response did not repair every required target.",
            internal_code="ARCH_REPAIR_TARGET_MISSING",
            failure_stage="architecture_repair_target_whitelist",
            diagnostics={"patch_count": len(patches), "expected_target_count": len(expected), "applied_target_count": len(seen)},
        )
    removal_lessons = {_repair_parent_lesson_path(path) for path in removal_paths}
    replacement_lessons = {_repair_parent_lesson_path(path) for path in replacement_paths}
    if removal_lessons & replacement_lessons:
        raise _repair_scope_violation("Repair response mixed unit removal and replacement in one lesson.", patch_count=len(patches))

    # Commit to a copy only after every patch passed the structural, authority,
    # canonical-fact and source-scope guards above. The mandatory V5 domain
    # validation below can still reject the whole set without altering the
    # pre-repair workflow candidate.
    result = deepcopy(_semantic_architecture_snapshot(blueprint))
    for path in sorted(removal_paths, reverse=True):
        location = _repair_unit_parent_and_index(result, path)
        if location is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        units, index = location
        units.pop(index)
    for path, replacement in replacements:
        node = _blueprint_path_object(result, path)
        if node is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node.update(replacement)
    if not _architecture_repair_preserves_unaffected_snapshot(
        blueprint,
        result,
        replacement_paths=replacement_paths,
        removal_paths=removal_paths,
    ):
        raise _repair_scope_violation("Repair response modified architecture outside approved target boundaries.", patch_count=len(patches))
    if blueprint.get("architecture_contract_version") == 5:
        _validate_v5_repair_patch_set_before_apply(
            result,
            target_paths=set(expected),
            replacement_fields=replacement_fields,
            patch_count=len(patches),
            baseline=blueprint,
        )
    # The replacement is semantic only.  Drop the previous server allocation
    # so the deterministic allocator must re-establish it from the repaired
    # Source Map scope; it is never preserved or edited by the provider.
    return _semantic_architecture_snapshot(result)
