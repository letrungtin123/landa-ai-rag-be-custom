"""Course-architecture repair prompts and the scoped repair source context."""

from __future__ import annotations

import json
from typing import Any, Literal

from app.lesson_author_blueprint import INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS, SEMANTIC_REPAIR_CONTRACT_VERSION
from app.services.lesson_author.architecture_shape import _blueprint_path_object, _semantic_delta_blocks
from app.services.lesson_author.source_context import MAX_WORKFLOW_REPAIR_TARGET_CHARS
from app.workflows.contracts import RepairTarget, WorkflowFailure, safe_workflow_path


def _v5_semantic_delta_operations_for_targets(targets: list[RepairTarget]) -> set[str]:
    """Return typed V5 operations only when every target can avoid replacement."""

    operations: set[str] = set()
    for target in targets:
        target_operations = target.get("semantic_operations")
        if not isinstance(target_operations, list) or not target_operations:
            return set()
        operations.update(
            str(operation).strip()
            for operation in target_operations
            if isinstance(operation, str) and operation.strip()
        )
    return operations


def _semantic_delta_target_snapshot(
    node: dict[str, Any],
    target: RepairTarget,
) -> dict[str, Any]:
    """Give a coherence repair only semantic IDs, never provenance to rewrite."""

    blocks = node.get("learning_blocks") if isinstance(node.get("learning_blocks"), list) else []
    allowed_block_ids = {
        str(value).strip()
        for value in target.get("allowed_block_ids", [])
        if isinstance(value, str) and value.strip()
    }
    snapshot: dict[str, Any] = {
        "path": target["path"],
        "scope": target["scope"],
        "codes": target["codes"],
        "semantic_operations": list(target.get("semantic_operations", [])),
        "allowed_block_ids": sorted(allowed_block_ids),
        "allowed_objective_ids": sorted({
            str(value).strip()
            for value in target.get("allowed_objective_ids", [])
            if isinstance(value, str) and value.strip()
        }),
        "current_blocks": [
            {
                "id": str(block.get("id") or "").strip(),
                "intent": str(block.get("intent") or "").strip(),
                "learning_objective_refs": [
                    str(value).strip()
                    for value in block.get("learning_objective_refs", [])
                    if isinstance(value, str) and value.strip()
                ],
            }
            for block in blocks
            if isinstance(block, dict)
            and str(block.get("id") or "").strip() in allowed_block_ids
        ],
    }
    if "align_concepts_to_evidence" in set(target.get("semantic_operations") or []):
        # These are server-derived, source-map-compatible identifier lists.
        # The provider may select only one exact candidate pair; it never
        # receives or changes a block's evidence/fact ownership fields.
        snapshot.update({
            "required_concept_ids": sorted({
                str(value).strip()
                for value in target.get("required_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }),
            "allowed_concept_ids": sorted({
                str(value).strip()
                for value in target.get("allowed_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }),
            "primary_concept_options": [
                sorted({str(value).strip() for value in option if isinstance(value, str) and value.strip()})
                for option in target.get("primary_concept_options", [])
                if isinstance(option, list)
            ],
        })
    if "set_block_intent" in set(target.get("semantic_operations") or []):
        # The provider receives only the server-derived choices for the one
        # existing block it may change. These compact labels cannot expand
        # into the general Blueprint intent enum.
        allowed_intents = target.get("allowed_intents_by_block_id")
        snapshot["allowed_intents_by_block_id"] = {
            block_id: sorted({
                str(intent).strip()
                for intent in intents
                if isinstance(intent, str) and intent.strip()
            })
            for block_id, intents in allowed_intents.items()
            if isinstance(block_id, str) and block_id in allowed_block_ids and isinstance(intents, list)
        } if isinstance(allowed_intents, dict) else {}
        snapshot["assessment_dependency_objective_count"] = max(
            0,
            int(target.get("assessment_dependency_objective_count") or 0),
        )
    if "repair_assessment_alignment" in set(target.get("semantic_operations") or []):
        # Python retains exact mutation paths. Intent repair additionally
        # receives bounded instructional descriptors, never evidence bodies.
        snapshot["assessment_alignment_candidates"] = [
            {
                "knowledge_check_block_id": str(candidate.get("knowledge_check_block_id") or ""),
                "teaching_block_id": str(candidate.get("teaching_block_id") or ""),
                "learning_objective_refs": [
                    str(value).strip()
                    for value in candidate.get("learning_objective_refs", [])
                    if isinstance(value, str) and value.strip()
                ],
                **({
                    "allowed_intents": list(candidate["allowed_intents"]),
                    "semantic_descriptor": candidate["semantic_descriptor"],
                    "objective_text": candidate["objective_text"],
                } if candidate.get("allowed_intents") else {}),
            }
            for candidate in target.get("assessment_alignment_candidates", [])
            if isinstance(candidate, dict)
        ]
    if "select_assessment_teaching_alignment" in set(target.get("semantic_operations") or []):
        # The compiler created this allowlist from the immutable Blueprint.
        # It contains compact provider-owned instructional semantics but no
        # evidence scopes, source refs, concepts or canonical fact IDs.
        snapshot["assessment_plan_fingerprint"] = str(target.get("assessment_plan_fingerprint") or "")
        snapshot["assessment_plan_candidates"] = [
            {
                "objective_ref": str(candidate.get("objective_ref") or ""),
                "unit_path": str(candidate.get("unit_path") or ""),
                "teaching_block_id": str(candidate.get("teaching_block_id") or ""),
                "semantic_descriptor": candidate.get("semantic_descriptor")
                if isinstance(candidate.get("semantic_descriptor"), dict) else {},
                **({"allowed_intents": list(candidate["allowed_intents"])} if candidate.get("allowed_intents") else {}),
            }
            for candidate in target.get("assessment_plan_candidates", [])
            if isinstance(candidate, dict)
        ]
        snapshot["assessment_plan_objectives"] = {
            str(objective_ref): str(objective).strip()[:480]
            for objective_ref, objective in (target.get("assessment_plan_objectives") or {}).items()
            if isinstance(objective_ref, str) and isinstance(objective, str)
            and objective_ref.strip() and objective.strip()
        }
    # Post-allocation depth repair is lesson-scoped. The provider sees only
    # server-approved unit/block IDs plus local objective IDs; source scope
    # and canonical fact ownership are intentionally absent.
    if target.get("scope") == "lesson":
        allowed_unit_paths = {
            str(value).strip()
            for value in target.get("allowed_unit_paths", [])
            if isinstance(value, str) and value.strip()
        }
        current_units: list[dict[str, Any]] = []
        for unit_index, unit in enumerate(node.get("units", []), start=1):
            if not isinstance(unit, dict):
                continue
            unit_path = f"{target['path']}.unit_{unit_index}"
            if unit_path not in allowed_unit_paths:
                continue
            current_units.append({
                "unit_path": unit_path,
                "current_blocks": [
                    {
                        "id": str(block.get("id") or "").strip(),
                        "intent": str(block.get("intent") or "").strip(),
                        "learning_objective_refs": [
                            str(value).strip()
                            for value in block.get("learning_objective_refs", [])
                            if isinstance(value, str) and value.strip()
                        ],
                    }
                    for block in _semantic_delta_blocks(unit)
                    if str(block.get("id") or "").strip() in allowed_block_ids
                ],
            })
        snapshot["allowed_unit_paths"] = sorted(allowed_unit_paths)
        snapshot["allowed_support_intents"] = sorted(INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS)
        snapshot["semantic_repair_contract_version"] = SEMANTIC_REPAIR_CONTRACT_VERSION
        snapshot["current_units"] = current_units
        snapshot.pop("current_blocks", None)
    return snapshot


def build_course_architecture_repair_prompt(
    *,
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    source_map_context: str,
    locale: Literal["vi", "en"],
) -> str:
    semantic_delta_operations = _v5_semantic_delta_operations_for_targets(targets)
    target_snapshots = []
    for target in targets:
        node = _blueprint_path_object(blueprint, target["path"])
        if node is None:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "A requested repair path does not exist in the blueprint.",
                internal_code="ARCH_REPAIR_TARGET_MISSING",
                failure_stage="architecture_repair_target_snapshot",
                diagnostics={"repair_target_path": safe_workflow_path(target["path"])},
            )
        if semantic_delta_operations:
            target_snapshots.append(_semantic_delta_target_snapshot(node, target))
        else:
            target_snapshots.append({
                "path": target["path"], "scope": target["scope"], "codes": target["codes"],
                "allowed_fields": target["allowed_fields"],
                "allowed_operations": target.get("allowed_operations", ["replace"]),
                **({"allowed_evidence_scope_ids": target.get("allowed_evidence_scope_ids", [])}
                   if target.get("allowed_evidence_scope_ids") else {}),
                "diagnostics": target.get("diagnostics", []),
                "current": _semantic_architecture_snapshot(node),
            })
    serialized_targets = json.dumps(target_snapshots, ensure_ascii=False, separators=(",", ":"))
    if len(serialized_targets) > MAX_WORKFLOW_REPAIR_TARGET_CHARS:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE",
            "The affected architecture scope is too large for a bounded repair prompt.",
            internal_code="ARCH_REPAIR_SCOPE_TOO_LARGE",
            failure_stage="architecture_repair_target_snapshot",
            diagnostics={"repair_target_count": len(targets), "target_snapshot_chars": len(serialized_targets)},
        )
    language = "Vietnamese" if locale == "vi" else "English"
    instructions = [
        "You are repairing a bounded course-architecture review artifact.",
        f"Write in {language}. Do not reason aloud. Return JSON only.",
        "Preserve canonical IDs, authoritative source/course titles and source quotations; output language is not permission to change protected fields.",
    ]
    if semantic_delta_operations:
        instructions.append(
            "You may submit only one typed semantic delta for every listed target. Never return replacement, learning_blocks, source refs, evidence scope IDs, source fact IDs, allocation metadata, or any other provenance field."
        )
        if "align_concepts_to_evidence" in semantic_delta_operations:
            instructions.append(
                "align_concepts_to_evidence may return only the exact listed required_concept_ids and one listed primary_concept_options value. Do not return or alter blocks, source refs, evidence scopes, facts, allocation data, components, titles, objectives, or hierarchy."
            )
        if "set_block_intent" in semantic_delta_operations:
            instructions.append(
                "set_block_intent may select only an allowed existing block_id and its exact listed intent in allowed_intents_by_block_id. Those choices preserve any dependent assessment teaching anchor."
            )
        if "repair_knowledge_check" in semantic_delta_operations:
            instructions.append(
                "repair_knowledge_check may select only an allowed existing knowledge_check block_id and only listed local objective IDs. The server derives supporting evidence."
            )
        if "repair_assessment_alignment" in semantic_delta_operations:
            instructions.append(
                "repair_assessment_alignment may select one listed existing knowledge_check_block_id and one approved teaching_selections entry for every listed local objective. Each selection may use only its listed teaching_block_id and exact local objective IDs. Only a candidate listing allowed_intents requires an intent choice: choose the teaching treatment justified by its current instructional purpose and the selected objectives; use the same intent for all selections of that block. Omit intent for all other candidates. Do not relabel a block just to pass if its semantics cannot teach those objectives from existing evidence. The server owns the exact approved block paths and derives supporting evidence. Never return source refs, evidence scopes, concepts, facts, allocation data, content, hierarchy, or any other block field."
            )
        if "add_knowledge_check" in semantic_delta_operations:
            instructions.append(
                "add_knowledge_check may select only the listed eligible teaching after_block_id and only listed local objective IDs. The server creates the block and derives supporting evidence."
            )
        if "select_assessment_teaching_alignment" in semantic_delta_operations:
            instructions.append(
                "select_assessment_teaching_alignment may select only one listed teaching block for each listed objective. Compare the listed objective descriptor with the candidate's compact semantic descriptor. For a candidate listing allowed_intents, return intent from that list only if its existing evidence-backed instructional purpose can teach the objective using that treatment; use the same intent for every selection of that block. Do not relabel mere practice/reflection as teaching just to pass. Omit intent for all other candidates. Use SELECT only when this semantic alignment is justified; otherwise use NO_MATCH. The server validates the exact candidate fingerprint, inserts any knowledge_check, and derives supporting evidence. Never return source refs, evidence scopes, concepts, facts, allocation data, content, components, titles, objectives, or hierarchy."
            )
        if "add_instructional_support_block" in semantic_delta_operations:
            instructions.append(
                "add_instructional_support_block may select only a listed unit_path, its listed eligible teaching after_block_id, an exact intent from allowed_support_intents, and listed local objective IDs. Return compact semantic content only. The server creates the block and derives supporting evidence; never return evidence scopes, source references, concepts, facts, allocation data, HTML, or component payloads."
            )
        instructions.append(
            "Return exactly one JSON patch per target using only the operation-specific schema supplied by the server."
        )
    else:
        instructions.extend([
            "You may repair only the listed paths, operations, and allowed fields. Do not add chapters, move unrelated lessons, invent source facts, or change fields outside replacement.",
            "When units is an allowed replacement field, return one to three complete units only. Preserve every approved objective, concept, source reference and evidence-scope owner in a coherent unit; never truncate, positionally drop, or fabricate provenance merely to meet cardinality.",
            "For a V5 missing-evidence-owner target, add only its allowed_evidence_scope_ids to primary_evidence_scope_ids of a compatible existing learning block. Never return source_fact_ids, allocation metadata, unknown scope IDs, or changed source-scope membership.",
            "For unit-source-fact ownership, never add source_fact_ids yourself. A replace operation may change only its listed unit fields. Use remove_unit only for an evidence-free reinforcement unit that cannot receive a unique primary semantic scope; it removes only that exact target unit.",
            "Return exactly: {\"patches\":[{\"path\":\"...\",\"operation\":\"replace|remove_unit\",\"replacement\":{...}}]}. replacement is required only for replace. Include one patch for every listed target.",
        ])
    instructions.extend([
        "GLOBAL SOURCE MAP (evidence; not a heading-to-course template):",
        source_map_context,
        "REPAIR TARGETS:",
        serialized_targets,
    ])
    return "\n".join(instructions)


def build_v5_scoped_repair_source_context(
    *,
    blueprint: dict[str, Any],
    targets: list[RepairTarget],
    source_map: dict[str, Any],
) -> str:
    """Build V5 repair evidence from only the affected scope inventory.

    The provider sees the exact missing scope IDs plus compatible provenance
    metadata, never canonical fact IDs or the full global Source Map. Existing
    scope IDs inside a schema-repair target are included solely so a local
    lesson reorganization can preserve ownership boundaries.
    """

    semantic_operations = _v5_semantic_delta_operations_for_targets(targets)
    requested_scope_ids: set[str] = set()
    for target in targets:
        requested_scope_ids.update(
            str(scope_id).strip()
            for scope_id in target.get("allowed_evidence_scope_ids", [])
            if isinstance(scope_id, str) and scope_id.strip()
        )
        node = _blueprint_path_object(blueprint, target["path"])
        if isinstance(node, dict):
            requested_scope_ids.update(_collect_nested_source_values(node, "primary_evidence_scope_ids"))
            requested_scope_ids.update(_collect_nested_source_values(node, "supporting_evidence_scope_ids"))

    scopes_by_id = {
        str(scope.get("id") or "").strip(): scope
        for scope in source_map.get("source_evidence_scopes", [])
        if isinstance(scope, dict) and str(scope.get("id") or "").strip()
    }
    if requested_scope_ids - set(scopes_by_id):
        raise WorkflowFailure(
            "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
            "A V5 repair target references an evidence scope outside the immutable Source Map.",
            internal_code="V5_REPAIR_SCOPE_NOT_IN_CONTEXT",
            failure_stage="architecture_repair_target_snapshot",
            diagnostics={"repair_target_count": len(targets)},
        )
    # A post-allocation depth delta never chooses provenance. Its selected
    # teaching-block anchor is already server-validated during apply, so keep
    # immutable evidence-scope IDs out of the provider context altogether.
    # This differs from schema/evidence repairs, which need an explicit safe
    # scope inventory to repair a missing owner.
    if (
        semantic_operations == {"add_instructional_support_block"}
        or semantic_operations == {"select_assessment_teaching_alignment"}
    ):
        return json.dumps({
            "source_map_version": source_map.get("version"),
            "v5_server_owned_semantic_selection": True,
            "v5_post_allocation_depth_repair": semantic_operations == {"add_instructional_support_block"},
            "provider_evidence_scope_selection": False,
            "server_validates_grounding_from_selected_anchor": True,
        }, ensure_ascii=False, separators=(",", ":"))
    descriptors = [{
        "id": scope_id,
        "document_id": str(scope.get("document_id") or "").strip(),
        "section_id": str(scope.get("section_id") or "").strip(),
        "source_ref": str(scope.get("source_ref") or "").strip(),
        "concept_ids": [
            str(value).strip()
            for value in scope.get("concept_ids", [])
            if isinstance(value, str) and value.strip()
        ],
        "heading_path": [
            str(value)[:160]
            for value in scope.get("heading_path", [])
            if isinstance(value, str) and value.strip()
        ][:8],
        "evidence_char_count": int(scope.get("evidence_char_count") or 0),
        "evidence_token_estimate": int(scope.get("evidence_token_estimate") or 0),
    } for scope_id, scope in sorted(scopes_by_id.items()) if scope_id in requested_scope_ids]
    context = json.dumps({
        "source_map_version": source_map.get("version"),
        "v5_scoped_repair": True,
        "source_evidence_scopes": descriptors,
    }, ensure_ascii=False, separators=(",", ":"))
    if len(context) > MAX_WORKFLOW_REPAIR_TARGET_CHARS:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE",
            "The V5 local evidence-scope repair context exceeds the bounded prompt limit.",
            internal_code="V5_REPAIR_SCOPE_CONTEXT_TOO_LARGE",
            failure_stage="architecture_repair_target_snapshot",
            diagnostics={
                "repair_target_count": len(targets),
                "scoped_evidence_scope_count": len(descriptors),
                "serialized_target_chars": len(context),
            },
        )
    return context


def _collect_nested_source_values(value: Any, key: str) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        raw = value.get(key)
        if isinstance(raw, list):
            values.update(str(item).strip() for item in raw if str(item).strip())
        for child in value.values():
            values.update(_collect_nested_source_values(child, key))
    elif isinstance(value, list):
        for child in value:
            values.update(_collect_nested_source_values(child, key))
    return values


def _semantic_architecture_snapshot(value: Any) -> Any:
    """Remove server-owned allocation artifacts before a provider repair call."""
    if isinstance(value, dict):
        return {
            key: _semantic_architecture_snapshot(child)
            for key, child in value.items()
            if key not in {
                "source_fact_ids",
                "covered_source_fact_ids",
                "source_fact_allocation",
                "source_evidence_scope_allocation",
                "component_plan",
            }
        }
    if isinstance(value, list):
        return [_semantic_architecture_snapshot(child) for child in value]
    return value
