from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from time import monotonic, perf_counter
from typing import Any, Callable, Literal

import asyncpg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.api.deps import get_db, require_internal_token
from app.api.routes import chat as chat_routes
from app.api.routes import health
from app.api.routes import kb as kb_routes
from app.assessment_planner import (
    assessment_intent_repair_options,
    assessment_plan_fingerprint,
    assessment_teaching_semantic_descriptor,
    compile_v5_assessment_plan,
    materialize_assessment_source_refs,
    preserve_assessment_visual_support,
)
from app.assessment_selection_contract import VERSION as ASSESSMENT_SELECTION_CONTRACT_VERSION
from app.assessment_selection_contract import build_assessment_selection_contract
from app.core.config import settings
from app.core.errors import AppError, app_error_handler, request_validation_error_handler, unhandled_error_handler
from app.core.lifespan import build_lifespan
from app.core.logging import configure_application_logging
from app.core.middleware import RequestBodyLimitMiddleware
from app.core.request_context import DisconnectCancellationMiddleware, RequestContextMiddleware
from app.idm.course_design import run_idm_course_design
from app.idm.diagram import idm_source_step_diagram
from app.idm.module_design import run_idm_module_design
from app.idm.runtime import (
    RUN_STOPPING_PROVIDER_CODES,
    TRANSIENT_PROVIDER_CODES,
    IdmError,
    IdmProviderError,
    IdmRuntime,
    IdmStageError,
    IdmTokenAllowance,
)
from app.idm.source_locked import idm_source_faq, idm_source_grounded_single_choice, render_idm_source_locked_html
from app.idm.storyboard import IdmUnitDeps, run_idm_unit
from app.instructional_density import (
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    SOURCE_SCOPE_CHUNKS_PER_GROUP,
    partition_chunk_facts_for_density,
)
from app.instructional_opportunities import VERSION as EVIDENCE_TREATMENT_VERSION
from app.instructional_opportunities import compile_evidence_treatments
from app.instructional_quality import source_relationship_pairs
from app.learner_content_purity import build_learner_content_purity_context
from app.lesson_author_blueprint import (
    ACTION_OBJECTIVE_REPAIR_INTENTS,
    INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS,
    INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS,
    SEMANTIC_LEARNING_BLOCK_INTENTS,
    SEMANTIC_REPAIR_CONTRACT_VERSION,
    V5_SEMANTIC_DELTA_REPAIR_OPERATIONS,
    LessonAuthorBlueprintValidationError,
    build_course_architecture_repair_response_schema,
    build_v5_semantic_delta_repair_response_schema,
    semantic_delta_required_fields,
    validate_lesson_author_blueprint,
)
from app.lesson_author_checkpoint import assemble_checkpoint_chapter, checkpoint_expected_units
from app.lesson_author_orchestration_v2 import OrchestrationContractError
from app.lesson_author_orchestration_v2 import canonical_hash as orchestration_v2_canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    ChapterShardProviderWireV2,
    CourseSkeletonDraftV2,
    CourseSkeletonProviderWireV2,
    SourceOutlineAuthorityV2,
    SourceOutlineChapterV2,
    SourceSnapshotFactV2,
    UnitGenerationContractV2,
    bind_chapter_shard_v2,
    bind_course_skeleton_v2,
    chapter_shard_prompt_v2,
    fallback_chapter_shard_draft_v2,
    fallback_course_skeleton_draft_v2,
    parse_chapter_shard_draft_v2,
    salvage_chapter_shard_draft_v2,
    skeleton_prompt_v2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from app.lesson_prompt_policy import lesson_output_language_policy
from app.lesson_quality import (
    duplicate_validation_result,
    instructional_plan_validation_result,
    pedagogical_validation_result,
)
from app.schemas.chat import RagChatRequest
from app.schemas.common import AiUsage
from app.schemas.lesson_author import (
    RagLessonAuthorBlueprintRequest,
    RagLessonAuthorCheckpointRequest,
    RagLessonAuthorRequest,
)
from app.schemas.orchestration_v2 import (
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
)
from app.semantic_review import (
    SemanticReviewResponse,
    SemanticReviewRunOutcome,
    build_semantic_repair_prompt,
    build_semantic_review_prompt,
    run_bounded_semantic_review,
    safe_semantic_review_summary,
    semantic_review_config_hash,
    validate_semantic_review_response,
)
from app.services import provider
from app.services import runtime as service_runtime
from app.services.deadlines import record_fallback, run_with_deadline
from app.services.ingestion.chunking import SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED, SOURCE_EVIDENCE_READY
from app.services.lesson_author import blueprint as blueprint_service
from app.services.lesson_author.architecture_shape import (
    _V5_GENERIC_EXPLANATION_INTENTS,
    _V5_TEACHING_INTENTS,
    _blueprint_path_object,
    _repair_parent_lesson_path,
    _semantic_delta_blocks,
    _v5_text_ids,
    _workflow_issue,
    _workflow_issue_from_blueprint_validation_error,
)
from app.services.lesson_author.architecture_validation import (
    _v5_base_teaching_anchor_candidates,
    _v5_fully_aligned_teaching_anchor_candidates,
    _v5_lesson_block_records,
    _v5_objective_ids_are_local,
    validate_course_architecture_workflow,
    validate_v5_instructional_coherence,
)
from app.services.lesson_author.blueprint import (
    CourseArchitectSemanticScopeError,
    LessonAuthorBlueprintGenerationError,
    lesson_author_blueprint_failure_message,
    validate_lesson_author_source_refs,
)
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    source_fact_allocation_diagnostics,
    validate_course_architecture_evidence_scope,
)
from app.services.lesson_author.granularity import (
    allocate_blueprint_source_fact_ids,
    ensure_blueprint_source_granularity,
)
from app.services.lesson_author.media_review import MEDIA_REVIEW_VERSION, enrich_lesson_author_blueprint_media_review
from app.services.lesson_author.prompts import build_lesson_author_blueprint_prompt, build_lesson_author_prompt
from app.services.lesson_author.proposal_validation import (
    normalize_lesson_author_proposal_tree,
    parse_course_architecture_repair_payload,
    parse_lesson_author_json,
    validate_lesson_author_proposal_shape,
)
from app.services.lesson_author.proposal_validation import (
    semantic_learning_visible_text as semantic_learning_visible_text,
)
from app.services.lesson_author.source_context import (
    MAX_WORKFLOW_REPAIR_TARGET_CHARS,
    assert_v5_immutable_source_context,
    assert_v5_scoped_repair_target_bound,
    create_v5_immutable_source_context,
)
from app.services.lesson_author.source_refs import (
    drop_invalid_lesson_author_proposal_source_refs,
    update_blueprint_source_coverage,
    validate_lesson_author_proposal_source_refs,
)
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.provider_schemas import (
    bind_staged_instance_payload,
    build_lesson_author_proposal_response_schema,
    build_staged_instance_response_model,
    build_staged_multi_repair_model,
    decode_staged_multi_repair,
)
from app.services.lesson_author.staged.provider_schemas import (
    staged_component_payload_code as staged_component_payload_code,
)
from app.services.lesson_author.staged.skeleton import (
    extract_lesson_author_unit_batches,
    should_stage_lesson_author_proposal,
    validate_staged_skeleton_source_facts,
)
from app.services.lesson_author.staged.source_locked import build_source_locked_staged_skeleton
from app.services.lesson_author.staged.validation import (
    StagedUnitFinding,
    merge_staged_component_payload_delta,
    validate_staged_unit_content,
)
from app.services.meta import API_VERSION
from app.services.orchestration_v2.source_locked import (
    build_orchestration_v2_source_locked_components,
    build_orchestration_v2_source_locked_unit,
)
from app.services.provider import (
    PROVIDER_TRANSIENT_MAX_ATTEMPTS,
    combine_usage,
    is_non_retryable_provider_error,
    normalize_embedding_model,
    provider_http_error_status,
    require_provider_api_key,
    safe_provider_error_diagnostics,
)
from app.services.provider import generate_content as generate_content
from app.services.retrieval import search as retrieval_search
from app.services.retrieval.query import decode_json_object, retrieval_limits
from app.services.retrieval.search import build_retrieval_diagnostics, format_sources
from app.services.retrieval.source_coverage import (
    extract_source_coverage_facts,
    format_course_architecture_coverage_contract,
    format_source_coverage_manifest,
    source_coverage_metrics,
    validate_lesson_author_source_coverage,
)
from app.services.text import clean_text
from app.source_chapter_policy import bind_source_chapters, resolve_source_chapter_policy
from app.source_map import build_course_architect_context, build_source_map
from app.workflows.contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowIssue,
    WorkflowValidationResult,
    safe_workflow_issue_summary,
    safe_workflow_path,
)
from app.workflows.course_architecture import (
    V5_MAX_PROVIDER_REPAIR_CALLS,
    V5_MAX_REPAIR_ATTEMPTS_PER_LAYER,
    CourseArchitectureWorkflowCallbacks,
    run_course_architecture_workflow,
)
from app.workflows.lesson_generation import LessonGenerationWorkflowCallbacks, run_lesson_generation_workflow

app = FastAPI(
    title="Internal AI RAG Service",
    version=API_VERSION,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    # Resolved at call time: startup/shutdown are defined further down.
    lifespan=build_lifespan(
        state=service_runtime.runtime_state,
        startup=lambda: service_runtime.startup(),
        shutdown=lambda: service_runtime.shutdown(),
        grace_seconds=settings.shutdown_grace_seconds,
    ),
)


# add_middleware wraps outward: request context (outermost) -> body limit ->
# disconnect cancellation (buffers the already size-checked body) -> routes.
app.add_middleware(DisconnectCancellationMiddleware)


app.add_middleware(
    RequestBodyLimitMiddleware,
    max_request_bytes=settings.max_request_bytes,
    idm_max_request_bytes=settings.idm_max_request_bytes,
)


app.add_middleware(RequestContextMiddleware, state=service_runtime.runtime_state)


app.add_exception_handler(AppError, app_error_handler)


app.add_exception_handler(Exception, unhandled_error_handler)


app.add_exception_handler(RequestValidationError, request_validation_error_handler)


app.include_router(health.router)


app.include_router(kb_routes.router)


app.include_router(chat_routes.router)


# Configure the package logger so every app.* module (app.idm, app.core.request_context, ...)
# emits through the JSON handler, not only this module.
configure_application_logging("app")


logger = logging.getLogger(__name__)


# Keep the outer Node -> Python request alive long enough to serialize and
# return a deterministic source-locked fallback after an inner provider
# deadline. Without this gap, a provider timeout and the HTTP client timeout
# race at the same millisecond and turn a valid fallback into outcome_unknown.
ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS = 5_000


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


def _proposal_path_object(proposal: dict[str, Any], path: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"lesson|chapter_(\d+)\.lesson_(\d+)(?:\.unit_(\d+)(?:\.component_(\d+))?)?", path)
    if match is None:
        return None
    if path == "lesson":
        return proposal
    chapters = proposal.get("chapters") if isinstance(proposal.get("chapters"), list) else []
    chapter_index, lesson_index = int(match.group(1)) - 1, int(match.group(2)) - 1
    if not 0 <= chapter_index < len(chapters) or not isinstance(chapters[chapter_index], dict):
        return None
    lessons = chapters[chapter_index].get("lessons") if isinstance(chapters[chapter_index].get("lessons"), list) else []
    if not 0 <= lesson_index < len(lessons) or not isinstance(lessons[lesson_index], dict):
        return None
    node: dict[str, Any] = lessons[lesson_index]
    if match.group(3) is None:
        return node
    units = node.get("units") if isinstance(node.get("units"), list) else []
    unit_index = int(match.group(3)) - 1
    if not 0 <= unit_index < len(units) or not isinstance(units[unit_index], dict):
        return None
    node = units[unit_index]
    if match.group(4) is None:
        return node
    components = node.get("components") if isinstance(node.get("components"), list) else node.get("blocks")
    component_index = int(match.group(4)) - 1
    return components[component_index] if isinstance(components, list) and 0 <= component_index < len(components) and isinstance(components[component_index], dict) else None


def validate_lesson_generation_workflow(
    proposal: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
) -> WorkflowValidationResult:
    issues: list[WorkflowIssue] = []
    try:
        validate_lesson_author_proposal_shape(proposal)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("LESSON_VALIDATION_FAILED", str(error), path="lesson"))
    try:
        validate_lesson_author_proposal_source_refs(proposal, known_source_refs)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("INVALID_SOURCE_REF", str(error), path="lesson"))
    try:
        coverage = validate_lesson_author_source_coverage(proposal, source_coverage_manifest)
    except LessonAuthorProposalValidationError as error:
        coverage = source_coverage_metrics(proposal, source_coverage_manifest)
        issues.append(_workflow_issue("SOURCE_EVIDENCE_INSUFFICIENT", str(error), path="lesson"))
    return WorkflowValidationResult(issues, {"source_coverage": coverage.get("coverage_ratio")})


def build_lesson_generation_repair_prompt(
    *,
    proposal: dict[str, Any],
    targets: list[RepairTarget],
    evidence_context: str,
    locale: Literal["vi", "en"],
) -> str:
    snapshots = []
    for target in targets:
        node = _proposal_path_object(proposal, target["path"])
        if node is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "A requested lesson repair path does not exist.",
                internal_code="LESSON_REPAIR_TARGET_MISSING",
                failure_stage="lesson_repair_target_snapshot",
                diagnostics={"repair_target_path": safe_workflow_path(target["path"])},
            )
        snapshots.append({
            "path": target["path"], "scope": target["scope"], "codes": target["codes"],
            "allowed_fields": target["allowed_fields"], "current": node,
        })
    serialized_targets = json.dumps(snapshots, ensure_ascii=False, separators=(",", ":"))
    if len(serialized_targets) > MAX_WORKFLOW_REPAIR_TARGET_CHARS:
        raise WorkflowFailure(
            "LESSON_REPAIR_SCOPE_TOO_LARGE",
            "The affected lesson scope is too large for a bounded repair prompt.",
            internal_code="LESSON_REPAIR_SCOPE_TOO_LARGE",
            failure_stage="lesson_repair_target_snapshot",
            diagnostics={"repair_target_count": len(targets)},
        )
    language = "Vietnamese" if locale == "vi" else "English"
    return "\n".join([
        "Repair only the listed parts of this source-grounded lesson proposal.",
        f"Write in {language}; return JSON only and do not include reasoning.",
        lesson_output_language_policy(locale),
        "Return {\"patches\":[{\"path\":\"...\",\"replacement\":{...}}]}. Every patch must match a listed path and only use allowed_fields. Do not change course hierarchy, source scope, assets, or unrelated components.",
        "SOURCE EVIDENCE (bounded to the approved lesson scope):",
        evidence_context,
        "REPAIR TARGETS:",
        serialized_targets,
    ])


def apply_lesson_generation_repair_patches(
    proposal: dict[str, Any],
    targets: list[RepairTarget],
    payload: dict[str, Any],
) -> dict[str, Any]:
    patches = payload.get("patches") if isinstance(payload.get("patches"), list) else []
    expected = {target["path"]: target for target in targets}
    result = deepcopy(proposal)
    seen: set[str] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response contains an invalid patch.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
                diagnostics={"patch_count": len(patches)},
            )
        path = str(patch.get("path") or "")
        target = expected.get(path)
        replacement = patch.get("replacement")
        if target is None or path in seen:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to change an unapproved scope.",
                internal_code="LESSON_REPAIR_TARGET_OUT_OF_SCOPE",
                failure_stage="lesson_repair_target_whitelist",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if not isinstance(replacement, dict):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response contains an invalid replacement patch.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if set(replacement) - set(target["allowed_fields"]):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to change fields outside its allowed scope.",
                internal_code="LESSON_REPAIR_FIELD_OUT_OF_SCOPE",
                failure_stage="lesson_repair_field_whitelist",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        original = _proposal_path_object(proposal, path)
        if original is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="LESSON_REPAIR_TARGET_MISSING",
                failure_stage="lesson_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if not _repair_keeps_existing_source_scope(original, replacement):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to expand the approved source scope.",
                internal_code="LESSON_REPAIR_SOURCE_SCOPE_EXPANSION",
                failure_stage="lesson_repair_source_scope_guard",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node = _proposal_path_object(result, path)
        if node is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="LESSON_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="lesson_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node.update(replacement)
        seen.add(path)
    if seen != set(expected):
        raise WorkflowFailure(
            "LESSON_REPAIR_INVALID",
            "Repair response did not repair every required target.",
            internal_code="LESSON_REPAIR_TARGET_MISSING",
            failure_stage="lesson_repair_target_whitelist",
            diagnostics={"patch_count": len(patches), "expected_target_count": len(expected), "applied_target_count": len(seen)},
        )
    return result


async def build_lesson_author_checkpoint_result(
    request: RagLessonAuthorCheckpointRequest,
    *,
    context: str,
    source_outline: str,
    source_coverage: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    known_source_refs: set[str],
    retrieval: dict[str, Any],
    retrieval_usage: AiUsage,
    elapsed_ms: int,
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Generation per unit OR full deterministic acceptance; never DB persistence."""
    started = perf_counter()
    remaining = request.remaining_workflow_budget_ms - elapsed_ms
    if remaining <= 0:
        raise WorkflowFailure("PROVIDER_ERROR", "The chapter invocation deadline expired.",
                              internal_code="AI_STAGED_LESSON_WORKFLOW_TIMEOUT", failure_stage="chapter_checkpoint_deadline")
    if request.checkpoint_action == "generate_unit":
        preflight = instructional_plan_validation_result(request.blueprint_architecture.model_dump())
        codes = sorted({issue["code"] for issue in preflight.issues})
        emit({"stage": "chapter_instructional_plan_preflight", "event": "rejected" if preflight.errors else "passed",
              "validation_codes": codes, "error_count": len(preflight.errors),
              "findings": [safe_workflow_issue_summary(issue, repairable=False) for issue in preflight.errors[:32]],
              "omitted_finding_count": max(0, len(preflight.errors) - 32),
              "target_paths": sorted({safe_workflow_path(issue.get("path")) for issue in preflight.errors})})
        if preflight.errors:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The approved teaching plan has inconsistent objective bindings.",
                                  internal_code="CHAPTER_INSTRUCTIONAL_PLAN_INVALID", failure_stage="chapter_instructional_plan_preflight",
                                  diagnostics={"validation_codes": codes})
        value, generated_usage = await staged_writer.generate_staged_lesson_author_proposal(
            request, context, source_outline, source_coverage, source_rows=rows,
            source_coverage_manifest=manifest, checkpoint_unit_index=request.checkpoint_unit_index,
            remaining_workflow_budget_ms=remaining,
        )
        validate_lesson_author_proposal_source_refs(
            {"chapters": [{"lessons": [{"units": [value["unit"]]}]}]}, known_source_refs,
        )
        approved_title = [unit.title for lesson in request.blueprint_architecture.lessons for unit in lesson.units][value["unit_index"]]
        if value["unit"].get("title") != approved_title:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The generated unit changed its approved identity.",
                                  internal_code="CHAPTER_CHECKPOINT_UNIT_INVALID", failure_stage="chapter_checkpoint_unit_validation")
        usage = combine_usage(retrieval_usage, generated_usage)
        provider_known = value.pop("provider_usage_complete", False) is True
        # Existing retrieval embedding accounting uses local estimates. Do not
        # mislabel that aggregate as complete provider usage when settling holds.
        complete_usage = provider_known and retrieval_usage.totalTokens == 0
        emit({"stage": "chapter_checkpoint_unit_validation", "event": "unit_ready",
              "unit_index": value["unit_index"], "unit_path": value["unit_path"],
              "component_count": len(value["unit"].get("components", [])),
              "usage_source": "provider" if complete_usage else "mixed_or_unavailable"})
        return {**value, "correlation_id": request.correlation_id, "status": "unit_ready", "usage": usage.model_dump(),
                "usage_complete": complete_usage, "usage_source": "provider" if complete_usage else "mixed_or_unavailable",
                "retrieval": retrieval}

    skeleton = build_source_locked_staged_skeleton(request, manifest)
    batches = extract_lesson_author_unit_batches(skeleton, manifest, request.locale)
    validate_staged_skeleton_source_facts(batches, manifest)
    expected = checkpoint_expected_units(batches)
    for checkpoint in request.checkpoint_units:
        approved = expected[checkpoint.unit_index]
        if checkpoint.unit.get("title") != approved["unit_title"] or validate_staged_unit_content(checkpoint.unit, approved, strict_payload=True):
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "A stored unit no longer satisfies its approved contract.",
                                  internal_code="CHAPTER_CHECKPOINT_UNIT_INVALID", failure_stage="chapter_checkpoint_revalidation",
                                  diagnostics={"unit_index": checkpoint.unit_index})
    proposal = normalize_lesson_author_proposal_tree(assemble_checkpoint_chapter(skeleton, expected, request.checkpoint_units))
    # Ordered full-chapter gates. No implicit provider repair is allowed to modify
    # immutable checkpoints during finalization; failures remain fail-closed.
    all_codes: list[str] = []
    for stage, validate in (
        ("chapter_checkpoint_content_validation", lambda: validate_lesson_generation_workflow(proposal, manifest, known_source_refs)),
        ("chapter_checkpoint_pedagogical_validation", lambda: pedagogical_validation_result(proposal, request.blueprint_architecture.model_dump())),
        ("chapter_checkpoint_duplication_validation", lambda: duplicate_validation_result(proposal)),
    ):
        result = validate()
        codes = sorted({str(issue.get("code") or "LESSON_VALIDATION_FAILED") for issue in result.issues})
        all_codes.extend(codes)
        emit({"stage": stage, "event": "rejected" if result.errors else "passed", "validation_codes": codes,
              "error_count": len(result.errors), "unit_count": len(expected),
              "findings": [safe_workflow_issue_summary(issue, repairable=False) for issue in result.errors[:32]],
              "omitted_finding_count": max(0, len(result.errors) - 32),
              "target_paths": sorted({safe_workflow_path(issue.get("path")) for issue in result.errors})})
        if result.errors:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The complete chapter did not pass acceptance.",
                                  internal_code="CHAPTER_CHECKPOINT_REVALIDATION_FAILED", failure_stage=stage,
                                  diagnostics={"validation_codes": codes})
    coverage = validate_lesson_author_source_coverage(proposal, manifest)
    retrieval.update({"source_coverage_required_count": coverage["required_count"],
                      "source_coverage_covered_count": coverage["covered_count"],
                      "source_coverage_ratio": coverage["coverage_ratio"],
                      "source_coverage_missing_fact_ids": coverage["missing_fact_ids"], "source_coverage_status": coverage["status"]})
    emit({"stage": "chapter_checkpoint_finalize", "event": "ready", "unit_count": len(expected)})
    return {"checkpoint_version": 1, "correlation_id": request.correlation_id, "status": "ready", "proposal": proposal, "retrieval": retrieval,
            "usage": retrieval_usage.model_dump(), "usage_complete": retrieval_usage.totalTokens == 0,
            "usage_source": "no_generation" if retrieval_usage.totalTokens == 0 else "local_estimate",
            "workflow": {"workflow": "lesson_generation", "workflow_version": "chapter-checkpoint-1",
                         "status": "ready", "repair_count": 0, "validation_codes": all_codes,
                         "duration_ms": elapsed_ms + round((perf_counter() - started) * 1000), "node_durations_ms": {}}}


@app.post("/v1/lesson-author/chapter-checkpoint", dependencies=[Depends(require_internal_token)])
async def lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/chapter-checkpoint",
        settings.lesson_author_deadline_ms,
        _lesson_author_chapter_checkpoint(request, pool),
    )


async def _lesson_author_chapter_checkpoint(
    request: RagLessonAuthorCheckpointRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        # Covers retrieval and final validation too. Cancellation of a to_thread
        # await is NOT proof that the synchronous Google HTTP request was stopped.
        return await asyncio.wait_for(lesson_author_proposal(request, pool), request.remaining_workflow_budget_ms / 1000)
    except asyncio.TimeoutError as error:
        failure = WorkflowFailure("PROVIDER_ERROR", "The chapter invocation deadline expired.",
                                  internal_code="AI_STAGED_LESSON_WORKFLOW_TIMEOUT", failure_stage="chapter_checkpoint_deadline")
        cause: Exception = error
    except WorkflowFailure as error:
        failure, cause = error, error
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        code = str(detail.get("code") or "SOURCE_SCOPE_INCOMPLETE")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", code):
            code = "SOURCE_SCOPE_INCOMPLETE"
        provider_error = code.startswith("AI_PROVIDER_") or code == "AI_STAGED_LESSON_WORKFLOW_TIMEOUT"
        failure = WorkflowFailure("PROVIDER_ERROR" if provider_error else "SOURCE_SCOPE_INCOMPLETE",
                                  "The chapter checkpoint request could not complete.", internal_code=code,
                                  failure_stage="chapter_checkpoint_provider" if provider_error else "chapter_checkpoint_source_validation")
        cause = error
    except (ValueError, LessonAuthorProposalValidationError) as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint contract is invalid.",
                                  internal_code="CHAPTER_CHECKPOINT_CONTRACT_INVALID", failure_stage="chapter_checkpoint_contract")
        cause = error
    except Exception as error:
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "The chapter checkpoint operation failed.",
                                  internal_code="CHAPTER_CHECKPOINT_INTERNAL_ERROR", failure_stage="chapter_checkpoint_internal")
        cause = error
    diagnostics = {"workflow": "lesson_generation", "event": "final_failure",
                   "correlation_id": request.correlation_id, "conversation_id": request.conversation_id,
                   "checkpoint_action": request.checkpoint_action, "unit_index": request.checkpoint_unit_index,
                   "failure_stage": failure.failure_stage, "internal_failure_code": failure.internal_code,
                   "external_failure_code": failure.code, "duration_ms": round((perf_counter() - started) * 1000)}
    if failure.internal_code in {"CHAPTER_UNIT_CONTRACT_REJECTED", "CHAPTER_COMPONENT_REPAIR_EXHAUSTED"}:
        diagnostics.update(failure.diagnostics)
    if failure.code == "PROVIDER_ERROR":
        provider_keys = {"provider_http_status", "provider_error_type", "provider_status", "provider_error_category",
                         "provider_message_class", "provider_error_detail_count",
                         "provider_schema_constraint", "provider_error_markers", "usage_source", "provider_input_tokens", "provider_output_tokens",
                         "provider_total_tokens", "provider_finish_reason", "provider_schema_fingerprint"}
        diagnostics.update({key: value for key, value in failure.diagnostics.items() if key in provider_keys})
    logger.info("lesson_author_checkpoint_diagnostic %s", json.dumps(diagnostics, sort_keys=True))
    raise HTTPException(status_code=502 if failure.code == "PROVIDER_ERROR" else 422, detail={
        "code": failure.code, "message": "The chapter could not complete its validated checkpoint operation.",
        "internal_failure_code": failure.internal_code, "failure_stage": failure.failure_stage,
        "correlation_id": request.correlation_id,
    }) from cause


@app.post("/v1/lesson-author/proposal", dependencies=[Depends(require_internal_token)])
async def lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/proposal",
        settings.lesson_author_deadline_ms,
        _lesson_author_proposal(request, pool),
    )


async def _lesson_author_proposal(
    request: RagLessonAuthorRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    workflow_started = perf_counter()

    def emit_lesson_diagnostic(metadata: dict[str, Any]) -> None:
        """Log request-correlated, metadata-only lesson workflow diagnostics."""

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "lesson_generation",
            "workflow_version": "chapter-checkpoint-1" if isinstance(request, RagLessonAuthorCheckpointRequest) else "langgraph-v1",
            "architecture_contract_version": (
                request.blueprint_architecture.architecture_contract_version
                if request.blueprint_architecture is not None else None
            ),
            "correlation_id": request.correlation_id,
            "tenant_id": request.tenant_id,
            "kb_id": request.kb_id,
            "conversation_id": request.conversation_id,
            "source_document_ids": [document.document_id for document in request.source_documents],
            **metadata,
        }
        logger.info("lesson_author_proposal_diagnostic %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    emit_lesson_diagnostic({
        "stage": "lesson_author_request",
        "event": "received",
        "repair_pass_number": 0,
        "output_locale": request.locale,
        "language_policy_version": "lesson-language-1",
    })
    rows, retrieval_usage, structure_context = await retrieval_search.retrieve_chunks(pool, request)
    emit_lesson_diagnostic({
        "stage": "rag_retrieval",
        "event": "completed",
        "repair_pass_number": 0,
        "retrieved_chunk_count": len(rows),
        "target_source_scope_hard_locked": bool(structure_context.get("target_source_scope_hard_locked")),
        "target_source_scope_truncated": bool(structure_context.get("target_source_scope_truncated")),
    })
    context, sources = format_sources(rows, max_context_chars=retrieval_limits(request)["max_context_chars"])
    retrieval = build_retrieval_diagnostics(request, rows, sources, structure_context)
    source_coverage_manifest = structure_context.get("source_coverage_manifest")
    source_coverage = format_source_coverage_manifest(source_coverage_manifest)
    missing_blueprint_fact_ids = structure_context.get("blueprint_draft_missing_source_fact_ids", [])
    if missing_blueprint_fact_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                "message": "Không thể soạn chương vì một phần nguồn đã khóa trong Bản thiết kế không còn khả dụng. Vui lòng tạo lại Bản thiết kế khóa học.",
                "missing_source_fact_ids": missing_blueprint_fact_ids[:24],
                "retrieval": retrieval,
            },
        )
    if retrieval_search.target_source_scope_is_incomplete(structure_context, rows, sources, source_coverage_manifest):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_SOURCE_SCOPE_INCOMPLETE",
                "message": "Không thể soạn chương vì phạm vi tài liệu nguồn chưa được tải đầy đủ. Vui lòng re-index tài liệu rồi thử lại.",
                "retrieval": retrieval,
            },
        )
    if source_coverage_manifest and source_coverage_manifest.get("scope_unresolved"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_INFERRED_SCOPE_UNRESOLVED",
                "message": "Không thể ánh xạ đầy đủ các mục của chương vào tài liệu nguồn không có mục lục. Vui lòng re-index tài liệu rồi thử lại.",
                "missing_source_refs": sorted(
                    set(source_coverage_manifest.get("target_source_refs", []))
                    - set(source_coverage_manifest.get("resolved_source_refs", []))
                )[:20],
                "retrieval": retrieval,
            },
        )
    if source_coverage_manifest and source_coverage_manifest.get("truncated"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LESSON_AUTHOR_SOURCE_COVERAGE_TRUNCATED",
                "message": "Phạm vi nội dung nguồn vượt giới hạn kiểm tra đầy đủ; hệ thống không tạo bài học để tránh lược bỏ dữ liệu.",
                "retrieval": retrieval,
            },
        )
    if isinstance(request, RagLessonAuthorCheckpointRequest):
        # The normal endpoint applies this gate in its LangGraph evidence node.
        # Checkpoint mode branches before that graph, so retain the same gate here
        # and require the canonical manifest used to validate cached ownership.
        if not rows or not isinstance(source_coverage_manifest, dict) or not (
            source_coverage_manifest.get("facts") or source_coverage_manifest.get("supporting_evidence_facts")
        ):
            raise WorkflowFailure("SOURCE_EVIDENCE_INSUFFICIENT", "The approved chapter evidence is unavailable.",
                                  internal_code="SOURCE_EVIDENCE_INSUFFICIENT", failure_stage="chapter_checkpoint_evidence_validation")
        return await build_lesson_author_checkpoint_result(
            request, context=context, source_outline=structure_context.get("outline", ""), source_coverage=source_coverage,
            rows=rows, manifest=source_coverage_manifest, known_source_refs=set(structure_context.get("known_source_refs", set())),
            retrieval=retrieval, retrieval_usage=retrieval_usage, elapsed_ms=round((perf_counter() - workflow_started) * 1000),
            emit=emit_lesson_diagnostic,
        )
    prompt = build_lesson_author_prompt(
        request,
        context,
        structure_context.get("outline", ""),
        source_coverage,
    )
    async def retrieve_lesson_evidence() -> dict[str, Any]:
        # Retrieval occurred immediately before this request-local graph is
        # entered. Keep raw chunks in the endpoint closure, not graph state.
        return {
            "source_document_count": len(request.source_documents),
            "retrieved_count": len(rows),
            "returned_source_count": len(sources),
            "target_source_scope_hard_locked": bool(structure_context.get("target_source_scope_hard_locked")),
            "target_source_scope_truncated": bool(structure_context.get("target_source_scope_truncated")),
        }

    def validate_lesson_evidence(evidence: dict[str, Any]) -> WorkflowValidationResult:
        if not evidence.get("retrieved_count") or retrieval_search.target_source_scope_is_incomplete(structure_context, rows, sources, source_coverage_manifest):
            return WorkflowValidationResult([
                _workflow_issue("SOURCE_EVIDENCE_INSUFFICIENT", "The approved lesson source scope has insufficient retrieved evidence.", path="lesson"),
            ])
        return WorkflowValidationResult([], {"source_coverage": None})

    async def generate_lesson_candidate() -> WorkflowGenerationResult:
        """The generation node preserves existing staged and JSON safeguards."""
        total_generation_usage = AiUsage()
        proposal: dict[str, Any] | None = None
        last_error: str | None = None
        should_stage = should_stage_lesson_author_proposal(request, context)
        logger.info(
            "lesson_author_proposal_generation_mode conversation_id=%s operation=%s target_type=%s mode=%s context_chars=%s max_output_tokens=%s has_outline_context=%s has_target_scope=%s",
            request.conversation_id,
            request.operation,
            request.target_type or "none",
            "staged" if should_stage else "single",
            len(context),
            request.max_output_tokens,
            bool(request.outline_context.strip()),
            bool(request.target_scope_instruction.strip()),
        )
        if should_stage:
            try:
                staged_candidate, staged_usage = await staged_writer.generate_staged_lesson_author_proposal(
                    request,
                    context,
                    structure_context.get("outline", ""),
                    source_coverage,
                    source_rows=rows,
                    source_coverage_manifest=source_coverage_manifest,
                )
                total_generation_usage = combine_usage(total_generation_usage, staged_usage)
                allowed_source_refs = set(structure_context.get("known_source_refs", set()))
                staged_candidate, dropped_refs = drop_invalid_lesson_author_proposal_source_refs(staged_candidate, allowed_source_refs)
                if dropped_refs:
                    logger.warning("lesson_author_staged_proposal_dropped_unknown_refs refs=%s", ",".join(dropped_refs[:8]))
                validate_lesson_author_proposal_source_refs(staged_candidate, allowed_source_refs)
                validate_lesson_author_source_coverage(staged_candidate, source_coverage_manifest)
                proposal = staged_candidate
                logger.info("lesson_author_proposal_staged_valid context_chars=%s", len(context))
            except (HTTPException, LessonAuthorProposalValidationError) as error:
                if isinstance(error, HTTPException) and is_non_retryable_provider_error(error):
                    raise WorkflowFailure("PROVIDER_ERROR", "The provider rejected the staged lesson generation request.") from error
                last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                logger.warning("lesson_author_proposal_staged_failed conversation_id=%s reason=%s", request.conversation_id, last_error or "unknown")
                raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The staged lesson candidate did not satisfy its generation contract.") from error

        if proposal is None:
            for attempt in range(request.max_attempts):
                attempt_prompt = prompt if attempt == 0 else "\n\n".join([
                    prompt,
                    "The previous proposal did not satisfy the server validation contract.",
                    "Regenerate one complete, compact proposal now. Preserve the requested scope, include every chapter, lesson, unit and component required by the schema, ensure every lesson has at least one non-empty unit, use plain title fields without structural numbering, and return only one JSON object.",
                    f"Validation feedback from the previous response: {last_error}. Correct this exact issue in the new JSON; do not repeat the invalid component." if last_error else "",
                ])
                text, generation_usage = await provider.generate_content(
                    request.api_key,
                    request.model,
                    attempt_prompt,
                    max_output_tokens=request.max_output_tokens,
                    json_mode=True,
                    response_schema=build_lesson_author_proposal_response_schema(),
                    thinking_config=types.ThinkingConfig(include_thoughts=False),
                )
                total_generation_usage = combine_usage(total_generation_usage, generation_usage)
                try:
                    candidate = normalize_lesson_author_proposal_tree(parse_lesson_author_json(text, "proposal"))
                    validate_lesson_author_proposal_shape(candidate)
                    allowed_source_refs = set(structure_context.get("known_source_refs", set()))
                    candidate, dropped_refs = drop_invalid_lesson_author_proposal_source_refs(candidate, allowed_source_refs)
                    if dropped_refs:
                        logger.warning("lesson_author_proposal_dropped_unknown_refs refs=%s", ",".join(dropped_refs[:8]))
                    validate_lesson_author_proposal_source_refs(candidate, allowed_source_refs)
                    validate_lesson_author_source_coverage(candidate, source_coverage_manifest)
                    proposal = candidate
                    break
                except HTTPException as error:
                    if is_non_retryable_provider_error(error):
                        raise WorkflowFailure("PROVIDER_ERROR", "The provider rejected lesson proposal generation.") from error
                    last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                except LessonAuthorProposalValidationError as error:
                    last_error = re.sub(r"\s+", " ", str(error)).strip()[:240]
                logger.warning("lesson_author_proposal_invalid attempt=%s reason=%s", attempt + 1, last_error or "unknown")
        if proposal is None:
            raise WorkflowFailure("LESSON_VALIDATION_FAILED", "The provider did not return a valid lesson proposal after bounded attempts.")
        return WorkflowGenerationResult(proposal, total_generation_usage.model_dump())

    async def repair_lesson_candidate(
        candidate: dict[str, Any],
        targets: list[RepairTarget],
    ) -> WorkflowGenerationResult:
        repair_prompt = build_lesson_generation_repair_prompt(
            proposal=candidate,
            targets=targets,
            evidence_context=context,
            locale=request.locale,
        )
        text, repair_usage = await provider.generate_content(
            request.api_key,
            request.model,
            repair_prompt,
            max_output_tokens=min(request.max_output_tokens, 16_384),
            json_mode=True,
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )
        try:
            payload = parse_lesson_author_json(text, "lesson repair")
            repaired = apply_lesson_generation_repair_patches(candidate, targets, payload)
        except WorkflowFailure as error:
            logger.warning(
                "lesson_author_lesson_repair_patch_rejected failure_code=%s",
                error.internal_code or error.code,
            )
            raise
        except HTTPException as error:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Provider returned invalid JSON for scoped lesson repair.",
                internal_code="LESSON_REPAIR_JSON_INVALID",
                failure_stage="lesson_repair_json_parser",
                diagnostics={"provider_status_code": error.status_code},
            ) from error
        except LessonAuthorProposalValidationError as error:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Provider returned a repair payload that failed the lesson contract.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
            ) from error
        return WorkflowGenerationResult(repaired, repair_usage.model_dump())

    try:
        proposal, workflow = await run_lesson_generation_workflow(
            LessonGenerationWorkflowCallbacks(
                validate_contract=lambda: [],
                retrieve_evidence=retrieve_lesson_evidence,
                validate_evidence=validate_lesson_evidence,
                generate_proposal=generate_lesson_candidate,
                validate_content=lambda candidate: validate_lesson_generation_workflow(
                    candidate,
                    source_coverage_manifest,
                    set(structure_context.get("known_source_refs", set())),
                ),
                validate_pedagogy=lambda candidate: pedagogical_validation_result(
                    candidate,
                    request.blueprint_architecture.model_dump() if request.blueprint_architecture else None,
                ),
                validate_duplicates=duplicate_validation_result,
                repair_content=repair_lesson_candidate,
                emit_diagnostic=emit_lesson_diagnostic,
            ),
            request_context={
                "correlation_id": request.correlation_id,
                "tenant_id": request.tenant_id,
                "kb_id": request.kb_id,
                "conversation_id": request.conversation_id,
                "operation": request.operation,
                "target_type": request.target_type,
                "source_document_count": len(request.source_documents),
            },
            max_repair_attempts=min(2, max(0, settings.lesson_workflow_max_repair_attempts)),
        )
    except WorkflowFailure as error:
        emit_lesson_diagnostic({
            "stage": error.failure_stage or "lesson_validation",
            "event": "final_failure",
            "repair_pass_number": int(error.diagnostics.get("repair_pass_number") or 0),
            "failure_stage": error.failure_stage or "lesson_validation",
            "internal_failure_code": error.internal_code,
            "external_failure_code": error.code,
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        })
        raise HTTPException(
            status_code=422 if error.code in {"SOURCE_EVIDENCE_INSUFFICIENT", "SOURCE_SCOPE_INCOMPLETE"} else 502,
            detail={
                "code": error.code,
                "message": "AI chưa thể tạo đề xuất nội dung bài học hợp lệ.",
                "workflow_issues": error.issues[:20],
            },
        ) from error
    coverage_metrics = validate_lesson_author_source_coverage(proposal, source_coverage_manifest)
    retrieval.update(
        {
            "source_coverage_required_count": coverage_metrics["required_count"],
            "source_coverage_covered_count": coverage_metrics["covered_count"],
            "source_coverage_ratio": coverage_metrics["coverage_ratio"],
            "source_coverage_missing_fact_ids": coverage_metrics["missing_fact_ids"],
            "source_coverage_status": coverage_metrics["status"],
        },
    )
    usage = combine_usage(retrieval_usage, AiUsage(**(workflow.get("usage") or {})))
    return {"proposal": proposal, "usage": usage.model_dump(), "sources": sources, "retrieval": retrieval, "workflow": workflow}


# Legacy V2 skeleton/shard providers fall back on these codes; AI_PROVIDER_RATE_LIMITED was
# reported as AI_PROVIDER_QUOTA_EXHAUSTED before the 429 classification and keeps that branch.
ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES = frozenset({
    "AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE", "AI_PROVIDER_QUOTA_EXHAUSTED", "AI_PROVIDER_RATE_LIMITED",
})


def _orchestration_v2_http_error(code: str, message: str, *, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


async def _orchestration_v2_generate_content(
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
    response_schema: type[BaseModel],
    correlation_id: str,
    generation_stage: str,
) -> tuple[str, AiUsage]:
    """Call Gemini with a safe, terminal classification for definitive refusals.

    A provider HTTP 400/401/403 proves that this request was rejected rather
    than accepted for generation. Expose only a stable internal code so the
    durable worker can release its reservation and fail immediately. Network,
    timeout, and 5xx ambiguity retain the existing outcome-unknown path.
    """

    try:
        return await provider.generate_content(
            api_key,
            model,
            prompt,
            max_output_tokens=max_output_tokens,
            json_mode=True,
            response_schema=response_schema,
        )
    except HTTPException:
        raise
    except Exception as error:
        status = provider_http_error_status(error)
        diagnostics = safe_provider_error_diagnostics(error)
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "provider_request_rejected" if status in {400, 401, 403} else "provider_request_failed",
            "correlation_id": correlation_id,
            "generation_stage": generation_stage,
            "model": model,
            **diagnostics,
        }, sort_keys=True))
        if status == 400:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_REQUEST_REJECTED",
                "AI provider rejected the structured generation request.",
                status_code=502,
            ) from error
        if status in {401, 403}:
            raise _orchestration_v2_http_error(
                "AI_PROVIDER_AUTH_REJECTED",
                "AI provider rejected the configured credentials or access policy.",
                status_code=502,
            ) from error
        raise


@app.post("/v1/lesson-author/orchestration-v2/source-snapshot", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_source_snapshot(
    request: RagLessonAuthorSourceSnapshotV2Request,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    """Return one deterministic source page without a provider call or whole-source materialization."""

    document_ids = sorted(document.document_id for document in request.source_documents)
    cursor = request.cursor
    if cursor is not None and str(cursor["document_id"]) not in set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_INVALID", "The source cursor is outside the selected source authority.",
        )
    indexes = await pool.fetch(
        """
        SELECT r.id::text AS index_id,r.document_id::text AS document_id,r.content_sha256,
               r.chunk_count,r.embedding_model,r.embedding_dimensions
        FROM rag_document_indexes r
        WHERE r.tenant_id=$1::uuid AND r.kb_id=$2::uuid AND r.document_id=ANY($3::uuid[])
          AND r.engine='self_built_rag' AND r.status='learned' AND r.is_active=true
          AND r.embedding_model=$4 AND r.embedding_dimensions=$5::int
        ORDER BY r.document_id
        """,
        request.tenant_id, request.kb_id, document_ids,
        normalize_embedding_model(request.embedding_model), request.embedding_dimensions,
    )
    if len(indexes) != len(document_ids) or {str(row["document_id"]) for row in indexes} != set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_UNAVAILABLE", "The selected learned source revision is unavailable.",
        )
    structure_rows = await pool.fetch(
        """
        SELECT c.document_id::text AS document_id,
               (jsonb_agg(c.metadata->'source_structure' ORDER BY c.chunk_no)
                 FILTER (WHERE c.metadata ? 'source_structure'))->0 AS source_structure,
               min(lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               ) AS source_evidence_revision,
               count(*)::integer AS actual_chunk_count,
               count(*) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_chunk_count,
               count(DISTINCT lower(c.metadata->>'source_evidence_revision')) FILTER (
                 WHERE coalesce(c.metadata->>'source_evidence_revision','') ~ '^[0-9a-fA-F]{64}$'
               )::integer AS evidence_revision_distinct_count
        FROM rag_chunks c
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
        GROUP BY c.document_id
        ORDER BY c.document_id
        """,
        request.tenant_id, request.kb_id, [str(row["index_id"]) for row in indexes],
    )
    structures_by_document = {
        str(row["document_id"]): decode_json_object(row["source_structure"])
        for row in structure_rows
        if row.get("source_structure") is not None
    }
    structure_rows_by_document = {str(row["document_id"]): row for row in structure_rows}
    evidence_revisions_by_document: dict[str, str] = {}
    evidence_status_by_document: dict[str, str] = {}
    for index_row in indexes:
        document_id = str(index_row["document_id"])
        declared_chunks = int(index_row["chunk_count"] or 0)
        summary = structure_rows_by_document.get(document_id) or {}
        revision = str(summary.get("source_evidence_revision") or "").strip().casefold()
        valid_revision = revision if re.fullmatch(r"[0-9a-f]{64}", revision) else None
        actual_chunks = int(summary.get("actual_chunk_count") if summary.get("actual_chunk_count") is not None
                            else declared_chunks)
        revision_chunks = int(summary.get("evidence_revision_chunk_count")
                              if summary.get("evidence_revision_chunk_count") is not None
                              else declared_chunks if valid_revision else 0)
        revision_count = int(summary.get("evidence_revision_distinct_count")
                             if summary.get("evidence_revision_distinct_count") is not None
                             else 1 if valid_revision else 0)
        if declared_chunks < 1 or actual_chunks != declared_chunks:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index does not match its declared chunk inventory.",
                status_code=409,
            )
        if revision_chunks == 0 and revision_count == 0 and valid_revision is None:
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        elif (revision_chunks == declared_chunks and revision_count == 1
              and valid_revision is not None):
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_READY
            evidence_revisions_by_document[document_id] = valid_revision
        else:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index contains mixed structured-evidence revisions.",
                status_code=409,
            )
    policy_documents = [{
        "document_id": document_id,
        "structure": structures_by_document.get(document_id),
    } for document_id in document_ids]
    chapter_policy = resolve_source_chapter_policy(policy_documents)
    policy_mode = str(chapter_policy.get("mode") or "NEEDS_STRUCTURE_REVIEW")
    authority_mode: Literal["locked", "model_designed", "needs_review"] = (
        "locked" if policy_mode.startswith("SOURCE_LOCKED")
        else "model_designed" if policy_mode == "MODEL_DESIGNED"
        else "needs_review"
    )
    authority_source: Literal["toc", "headings", "none", "ambiguous"] = (
        "toc" if policy_mode == "SOURCE_LOCKED_TOC"
        else "headings" if policy_mode == "SOURCE_LOCKED_HEADINGS"
        else "none" if authority_mode == "model_designed"
        else "ambiguous"
    )
    chapters = [SourceOutlineChapterV2(
        order=index,
        document_id=str(chapter["document_id"]),
        source_ref=str(chapter["source_ref"]),
        title=str(chapter["title"]),
    ) for index, chapter in enumerate(chapter_policy.get("chapters", []))] if authority_mode == "locked" else []
    confidences = [float(structure.get("confidence") or 0) for structure in structures_by_document.values()
                   if isinstance(structure, dict)]
    reason_codes = [str(value)[:120] for value in chapter_policy.get("reason_codes", [])[:32]]
    legacy_document_count = sum(
        status == SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        for status in evidence_status_by_document.values()
    )
    if legacy_document_count and "STRUCTURED_EVIDENCE_REVISION_MISSING" not in reason_codes:
        reason_codes = [*reason_codes[:31], "STRUCTURED_EVIDENCE_REVISION_MISSING"]
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "source_snapshot_legacy_evidence_review_required",
            "correlation_id": request.correlation_id,
            "source_snapshot_hash": request.source_snapshot_hash,
            "legacy_document_count": legacy_document_count,
            "selected_document_count": len(document_ids),
        }, sort_keys=True))
    authority_payload = {
        "mode": authority_mode,
        "source": authority_source,
        "complete": bool(chapter_policy.get("complete")),
        "confidence": min(confidences) if confidences else 0.0,
        "reason_codes": reason_codes,
        "chapters": [chapter.model_dump(mode="json") for chapter in chapters],
    }
    source_authority = SourceOutlineAuthorityV2(
        **authority_payload,
        structure_hash=orchestration_v2_canonical_hash(authority_payload),
    )
    revision_payload: list[dict[str, Any]] = []
    for row in indexes:
        document_id = str(row["document_id"])
        revision_entry: dict[str, Any] = {
            "document_id": document_id,
            "index_id": str(row["index_id"]),
            "content_sha256": str(row["content_sha256"] or ""),
            "chunk_count": int(row["chunk_count"] or 0),
            "embedding_model": str(row["embedding_model"]),
            "embedding_dimensions": int(row["embedding_dimensions"]),
            "source_evidence_status": evidence_status_by_document[document_id],
        }
        # Preserve the exact legacy revision for already learned documents.
        # A new revision is materialized only after re-indexing has produced
        # explicit structured-evidence metadata.
        evidence_revision = evidence_revisions_by_document.get(document_id)
        if evidence_revision is not None:
            revision_entry["source_evidence_revision"] = evidence_revision
        revision_payload.append(revision_entry)
    source_revision = orchestration_v2_canonical_hash(revision_payload)
    if request.expected_source_revision and request.expected_source_revision != source_revision:
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_CHANGED", "The learned source revision changed while paging.", status_code=409,
        )

    index_ids = [str(row["index_id"]) for row in indexes]
    after_document_id = str(cursor["document_id"]) if cursor else None
    after_chunk_no = int(cursor["chunk_no"]) if cursor else -1
    row_limit = 128
    fetched = await pool.fetch(
        """
        SELECT c.content,c.source_page,c.source_section,c.metadata,c.chunk_no,c.index_id::text AS index_id,
               c.document_id::text AS document_id,d.name AS document_name
        FROM rag_chunks c
        JOIN kb_documents d ON d.id=c.document_id AND d.tenant_id=c.tenant_id AND d.kb_id=c.kb_id
        WHERE c.tenant_id=$1::uuid AND c.kb_id=$2::uuid AND c.index_id=ANY($3::uuid[])
          AND ($4::uuid IS NULL OR c.document_id>$4::uuid
               OR (c.document_id=$4::uuid AND c.chunk_no>$5::int))
        ORDER BY c.document_id,c.chunk_no
        LIMIT $6::int
        """,
        request.tenant_id, request.kb_id, index_ids, after_document_id, after_chunk_no, row_limit + 1,
    )
    document_order = {document_id: index + 1 for index, document_id in enumerate(document_ids)}
    facts: list[SourceSnapshotFactV2] = []
    content_bytes = 0
    last_cursor: dict[str, Any] | None = None
    has_more = len(fetched) > row_limit
    for raw_row in fetched[:row_limit]:
        row = dict(raw_row)
        document_id = str(row["document_id"])
        chunk_no = int(row["chunk_no"])
        metadata = decode_json_object(row.get("metadata")) or {}
        content_kinds = {
            str(value).strip().casefold()
            for value in metadata.get("content_kinds", [])
            if isinstance(value, str) and value.strip()
        } if isinstance(metadata.get("content_kinds"), list) else set()
        table_count = max(0, min(100, int(metadata.get("table_count") or 0))) \
            if str(metadata.get("table_count") or "0").isdigit() else 0
        fact_texts = extract_source_coverage_facts(
            clean_text(str(row.get("content") or "")),
            preserve_table_numeric=table_count > 0 or "table" in content_kinds,
        )
        page = row.get("source_page") if isinstance(row.get("source_page"), int) and row.get("source_page") > 0 else None
        source_ref = str(metadata.get("source_ref") or row.get("source_section") or "").strip()[:255] or None
        heading = metadata.get("heading_path")
        if isinstance(heading, list):
            heading = " › ".join(
                str(value).strip()[:120] for value in heading[:8] if str(value).strip()
            )
        scope_title = str(heading or row.get("source_section") or "").strip()
        if not scope_title:
            suffix = f" · page {page}" if page is not None else f" · chunk {chunk_no + 1}"
            scope_title = f"{str(row.get('document_name') or 'Source document')}{suffix}"
        scope_title = scope_title[:500]
        visual_regions = metadata.get("visual_regions")
        if not isinstance(visual_regions, list):
            visual_regions = []
        visual_regions = [region for region in visual_regions[:4] if isinstance(region, dict)]
        visual_prompt_text = str(metadata.get("visual_prompt_text") or "").strip()[:600] or None
        # Scope V3 is a deterministic density lane. Four adjacent chunks share
        # one scope only when their same-index fact partitions remain within a
        # complete instructional-unit budget. This is pagination independent:
        # no cross-request mutable counter can change scope identity.
        scope_group = chunk_no // SOURCE_SCOPE_CHUNKS_PER_GROUP
        if source_ref:
            scope_owner = f"source-ref:{source_ref}"
        else:
            scope_owner = "title:" + hashlib.sha256(scope_title.encode("utf-8")).hexdigest()[:24]
        candidates: list[SourceSnapshotFactV2] = []
        fact_index = 0
        try:
            fact_partitions = partition_chunk_facts_for_density(fact_texts)
        except ValueError as error:
            raise _orchestration_v2_http_error(
                "SOURCE_FACT_DENSITY_INVALID",
                "One canonical source fact exceeds the instructional density contract.",
            ) from error
        for lane_index, partition in enumerate(fact_partitions):
            scope_locator = f"{scope_owner}:chunk-group:{scope_group}:lane:{lane_index}"
            scope_key = "scope3_" + hashlib.sha256(
                f"{document_id}\x1e{scope_locator}".encode("utf-8"),
            ).hexdigest()[:32]
            # Chunk metadata describes every structure found anywhere in the
            # chunk. Density partitioning creates independent scopes, so
            # copying a chunk-level table marker onto a prose-only lane makes
            # downstream contracts require a table that cannot be faithfully
            # reconstructed from that lane. Keep only structure represented by
            # the canonical facts in this partition.
            partition_table_row_count = sum(
                bool(re.match(r"^Row\s+\d+\s*:\s*.+\|.+$", text, re.IGNORECASE))
                for text in partition
            )
            partition_has_table = partition_table_row_count >= 2
            partition_content_kinds = set(content_kinds)
            if partition_has_table:
                partition_content_kinds.add("table")
            else:
                partition_content_kinds.discard("table")
            partition_table_count = max(1, table_count) if partition_has_table else 0
            for text in partition:
                fact_index += 1
                candidates.append(SourceSnapshotFactV2(
                    document_id=document_id,
                    fact_key=f"d{document_order[document_id]}-c{chunk_no + 1}-f{fact_index}",
                    scope_key=scope_key,
                    fact_text=text,
                    source_ref=source_ref,
                    source_page=page,
                    source_chunk=chunk_no,
                    locator={
                        "index_id": str(row["index_id"]),
                        "source_revision": source_revision,
                        "scope_title": scope_title,
                        "parser_version": str(metadata.get("parser_version") or "")[:80] or None,
                        "content_kinds": sorted(partition_content_kinds)[:8],
                        "table_count": partition_table_count,
                        "source_evidence_revision": (
                            str(metadata.get("source_evidence_revision") or "").strip().casefold()
                            if re.fullmatch(
                                r"[0-9a-f]{64}",
                                str(metadata.get("source_evidence_revision") or "").strip().casefold(),
                            )
                            else None
                        ),
                        "source_evidence_status": evidence_status_by_document[document_id],
                        "structured_evidence_contract_version": str(
                            metadata.get("structured_evidence_contract_version") or ""
                        )[:80] or None,
                        "visual_regions": visual_regions,
                        "visual_prompt_text": visual_prompt_text,
                        "instructional_density_policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
                        "scope_partition_lane": lane_index,
                    },
                ))
        candidate_bytes = sum(len(fact.fact_text.encode("utf-8")) for fact in candidates)
        exceeds = (len(facts) + len(candidates) > request.page_max_facts
                   or content_bytes + candidate_bytes > request.page_max_bytes)
        if exceeds and facts:
            has_more = True
            break
        if exceeds:
            raise _orchestration_v2_http_error(
                "SOURCE_CHUNK_EXCEEDS_PAGE", "One source chunk exceeds the bounded page contract.",
            )
        facts.extend(candidates)
        content_bytes += candidate_bytes
        last_cursor = {"document_id": document_id, "chunk_no": chunk_no}

    if has_more and last_cursor is None:
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_STALLED", "The source cursor could not advance.",
        )
    facts_wire = [fact.model_dump(mode="json") for fact in facts]
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "source_snapshot_page_ready", "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash, "source_revision": source_revision,
        "source_fact_count": len(facts_wire), "source_content_bytes": content_bytes,
        "source_evidence_ready_document_count": len(document_ids) - legacy_document_count,
        "source_evidence_legacy_document_count": legacy_document_count,
        "has_more": has_more, "provider_call_count": 0,
    }, sort_keys=True))
    return {
        "contract_version": 2,
        "source_snapshot_hash": request.source_snapshot_hash,
        "source_revision": source_revision,
        "source_authority": source_authority.model_dump(mode="json"),
        "facts": facts_wire,
        "next_cursor": last_cursor if has_more else None,
        "has_more": has_more,
        "page_content_bytes": content_bytes,
        "usage": AiUsage().model_dump(),
    }


async def _idm_generate(api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, AiUsage]:
    """Provider transport for ``app.idm``: map service errors onto ``IdmProviderError``.

    An exhausted key or a rate limit that outlived the bounded wait is terminal for the
    stage (``RUN_STOPPING_PROVIDER_CODES``): it reaches Node as a 503 with that code
    instead of turning into deterministic course content.
    """

    try:
        return await provider.generate_content(api_key, model, prompt, **options)
    except HTTPException as error:
        code = str((error.detail if isinstance(error.detail, dict) else {}).get("code") or "AI_PROVIDER_UNAVAILABLE")
        if code in RUN_STOPPING_PROVIDER_CODES:
            raise IdmProviderError(code, terminal=True, http_status=503) from error
        raise IdmProviderError(code, terminal=code not in TRANSIENT_PROVIDER_CODES,
                               http_status=error.status_code) from error
    except AppError as error:
        raise IdmProviderError(error.code, terminal=False, http_status=error.http_status) from error
    except Exception as error:
        status = provider_http_error_status(error)
        logger.warning("lesson_author_idm %s", json.dumps({
            "event": "provider_request_failed", "model": model, **safe_provider_error_diagnostics(error),
        }, sort_keys=True))
        if status == 400:
            raise IdmProviderError("AI_PROVIDER_REQUEST_REJECTED", terminal=True) from error
        if status in {401, 403}:
            raise IdmProviderError("AI_PROVIDER_AUTH_REJECTED", terminal=True) from error
        raise IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False) from error


def _idm_runtime(request: RagChatRequest, *, budget_ms: int, allowance: Any | None) -> IdmRuntime:
    return IdmRuntime(
        generate=_idm_generate, api_key=require_provider_api_key(request.api_key), model=request.model,
        locale=request.locale, deadline=monotonic() + budget_ms / 1000, correlation_id=request.correlation_id,
        token_allowance=(IdmTokenAllowance(allowance.input_tokens, allowance.output_tokens)
                         if allowance is not None else None),
        provider_call_timeout_ms=settings.idm_provider_call_timeout_ms,
        rate_limit_max_wait_ms=settings.provider_rate_limit_max_wait_ms,
    )


def _idm_http_error(error: IdmError) -> HTTPException:
    if isinstance(error, IdmProviderError):
        return _orchestration_v2_http_error(error.code, "AI provider failed for the IDM stage.",
                                            status_code=error.http_status)
    return _orchestration_v2_http_error(error.code, "The IDM stage cannot be completed for this source.")


@app.post("/v1/lesson-author/orchestration-v2/course-skeleton", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    if request.idm is not None:
        try:
            result = await run_idm_course_design(
                request.idm, source_snapshot_hash=request.source_snapshot_hash,
                runtime=_idm_runtime(request, budget_ms=request.idm.remaining_budget_ms,
                                     allowance=request.idm.token_allowance),
                parallelism=settings.idm_w1_parallelism, max_sections=settings.idm_max_sections,
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_course_skeleton(request=request)
    record_fallback("course_skeleton", result)
    return result


async def _lesson_author_orchestration_v2_course_skeleton(
    request: RagLessonAuthorCourseSkeletonV2Request,
) -> dict[str, Any]:
    """Generate only global structure; chapter content is delegated to bounded shards."""

    prompt = skeleton_prompt_v2(request.locale, request.scope_catalog, request.source_authority)
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "course_skeleton", **COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    }, sort_keys=True))
    usage = AiUsage()
    last_code = "ARCHITECTURE_SKELETON_INVALID"
    provider_failure_code: str | None = None
    for attempt in range(1, request.max_attempts + 1):
        try:
            text, attempt_usage = await _orchestration_v2_generate_content(
                request.api_key, request.model, prompt,
                max_output_tokens=request.max_output_tokens,
                response_schema=CourseSkeletonProviderWireV2,
                correlation_id=request.correlation_id,
                generation_stage="course_skeleton",
            )
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            # The Node worker persists its dispatch fence before this HTTP
            # request. Unknown SDK/transport failures must therefore settle
            # conservatively instead of turning a valid source fallback into a
            # blank 5xx response.
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_provider_fallback",
                "correlation_id": request.correlation_id,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            }, sort_keys=True))
            break
        usage = combine_usage(usage, attempt_usage)
        try:
            draft = CourseSkeletonDraftV2.model_validate_json(text)
            skeleton = bind_course_skeleton_v2(
                draft,
                source_snapshot_hash=request.source_snapshot_hash,
                locale=request.locale,
                scope_catalog=request.scope_catalog,
                source_authority=request.source_authority,
            )
            logger.info("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.source_snapshot_hash,
                "scope_count": len(request.scope_catalog), "chapter_count": len(skeleton.chapters),
                "provider_attempt": attempt,
            }, sort_keys=True))
            return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
            }, sort_keys=True))
        except ValidationError as error:
            last_code = "ARCHITECTURE_SKELETON_INVALID"
            safe_errors = [{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                           for item in error.errors(include_url=False, include_input=False)[:10]]
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "course_skeleton_validation_failed",
                "correlation_id": request.correlation_id,
                "provider_attempt": attempt,
                "failure_code": last_code,
                "validation_errors": safe_errors,
            }, sort_keys=True))
    fallback = fallback_course_skeleton_draft_v2(
        request.locale, request.scope_catalog, request.source_authority,
    )
    skeleton = bind_course_skeleton_v2(
        fallback,
        source_snapshot_hash=request.source_snapshot_hash,
        locale=request.locale,
        scope_catalog=request.scope_catalog,
        source_authority=request.source_authority,
    )
    logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "course_skeleton_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash,
        "scope_count": len(request.scope_catalog),
        "chapter_count": len(skeleton.chapters),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
    }, sort_keys=True))
    return {"contract_version": 2, "skeleton": skeleton.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}


@app.post("/v1/lesson-author/orchestration-v2/chapter-shard", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    context = request.idm_module_context
    if context is not None:
        try:
            result = await run_idm_module_design(
                context=context, skeleton=request.skeleton, plan=request.shard_plan, facts=request.source_facts,
                runtime=_idm_runtime(request, budget_ms=context.remaining_budget_ms,
                                     allowance=context.token_allowance),
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_chapter_shard(request=request)
    record_fallback("chapter_shard", result)
    return result


async def _lesson_author_orchestration_v2_chapter_shard(
    request: RagLessonAuthorChapterShardV2Request,
) -> dict[str, Any]:
    """Generate one independently retryable, source-bounded chapter shard."""

    try:
        prompt = chapter_shard_prompt_v2(
            request.locale, request.skeleton, request.shard_plan, request.source_facts,
        )
    except (OrchestrationContractError, StopIteration) as error:
        code = error.code if isinstance(error, OrchestrationContractError) else "ARCHITECTURE_SHARD_IDENTITY_MISMATCH"
        raise _orchestration_v2_http_error(code, "The chapter shard context is invalid.") from error
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "provider_schema_projection_ready", "correlation_id": request.correlation_id,
        "generation_stage": "chapter_shard", "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index, **CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    }, sort_keys=True))
    usage = AiUsage()
    last_code = "ARCHITECTURE_SHARD_INVALID"
    provider_failure_code: str | None = None
    repair_hint = ""
    last_provider_text: str | None = None
    best_salvage: tuple[Any, dict[str, Any]] | None = None
    for attempt in range(1, request.max_attempts + 1):
        try:
            text, attempt_usage = await _orchestration_v2_generate_content(
                request.api_key, request.model, prompt + repair_hint,
                max_output_tokens=request.max_output_tokens,
                response_schema=ChapterShardProviderWireV2,
                correlation_id=request.correlation_id,
                generation_stage="chapter_shard",
            )
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, dict) else {}
            code = detail.get("code")
            if code not in ORCHESTRATION_V2_PROVIDER_FALLBACK_CODES:
                raise
            provider_failure_code = str(code)
            last_code = provider_failure_code
            break
        except Exception as error:
            provider_failure_code = "AI_PROVIDER_UNAVAILABLE"
            last_code = provider_failure_code
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_provider_fallback",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": provider_failure_code,
                "exception_type": type(error).__name__,
            }, sort_keys=True))
            break
        usage = combine_usage(usage, attempt_usage)
        last_provider_text = text
        try:
            draft = parse_chapter_shard_draft_v2(text)
            compiler_diagnostics: list[dict[str, Any]] = []
            shard = bind_chapter_shard_v2(
                draft,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
                compiler_diagnostics=compiler_diagnostics,
            )
            logger.info("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_ready", "correlation_id": request.correlation_id,
                "source_snapshot_hash": request.skeleton.source_snapshot_hash,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "shard_count": request.shard_plan.shard_count,
                "source_fact_count": len(request.source_facts), "provider_attempt": attempt,
                "component_compiler": compiler_diagnostics[:24],
                "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
            }, sort_keys=True))
            return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                    "usage": usage.model_dump(), "usage_complete": True,
                    "usage_source": "provider", "content_origin": "provider_validated",
                    "quality_state": "validated"}
        except OrchestrationContractError as error:
            last_code = error.code
            safe_errors = [{"type": "contract_error", "loc": [], "code": error.code}]
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_code = "ARCHITECTURE_SHARD_INVALID"
            safe_errors = ([{"type": item.get("type"), "loc": list(item.get("loc") or ())}
                            for item in error.errors(include_url=False, include_input=False)[:12]]
                           if isinstance(error, ValidationError)
                           else [{"type": type(error).__name__, "loc": []}])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "chapter_shard_validation_failed", "correlation_id": request.correlation_id,
            "chapter_key": request.shard_plan.chapter_key, "shard_index": request.shard_plan.shard_index,
            "provider_attempt": attempt, "failure_code": last_code, "validation_errors": safe_errors,
        }, sort_keys=True))
        try:
            candidate_salvage = salvage_chapter_shard_draft_v2(
                text,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
            )
            if (candidate_salvage is not None
                    and (best_salvage is None
                         or candidate_salvage[1]["accepted_provider_unit_count"]
                         > best_salvage[1]["accepted_provider_unit_count"])):
                best_salvage = candidate_salvage
        except (OrchestrationContractError, ValidationError, ValueError, json.JSONDecodeError):
            pass
        repair_hint = (" REPAIR_REQUIREMENTS: Return the complete schema again. Correct these validation locations: "
                       + json.dumps(safe_errors, separators=(",", ":")) + ".")

    if best_salvage is None and last_provider_text is not None:
        try:
            best_salvage = salvage_chapter_shard_draft_v2(
                last_provider_text,
                skeleton=request.skeleton,
                plan=request.shard_plan,
                source_facts=request.source_facts,
            )
        except (OrchestrationContractError, ValidationError, ValueError, json.JSONDecodeError) as error:
            logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "chapter_shard_partial_salvage_rejected",
                "correlation_id": request.correlation_id,
                "chapter_key": request.shard_plan.chapter_key,
                "shard_index": request.shard_plan.shard_index,
                "failure_code": error.code if isinstance(error, OrchestrationContractError) else type(error).__name__,
            }, sort_keys=True))
            best_salvage = None
    if best_salvage is not None:
        shard, diagnostics = best_salvage
        component_decisions = diagnostics.pop("component_decisions", [])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "chapter_shard_partial_salvage_ready",
            "correlation_id": request.correlation_id,
            "chapter_key": request.shard_plan.chapter_key,
            "shard_index": request.shard_plan.shard_index,
            "source_fact_count": len(request.source_facts),
            "provider_attempts": request.max_attempts,
            "last_failure_code": last_code,
            **diagnostics,
            "component_compiler": component_decisions[:24],
            "omitted_component_compiler_unit_count": max(0, len(component_decisions) - 24),
        }, sort_keys=True))
        return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
                "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
                "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
                "content_origin": "structured_fallback", "quality_state": "review_required"}

    fallback = fallback_chapter_shard_draft_v2(
        request.skeleton, request.shard_plan, request.source_facts,
    )
    compiler_diagnostics: list[dict[str, Any]] = []
    shard = bind_chapter_shard_v2(
        fallback,
        skeleton=request.skeleton,
        plan=request.shard_plan,
        source_facts=request.source_facts,
        compiler_diagnostics=compiler_diagnostics,
    )
    logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "chapter_shard_deterministic_fallback_ready",
        "correlation_id": request.correlation_id,
        "chapter_key": request.shard_plan.chapter_key,
        "shard_index": request.shard_plan.shard_index,
        "source_fact_count": len(request.source_facts),
        "provider_attempts": request.max_attempts,
        "last_failure_code": last_code,
        "component_compiler": compiler_diagnostics[:24],
        "omitted_component_compiler_unit_count": max(0, len(compiler_diagnostics) - 24),
    }, sort_keys=True))
    return {"contract_version": 2, "shard": shard.model_dump(mode="json"),
            "usage": usage.model_dump(), "usage_complete": provider_failure_code is None,
            "usage_source": "reserved_upper_bound" if provider_failure_code else "provider",
            "content_origin": "structured_fallback", "quality_state": "review_required"}


@app.post("/v1/lesson-author/orchestration-v2/unit", dependencies=[Depends(require_internal_token)])
async def lesson_author_orchestration_v2_unit(
    request: RagLessonAuthorUnitV2Request,
) -> dict[str, Any]:
    if request.unit_contract.idm_unit_brief is not None:
        budget_ms = max(1_000, request.remaining_workflow_budget_ms - ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS)
        try:
            result = await run_idm_unit(
                contract=request.unit_contract, runtime=_idm_runtime(request, budget_ms=budget_ms, allowance=None),
                deps=_idm_unit_deps(request), fallback_only=request.fallback_only,
            )
        except IdmError as error:
            raise _idm_http_error(error) from error
        except ValidationError as error:
            raise _idm_http_error(IdmStageError("IDM_STAGE_OUTPUT_INVALID")) from error
    else:
        result = await _lesson_author_orchestration_v2_unit(request=request)
    record_fallback("unit", result)
    return result


def _idm_unit_deps(request: RagLessonAuthorUnitV2Request) -> IdmUnitDeps:
    contract = request.unit_contract
    manifest = unit_contract_manifest_v2(contract, locale=request.locale)
    bundle = manifest.get("source_evidence_bundle")
    supporting = list(dict.fromkeys(item for plan in contract.component_plan
                                    for item in plan.supporting_evidence_fact_ids))
    source_locked_components = build_orchestration_v2_source_locked_components(
        contract, request.locale, html_renderer=render_idm_source_locked_html,
        single_choice_builder=idm_source_grounded_single_choice, diagram_builder=idm_source_step_diagram,
        faq_builder=idm_source_faq)
    return IdmUnitDeps(
        build_instance_model=build_staged_instance_response_model,
        bind_instance_payload=bind_staged_instance_payload,
        validate_unit=lambda unit, expected: validate_staged_unit_content(unit, expected, strict_payload=True),
        build_repair_model=build_staged_multi_repair_model,
        decode_repair=decode_staged_multi_repair,
        merge_repair=lambda unit, delta, targets, coverage: merge_staged_component_payload_delta(
            unit, delta, targets, coverage_targets=coverage),
        source_locked_unit=build_orchestration_v2_source_locked_unit(
            contract, request.locale, components=source_locked_components),
        source_locked_components=source_locked_components,
        purity_context=build_learner_content_purity_context(
            {"source_fact_ids": list(contract.unit_source_fact_ids), "supporting_evidence_fact_ids": supporting},
            manifest, _orchestration_v2_unit_source_rows(request)),
        evidence_review_required=isinstance(bundle, dict) and bundle.get("status") == "review_required",
        judge_mode=settings.idm_judge_mode,
        recoverable_errors=(LessonAuthorProposalValidationError, WorkflowFailure, ValueError, TypeError, KeyError),
    )


async def _lesson_author_orchestration_v2_unit(
    request: RagLessonAuthorUnitV2Request,
) -> dict[str, Any]:
    """Generate exactly one immutable inventory unit with the proven Stage-2 writer."""

    unit_started = perf_counter()
    contract = request.unit_contract
    manifest = unit_contract_manifest_v2(contract, locale=request.locale)
    evidence_bundle = manifest.get("source_evidence_bundle")
    evidence_review_required = (
        isinstance(evidence_bundle, dict)
        and evidence_bundle.get("status") == "review_required"
    )
    evidence_quality_state = "review_required" if evidence_review_required else "validated"
    source_rows = _orchestration_v2_unit_source_rows(request)
    context = "\n".join(f"[{fact.fact_key}] {fact.fact_text}" for fact in contract.source_facts)
    source_outline = f"{contract.chapter_title} > {contract.lesson_title} > {contract.unit_title}"
    source_coverage = format_source_coverage_manifest(manifest)
    adapted_architecture = unit_contract_v5_architecture_v2(contract)
    return await _lesson_author_orchestration_v2_unit_legacy(
        request, contract, manifest, evidence_quality_state, source_rows, context, source_outline,
        source_coverage, adapted_architecture, unit_started,
    )


def _orchestration_v2_unit_source_rows(request: RagLessonAuthorUnitV2Request) -> list[dict[str, Any]]:
    contract = request.unit_contract
    document_names = {document.document_id: document.name for document in request.source_documents}
    # Adapt the immutable V2 source ledger to the complete legacy formatter
    # row contract. Omitting any of these display/retrieval fields used to turn
    # a valid source fact into an ASGI 500 before the provider was called.
    return [{
        "document_id": fact.document_id,
        "document_name": document_names[fact.document_id],
        "source_page": fact.source_page,
        "source_section": fact.locator.get("source_section") or fact.source_ref,
        "chunk_no": fact.source_chunk,
        "content": fact.fact_text,
        "score": 1.0,
        "vector_score": 1.0,
        "keyword_score": 0.0,
        "method": "orchestration_v2_source_ledger",
        "methods": ["orchestration_v2_source_ledger"],
        "metadata": {
            "source_ref": fact.source_ref,
            "fact_key": fact.fact_key,
            "heading_path": fact.locator.get("heading_path"),
        },
    } for fact in contract.source_facts]


async def _lesson_author_orchestration_v2_unit_legacy(
    request: RagLessonAuthorUnitV2Request,
    contract: UnitGenerationContractV2,
    manifest: dict[str, Any],
    evidence_quality_state: str,
    source_rows: list[dict[str, Any]],
    context: str,
    source_outline: str,
    source_coverage: str,
    adapted_architecture: dict[str, Any],
    unit_started: float,
) -> dict[str, Any]:
    adapted = RagLessonAuthorRequest.model_validate({
        **request.model_dump(exclude={"contract_version", "unit_contract", "remaining_workflow_budget_ms"}),
        "outline_context": contract.unit_path,
        "target_scope_instruction": "Generate only the approved immutable orchestration V2 unit.",
        "output_schema_hint": "Return the server-supplied staged unit schema.",
        "operation": "create", "target_type": "chapter", "generation_mode": "staged",
        "blueprint_architecture": adapted_architecture,
    })
    source_locked_fallback_unit = build_orchestration_v2_source_locked_unit(contract, request.locale)
    attempt_trace: list[dict[str, Any]] = []
    semantic_review_mode = settings.semantic_review_mode
    semantic_review_model = settings.semantic_review_model or request.model
    semantic_config_hash = semantic_review_config_hash(
        mode=semantic_review_mode,
        model=semantic_review_model,
        timeout_ms=settings.semantic_review_timeout_ms,
        max_output_tokens=settings.semantic_review_max_output_tokens,
        provider_attempt_cap=settings.semantic_review_provider_attempt_cap,
    )
    unit_supporting_evidence_fact_ids = list(dict.fromkeys(
        fact_id
        for plan in contract.component_plan
        for fact_id in plan.supporting_evidence_fact_ids
    ))
    fact_text_by_id = {fact.fact_key: fact.fact_text for fact in contract.source_facts}
    diagram_relationships_by_plan_id = {
        plan.component_plan_id: [
            [left, right, *([relation] if relation else [])]
            for left, right, relation in source_relationship_pairs(
                fact_text_by_id[fact_id]
                for fact_id in dict.fromkeys([
                    *plan.source_fact_ids,
                    *plan.supporting_evidence_fact_ids,
                ])
                if fact_id in fact_text_by_id
            )
        ]
        for plan in contract.component_plan
        if plan.type == "la_diagram"
    }
    expected = {
        "unit_title": contract.unit_title,
        "unit_purpose": contract.unit_purpose,
        "source_fact_ids": list(contract.unit_source_fact_ids),
        "supporting_evidence_fact_ids": unit_supporting_evidence_fact_ids,
        "component_types": [plan.type for plan in contract.component_plan],
        "component_plan": [plan.model_dump(mode="json") for plan in contract.component_plan],
        "diagram_relationships_by_plan_id": diagram_relationships_by_plan_id,
        "learning_objectives": list(contract.lesson_learning_objectives),
        "learning_objective_refs": list(contract.unit_learning_objective_refs),
        "locale": request.locale,
        "learner_content_purity": build_learner_content_purity_context(
            {
                "source_fact_ids": list(contract.unit_source_fact_ids),
                "supporting_evidence_fact_ids": unit_supporting_evidence_fact_ids,
            },
            manifest,
            source_rows,
        ),
        "instructional_output_budget": {
            "policy_version": adapted_architecture["lessons"][0]["units"][0]["instructional_density_policy_version"],
            "source_content_chars": adapted_architecture["lessons"][0]["units"][0]["source_content_chars"],
            "source_estimated_words": adapted_architecture["lessons"][0]["units"][0]["source_estimated_words"],
            "max_visible_chars": adapted_architecture["lessons"][0]["units"][0]["max_generated_visible_chars"],
            "max_words": adapted_architecture["lessons"][0]["units"][0]["max_generated_words"],
        } if adapted_architecture["lessons"][0]["units"][0]["instructional_density_policy_version"] else None,
    }

    def semantic_review_unavailable(unit: dict[str, Any], code: str) -> dict[str, Any]:
        return safe_semantic_review_summary(
            SemanticReviewRunOutcome(
                unit=deepcopy(unit),
                quality_state="review_required",
                review_status="unavailable",
                first_review=None,
                scoped_review=None,
                repair_attempted=False,
                repair_applied=False,
                repair_component_indices=(),
                failure_code=code,
            ),
            config_hash=semantic_config_hash,
        )

    def deterministic_fallback(reason_code: str, *, provider_dispatched: bool) -> dict[str, Any]:
        unit = deepcopy(source_locked_fallback_unit)
        finding = (validate_staged_unit_content(unit, expected, strict_payload=True)
                   if isinstance(unit, dict) else
                   StagedUnitFinding("Deterministic unit fallback is unavailable.",
                                     "UNIT_FALLBACK_UNAVAILABLE", "unit", False))
        if finding is not None:
            logger.error("lesson_author_orchestration_v2 %s", json.dumps({
                "event": "unit_deterministic_fallback_rejected",
                "correlation_id": request.correlation_id,
                "chapter_key": contract.chapter_key,
                "unit_path": contract.unit_path,
                "reason_code": reason_code,
                "validation_finding": finding.diagnostic() if isinstance(finding, StagedUnitFinding) else {
                    "code": "UNIT_FALLBACK_INVALID", "path": "unit", "repairable": False,
                },
            }, sort_keys=True))
            raise _orchestration_v2_http_error(
                "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID",
                "The source-locked unit fallback did not pass server validation.",
                status_code=422,
            )
        # This is a whole-unit deterministic draft, not a provider-validated
        # candidate. It must remain reviewable even when every source fact is
        # text-only and otherwise validated. The V2 persistence fence relies
        # on this exact provenance/quality pair for provider-free replay.
        fallback_quality_state = "review_required"
        semantic_summary = (
            semantic_review_unavailable(unit, "SEMANTIC_REVIEW_NOT_RUN")
            if semantic_review_mode != "off" else None
        )
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_deterministic_fallback_ready",
            "correlation_id": request.correlation_id,
            "source_snapshot_hash": contract.source_snapshot_hash,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "source_fact_count": len(contract.source_facts),
            "component_count": len(contract.component_plan),
            "reason_code": reason_code,
            "provider_dispatched": provider_dispatched,
            "content_origin": "structured_fallback",
            "quality_state": fallback_quality_state,
            **({"semantic_review": semantic_summary} if semantic_summary is not None else {}),
        }, sort_keys=True))
        usage = AiUsage().model_dump()
        # The Node dispatch fence is persisted before this HTTP request. A
        # first-attempt fallback therefore settles the reserved upper bound
        # even when Python rejected the request before reaching Gemini. The
        # durable fallback-only replay is provider-free and fully complete.
        first_attempt = not request.fallback_only
        if len(attempt_trace) < 64:
            attempt_trace.append({
                "sequence": len(attempt_trace) + 1,
                "invocation_kind": "deterministic",
                "invocation_index": 1,
                "provider_attempt": None,
                "phase": "fallback",
                "outcome": "fallback",
                "event_code": "deterministic_fallback",
                "failure_stage": ("provider_usage" if reason_code == "PROVIDER_USAGE_INCOMPLETE"
                                  else "durable_replay" if reason_code == "DURABLE_FINAL_ATTEMPT"
                                  else "unit_generation"),
                "failure_code": reason_code[:100] if re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", reason_code) else "FALLBACK_REQUIRED",
                "failure_path": None,
                "provider_dispatched": provider_dispatched,
                "usage_source": "unknown",
                "observed_usage": {},
                "duration_ms": 0,
                "diagnostics": {},
            })
        return {"contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
                "unit_path": contract.unit_path, "unit": unit, "usage": usage,
                "usage_complete": not first_attempt,
                "usage_source": "reserved_upper_bound" if first_attempt else "deterministic_fallback",
                "content_origin": "structured_fallback", "quality_state": fallback_quality_state,
                "attempt_trace": attempt_trace,
                **({"semantic_review": semantic_summary} if semantic_summary is not None else {})}

    if request.fallback_only:
        return deterministic_fallback("DURABLE_FINAL_ATTEMPT", provider_dispatched=False)
    provider_dispatched = False

    def mark_provider_dispatched() -> None:
        nonlocal provider_dispatched
        provider_dispatched = True

    semantic_usage = AiUsage()
    semantic_usage_complete = True
    semantic_provider_dispatched = False

    def semantic_provider_attempts_used() -> int:
        return sum(
            event.get("event_code") == "provider_http_attempt_started"
            for event in attempt_trace
            if isinstance(event, dict)
        )

    async def call_semantic_provider(
        *,
        prompt: str,
        response_schema: type[BaseModel],
        invocation_kind: Literal["evaluator", "repair"],
        invocation_index: int,
        max_output_tokens: int,
    ) -> str:
        """Dispatch one frozen semantic invocation inside the shared request budget."""

        nonlocal semantic_usage, semantic_usage_complete, semantic_provider_dispatched
        # `generate_content` may retry one transient 5xx response. Reserve both
        # transport attempts before dispatch so nested retries cannot exceed the
        # writer/evaluator/repair request ceiling.
        if (
            semantic_provider_attempts_used() + PROVIDER_TRANSIENT_MAX_ATTEMPTS
            > settings.semantic_review_provider_attempt_cap
        ):
            raise RuntimeError("SEMANTIC_PROVIDER_ATTEMPT_BUDGET_EXHAUSTED")
        elapsed_ms = max(0, int((perf_counter() - unit_started) * 1000))
        remaining_ms = request.remaining_workflow_budget_ms - elapsed_ms
        timeout_ms = min(settings.semantic_review_timeout_ms, remaining_ms)
        if timeout_ms < 1_000:
            raise RuntimeError("SEMANTIC_WORKFLOW_BUDGET_EXHAUSTED")
        invocation_dispatched = False
        invocation_usage_complete = False

        def capture(metadata: dict[str, Any]) -> None:
            nonlocal invocation_dispatched, invocation_usage_complete, semantic_provider_dispatched
            event = str(metadata.get("event") or "")
            provider_attempt = metadata.get("provider_attempt")
            if event == "provider_http_attempt_started":
                invocation_dispatched = True
                semantic_provider_dispatched = True
            counts = [
                metadata.get(key)
                for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")
            ]
            if (
                metadata.get("provider_http_status") == 200
                and metadata.get("usage_source") == "provider"
                and all(type(value) is int and value >= 0 for value in counts)
                and counts[2] >= counts[0] + counts[1]
            ):
                invocation_usage_complete = True
            outcome = {
                "provider_http_attempt_started": "started",
                "provider_http_attempt_succeeded": "succeeded",
                "provider_response_received": "succeeded",
                "provider_transient_retry": "retrying",
                "provider_timeout": "failed",
                "provider_quota_exhausted": "failed",
                "provider_unavailable": "failed",
                "provider_request_failed": "failed",
            }.get(event)
            if (
                outcome is None
                or type(provider_attempt) is not int
                or not 1 <= provider_attempt <= 8
                or len(attempt_trace) >= 64
            ):
                return
            raw_code = str(metadata.get("internal_failure_code") or event).upper()
            failure_code = re.sub(r"[^A-Z0-9_]", "_", raw_code)[:100] or "SEMANTIC_PROVIDER_FAILED"
            observed_usage = {
                key: metadata[key]
                for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")
                if type(metadata.get(key)) is int and metadata[key] >= 0
            }
            diagnostics = {
                key: metadata[key]
                for key in ("provider_http_status", "provider_status", "provider_error_category",
                            "provider_schema_constraint", "provider_finish_reason")
                if metadata.get(key) is not None
            }
            attempt_trace.append({
                "sequence": len(attempt_trace) + 1,
                "invocation_kind": invocation_kind,
                "invocation_index": invocation_index,
                "provider_attempt": provider_attempt,
                "phase": "semantic_review" if invocation_kind == "evaluator" else "semantic_repair",
                "outcome": outcome,
                "event_code": event[:96],
                "failure_stage": "provider_transport" if outcome == "failed" else None,
                "failure_code": failure_code if outcome == "failed" else None,
                "failure_path": None,
                "provider_dispatched": True,
                "usage_source": (
                    "provider_reported" if metadata.get("usage_source") == "provider"
                    else "estimated" if metadata.get("usage_source") == "local_estimate"
                    else "unknown"
                ),
                "observed_usage": observed_usage,
                "duration_ms": max(0, int(metadata.get("duration_ms") or 0)),
                "diagnostics": diagnostics,
            })

        try:
            text, invocation_usage = await provider.generate_content(
                request.api_key,
                semantic_review_model,
                prompt,
                max_output_tokens=max_output_tokens,
                json_mode=True,
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(include_thoughts=False),
                request_timeout_ms=timeout_ms,
                on_provider_telemetry=capture,
            )
            semantic_usage = combine_usage(semantic_usage, invocation_usage)
            return text
        finally:
            if invocation_dispatched and not invocation_usage_complete:
                semantic_usage_complete = False

    # The durable caller has its own request deadline. End generation before
    # that boundary so timeout handling can still build and return the
    # deterministic fallback over the same HTTP request.
    generation_budget_ms = max(
        1,
        request.remaining_workflow_budget_ms
        - ORCHESTRATION_V2_UNIT_FALLBACK_RESPONSE_HEADROOM_MS,
    )
    try:
        value, usage = await asyncio.wait_for(staged_writer.generate_staged_lesson_author_proposal(
            adapted, context, source_outline, source_coverage, source_rows=source_rows,
            source_coverage_manifest=manifest, checkpoint_unit_index=0,
            remaining_workflow_budget_ms=generation_budget_ms,
            checkpoint_component_fallback_unit=source_locked_fallback_unit,
            on_provider_dispatch=mark_provider_dispatched,
            attempt_trace_sink=attempt_trace,
        ), generation_budget_ms / 1000)
        unit = value.get("unit") if isinstance(value, dict) else None
        if not isinstance(unit, dict) or unit.get("title") != contract.unit_title:
            raise LessonAuthorProposalValidationError("ORCHESTRATION_V2_UNIT_IDENTITY_CHANGED")
        provider_usage_complete = value.pop("provider_usage_complete", False) is True
        if not provider_usage_complete:
            return deterministic_fallback("PROVIDER_USAGE_INCOMPLETE", provider_dispatched=provider_dispatched)
        fallback_component_indices = value.pop("fallback_component_indices", [])
        partial_fallback = bool(fallback_component_indices)
        content_origin = "structured_fallback" if partial_fallback else "provider_validated"
        semantic_summary: dict[str, Any] | None = None
        semantic_outcome: SemanticReviewRunOutcome | None = None
        if semantic_review_mode != "off":
            reviewer_invocation_index = 0
            allowed_review_fact_ids = {
                fact.fact_key for fact in contract.source_facts
            }

            async def semantic_reviewer(
                candidate: dict[str, Any],
                component_indices: tuple[int, ...],
            ) -> SemanticReviewResponse:
                nonlocal reviewer_invocation_index
                reviewer_invocation_index += 1
                prompt = build_semantic_review_prompt(
                    unit=candidate,
                    expected=expected,
                    manifest=manifest,
                    component_indices=component_indices,
                    locale=request.locale,
                )
                text = await call_semantic_provider(
                    prompt=prompt,
                    response_schema=SemanticReviewResponse,
                    invocation_kind="evaluator",
                    invocation_index=reviewer_invocation_index,
                    max_output_tokens=settings.semantic_review_max_output_tokens,
                )
                parsed = SemanticReviewResponse.model_validate_json(text)
                return validate_semantic_review_response(
                    parsed,
                    component_count=len(candidate.get("components", [])),
                    allowed_source_fact_ids=allowed_review_fact_ids,
                    requested_component_indices=component_indices,
                )

            async def semantic_repairer(
                candidate: dict[str, Any],
                component_indices: tuple[int, ...],
                findings: tuple[Any, ...],
            ) -> dict[str, Any]:
                prompt = build_semantic_repair_prompt(
                    unit=candidate,
                    expected=expected,
                    manifest=manifest,
                    component_indices=component_indices,
                    findings=findings,
                    locale=request.locale,
                )
                response_model = build_staged_multi_repair_model(
                    candidate, list(component_indices), [],
                )
                text = await call_semantic_provider(
                    prompt=prompt,
                    response_schema=response_model,
                    invocation_kind="repair",
                    invocation_index=1,
                    max_output_tokens=min(
                        max(settings.semantic_review_max_output_tokens, 8_192),
                        request.max_output_tokens,
                        16_384,
                    ),
                )
                diagnostics: dict[str, Any] = {}
                delta = decode_staged_multi_repair(
                    text, candidate, list(component_indices), [], diagnostics,
                )
                return merge_staged_component_payload_delta(
                    candidate, delta, list(component_indices), coverage_targets=[],
                )

            def deterministic_semantic_recheck(candidate: dict[str, Any]) -> str | None:
                finding = validate_staged_unit_content(candidate, expected, strict_payload=True)
                return str(finding) if finding is not None else None

            semantic_outcome = await run_bounded_semantic_review(
                unit,
                reviewer=semantic_reviewer,
                repairer=semantic_repairer,
                deterministic_validate=deterministic_semantic_recheck,
                allow_repair=semantic_review_mode == "repair",
            )
            unit = semantic_outcome.unit
            semantic_summary = safe_semantic_review_summary(
                semantic_outcome,
                config_hash=semantic_config_hash,
            )
        # Component fallback passes the same instructional validator as the
        # provider-authored components, so provenance does not reduce its
        # publication readiness.
        quality_state = evidence_quality_state
        if (
            semantic_review_mode == "repair"
            and semantic_outcome is not None
            and semantic_outcome.quality_state == "review_required"
        ):
            quality_state = "review_required"
        combined_usage = combine_usage(usage, semantic_usage)
        combined_usage_complete = provider_usage_complete and semantic_usage_complete
        logger.info("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_ready", "correlation_id": request.correlation_id,
            "source_snapshot_hash": contract.source_snapshot_hash, "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path, "source_fact_count": len(contract.source_facts),
            "component_count": len(unit.get("components", [])),
            "provider_usage_complete": provider_usage_complete,
            "fallback_component_indices": fallback_component_indices,
            "content_origin": content_origin,
            "quality_state": quality_state,
            "semantic_review_mode": semantic_review_mode,
            "semantic_provider_dispatched": semantic_provider_dispatched,
            **({"semantic_review": semantic_summary} if semantic_summary is not None else {}),
        }, sort_keys=True))
        return {"contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
                "unit_path": contract.unit_path, "unit": unit, "usage": combined_usage.model_dump(),
                "usage_complete": combined_usage_complete,
                "usage_source": "provider" if combined_usage_complete else "reserved_upper_bound",
                "content_origin": content_origin, "quality_state": quality_state,
                "attempt_trace": attempt_trace,
                **({"semantic_review": semantic_summary} if semantic_summary is not None else {})}
    except asyncio.TimeoutError:
        return deterministic_fallback("AI_STAGED_LESSON_WORKFLOW_TIMEOUT", provider_dispatched=provider_dispatched)
    except WorkflowFailure as error:
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_generation_failed",
            "correlation_id": request.correlation_id,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "failure_stage": error.failure_stage,
            "failure_code": error.internal_code or error.code,
            "provider_dispatched": provider_dispatched,
        }, sort_keys=True))
        return deterministic_fallback(error.internal_code or error.code, provider_dispatched=provider_dispatched)
    except (LessonAuthorProposalValidationError, ValidationError, ValueError) as error:
        validation_errors = ([{"loc": list(item.get("loc", ())), "type": item.get("type")}
                              for item in error.errors()[:16]] if isinstance(error, ValidationError) else [])
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "unit_generation_invalid",
            "correlation_id": request.correlation_id,
            "chapter_key": contract.chapter_key,
            "unit_path": contract.unit_path,
            "exception_type": type(error).__name__,
            "validation_errors": validation_errors,
            "provider_dispatched": provider_dispatched,
        }, sort_keys=True))
        return deterministic_fallback("ORCHESTRATION_V2_UNIT_INVALID", provider_dispatched=provider_dispatched)


@app.post("/v1/lesson-author/blueprint", dependencies=[Depends(require_internal_token)])
async def lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: asyncpg.Pool = Depends(get_db),
) -> dict[str, Any]:
    return await run_with_deadline(
        "/v1/lesson-author/blueprint",
        settings.lesson_author_deadline_ms,
        _lesson_author_blueprint(request, pool),
    )


async def _lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    workflow_started = perf_counter()

    def emit_blueprint_diagnostic(metadata: dict[str, Any]) -> None:
        """Emit only structured operational metadata for the correlated run."""

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "course_architecture",
            "workflow_version": "langgraph-v1",
            "architecture_contract_version": 5,
            "repair_provider_call_cap": V5_MAX_PROVIDER_REPAIR_CALLS,
            "semantic_repair_contract_version": SEMANTIC_REPAIR_CONTRACT_VERSION,
            "repair_attempt_cap_per_layer": V5_MAX_REPAIR_ATTEMPTS_PER_LAYER,
            "correlation_id": request.correlation_id,
            "tenant_id": request.tenant_id,
            "kb_id": request.kb_id,
            "conversation_id": request.conversation_id,
            "course_id": request.course_id,
            "source_document_ids": [document.document_id for document in request.source_documents],
            **metadata,
        }
        logger.info("lesson_author_blueprint_diagnostic %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    emit_blueprint_diagnostic({
        "stage": "lesson_author_request",
        "event": "received",
        "repair_pass_number": 0,
    })
    rows, retrieval_usage, structure_context = await retrieval_search.retrieve_chunks(pool, request)
    emit_blueprint_diagnostic({
        "stage": "rag_retrieval",
        "event": "completed",
        "repair_pass_number": 0,
        "retrieved_chunk_count": len(rows),
        "source_scope_truncated": bool(structure_context.get("course_blueprint_source_scope_truncated")),
    })
    context, sources = format_sources(rows, max_context_chars=retrieval_limits(request)["max_context_chars"])
    retrieval = build_retrieval_diagnostics(request, rows, sources, structure_context)
    source_coverage_manifest = structure_context.get("source_coverage_manifest")
    source_coverage = format_course_architecture_coverage_contract(source_coverage_manifest)
    runtime: dict[str, Any] = {}

    def validate_source_scope() -> list[WorkflowIssue]:
        if structure_context.get("course_blueprint_source_scope_truncated"):
            return [_workflow_issue(
                "SOURCE_SCOPE_INCOMPLETE",
                "The complete source scope is unavailable for global course architecture.",
            )]
        policy = structure_context.get("source_chapter_policy")
        if policy is not None:
            represented_documents = {str(node.get("document_id")) for node in structure_context.get("source_structure_nodes", [])}
            missing_documents = {doc.document_id for doc in request.source_documents} - represented_documents
            emit_blueprint_diagnostic({
                "stage": "source_chapter_policy", "event": "resolved",
                "mode": policy["mode"], "complete": policy["complete"],
                "expected_chapter_count": len(policy["chapters"]),
                "reason_codes": policy["reason_codes"],
                "missing_document_count": len(missing_documents),
                "source_structure_node_count": structure_context.get("structure_node_count", 0),
                "source_outline_display_truncated": bool(structure_context.get("source_outline_display_truncated")),
                "source_outline_display_chars": len(structure_context.get("outline") or ""),
            })
            if not policy["complete"] or missing_documents:
                raise WorkflowFailure(
                    "SOURCE_STRUCTURE_REVIEW_REQUIRED", "Source chapter authority is incomplete or ambiguous.",
                    internal_code="SOURCE_STRUCTURE_REVIEW_REQUIRED", failure_stage="source_chapter_policy",
                    diagnostics={"reason_codes": policy["reason_codes"], "missing_document_count": len(missing_documents)},
                )
        return []

    def build_global_source_map() -> dict[str, Any]:
        source_map = build_source_map(
            structure_context.get("source_structure_nodes", []),
            source_coverage_manifest,
            locale=request.locale,
        )
        architect_context = build_course_architect_context(
            source_map,
            source_coverage_manifest,
            max_chars=max(1, settings.source_map_architect_context_max_chars),
        )
        coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
        if not coverage.get("section_scope_complete"):
            raise WorkflowFailure(
                "SOURCE_MAP_SCOPE_INCOMPLETE",
                "The global Source Map does not represent the complete selected source scope.",
            )
        if not coverage.get("fact_scope_complete"):
            incomplete_reason = str(coverage.get("incomplete_reason") or "SOURCE_MAP_SCOPE_INCOMPLETE")
            raise WorkflowFailure(
                incomplete_reason if incomplete_reason == "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED" else "SOURCE_MAP_SCOPE_INCOMPLETE",
                "The global Source Map does not represent every canonical source fact.",
            )
        if not architect_context.get("context_complete"):
            raise WorkflowFailure(
                str(architect_context.get("error_code") or "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE"),
                "The global Source Map hierarchy cannot fit within the configured architect context budget.",
            )
        v5_source_context = create_v5_immutable_source_context(
            source_map,
            source_coverage_manifest,
        )
        runtime["v5_source_context"] = v5_source_context
        runtime["source_map_context"] = architect_context["context"]
        # Never give a graph node the immutable object itself. The graph gets
        # a disposable copy while validation/allocation always return to the
        # request-scoped source authority below.
        runtime["source_map"] = v5_source_context.source_map_copy()
        runtime["source_map_diagnostics"] = architect_context["diagnostics"]
        emit_blueprint_diagnostic({
            "stage": "source_map_build",
            "event": "completed",
            "repair_pass_number": 0,
            "canonical_fact_count": int(coverage.get("total_fact_count") or 0),
            "represented_fact_count": int(coverage.get("represented_fact_count") or 0),
            "source_section_count": int(coverage.get("section_count") or 0),
            "source_concept_count": int(coverage.get("concept_count") or 0),
            "fact_scope_complete": bool(coverage.get("fact_scope_complete")),
            "source_map_complete": bool(coverage.get("fact_scope_complete")) and bool(coverage.get("section_scope_complete")),
            "architect_context_mode": architect_context["diagnostics"].get("architect_context_mode"),
            "architect_context_chars": architect_context["diagnostics"].get("architect_context_size"),
            "architect_hierarchy_encoding": architect_context["diagnostics"].get("architect_hierarchy_encoding"),
            "architect_uncompressed_context_chars": architect_context["diagnostics"].get("architect_uncompressed_context_chars"),
            "architect_detail_fact_count": architect_context["diagnostics"].get("architect_detail_fact_count"),
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            "source_context_fingerprint": v5_source_context.fingerprint,
        })
        return v5_source_context.source_map_copy()

    async def architect_course(source_map: dict[str, Any]) -> WorkflowGenerationResult:
        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="course_architect_generation",
        )
        prompt = build_lesson_author_blueprint_prompt(
            request,
            context,
            structure_context.get("outline", ""),
            source_coverage,
            runtime["source_map_context"],
            structure_source=structure_context.get("structure_source"),
            authoritative_source_nodes=structure_context.get("authoritative_source_nodes"),
            source_chapter_policy=structure_context.get("source_chapter_policy"),
        )
        emit_blueprint_diagnostic({
            "stage": "course_architect_policy", "event": "composed",
            "policy_mode": "self_built_rag_v5",
            "policy_version": "v5-blueprint-policy-1" if request.system_prompt.startswith("SERVER POLICY VERSION: v5-blueprint-policy-1.") else "legacy_internal_caller",
            "policy_chars": len(request.system_prompt.strip()),
            "policy_sha256": hashlib.sha256(request.system_prompt.strip().encode("utf-8")).hexdigest(),
            "output_locale": request.locale,
        })
        try:
            blueprint, generation_usage = await blueprint_service.generate_validated_lesson_author_blueprint(
                request,
                prompt,
                set(structure_context.get("known_source_refs", set())),
                structure_source=structure_context.get("structure_source"),
                authoritative_source_nodes=structure_context.get("authoritative_source_nodes"),
                source_chapter_policy=structure_context.get("source_chapter_policy"),
                source_structure_nodes=structure_context.get("source_structure_nodes"),
                allow_server_fact_allocation=True,
                # V5 semantic findings are intentionally returned to the
                # graph as local patch targets. V3/V4 callers retain the
                # existing whole-candidate compatibility behavior.
                v5_immutable_source_context=v5_source_context,
                defer_semantic_scope_validation=True,
                emit_diagnostic=emit_blueprint_diagnostic,
            )
            if blueprint.get("architecture_contract_version") not in {3, 4, 5}:
                blueprint = ensure_blueprint_source_granularity(
                    blueprint,
                    source_coverage_manifest,
                    structure_context.get("source_structure_nodes"),
                    request.locale,
                )
            return WorkflowGenerationResult(blueprint, generation_usage.model_dump())
        except CourseArchitectSemanticScopeError as error:
            raise WorkflowFailure(
                "ARCHITECTURE_SCOPE_INCOMPLETE",
                "Course Architect output did not satisfy canonical semantic ownership.",
                issues=error.issues,
                internal_code="ARCH_SEMANTIC_SCOPE_ATTEMPTS_EXHAUSTED",
                failure_stage="course_architect_semantic_scope_validation",
                diagnostics={"architect_attempt_count": error.attempt_count},
            ) from error
        except LessonAuthorBlueprintGenerationError as error:
            raise WorkflowFailure(
                "PROVIDER_ERROR",
                error.reason or error.code,
                internal_code="ARCH_PROVIDER_OUTPUT_INVALID",
                failure_stage="course_architect_output_validation",
                diagnostics={"architect_attempt_count": request.max_attempts},
            ) from error
        except LessonAuthorBlueprintValidationError as error:
            raise WorkflowFailure(
                error.code,
                str(error),
                internal_code="ARCH_FACT_ALLOCATION_FAILED",
                failure_stage="canonical_fact_allocation",
            ) from error
        except HTTPException as error:
            raise WorkflowFailure(
                "PROVIDER_ERROR",
                "Course Architect provider call failed.",
                internal_code="AI_PROVIDER_TIMEOUT" if isinstance(error.detail, dict) and error.detail.get("code") == "AI_PROVIDER_TIMEOUT" else "ARCH_PROVIDER_ERROR",
                failure_stage="course_architect_provider",
                diagnostics={
                    "provider_http_status": None if isinstance(error.detail, dict) and error.detail.get("code") == "AI_PROVIDER_TIMEOUT" else error.status_code,
                    "application_http_status": error.status_code,
                },
            ) from error

    def validate_blueprint_layers(
        candidate: dict[str, Any],
        _workflow_source_map: dict[str, Any],
    ) -> WorkflowValidationResult:
        """Run V5 validation in repairable layers against immutable source state.

        Schema, semantic ownership and instructional coherence are evaluated
        before server allocation. This prevents a four-scope local omission
        from being inflated into hundreds of fact findings or a whole-course
        repair. Only a semantically complete candidate can be allocated.
        """

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="blueprint_validation",
        )
        def emit_layer(event: str, issues: list[WorkflowIssue] | None = None) -> None:
            emit_blueprint_diagnostic({
                "stage": "v5_blueprint_layer_validation",
                "event": event,
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
                "architecture_contract_version": 5,
                "canonical_fact_count": v5_source_context.canonical_fact_count,
                "evidence_scope_count": v5_source_context.evidence_scope_count,
                "candidate_status": event,
                "validation_codes": sorted({
                    str(issue.get("code") or "")
                    for issue in (issues or [])
                    if issue.get("code")
                }),
            })

        def mark_repair_layer(
            result: WorkflowValidationResult,
            repair_layer: Literal[
                "SCHEMA",
                "EVIDENCE_SEMANTIC",
                "PRE_ALLOCATION_COHERENCE",
                "POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
            ],
        ) -> WorkflowValidationResult:
            """Annotate deterministic V5 findings for the graph scheduler only."""

            for issue in result.issues:
                if issue.get("severity") == "error":
                    issue["repair_layer"] = repair_layer
            return result

        if candidate.get("architecture_contract_version") != 5:
            result = WorkflowValidationResult([{
                "code": "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
                "severity": "error",
                "message": "The V5 Course Architect candidate did not retain the required V5 contract.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("contract_failed", result.issues)
            return result
        try:
            normalized = validate_lesson_author_blueprint(
                candidate,
                require_source_fact_ownership=False,
                forbid_provider_fact_ownership=True,
            )
            validate_lesson_author_source_refs(
                normalized,
                set(structure_context.get("known_source_refs", set())),
            )
            if structure_context.get("source_chapter_policy") is not None:
                normalized = bind_source_chapters(normalized, structure_context["source_chapter_policy"], v5_source_context.source_map_copy())
        except LessonAuthorBlueprintValidationError as error:
            issue = _workflow_issue_from_blueprint_validation_error(error)
            # A bounded lesson array violation (including the provider's four
            # units) is local. Root/chapter failures are globally unusable and
            # must not become an oversized repair request.
            issue["repairable"] = bool(re.fullmatch(
                r"chapters\[\d+\]\.lessons\[\d+\](?:\.units(?:\[\d+\])?)?",
                error.path or "",
            ))
            result = WorkflowValidationResult([issue])
            emit_layer("schema_failed", result.issues)
            return mark_repair_layer(result, "SCHEMA")
        except LessonAuthorProposalValidationError:
            result = WorkflowValidationResult([{
                "code": "INVALID_SOURCE_REF",
                "severity": "error",
                "message": "Blueprint source references are outside the approved source scope.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("source_reference_failed", result.issues)
            return result

        # The normalizer creates a provider-safe semantic candidate; any old
        # allocation must be absent at this stage and is re-created below.
        candidate.clear()
        candidate.update(normalized)

        # Provider output never grants capability; only the internal Node request does.
        if request.component_capabilities is not None:
            candidate["component_capabilities"] = request.component_capabilities.model_dump()

        semantic = validate_course_architecture_evidence_scope(
            candidate,
            v5_source_context.source_map_copy(),
        )
        if semantic.errors:
            emit_layer("semantic_scope_failed", semantic.issues)
            return mark_repair_layer(semantic, "EVIDENCE_SEMANTIC")
        emit_layer("evidence_semantic_passed", [])

        # Freeze direct assessment references from the same canonical scopes
        # already accepted above, before candidate fingerprints/mutation guards.
        # Explicit provider refs and all ownership remain protected unchanged.
        projected, provenance_diagnostics, provenance_issues = materialize_assessment_source_refs(
            candidate, v5_source_context.source_map_copy(),
        )
        emit_blueprint_diagnostic({
            "stage": "v5_assessment_provenance", "event": "failed" if provenance_issues else "resolved",
            "materialized_block_count": 0 if provenance_issues else len(provenance_diagnostics),
            "blocks": provenance_diagnostics[:24],
            "omitted_block_count": max(0, len(provenance_diagnostics) - 24),
            "reason_codes": sorted({issue.safe_reason for issue in provenance_issues}),
        })
        if provenance_issues:
            result = WorkflowValidationResult([issue.workflow_issue() for issue in provenance_issues])
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")
        projected_semantic = validate_course_architecture_evidence_scope(projected, v5_source_context.source_map_copy())
        if projected_semantic.errors:
            emit_layer("semantic_scope_failed", projected_semantic.issues)
            return mark_repair_layer(projected_semantic, "EVIDENCE_SEMANTIC")
        candidate.clear()
        candidate.update(projected)

        # Assessment intent is compiled before coherence, one local objective
        # at a time. The Architect declares *that* assessment is required; the
        # compiler is the server authority that creates factless supporting
        # knowledge-check blocks from existing teaching provenance.  It never
        # assigns canonical facts or expands source scope.
        assessment_plan = compile_v5_assessment_plan(candidate)
        unit_diagnostics = assessment_plan.safe_unit_diagnostics(candidate)
        emit_blueprint_diagnostic({
            "stage": "v5_assessment_plan_compiler",
            "event": assessment_plan.status.lower(),
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "architecture_contract_version": 5,
            **assessment_plan.metrics,
            "units": unit_diagnostics[:24],
            "omitted_unit_count": max(0, len(unit_diagnostics) - 24),
            "objective_diagnostics": assessment_plan.objective_diagnostics,
            "objective_diagnostic_count": assessment_plan.objective_diagnostic_count,
            "omitted_objective_diagnostic_count": max(
                0, assessment_plan.objective_diagnostic_count - len(assessment_plan.objective_diagnostics),
            ),
        })
        if assessment_plan.status == "TERMINAL_GAP":
            result = WorkflowValidationResult(
                [issue.workflow_issue() for issue in assessment_plan.issues],
                assessment_plan.metrics,
            )
            emit_layer("assessment_plan_terminal_gap", result.issues)
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")
        # The compiler is copy-on-write. Applying this safe candidate here
        # does not persist anything; final review persistence remains after
        # all evidence/coherence/allocation validation passes.
        candidate.clear()
        candidate.update(assessment_plan.blueprint)
        if assessment_plan.status == "NEEDS_SEMANTIC_RESOLUTION":
            result = WorkflowValidationResult(
                [issue.workflow_issue() for issue in assessment_plan.issues],
                assessment_plan.metrics,
            )
            emit_layer("assessment_plan_semantic_resolution_required", result.issues)
            return mark_repair_layer(result, "PRE_ALLOCATION_COHERENCE")

        treatment_candidate, treatment_diagnostics = compile_evidence_treatments(
            candidate, v5_source_context.source_map_copy(), v5_source_context.manifest_copy(),
        )
        # Both the existing evidence and coherence validators still decide
        # acceptance. Treatment discovery cannot grant ownership or bypass them.
        treatment_semantic = validate_course_architecture_evidence_scope(
            treatment_candidate, v5_source_context.source_map_copy(),
        )
        if treatment_semantic.errors:
            emit_layer("semantic_scope_failed", treatment_semantic.issues)
            return mark_repair_layer(treatment_semantic, "EVIDENCE_SEMANTIC")
        candidate.clear()
        candidate.update(treatment_candidate)
        emit_blueprint_diagnostic({
            "stage": "evidence_treatment_discovery", "event": "completed",
            "version": EVIDENCE_TREATMENT_VERSION,
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "units": treatment_diagnostics[:24],
            "omitted_unit_count": max(0, len(treatment_diagnostics) - 24),
        })

        coherence = validate_v5_instructional_coherence(candidate)
        if coherence.errors:
            emit_layer("pre_allocation_coherence_failed", coherence.issues)
            return mark_repair_layer(coherence, "PRE_ALLOCATION_COHERENCE")
        emit_layer("instructional_coherence_passed", [])

        allocated = allocate_source_map_architecture_facts(
            candidate,
            v5_source_context.source_map_copy(),
            v5_source_context.manifest_copy(),
        )
        allocation = allocated.get("source_fact_allocation")
        if (
            not isinstance(allocation, dict)
            or int(allocation.get("required_count") or -1) != v5_source_context.canonical_fact_count
        ):
            result = WorkflowValidationResult([{
                "code": "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
                "severity": "error",
                "message": "V5 allocation did not retain the immutable canonical fact manifest.",
                "path": "course",
                "repairable": False,
            }])
            emit_layer("allocation_context_failed", result.issues)
            return result
        if allocation.get("complete"):
            allocated = allocate_blueprint_source_fact_ids(
                allocated,
                v5_source_context.manifest_copy(),
                structure_context.get("source_structure_nodes"),
            )
        candidate.clear()
        candidate.update(allocated)
        emit_blueprint_diagnostic({
            "stage": "canonical_fact_allocation",
            "event": "completed",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "canonical_fact_count": v5_source_context.canonical_fact_count,
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            **source_fact_allocation_diagnostics(candidate),
        })
        result = validate_course_architecture_workflow(
            candidate,
            v5_source_context.source_map_copy(),
            v5_source_context.manifest_copy(),
            set(structure_context.get("known_source_refs", set())),
        )
        depth_only = bool(result.errors) and all(
            str(issue.get("code") or "") == "INSTRUCTIONAL_DEPTH_INSUFFICIENT"
            for issue in result.errors
        )
        emit_layer(
            "final_validation_passed" if not result.errors
            else "post_allocation_instructional_depth_failed" if depth_only
            else "final_validation_failed",
            result.issues,
        )
        if depth_only:
            return mark_repair_layer(result, "POST_ALLOCATION_INSTRUCTIONAL_DEPTH")
        return result

    async def repair_course(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> WorkflowGenerationResult:
        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_target_snapshot",
        )
        assert_v5_scoped_repair_target_bound(targets, v5_source_context)
        targets = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        deterministic_targets = [
            target for target in targets
            if target.get("deterministic_semantic_delta") is True
        ]
        provider_targets = [
            target for target in targets
            if target.get("deterministic_semantic_delta") is not True
        ]
        working_blueprint = blueprint
        deterministic_payload = _v5_deterministic_assessment_alignment_payload(deterministic_targets)
        if deterministic_payload is not None:
            working_blueprint = apply_course_architecture_repair_patches(
                working_blueprint,
                deterministic_targets,
                deterministic_payload,
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(deterministic_targets),
                "provider_called": False,
                "semantic_operation": "repair_assessment_alignment",
                "deterministic_reason": "EXACT_BASE_ELIGIBLE_TEACHING_ANCHOR",
            })
        if not provider_targets:
            return WorkflowGenerationResult(working_blueprint)
        blueprint = working_blueprint
        targets = provider_targets
        repair_layers = {
            str(target.get("repair_layer") or "")
            for target in targets
            if isinstance(target, dict)
        }
        repair_layer = next(iter(repair_layers)) if len(repair_layers) == 1 else "UNCLASSIFIED"
        semantic_delta_operations = _v5_semantic_delta_operations_for_targets(targets)
        layer_attempt_number = max(
            (int(target.get("layer_attempt_number") or 0) for target in targets),
            default=0,
        )
        total_repair_provider_calls = max(
            (int(target.get("total_repair_provider_calls") or 0) for target in targets),
            default=0,
        )
        if semantic_delta_operations:
            operation_groups: list[tuple[str | None, list[RepairTarget]]] = []
            for operation in sorted(semantic_delta_operations):
                group = [
                    target for target in targets
                    if list(target.get("semantic_operations") or []) == [operation]
                ]
                if group:
                    operation_groups.append((operation, group))
            if sum(len(group) for _operation, group in operation_groups) != len(targets):
                raise WorkflowFailure(
                    "ARCHITECTURE_REPAIR_INVALID",
                    "A V5 coherence target did not map to one exact semantic-delta operation.",
                    internal_code="ARCH_REPAIR_SEMANTIC_OPERATION_CONFLICT",
                    failure_stage="architecture_repair_target_snapshot",
                    diagnostics={"repair_target_count": len(targets)},
                )
        else:
            operation_groups = [(None, targets)]
        repair_pass_number = int(runtime.get("repair_provider_pass") or 0) + 1
        runtime["repair_provider_pass"] = repair_pass_number
        repair_output_tokens = min(request.max_output_tokens, 16_384)
        scoped_contexts: list[tuple[str | None, list[RepairTarget], str, str]] = []
        for operation, operation_targets in operation_groups:
            scoped_repair_context = build_v5_scoped_repair_source_context(
                blueprint=blueprint,
                targets=operation_targets,
                source_map=v5_source_context.source_map_copy(),
            )
            scoped_contexts.append((
                operation,
                operation_targets,
                scoped_repair_context,
                build_course_architecture_repair_prompt(
                    blueprint=blueprint,
                    targets=operation_targets,
                    source_map_context=scoped_repair_context,
                    locale=request.locale,
                ),
            ))
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_target_snapshot",
            "event": "completed",
            "repair_pass_number": repair_pass_number,
            "repair_target_count": len(targets),
            "repair_scope": sorted({target["scope"] for target in targets}),
            **({"semantic_delta_operations": sorted(semantic_delta_operations)} if semantic_delta_operations else {}),
            "repair_layer": repair_layer,
            "layer_attempt_number": layer_attempt_number,
            "total_repair_provider_calls": total_repair_provider_calls,
            "serialized_target_chars": sum(len(context) for _operation, _targets, context, _prompt in scoped_contexts),
            "requested_output_tokens": repair_output_tokens,
            "canonical_fact_count": v5_source_context.canonical_fact_count,
            "evidence_scope_count": v5_source_context.evidence_scope_count,
            "provider_operation_count": len(scoped_contexts),
        })
        repair_usages: list[AiUsage] = []
        all_patches: list[dict[str, Any]] = []
        executed_provider_call_count = 0
        # Targets carry the actual count BEFORE dispatch, not a reservation.
        prior_repair_provider_calls = total_repair_provider_calls
        try:
            for operation_index, (operation, operation_targets, _context, prompt) in enumerate(scoped_contexts, start=1):
                repair_provider_telemetry: dict[str, Any] = {}
                selection_contract = None
                if operation == "select_assessment_teaching_alignment":
                    selection_contract = build_assessment_selection_contract(
                        operation_targets, max_context_chars=MAX_WORKFLOW_REPAIR_TARGET_CHARS,
                        blueprint=blueprint, source_map=v5_source_context.source_map_copy(),
                        manifest=v5_source_context.manifest_copy(),
                    )
                    prompt = selection_contract.prompt(request.locale)
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_selection_contract", "event": "prepared",
                        "repair_pass_number": repair_pass_number, "repair_layer": repair_layer,
                        "semantic_operation": operation, "selection_contract_version": ASSESSMENT_SELECTION_CONTRACT_VERSION,
                        "selection_slot_count": len(selection_contract.bindings), "repair_target_count": len(operation_targets),
                        **selection_contract.diagnostics,
                        "total_repair_provider_calls": total_repair_provider_calls,
                    })

                def record_repair_provider_telemetry(telemetry: dict[str, Any]) -> None:
                    repair_provider_telemetry.update(telemetry)
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_provider",
                        "event": "completed",
                        "repair_pass_number": repair_pass_number,
                        "repair_layer": repair_layer,
                        "layer_attempt_number": layer_attempt_number,
                        "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                        "provider_operation_index": operation_index,
                        "provider_operation_count": len(scoped_contexts),
                        **({"semantic_operation": operation} if operation else {}),
                        **telemetry,
                    })

                try:
                    # Count the provider boundary before awaiting it: an HTTP
                    # failure, parser failure or mutation-guard rejection must
                    # still be visible as an executed call in terminal state.
                    executed_provider_call_count = operation_index
                    total_repair_provider_calls = prior_repair_provider_calls + executed_provider_call_count
                    text, usage = await provider.generate_content(
                        request.api_key,
                        request.model,
                        prompt,
                        max_output_tokens=repair_output_tokens,
                        json_mode=True,
                        response_schema=(
                            selection_contract.schema if selection_contract else build_v5_semantic_delta_repair_response_schema({operation})
                            if operation else build_course_architecture_repair_response_schema(
                                set().union(*(
                                    set(target.get("allowed_fields", []))
                                    for target in operation_targets
                                ))
                            )
                        ),
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                        request_timeout_ms=settings.blueprint_provider_request_timeout_ms,
                        on_provider_telemetry=record_repair_provider_telemetry,
                    )
                except HTTPException as error:
                    emit_blueprint_diagnostic({
                        "stage": "architecture_repair_provider",
                        "event": "failed",
                        "repair_pass_number": repair_pass_number,
                        "repair_layer": repair_layer,
                        "layer_attempt_number": layer_attempt_number,
                        "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                        "provider_operation_index": operation_index,
                        "provider_operation_count": len(scoped_contexts),
                        **({"semantic_operation": operation} if operation else {}),
                        "provider_http_status": error.status_code,
                        "provider_finish_reason": None,
                        "provider_finish_reason_available": False,
                        "usage_source": "unavailable",
                    })
                    raise WorkflowFailure(
                        "ARCHITECTURE_REPAIR_INVALID",
                        "Course architecture repair provider call failed.",
                        internal_code="ARCH_REPAIR_PROVIDER_ERROR",
                        failure_stage="architecture_repair_provider",
                        diagnostics={"provider_http_status": error.status_code, "semantic_operation": operation},
                    ) from error
                finish_reason = str(repair_provider_telemetry.get("provider_finish_reason") or "").upper()
                if finish_reason.endswith("MAX_TOKENS"):
                    raise WorkflowFailure(
                        "ARCHITECTURE_REPAIR_INVALID",
                        "Course architecture repair was truncated by the provider.",
                        internal_code="ARCH_REPAIR_PROVIDER_TRUNCATED",
                        failure_stage="architecture_repair_provider",
                        diagnostics={
                            "provider_finish_reason": repair_provider_telemetry.get("provider_finish_reason"),
                            "response_chars": repair_provider_telemetry.get("response_chars"),
                            "response_bytes": repair_provider_telemetry.get("response_bytes"),
                            "semantic_operation": operation,
                        },
                    )
                payload = selection_contract.decode_text(text) if selection_contract else parse_course_architecture_repair_payload(text)
                emit_blueprint_diagnostic({
                    "stage": "architecture_repair_json_parser",
                    "event": "passed",
                    "repair_pass_number": repair_pass_number,
                    "repair_layer": repair_layer,
                    "layer_attempt_number": layer_attempt_number,
                    "total_repair_provider_calls": prior_repair_provider_calls + operation_index,
                    "provider_operation_index": operation_index,
                    "provider_operation_count": len(scoped_contexts),
                    **({"semantic_operation": operation} if operation else {}),
                    "response_chars": len(text),
                })
                all_patches.extend(
                    patch for patch in payload.get("patches", []) if isinstance(patch, dict)
                )
                repair_usages.append(usage)

            action_intent_metadata = _v5_safe_action_intent_repair_metadata(
                blueprint,
                targets,
                all_patches,
            )
            repaired = apply_course_architecture_repair_patches(blueprint, targets, {"patches": all_patches})
            assert_v5_immutable_source_context(
                v5_source_context,
                stage="architecture_repair_patch_apply",
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_patch_apply",
                "event": "completed",
                "repair_pass_number": repair_pass_number,
                "repair_layer": repair_layer,
                "layer_attempt_number": layer_attempt_number,
                "total_repair_provider_calls": total_repair_provider_calls,
                "patch_count": len(all_patches),
                "accepted_patch_count": len(targets),
                **action_intent_metadata,
            })
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_candidate_validation",
                "event": "deferred_to_layered_validation",
                "repair_pass_number": repair_pass_number,
                "repair_layer": repair_layer,
                "layer_attempt_number": layer_attempt_number,
                "total_repair_provider_calls": total_repair_provider_calls,
                "repair_target_count": len(targets),
                "repair_validation_result": "PENDING_SCHEMA_SEMANTIC_COHERENCE_ALLOCATION",
            })
        except WorkflowFailure as error:
            # Preserve only operational call accounting. The exception and
            # logs intentionally never retain a provider patch or source text.
            error.diagnostics["executed_provider_call_count"] = executed_provider_call_count
            if error.failure_stage == "architecture_repair_semantic_delta_guard":
                emit_blueprint_diagnostic({
                    "stage": error.failure_stage,
                    "event": "rejected",
                    "repair_pass_number": repair_pass_number,
                    "repair_layer": repair_layer,
                    "layer_attempt_number": layer_attempt_number,
                    "total_repair_provider_calls": prior_repair_provider_calls + executed_provider_call_count,
                    "provider_calls_executed": executed_provider_call_count,
                    "external_failure_code": error.code,
                    "internal_failure_code": error.internal_code,
                    **error.diagnostics,
                })
            raise
        except LessonAuthorBlueprintValidationError as error:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Course architecture repair could not be applied to the canonical source scope.",
                internal_code="ARCH_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="architecture_repair_patch_apply",
            ) from error
        except HTTPException as error:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "Provider returned an invalid scoped architecture repair.",
                internal_code="ARCH_REPAIR_JSON_INVALID",
                failure_stage="architecture_repair_json_parser",
                diagnostics={"provider_http_status": error.status_code},
            ) from error
        return WorkflowGenerationResult(
            repaired,
            combine_usage(*repair_usages).model_dump(),
            provider_call_count=len(scoped_contexts),
        )

    async def deterministic_repair_course(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> WorkflowGenerationResult | None:
        """Apply server-proven factless-unit removals before any provider call."""

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_target_snapshot",
        )
        prepared_targets = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        deterministic_assessment = _v5_deterministic_assessment_alignment_payload(prepared_targets)
        if deterministic_assessment is not None and all(
            target.get("deterministic_semantic_delta") is True
            for target in prepared_targets
        ):
            repaired = apply_course_architecture_repair_patches(
                blueprint,
                prepared_targets,
                deterministic_assessment,
            )
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(prepared_targets),
                "provider_called": False,
                "semantic_operation": "repair_assessment_alignment",
                "deterministic_reason": "EXACT_BASE_ELIGIBLE_TEACHING_ANCHOR",
            })
            return WorkflowGenerationResult(repaired)
        evidence_alignment = _v5_deterministic_evidence_alignment_candidate(
            blueprint,
            prepared_targets,
        )
        if evidence_alignment is not None:
            emit_blueprint_diagnostic({
                "stage": "architecture_repair_deterministic",
                "event": "completed",
                "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
                "repair_target_count": len(prepared_targets),
                "provider_called": False,
                "semantic_operation": "align_concepts_to_evidence",
                "canonical_fact_count": v5_source_context.canonical_fact_count,
                "evidence_scope_count": v5_source_context.evidence_scope_count,
            })
            return WorkflowGenerationResult(evidence_alignment)

        outcome = _deterministic_factless_unit_removal_candidate(
            blueprint,
            prepared_targets,
            source_map=v5_source_context.source_map_copy(),
            source_coverage_manifest=source_coverage_manifest,
            known_source_refs=set(structure_context.get("known_source_refs", set())),
            source_structure_nodes=structure_context.get("source_structure_nodes"),
        )
        if outcome is None:
            return None
        repaired, repair_metadata = outcome
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_deterministic",
            "event": "completed",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0) + 1,
            "repair_target_count": len(targets),
            "provider_called": False,
            "baseline": source_fact_allocation_diagnostics(blueprint),
            "candidate": source_fact_allocation_diagnostics(repaired),
            **repair_metadata,
        })
        return WorkflowGenerationResult(repaired)

    def prepare_course_repair_targets(
        blueprint: dict[str, Any],
        targets: list[RepairTarget],
        _source_map: dict[str, Any],
    ) -> list[RepairTarget]:
        """Preflight V5 target authority before scheduler provider accounting."""

        v5_source_context = assert_v5_immutable_source_context(
            runtime.get("v5_source_context"),
            stage="architecture_repair_assessment_preflight",
        )
        assert_v5_scoped_repair_target_bound(targets, v5_source_context)
        prepared = _v5_prepare_semantic_repair_targets(
            blueprint,
            targets,
            v5_source_context.source_map_copy(),
        )
        assessment_targets = [
            target for target in prepared
            if target.get("semantic_operations") == ["repair_assessment_alignment"]
        ]
        action_intent_targets = [
            target for target in prepared
            if target.get("semantic_operations") == ["set_block_intent"]
        ]
        emit_blueprint_diagnostic({
            "stage": "architecture_repair_assessment_preflight",
            "event": "completed",
            "repair_target_count": len(prepared),
            "assessment_alignment_target_count": len(assessment_targets),
            "deterministic_assessment_alignment_count": sum(
                target.get("deterministic_semantic_delta") is True
                for target in assessment_targets
            ),
            "provider_assessment_choice_count": sum(
                target.get("deterministic_semantic_delta") is not True
                for target in assessment_targets
            ),
            "assessment_intent_repair_target_count": sum(
                any(candidate.get("allowed_intents") for candidate in target.get("assessment_alignment_candidates", []))
                for target in assessment_targets
            ),
            "action_intent_repair_target_count": len(action_intent_targets),
            "action_intent_allowed_choice_count": sum(
                len({
                    intent
                    for intents in (target.get("allowed_intents_by_block_id") or {}).values()
                    if isinstance(intents, list)
                    for intent in intents
                    if isinstance(intent, str) and intent.strip()
                })
                for target in action_intent_targets
            ),
            "action_intent_assessment_dependency_objective_count": sum(
                max(0, int(target.get("assessment_dependency_objective_count") or 0))
                for target in action_intent_targets
            ),
        })
        return prepared

    try:
        blueprint, workflow = await run_course_architecture_workflow(
            CourseArchitectureWorkflowCallbacks(
                validate_source_scope=validate_source_scope,
                build_source_map=build_global_source_map,
                architect=architect_course,
                validate_blueprint=validate_blueprint_layers,
                repair_blueprint=repair_course,
                prepare_repair_targets=prepare_course_repair_targets,
                deterministic_repair=deterministic_repair_course,
                emit_diagnostic=emit_blueprint_diagnostic,
                run_blocking=service_runtime.concurrency.run_cpu,
            ),
            request_context={
                "correlation_id": request.correlation_id,
                "tenant_id": request.tenant_id,
                "kb_id": request.kb_id,
                "conversation_id": request.conversation_id,
                "course_id": request.course_id,
                "source_document_count": len(request.source_documents),
            },
            max_repair_attempts=min(2, max(0, settings.course_workflow_max_repair_attempts)),
        )
        coverage_metrics = validate_lesson_author_source_coverage(
            {"chapters": blueprint.get("chapters", [])},
            source_coverage_manifest,
        )
        if request.component_capabilities is not None:
            blueprint["component_capabilities"] = request.component_capabilities.model_dump()
        if structure_context.get("source_chapter_policy") is not None:
            try:
                blueprint = bind_source_chapters(blueprint, structure_context["source_chapter_policy"], runtime.get("source_map"))
            except LessonAuthorBlueprintValidationError as error:
                raise WorkflowFailure(
                    error.code, "Final Blueprint violates source chapter authority.",
                    internal_code=error.code, failure_stage="source_chapter_policy_final",
                    diagnostics=error.safe_diagnostic(),
                ) from error
        source_map = runtime.get("source_map")
        if not isinstance(source_map, dict):
            # The graph owns the map in state; recompute deterministically only
            # for its response contract, never from top-K retrieval.
            source_map = await service_runtime.concurrency.run_cpu(build_global_source_map)
        source_map_coverage = source_map.get("coverage") if isinstance(source_map.get("coverage"), dict) else {}
        retrieval.update({
            "source_map_version": source_map.get("version"),
            "source_map_section_count": source_map_coverage.get("section_count", 0),
            "source_map_concept_count": len(source_map.get("concepts", [])),
            "source_map_fact_count": source_map_coverage.get("source_fact_count", 0),
            **{
                key: value
                for key, value in (runtime.get("source_map_diagnostics") or {}).items()
                if key in {
                    "source_total_facts", "source_total_sections", "source_total_concepts",
                    "source_map_complete", "architect_context_mode", "architect_context_size",
                    "architect_detail_fact_count",
                }
            },
        })
        retrieval.update({
            "source_coverage_required_count": coverage_metrics["required_count"],
            "source_coverage_covered_count": coverage_metrics["covered_count"],
            "source_coverage_status": coverage_metrics["status"],
        })
        retrieval = update_blueprint_source_coverage(retrieval, blueprint, structure_context)
        try:
            blueprint, media_metrics = enrich_lesson_author_blueprint_media_review(
                blueprint,
                source_coverage_manifest,
                request.locale,
            )
            emit_blueprint_diagnostic({
                "stage": "media_recommendation_evaluation",
                "event": "completed",
                "provider_called": False,
                "media_review_version": MEDIA_REVIEW_VERSION,
                **media_metrics,
            })
        except Exception as error:
            # Media is additive review metadata. A failure must not erase a
            # source-valid Blueprint or imply that no media is needed.
            emit_blueprint_diagnostic({
                "stage": "media_recommendation_evaluation",
                "event": "failed",
                "provider_called": False,
                "error_type": type(error).__name__,
                "media_review_available": False,
            })
    except WorkflowFailure as error:
        emit_blueprint_diagnostic({
            "stage": "endpoint_response",
            "event": "failed",
            "failure_stage": error.failure_stage or "blueprint_validation",
            "internal_failure_code": error.internal_code,
            "external_failure_code": error.code,
            "total_repair_provider_calls": int(error.diagnostics.get("repair_provider_calls", error.diagnostics.get("total_repair_provider_calls", 0))),
            "repair_pass_number": int((error.diagnostics or {}).get("repair_count") or runtime.get("repair_provider_pass") or 0),
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        })
        raise HTTPException(
            status_code=422 if error.code in {
                "SOURCE_SCOPE_INCOMPLETE", "SOURCE_MAP_SCOPE_INCOMPLETE", "SOURCE_EVIDENCE_INSUFFICIENT",
                "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED", "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE",
                "SOURCE_STRUCTURE_REVIEW_REQUIRED",
            } else 502,
            detail={
                "code": error.code,
                "message": (
                    ("Source chapter structure requires review before generation." if request.locale == "en"
                     else "Cần kiểm tra cấu trúc chương của nguồn trước khi tạo Bản thiết kế.")
                    if error.code == "SOURCE_STRUCTURE_REVIEW_REQUIRED"
                    else lesson_author_blueprint_failure_message(request.locale)
                ),
                "failure_stage": error.failure_stage or "blueprint_validation",
                "internal_failure_code": error.internal_code,
                "correlation_id": request.correlation_id,
                "total_repair_provider_calls": int(error.diagnostics.get("repair_provider_calls", error.diagnostics.get("total_repair_provider_calls", 0))),
                "workflow_issues": error.issues[:20],
            },
        ) from error
    except LessonAuthorBlueprintValidationError as error:
        emit_blueprint_diagnostic({
            "stage": "endpoint_response",
            "event": "failed",
            "failure_stage": "blueprint_source_coverage_validation",
            "internal_failure_code": "BLUEPRINT_SOURCE_COVERAGE_INVALID",
            "external_failure_code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INVALID",
            "repair_pass_number": int(runtime.get("repair_provider_pass") or 0),
            "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
            "validation_code": error.code,
        })
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LESSON_AUTHOR_BLUEPRINT_SOURCE_COVERAGE_INVALID",
                "message": lesson_author_blueprint_failure_message(request.locale),
                "usage": retrieval_usage.model_dump(),
            },
        ) from error
    usage = combine_usage(retrieval_usage, AiUsage(**(workflow.get("usage") or {})))
    emit_blueprint_diagnostic({
        "stage": "endpoint_response",
        "event": "completed",
        "repair_pass_number": int(workflow.get("repair_count") or 0),
        "duration_ms": max(0, round((perf_counter() - workflow_started) * 1000)),
        "workflow_status": workflow.get("status"),
        "workflow_validation_codes": list(workflow.get("validation_codes") or []),
        **source_fact_allocation_diagnostics(blueprint),
    })
    return {
        "blueprint": blueprint,
        "source_map": source_map,
        "usage": usage.model_dump(),
        "sources": sources,
        "retrieval": retrieval,
        "workflow": workflow,
    }
