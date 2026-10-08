"""Legacy course-architecture blueprint workflow: retrieval, architect, allocation, validation, repair."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

import asyncpg
from fastapi import HTTPException
from google.genai import types

from app.assessment_planner import compile_v5_assessment_plan, materialize_assessment_source_refs
from app.assessment_selection_contract import VERSION as ASSESSMENT_SELECTION_CONTRACT_VERSION
from app.assessment_selection_contract import build_assessment_selection_contract
from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME
from app.instructional_opportunities import VERSION as EVIDENCE_TREATMENT_VERSION
from app.instructional_opportunities import compile_evidence_treatments
from app.lesson_author_blueprint import (
    SEMANTIC_REPAIR_CONTRACT_VERSION,
    LessonAuthorBlueprintValidationError,
    build_course_architecture_repair_response_schema,
    build_v5_semantic_delta_repair_response_schema,
    validate_lesson_author_blueprint,
)
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorBlueprintRequest
from app.services import provider
from app.services import runtime as service_runtime
from app.services.deadlines import run_with_deadline
from app.services.lesson_author import blueprint as blueprint_service
from app.services.lesson_author.architecture_shape import (
    _workflow_issue,
    _workflow_issue_from_blueprint_validation_error,
)
from app.services.lesson_author.architecture_validation import (
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
from app.services.lesson_author.prompts import build_lesson_author_blueprint_prompt
from app.services.lesson_author.proposal_validation import parse_course_architecture_repair_payload
from app.services.lesson_author.repair.candidate import (
    _deterministic_factless_unit_removal_candidate,
    _v5_deterministic_evidence_alignment_candidate,
)
from app.services.lesson_author.repair.prompt import (
    _v5_semantic_delta_operations_for_targets,
    build_course_architecture_repair_prompt,
    build_v5_scoped_repair_source_context,
)
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.services.lesson_author.repair.targets import (
    _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_semantic_repair_targets,
    _v5_safe_action_intent_repair_metadata,
)
from app.services.lesson_author.source_context import (
    MAX_WORKFLOW_REPAIR_TARGET_CHARS,
    assert_v5_immutable_source_context,
    assert_v5_scoped_repair_target_bound,
    create_v5_immutable_source_context,
)
from app.services.lesson_author.source_refs import update_blueprint_source_coverage
from app.services.provider import combine_usage
from app.services.retrieval import search as retrieval_search
from app.services.retrieval.query import retrieval_limits
from app.services.retrieval.search import build_retrieval_diagnostics, format_sources
from app.services.retrieval.source_coverage import (
    format_course_architecture_coverage_contract,
    validate_lesson_author_source_coverage,
)
from app.source_chapter_policy import bind_source_chapters
from app.source_map import build_course_architect_context, build_source_map
from app.workflows.contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowIssue,
    WorkflowValidationResult,
)
from app.workflows.course_architecture import (
    V5_MAX_PROVIDER_REPAIR_CALLS,
    V5_MAX_REPAIR_ATTEMPTS_PER_LAYER,
    CourseArchitectureWorkflowCallbacks,
    run_course_architecture_workflow,
)

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_blueprint(
    request: RagLessonAuthorBlueprintRequest,
    pool: asyncpg.Pool,
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
