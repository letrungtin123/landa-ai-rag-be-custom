"""Chapter checkpoint result: unit/chapter validation of a checkpoint request."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from app.lesson_author_checkpoint import assemble_checkpoint_chapter, checkpoint_expected_units
from app.lesson_quality import (
    duplicate_validation_result,
    instructional_plan_validation_result,
    pedagogical_validation_result,
)
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest
from app.services.lesson_author.lesson_generation import validate_lesson_generation_workflow
from app.services.lesson_author.proposal_validation import normalize_lesson_author_proposal_tree
from app.services.lesson_author.source_refs import validate_lesson_author_proposal_source_refs
from app.services.lesson_author.staged import writer as staged_writer
from app.services.lesson_author.staged.skeleton import (
    extract_lesson_author_unit_batches,
    validate_staged_skeleton_source_facts,
)
from app.services.lesson_author.staged.source_locked import build_source_locked_staged_skeleton
from app.services.lesson_author.staged.validation import validate_staged_unit_content
from app.services.provider import combine_usage
from app.services.retrieval.source_coverage import validate_lesson_author_source_coverage
from app.workflows.contracts import WorkflowFailure, safe_workflow_issue_summary, safe_workflow_path


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
