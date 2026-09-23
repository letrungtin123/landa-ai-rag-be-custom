from __future__ import annotations

"""Bounded Course Architect orchestration over the global Phase-3 Source Map."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from time import perf_counter
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowIssue,
    WorkflowValidationResult,
    append_progress,
    node_duration_update,
    progress_event,
    safe_workflow_issue_summary,
    safe_workflow_path,
    sanitized_workflow_diagnostics,
)


RepairLayer = Literal[
    "SCHEMA",
    "EVIDENCE_SEMANTIC",
    "PRE_ALLOCATION_COHERENCE",
    "POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
]
V5_REPAIR_LAYERS: tuple[RepairLayer, ...] = (
    "SCHEMA",
    "EVIDENCE_SEMANTIC",
    "PRE_ALLOCATION_COHERENCE",
    "POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
)
# A layer gets one scoped provider repair. Repeating the same deterministic
# issue/target is a no-progress failure, not an invitation to retry. The global
# three-call cap deliberately remains lower than the four conceptual layers:
# a fourth repair remains fail closed rather than expanding retry behaviour.
V5_MAX_REPAIR_ATTEMPTS_PER_LAYER = 1
V5_MAX_PROVIDER_REPAIR_CALLS = 3


class CourseWorkflowState(TypedDict, total=False):
    request_context: dict[str, Any]
    source_map: dict[str, Any]
    blueprint: dict[str, Any]
    validation_issues: list[WorkflowIssue]
    repair_targets: list[RepairTarget]
    repair_attempts: int
    repair_provider_calls: int
    repair_layer_attempts: dict[str, int]
    repair_layer_statuses: dict[str, str]
    repair_layer_fingerprints: dict[str, str]
    current_repair_layer: RepairLayer
    failure_repair_pass_number: int
    max_repair_attempts: int
    metrics: dict[str, Any]
    progress: list[dict[str, Any]]
    errors: list[WorkflowIssue]
    failure_code: str
    failure_internal_code: str
    failure_stage: str
    status: Literal["running", "ready", "failed"]
    usage: dict[str, Any]
    node_durations_ms: dict[str, int]
    validation_fingerprint: str


@dataclass(frozen=True)
class CourseArchitectureWorkflowCallbacks:
    validate_source_scope: Callable[[], list[WorkflowIssue]]
    build_source_map: Callable[[], dict[str, Any]]
    architect: Callable[[dict[str, Any]], Awaitable[WorkflowGenerationResult]]
    validate_blueprint: Callable[[dict[str, Any], dict[str, Any]], WorkflowValidationResult]
    repair_blueprint: Callable[[dict[str, Any], list[RepairTarget], dict[str, Any]], Awaitable[WorkflowGenerationResult]]
    # Optional server-only preflight which may attach immutable repair
    # authority, select a deterministic delta, or fail before a provider call.
    # It runs before V5 provider-cap accounting so deterministic repairs never
    # reserve or consume a Gemini call.
    prepare_repair_targets: Callable[[dict[str, Any], list[RepairTarget], dict[str, Any]], list[RepairTarget]] | None = None
    # Server-only repair for deterministic, evidence-free targets. Returning
    # None delegates to the existing bounded provider repair path.
    deterministic_repair: Callable[[dict[str, Any], list[RepairTarget], dict[str, Any]], Awaitable[WorkflowGenerationResult | None]] | None = None
    emit_diagnostic: Callable[[dict[str, Any]], None] | None = None


NON_REPAIRABLE_CODES = {
    "SOURCE_SCOPE_INCOMPLETE",
    "SOURCE_MAP_SCOPE_INCOMPLETE",
    "SOURCE_EVIDENCE_INSUFFICIENT",
    "SOURCE_DOCUMENT_UNAVAILABLE",
    "SOURCE_OWNERSHIP_INVALID",
    "SOURCE_FACT_UNAVAILABLE",
    "SOURCE_MAP_PROVENANCE_INVALID",
    "ARCHITECT_CONTEXT_CANNOT_REPRESENT_SOURCE",
    "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED",
    "MISSING_INSTRUCTIONAL_SCOPE",
    "UNKNOWN_CONCEPT_ID",
    "UNKNOWN_SOURCE_REF",
    "CONCEPT_SOURCE_SCOPE_MISMATCH",
    "LESSON_PRIMARY_OWNERSHIP_WITHOUT_PRIMARY_UNIT",
    "ARCHITECTURE_SCOPE_INCOMPLETE",
    "AMBIGUOUS_INSTRUCTIONAL_SCOPE",
    "UNALLOCATED_SOURCE_FACT",
    "AMBIGUOUS_SOURCE_FACT_OWNERSHIP",
    "ARCHITECTURE_FACT_CAPACITY_EXCEEDED",
    "ARCHITECTURE_REPAIR_NO_PROGRESS",
    "SECURITY_VALIDATION_FAILED",
    "TENANT_MISMATCH",
    # A V5 repair is possible only when the immutable Source Map proves one
    # exact local destination. Otherwise the candidate is globally unusable.
    "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
    "EVIDENCE_SCOPE_REPAIR_TARGET_UNRESOLVED",
    "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE",
}

COURSE_WIDE_REPAIR_CODES = {
    "SOURCE_COVERAGE_INCOMPLETE",
    "CONCEPT_COVERAGE_INCOMPLETE",
    "PREREQUISITE_CYCLE",
}

UNIT_SOURCE_FACT_OWNERSHIP_ALLOWED_FIELDS = [
    "primary_concept_ids",
    "learning_blocks",
]

V5_SEMANTIC_DELTA_OPERATIONS_BY_CODE = {
    "EVIDENCE_SCOPE_CONCEPT_MISMATCH": "align_concepts_to_evidence",
    "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH": "set_block_intent",
    "ASSESSMENT_OBJECTIVE_NOT_COVERED": "repair_knowledge_check",
    "ASSESSMENT_EVIDENCE_NOT_GROUNDED": "repair_knowledge_check",
    "ASSESSMENT_BLOCK_REQUIRED": "add_knowledge_check",
    # The assessment-plan compiler has already proved source-safe candidate
    # anchors. Gemini may resolve only the remaining semantic ambiguity; it
    # cannot add provenance, facts, or arbitrary hierarchy nodes.
    "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED": "select_assessment_teaching_alignment",
    "INSTRUCTIONAL_DEPTH_INSUFFICIENT": "add_instructional_support_block",
}


def _path_scope(path: str) -> Literal["course", "chapter", "lesson", "unit"]:
    if ".unit_" in path:
        return "unit"
    if ".lesson_" in path:
        return "lesson"
    if path.startswith("chapter_"):
        return "chapter"
    return "course"


def _allowed_fields(scope: Literal["course", "chapter", "lesson", "unit"]) -> list[str]:
    if scope == "unit":
        # Canonical source facts are allocated by the server after the repair.
        # The provider can repair semantic scope, never fact ownership.
        return ["purpose", "concept_ids", "primary_concept_ids", "source_refs", "learning_objective_refs", "learning_blocks"]
    if scope == "lesson":
        return [
            "title",
            "learning_objectives",
            "primary_concept_ids",
            "supporting_concept_ids",
            "prerequisite_concept_ids",
            "estimated_minutes",
            "assessment_required",
            "assessment_objective_refs",
            "units",
        ]
    if scope == "chapter":
        return ["title", "learning_objectives", "concept_ids", "source_refs", "lessons"]
    return ["course_title", "course_outcomes", "chapters"]


def _is_lesson_units_array_length_issue(issue: WorkflowIssue) -> bool:
    """Return whether the validator proved this is exactly lesson.units 1..3."""

    return (
        str(issue.get("code") or "") == "BLUEPRINT_INVALID_SCHEMA"
        and str(issue.get("constraint") or "") == "ARRAY_LENGTH"
        and bool(re.fullmatch(
            r"chapters\[\d+\]\.lessons\[\d+\]\.units",
            str(issue.get("schema_path") or ""),
        ))
    )


def _allowed_fields_for_issue(
    scope: Literal["course", "chapter", "lesson", "unit"],
    issue: WorkflowIssue,
) -> list[str]:
    code = str(issue.get("code") or "")
    if scope == "lesson" and _is_lesson_units_array_length_issue(issue):
        # This is the exact latest-UAT case: no lesson metadata needs to move
        # merely because the provider emitted a fourth unit.
        return ["units"]
    if scope == "unit" and code == "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER":
        # The provider may only place an authorized evidence scope on a
        # semantic block. Canonical facts remain server-allocated.
        return ["learning_blocks"]
    if scope == "unit" and code == "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED":
        return list(UNIT_SOURCE_FACT_OWNERSHIP_ALLOWED_FIELDS)
    if scope == "unit" and code == "EVIDENCE_SCOPE_CONCEPT_MISMATCH":
        # A source/concept mismatch can be corrected only by the unit's
        # semantic scope; purpose/objective wording is not needed.
        return ["concept_ids", "primary_concept_ids", "source_refs", "learning_blocks"]
    if scope == "unit" and code == "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH":
        # V5 takes the typed semantic-delta path below. This fallback remains
        # only for compatibility callers that do not provide a V5 repair layer.
        return ["purpose", "learning_objective_refs", "learning_blocks"]
    if scope == "unit" and code in {
        "ASSESSMENT_OBJECTIVE_NOT_COVERED",
        "ASSESSMENT_EVIDENCE_NOT_GROUNDED",
    }:
        # The coherence validator locates the exact unit containing the
        # knowledge check. Assessment repair may alter only that unit's
        # semantic blocks; it cannot rewrite lesson metadata, concepts,
        # prerequisites, timing, or unrelated units.
        return ["learning_blocks"]
    if scope == "lesson" and code == "ASSESSMENT_BLOCK_REQUIRED":
        # There is no existing knowledge-check unit to target. Replacing the
        # ordered unit list is the smallest complete structural operation.
        return ["units"]
    if scope == "lesson" and code == "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED":
        return []
    return _allowed_fields(scope)


def _semantic_delta_operation_for_issue(
    scope: Literal["course", "chapter", "lesson", "unit"],
    issue: WorkflowIssue,
    *,
    repair_layer: RepairLayer | None,
) -> str | None:
    """Return the narrow V5 provider operation for one coherence finding."""

    operation = V5_SEMANTIC_DELTA_OPERATIONS_BY_CODE.get(str(issue.get("code") or ""))
    if operation is None:
        return None
    if operation == "align_concepts_to_evidence":
        return operation if repair_layer == "EVIDENCE_SEMANTIC" and scope == "unit" else None
    if operation == "add_instructional_support_block":
        return operation if repair_layer == "POST_ALLOCATION_INSTRUCTIONAL_DEPTH" and scope == "lesson" else None
    if repair_layer != "PRE_ALLOCATION_COHERENCE":
        return None
    if operation in {"set_block_intent", "repair_knowledge_check"} and scope != "unit":
        return None
    if operation == "add_knowledge_check" and scope != "unit":
        return None
    if operation == "select_assessment_teaching_alignment" and scope != "lesson":
        return None
    return operation


def _planned_v5_provider_repair_calls(targets: list[RepairTarget]) -> int:
    """Count exact operation schemas required without weakening their shape."""

    semantic_operations: set[str] = set()
    for target in targets:
        if target.get("deterministic_semantic_delta") is True:
            continue
        operations = target.get("semantic_operations")
        if not isinstance(operations, list) or len(operations) != 1:
            return 1
        operation = operations[0]
        if not isinstance(operation, str) or not operation:
            return 1
        semantic_operations.add(operation)
    return len(semantic_operations)


def _allowed_operations_for_issue(scope: Literal["course", "chapter", "lesson", "unit"], code: str) -> list[str]:
    if scope == "unit" and code == "UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED":
        # A pure reinforcement unit with no unique primary scope cannot own a
        # canonical fact without stealing it. Removing that exact invalid
        # unit is safer than allowing a lesson-wide rewrite.
        return ["replace", "remove_unit"]
    return ["replace"]


def _repair_layer_for_issue(issue: WorkflowIssue) -> RepairLayer | None:
    """Classify V5 repairable findings without inferring from provider text."""

    declared = issue.get("repair_layer")
    if declared in V5_REPAIR_LAYERS:
        return declared
    code = str(issue.get("code") or "")
    if code == "BLUEPRINT_INVALID_SCHEMA":
        return "SCHEMA"
    if code in {"MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER", "EVIDENCE_SCOPE_CONCEPT_MISMATCH"} or code.startswith("EVIDENCE_SCOPE_"):
        return "EVIDENCE_SEMANTIC"
    if code in {
        "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH",
        "OBJECTIVE_TOO_GENERIC",
        "ASSESSMENT_BLOCK_REQUIRED",
        "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED",
        "ASSESSMENT_OBJECTIVE_NOT_COVERED",
        "ASSESSMENT_EVIDENCE_NOT_GROUNDED",
    }:
        return "PRE_ALLOCATION_COHERENCE"
    if code == "INSTRUCTIONAL_DEPTH_INSUFFICIENT":
        return "POST_ALLOCATION_INSTRUCTIONAL_DEPTH"
    return None


def _is_v5_layer_aware_state(state: CourseWorkflowState) -> bool:
    blueprint = state.get("blueprint")
    return isinstance(blueprint, dict) and blueprint.get("architecture_contract_version") == 5


def _v5_repair_layer_for_blocking_issues(issues: list[WorkflowIssue]) -> RepairLayer | None:
    layers = {_repair_layer_for_issue(issue) for issue in issues}
    if len(layers) != 1:
        return None
    layer = next(iter(layers))
    return layer if layer in V5_REPAIR_LAYERS else None


def _v5_layer_statuses(layer: RepairLayer | None, *, blocking: bool) -> dict[str, str]:
    statuses = {candidate: "PENDING" for candidate in V5_REPAIR_LAYERS}
    if layer is None:
        return statuses
    index = V5_REPAIR_LAYERS.index(layer)
    for prior_layer in V5_REPAIR_LAYERS[:index]:
        statuses[prior_layer] = "PASS"
    statuses[layer] = "FAIL" if blocking else "PASS"
    if not blocking:
        for later_layer in V5_REPAIR_LAYERS[index + 1:]:
            statuses[later_layer] = "PASS"
    return statuses


def _safe_repair_diagnostic(issue: WorkflowIssue) -> dict[str, str] | None:
    """Keep model-facing repair findings structural and non-sensitive."""

    diagnostic = {
        "code": str(issue.get("code") or "UNKNOWN"),
        "path": safe_workflow_path(issue.get("path")),
    }
    for key in ("validator", "schema_error_code", "schema_path", "constraint", "expected_type", "actual_type", "evidence_scope_id"):
        value = issue.get(key)
        if isinstance(value, str) and value:
            diagnostic[key] = value
    return diagnostic if len(diagnostic) > 2 else None


def classify_course_repair_targets(
    issues: list[WorkflowIssue],
    *,
    repair_layer: RepairLayer | None = None,
) -> list[RepairTarget]:
    """Map deterministic findings to the smallest model-editable scope."""

    targets: dict[tuple[str, str], RepairTarget] = {}
    for issue in issues:
        if issue.get("severity") != "error":
            continue
        code = str(issue.get("code") or "")
        if code in NON_REPAIRABLE_CODES or issue.get("repairable") is False:
            continue
        paths = [str(issue.get("path") or "course")]
        paths.extend(str(path) for path in issue.get("related_paths", []) if path)
        if code in COURSE_WIDE_REPAIR_CODES:
            paths = ["course"]
        for path in paths:
            # Evidence validation reports block-level and unit-level symptoms
            # for the same mismatch. The server-owned typed delta always
            # operates on their common unit, never on a synthetic block path.
            if code == "EVIDENCE_SCOPE_CONCEPT_MISMATCH":
                unit_match = re.match(r"^(chapter_\d+\.lesson_\d+\.unit_\d+)", path)
                if unit_match is None:
                    continue
                path = unit_match.group(1)
            scope = _path_scope(path)
            key = (scope, path)
            issue_allowed_fields = _allowed_fields_for_issue(scope, issue)
            semantic_operation = _semantic_delta_operation_for_issue(
                scope,
                issue,
                repair_layer=repair_layer,
            )
            target = targets.get(key)
            if target is None:
                target = {
                    "scope": scope,
                    "path": path,
                    "codes": [],
                    # Typed V5 semantic deltas own their own strict operation
                    # contracts. Legacy/V3/V4 paths retain field replacement.
                    "allowed_fields": [] if semantic_operation is not None else issue_allowed_fields,
                    "diagnostics": [],
                    "allowed_operations": _allowed_operations_for_issue(scope, code),
                }
                if semantic_operation is not None:
                    target["semantic_operations"] = [semantic_operation]
                    target["allowed_operations"] = [semantic_operation]
                    target["allowed_block_ids"] = []
                    target["allowed_objective_ids"] = []
                    target["allowed_unit_paths"] = []
                if repair_layer is not None:
                    target["repair_layer"] = repair_layer
                targets[key] = target
            else:
                # Several findings may share one safe target. Union only the
                # fields individually authorized by their finding policies;
                # never fall back to the broad target-type permission list.
                if semantic_operation is None:
                    target["allowed_fields"] = list(dict.fromkeys([
                        *target["allowed_fields"],
                        *issue_allowed_fields,
                    ]))
                else:
                    semantic_operations = target.setdefault("semantic_operations", [])
                    if semantic_operation not in semantic_operations:
                        semantic_operations.append(semantic_operation)
                    target["allowed_operations"] = list(semantic_operations)
            if code not in target["codes"]:
                target["codes"].append(code)
            if semantic_operation is None:
                for operation in _allowed_operations_for_issue(scope, code):
                    if operation not in target["allowed_operations"]:
                        target["allowed_operations"].append(operation)
            diagnostic = _safe_repair_diagnostic(issue)
            if diagnostic is not None and diagnostic not in target["diagnostics"]:
                target["diagnostics"].append(diagnostic)
            if code == "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER":
                scope_id = issue.get("evidence_scope_id")
                if isinstance(scope_id, str) and scope_id:
                    allowed_scope_ids = target.setdefault("allowed_evidence_scope_ids", [])
                    if scope_id not in allowed_scope_ids:
                        allowed_scope_ids.append(scope_id)
            if semantic_operation is not None:
                allowed_block_ids = target.setdefault("allowed_block_ids", [])
                for block_id in issue.get("learning_block_ids", []):
                    if isinstance(block_id, str) and block_id and block_id not in allowed_block_ids:
                        allowed_block_ids.append(block_id)
                allowed_objective_ids = target.setdefault("allowed_objective_ids", [])
                for objective_id in issue.get("objective_ids", []):
                    if isinstance(objective_id, str) and objective_id and objective_id not in allowed_objective_ids:
                        allowed_objective_ids.append(objective_id)
                allowed_unit_paths = target.setdefault("allowed_unit_paths", [])
                for unit_path in issue.get("unit_paths", []):
                    if isinstance(unit_path, str) and unit_path and unit_path not in allowed_unit_paths:
                        allowed_unit_paths.append(unit_path)
    return list(targets.values())


def _blocking_issues(state: CourseWorkflowState) -> list[WorkflowIssue]:
    return [issue for issue in state.get("validation_issues", []) if issue.get("severity") == "error"]


def _is_non_repairable_issue(issue: WorkflowIssue) -> bool:
    return str(issue.get("code") or "") in NON_REPAIRABLE_CODES or issue.get("repairable") is False


def _validation_fingerprint(validation: WorkflowValidationResult) -> str:
    """Hash stable validation metadata; do not include blueprint/source content."""

    blocking = [
        {
            "code": str(issue.get("code") or ""),
            "path": safe_workflow_path(str(issue.get("path") or "course")),
            "related_paths": sorted(
                safe_workflow_path(str(path))
                for path in issue.get("related_paths", [])
                if isinstance(path, str)
            ),
            "constraint": str(issue.get("constraint") or ""),
            "expected_type": str(issue.get("expected_type") or ""),
            "actual_type": str(issue.get("actual_type") or ""),
        }
        for issue in validation.issues
        if issue.get("severity") == "error"
    ]
    payload = json.dumps(
        {"blocking": sorted(blocking, key=lambda item: (item["code"], item["path"], item["related_paths"])),
         "metrics": {key: validation.metrics.get(key) for key in ("source_coverage", "concept_coverage")}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _failure_code(state: CourseWorkflowState) -> str:
    errors = list(state.get("errors", [])) + _blocking_issues(state)
    non_repairable = [issue for issue in errors if _is_non_repairable_issue(issue)]
    if non_repairable:
        return str(non_repairable[0].get("code") or "ARCHITECTURE_VALIDATION_FAILED")
    if _is_v5_layer_aware_state(state):
        layer = state.get("current_repair_layer")
        layer_attempts = state.get("repair_layer_attempts") or {}
        if (
            int(state.get("repair_provider_calls") or 0) >= V5_MAX_PROVIDER_REPAIR_CALLS
            or (
                layer in V5_REPAIR_LAYERS
                and int(layer_attempts.get(layer) or 0) >= V5_MAX_REPAIR_ATTEMPTS_PER_LAYER
            )
        ):
            return "ARCHITECTURE_REPAIR_EXHAUSTED"
        return str(state.get("failure_code") or "ARCHITECTURE_VALIDATION_FAILED")
    if int(state.get("repair_attempts") or 0) >= int(state.get("max_repair_attempts") or 0):
        return "ARCHITECTURE_REPAIR_EXHAUSTED"
    return str(state.get("failure_code") or "ARCHITECTURE_VALIDATION_FAILED")


def _v5_scheduler_telemetry(
    state: CourseWorkflowState,
    *,
    provider_call_delta: int = 0,
) -> dict[str, Any]:
    """Return safe V5 scheduler state for correlated operational logs."""

    if not _is_v5_layer_aware_state(state):
        return {}
    layer = state.get("current_repair_layer")
    if layer not in V5_REPAIR_LAYERS:
        return {
            "repair_layer": "UNCLASSIFIED",
            "layer_attempt_number": 0,
            "total_repair_provider_calls": int(state.get("repair_provider_calls") or 0),
            "pre_allocation_coherence_attempts": 0,
            "post_allocation_instructional_depth_attempts": 0,
            "previous_layer_status": "NOT_APPLICABLE",
            "next_layer_status": "NOT_APPLICABLE",
        }
    attempts = state.get("repair_layer_attempts") or {}
    statuses = state.get("repair_layer_statuses") or {}
    index = V5_REPAIR_LAYERS.index(layer)
    previous_status = "START" if index == 0 else str(statuses.get(V5_REPAIR_LAYERS[index - 1]) or "PENDING")
    next_status = "COMPLETE" if index == len(V5_REPAIR_LAYERS) - 1 else str(statuses.get(V5_REPAIR_LAYERS[index + 1]) or "PENDING")
    return {
        "repair_layer": layer,
        "layer_attempt_number": int(attempts.get(layer) or 0) + provider_call_delta,
        "total_repair_provider_calls": int(state.get("repair_provider_calls") or 0) + provider_call_delta,
        "pre_allocation_coherence_attempts": (
            int(attempts.get("PRE_ALLOCATION_COHERENCE") or 0)
            + (provider_call_delta if layer == "PRE_ALLOCATION_COHERENCE" else 0)
        ),
        "post_allocation_instructional_depth_attempts": (
            int(attempts.get("POST_ALLOCATION_INSTRUCTIONAL_DEPTH") or 0)
            + (provider_call_delta if layer == "POST_ALLOCATION_INSTRUCTIONAL_DEPTH" else 0)
        ),
        "previous_layer_status": previous_status,
        "next_layer_status": next_status,
    }


def build_course_architecture_graph(callbacks: CourseArchitectureWorkflowCallbacks):
    graph = StateGraph(CourseWorkflowState)

    def emit(
        state: CourseWorkflowState,
        stage: str,
        event: str,
        *,
        started_at: float | None = None,
        **metadata: Any,
    ) -> None:
        """Emit metadata only; never attach prompts, source text, or model output."""

        request_context = state.get("request_context") or {}
        payload: dict[str, Any] = {
            "stage": stage,
            "event": event,
            "repair_pass_number": int(state.get("repair_attempts") or 0) + (1 if stage.startswith("architecture_repair") else 0),
            **_v5_scheduler_telemetry(
                state,
                provider_call_delta=1 if stage == "architecture_repair_provider" else 0,
            ),
            **metadata,
        }
        correlation_id = request_context.get("correlation_id")
        if isinstance(correlation_id, str) and correlation_id:
            payload["correlation_id"] = correlation_id
        if started_at is not None:
            payload["duration_ms"] = max(0, round((perf_counter() - started_at) * 1000))
        if callbacks.emit_diagnostic is not None:
            callbacks.emit_diagnostic(payload)

    async def validate_source_scope(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        issues = callbacks.validate_source_scope()
        errors = [issue for issue in issues if issue.get("severity") == "error"]
        emit(
            state,
            "lesson_author_request",
            "source_scope_validated",
            started_at=started,
            validation_finding_count=len(issues),
            blocking_finding_count=len(errors),
            findings=[safe_workflow_issue_summary(issue, repairable=not _is_non_repairable_issue(issue)) for issue in issues[:32]],
        )
        return {
            "validation_issues": issues,
            "errors": errors,
            "progress": append_progress(state, progress_event("ANALYZING_SOURCE", "Analyzing source scope", "course_architecture", current=1, total=6)),
            "node_durations_ms": node_duration_update(state, "validate_source_scope", started),
        }

    async def build_source_map(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        try:
            source_map = callbacks.build_source_map()
        except WorkflowFailure as error:
            issue: WorkflowIssue = {"code": error.code, "severity": "error", "message": error.message}
            emit(
                state,
                "source_map_build",
                "failed",
                started_at=started,
                internal_failure_code=error.internal_code,
                external_failure_code=error.code,
            )
            return {
                "errors": [issue], "failure_code": error.code,
                "failure_internal_code": error.internal_code,
                "failure_stage": error.failure_stage or "source_map_build",
                "node_durations_ms": node_duration_update(state, "build_source_map", started),
            }
        emit(state, "source_map_build", "completed", started_at=started)
        return {
            "source_map": source_map,
            "progress": append_progress(state, progress_event("BUILDING_SOURCE_MAP", "Building source map", "course_architecture", current=2, total=6)),
            "node_durations_ms": node_duration_update(state, "build_source_map", started),
        }

    async def course_architect(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        try:
            generated = await callbacks.architect(state["source_map"])
        except WorkflowFailure as error:
            issues = list(error.issues) or [{"code": error.code, "severity": "error", "message": error.message}]
            emit(
                state,
                "course_architect_generation",
                "failed",
                started_at=started,
                internal_failure_code=error.internal_code,
                external_failure_code=error.code,
                findings=[
                    safe_workflow_issue_summary(
                        issue,
                        repairable=not _is_non_repairable_issue(issue),
                    )
                    for issue in issues[:32]
                ],
            )
            return {
                "errors": issues, "failure_code": error.code,
                "failure_internal_code": error.internal_code,
                "failure_stage": error.failure_stage or "course_architect_generation",
                "node_durations_ms": node_duration_update(state, "course_architect", started),
            }
        emit(state, "course_architect_generation", "completed", started_at=started)
        return {
            "blueprint": generated.value,
            "usage": generated.usage,
            "progress": append_progress(state, progress_event("DESIGNING_COURSE", "Designing course architecture", "course_architecture", current=3, total=6)),
            "node_durations_ms": node_duration_update(state, "course_architect", started),
        }

    async def validate_blueprint(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        if not state.get("blueprint") or not state.get("source_map"):
            issue: WorkflowIssue = {"code": "ARCHITECTURE_VALIDATION_FAILED", "severity": "error", "message": "Blueprint or Source Map is unavailable."}
            validation = WorkflowValidationResult(issues=[issue])
        else:
            validation = callbacks.validate_blueprint(state["blueprint"], state["source_map"])
        blocking = [issue for issue in validation.issues if issue.get("severity") == "error"]
        fingerprint = _validation_fingerprint(validation)
        is_revalidation = int(state.get("repair_attempts") or 0) > 0
        is_v5 = _is_v5_layer_aware_state(state)
        repair_layer = _v5_repair_layer_for_blocking_issues(blocking) if is_v5 and blocking else None
        prior_layer_fingerprints = dict(state.get("repair_layer_fingerprints") or {})
        no_progress = (
            bool(blocking)
            and is_revalidation
            and (
                (
                    is_v5
                    and repair_layer is not None
                    and prior_layer_fingerprints.get(repair_layer) == fingerprint
                )
                or (
                    not is_v5
                    and isinstance(state.get("validation_fingerprint"), str)
                    and state.get("validation_fingerprint") == fingerprint
                )
            )
        )
        if no_progress:
            validation.issues.append({
                "code": "ARCHITECTURE_REPAIR_NO_PROGRESS",
                "severity": "error",
                "message": "Scoped architecture repair did not change deterministic blocking validation state.",
                "path": "course",
            })
            blocking = [issue for issue in validation.issues if issue.get("severity") == "error"]
        repairable_blocking = [issue for issue in blocking if not _is_non_repairable_issue(issue)]
        non_repairable_blocking = [issue for issue in blocking if _is_non_repairable_issue(issue)]
        layer_statuses = (
            {layer: "PASS" for layer in V5_REPAIR_LAYERS}
            if is_v5 and not blocking
            else _v5_layer_statuses(repair_layer, blocking=bool(blocking))
            if is_v5
            else {}
        )
        if is_v5 and repair_layer is not None:
            prior_layer_fingerprints[repair_layer] = fingerprint
        if is_v5 and repair_layer in V5_REPAIR_LAYERS:
            layer_index = V5_REPAIR_LAYERS.index(repair_layer)
            validation_scheduler_metadata = {
                "repair_layer": repair_layer,
                "layer_attempt_number": int((state.get("repair_layer_attempts") or {}).get(repair_layer, 0)),
                "total_repair_provider_calls": int(state.get("repair_provider_calls") or 0),
                "pre_allocation_coherence_attempts": int((state.get("repair_layer_attempts") or {}).get("PRE_ALLOCATION_COHERENCE", 0)),
                "post_allocation_instructional_depth_attempts": int((state.get("repair_layer_attempts") or {}).get("POST_ALLOCATION_INSTRUCTIONAL_DEPTH", 0)),
                "previous_layer_status": "START" if layer_index == 0 else layer_statuses[V5_REPAIR_LAYERS[layer_index - 1]],
                "next_layer_status": "COMPLETE" if layer_index == len(V5_REPAIR_LAYERS) - 1 else layer_statuses[V5_REPAIR_LAYERS[layer_index + 1]],
            }
        elif is_v5:
            validation_scheduler_metadata = {
                "repair_layer": "COMPLETE" if not blocking else "UNCLASSIFIED",
                "layer_attempt_number": 0,
                "total_repair_provider_calls": int(state.get("repair_provider_calls") or 0),
                "pre_allocation_coherence_attempts": int((state.get("repair_layer_attempts") or {}).get("PRE_ALLOCATION_COHERENCE", 0)),
                "post_allocation_instructional_depth_attempts": int((state.get("repair_layer_attempts") or {}).get("POST_ALLOCATION_INSTRUCTIONAL_DEPTH", 0)),
                "previous_layer_status": "PASS" if not blocking else "NOT_APPLICABLE",
                "next_layer_status": "COMPLETE" if not blocking else "NOT_APPLICABLE",
            }
        else:
            validation_scheduler_metadata = {}
        emit(
            state,
            "blueprint_revalidation" if is_revalidation else "blueprint_validation",
            "failed" if blocking else "passed",
            started_at=started,
            validation_finding_count=len(validation.issues),
            blocking_finding_count=len(blocking),
            repairable_blocking_finding_count=len(repairable_blocking),
            non_repairable_blocking_finding_count=len(non_repairable_blocking),
            findings=[safe_workflow_issue_summary(issue, repairable=not _is_non_repairable_issue(issue)) for issue in validation.issues[:32]],
            **validation_scheduler_metadata,
            **({
                "internal_failure_code": "ARCH_REPAIR_NO_PROGRESS" if no_progress else "ARCH_REPAIR_REVALIDATION_FAILED",
                "remaining_blocking_findings": len(blocking),
                "remaining_finding_codes": sorted({str(issue.get("code") or "") for issue in blocking if issue.get("code")}),
                "revalidation_status": "FAIL",
                "repair_progress": "unchanged" if no_progress else "changed",
            } if is_revalidation and blocking else ({"revalidation_status": "PASS"} if is_revalidation else {})),
        )
        return {
            "validation_issues": validation.issues,
            "metrics": validation.metrics,
            "validation_fingerprint": fingerprint,
            **({
                "current_repair_layer": repair_layer,
                "repair_layer_fingerprints": prior_layer_fingerprints,
                "repair_layer_statuses": layer_statuses,
            } if is_v5 else {}),
            **({
                "failure_internal_code": "ARCH_REPAIR_NO_PROGRESS" if no_progress else "ARCH_REPAIR_REVALIDATION_FAILED",
                "failure_stage": "blueprint_revalidation",
            } if is_revalidation and blocking else {}),
            "progress": append_progress(state, progress_event("VALIDATING_BLUEPRINT", "Validating course architecture", "course_architecture", current=4, total=6)),
            "node_durations_ms": node_duration_update(state, "validate_blueprint", started),
        }

    async def classify_repair(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        blocking = _blocking_issues(state)
        is_v5 = _is_v5_layer_aware_state(state)
        repair_layer = _v5_repair_layer_for_blocking_issues(blocking) if is_v5 else None
        errors: list[WorkflowIssue] = []
        preflight_failure: WorkflowFailure | None = None
        if is_v5 and repair_layer is None:
            errors.append({
                "code": "ARCHITECTURE_VALIDATION_FAILED",
                "severity": "error",
                "message": "The V5 validation result did not map to one bounded repair layer.",
                "path": "course",
            })
            targets: list[RepairTarget] = []
        else:
            targets = classify_course_repair_targets(blocking, repair_layer=repair_layer)
        planned_provider_calls = 0
        if (
            not errors
            and is_v5
            and targets
            and callbacks.prepare_repair_targets is not None
        ):
            try:
                targets = callbacks.prepare_repair_targets(
                    state["blueprint"], targets, state["source_map"],
                )
            except WorkflowFailure as error:
                preflight_failure = error
                issue: WorkflowIssue = {
                    "code": error.code,
                    "severity": "error",
                    "message": error.message,
                    "path": "course",
                }
                errors.append(issue)
                emit(
                    state,
                    error.failure_stage or "architecture_repair_preflight",
                    "failed",
                    started_at=started,
                    internal_failure_code=error.internal_code,
                    external_failure_code=error.code,
                    **error.diagnostics,
                )
        if is_v5 and repair_layer is not None:
            layer_attempts = state.get("repair_layer_attempts") or {}
            provider_calls = int(state.get("repair_provider_calls") or 0)
            planned_provider_calls = _planned_v5_provider_repair_calls(targets)
            if int(layer_attempts.get(repair_layer) or 0) >= V5_MAX_REPAIR_ATTEMPTS_PER_LAYER:
                errors.append({
                    "code": "ARCHITECTURE_REPAIR_EXHAUSTED",
                    "severity": "error",
                    "message": "The current V5 validation layer already used its bounded repair attempt.",
                    "path": "course",
                })
            if provider_calls + planned_provider_calls > V5_MAX_PROVIDER_REPAIR_CALLS:
                errors.append({
                    "code": "ARCHITECTURE_REPAIR_EXHAUSTED",
                    "severity": "error",
                    "message": "The V5 workflow reached its bounded total provider repair-call cap.",
                    "path": "course",
                })
            for target in targets:
                target["repair_layer"] = repair_layer
                target["layer_attempt_number"] = int(layer_attempts.get(repair_layer) or 0) + 1
                # Carry the post-batch total into the callback so every
                # provider event and terminal failure can report a monotonic,
                # authoritative counter even when a later patch guard fails.
                target["total_repair_provider_calls"] = provider_calls + planned_provider_calls
        if not targets:
            errors.append({"code": "ARCHITECTURE_VALIDATION_FAILED", "severity": "error", "message": "No safe scoped repair target was available."})
        emit(
            state,
            "repair_target_generation",
            "failed" if errors else "completed",
            started_at=started,
            repair_target_count=len(targets),
            **_v5_scheduler_telemetry(
                state,
                provider_call_delta=planned_provider_calls if is_v5 and repair_layer is not None else 0,
            ),
            **({"planned_provider_repair_calls": planned_provider_calls} if is_v5 and repair_layer is not None else {}),
            repair_targets=[
                {
                    "repair_target_id": f"repair_target_{index}",
                    "scope": target["scope"],
                    "safe_path": safe_workflow_path(target["path"]),
                    "allowed_fields": list(target["allowed_fields"]),
                    "allowed_operations": list(target.get("allowed_operations", ["replace"])),
                    **({"repair_layer": target["repair_layer"]} if target.get("repair_layer") else {}),
                    **({"layer_attempt_number": target["layer_attempt_number"]} if target.get("layer_attempt_number") else {}),
                    **({"total_repair_provider_calls": target["total_repair_provider_calls"]} if target.get("total_repair_provider_calls") else {}),
                    **({"allowed_evidence_scope_ids": list(target.get("allowed_evidence_scope_ids", []))}
                       if target.get("allowed_evidence_scope_ids") else {}),
                    "finding_codes": sorted(set(target["codes"])),
                    "diagnostics": list(target.get("diagnostics", [])),
                }
                for index, target in enumerate(targets, start=1)
            ],
        )
        return {
            "repair_targets": targets,
            "errors": errors,
            **({
                "failure_code": preflight_failure.code,
                "failure_internal_code": preflight_failure.internal_code,
                "failure_stage": preflight_failure.failure_stage or "architecture_repair_preflight",
            } if preflight_failure is not None else {}),
            "node_durations_ms": node_duration_update(state, "classify_repair", started),
        }

    async def scoped_repair(state: CourseWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        used_provider = False
        provider_call_count = 0
        try:
            generated = None
            if callbacks.deterministic_repair is not None:
                generated = await callbacks.deterministic_repair(
                    state["blueprint"], state["repair_targets"], state["source_map"],
                )
            if generated is None:
                emit(state, "architecture_repair_provider", "started")
                used_provider = True
                generated = await callbacks.repair_blueprint(
                    state["blueprint"], state["repair_targets"], state["source_map"],
                )
                provider_call_count = max(1, int(generated.provider_call_count or 1))
            else:
                emit(
                    state,
                    "architecture_repair_deterministic",
                    "completed",
                    started_at=started,
                    repair_target_count=len(state["repair_targets"]),
                )
        except WorkflowFailure as error:
            executed_provider_calls = max(
                0,
                int(error.diagnostics.get("executed_provider_call_count") or 0),
            )
            repair_layer_attempts = dict(state.get("repair_layer_attempts") or {})
            repair_provider_calls = int(state.get("repair_provider_calls") or 0)
            repair_layer = state.get("current_repair_layer")
            if (
                _is_v5_layer_aware_state(state)
                and used_provider
                and executed_provider_calls
                and repair_layer in V5_REPAIR_LAYERS
            ):
                repair_layer_attempts[repair_layer] = int(repair_layer_attempts.get(repair_layer) or 0) + 1
                repair_provider_calls += executed_provider_calls
            issue: WorkflowIssue = {"code": error.code, "severity": "error", "message": error.message}
            emit(
                state,
                error.failure_stage or "architecture_repair",
                "failed",
                started_at=started,
                internal_failure_code=error.internal_code,
                external_failure_code=error.code,
                **({"total_repair_provider_calls": repair_provider_calls,
                    "provider_calls_executed": executed_provider_calls}
                   if _is_v5_layer_aware_state(state) else {}),
                **error.diagnostics,
            )
            return {
                "errors": [issue], "failure_code": error.code,
                "failure_internal_code": error.internal_code,
                "failure_stage": error.failure_stage or "architecture_repair",
                "failure_repair_pass_number": int(state.get("repair_attempts") or 0) + 1,
                **({
                    "repair_provider_calls": repair_provider_calls,
                    "repair_layer_attempts": repair_layer_attempts,
                } if _is_v5_layer_aware_state(state) else {}),
                "node_durations_ms": node_duration_update(state, "scoped_repair", started),
            }
        emit(
            state,
            "architecture_repair_patch_apply",
            "completed",
            started_at=started,
            # Counters are committed in the state update below. Include the
            # provider call which just completed so this correlated terminal
            # repair event is not one call behind the scheduler state.
            **_v5_scheduler_telemetry(
                state,
                provider_call_delta=provider_call_count if used_provider else 0,
            ),
        )
        repair_layer_attempts = dict(state.get("repair_layer_attempts") or {})
        repair_provider_calls = int(state.get("repair_provider_calls") or 0)
        repair_layer = state.get("current_repair_layer")
        if _is_v5_layer_aware_state(state) and used_provider and repair_layer in V5_REPAIR_LAYERS:
            repair_layer_attempts[repair_layer] = int(repair_layer_attempts.get(repair_layer) or 0) + 1
            repair_provider_calls += provider_call_count
        return {
            "blueprint": generated.value,
            "usage": generated.usage,
            "repair_attempts": int(state.get("repair_attempts") or 0) + 1,
            **({
                "repair_layer_attempts": repair_layer_attempts,
                "repair_provider_calls": repair_provider_calls,
            } if _is_v5_layer_aware_state(state) else {}),
            "progress": append_progress(state, progress_event("REPAIRING_BLUEPRINT", "Repairing affected course architecture", "course_architecture", current=5, total=6)),
            "node_durations_ms": node_duration_update(state, "scoped_repair", started),
        }

    async def finalize(state: CourseWorkflowState) -> dict[str, Any]:
        emit(state, "response_finalize", "completed")
        return {"status": "ready", "progress": append_progress(state, progress_event("BLUEPRINT_READY", "Course blueprint is ready for review", "course_architecture", current=6, total=6))}

    async def fail(state: CourseWorkflowState) -> dict[str, Any]:
        code = _failure_code(state)
        internal_code = str(state.get("failure_internal_code") or code)
        failure_stage = str(state.get("failure_stage") or "blueprint_validation")
        emit(
            state,
            "response_finalize",
            "lesson_author_blueprint_failed",
            failure_stage=failure_stage,
            internal_failure_code=internal_code,
            external_failure_code=code,
            repair_pass_number=int(state.get("failure_repair_pass_number") or state.get("repair_attempts") or 0),
        )
        return {
            "status": "failed", "failure_code": code,
            "failure_internal_code": internal_code,
            "failure_stage": failure_stage,
        }

    def after_source_scope(state: CourseWorkflowState) -> str:
        return "fail" if state.get("errors") else "build_source_map"

    def after_source_map(state: CourseWorkflowState) -> str:
        return "fail" if state.get("errors") else "course_architect"

    def after_architect(state: CourseWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_blueprint"

    def after_validation(state: CourseWorkflowState) -> str:
        errors = _blocking_issues(state)
        if not errors:
            return "finalize"
        if any(_is_non_repairable_issue(issue) for issue in errors):
            return "fail"
        if _is_v5_layer_aware_state(state):
            repair_layer = _v5_repair_layer_for_blocking_issues(errors)
            if repair_layer is None:
                return "fail"
            layer_attempts = state.get("repair_layer_attempts") or {}
            if int(layer_attempts.get(repair_layer) or 0) >= V5_MAX_REPAIR_ATTEMPTS_PER_LAYER:
                return "fail"
            if int(state.get("repair_provider_calls") or 0) >= V5_MAX_PROVIDER_REPAIR_CALLS:
                return "fail"
            return "classify_repair"
        if int(state.get("repair_attempts") or 0) >= int(state.get("max_repair_attempts") or 0):
            return "fail"
        return "classify_repair"

    def after_classification(state: CourseWorkflowState) -> str:
        return "fail" if state.get("errors") or not state.get("repair_targets") else "scoped_repair"

    def after_repair(state: CourseWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_blueprint"

    graph.add_node("validate_source_scope", validate_source_scope)
    graph.add_node("build_source_map", build_source_map)
    graph.add_node("course_architect", course_architect)
    graph.add_node("validate_blueprint", validate_blueprint)
    graph.add_node("classify_repair", classify_repair)
    graph.add_node("scoped_repair", scoped_repair)
    graph.add_node("finalize", finalize)
    graph.add_node("fail", fail)
    graph.add_edge(START, "validate_source_scope")
    graph.add_conditional_edges("validate_source_scope", after_source_scope, {"build_source_map": "build_source_map", "fail": "fail"})
    graph.add_conditional_edges("build_source_map", after_source_map, {"course_architect": "course_architect", "fail": "fail"})
    graph.add_conditional_edges("course_architect", after_architect, {"validate_blueprint": "validate_blueprint", "fail": "fail"})
    graph.add_conditional_edges("validate_blueprint", after_validation, {"finalize": "finalize", "classify_repair": "classify_repair", "fail": "fail"})
    graph.add_conditional_edges("classify_repair", after_classification, {"scoped_repair": "scoped_repair", "fail": "fail"})
    graph.add_conditional_edges("scoped_repair", after_repair, {"validate_blueprint": "validate_blueprint", "fail": "fail"})
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile()


async def run_course_architecture_workflow(
    callbacks: CourseArchitectureWorkflowCallbacks,
    *,
    request_context: dict[str, Any],
    max_repair_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run without a checkpointer; this state exists only for the HTTP request."""

    started = perf_counter()
    graph = build_course_architecture_graph(callbacks)
    state = await graph.ainvoke({
        "request_context": request_context,
        "repair_attempts": 0,
        "repair_provider_calls": 0,
        "repair_layer_attempts": {},
        "repair_layer_statuses": {},
        "repair_layer_fingerprints": {},
        "max_repair_attempts": max(0, max_repair_attempts),
        "status": "running",
        "progress": [],
        "errors": [],
        "node_durations_ms": {},
    })
    diagnostics = sanitized_workflow_diagnostics(state, workflow="course_architecture", started_at=started)
    if state.get("status") != "ready":
        raise WorkflowFailure(
            str(state.get("failure_code") or "ARCHITECTURE_VALIDATION_FAILED"),
            "Course architecture workflow did not produce a valid review blueprint.",
            issues=list(state.get("validation_issues") or state.get("errors") or []),
            internal_code=str(state.get("failure_internal_code") or state.get("failure_code") or "ARCHITECTURE_VALIDATION_FAILED"),
            failure_stage=str(state.get("failure_stage") or "blueprint_validation"),
            diagnostics=diagnostics,
        )
    return dict(state["blueprint"]), {**diagnostics, "usage": dict(state.get("usage") or {})}
