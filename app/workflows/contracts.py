from __future__ import annotations

"""Typed, safe contracts shared by request-scoped authoring workflows."""

from dataclasses import dataclass, field
from time import perf_counter
import re
from typing import Any, Literal, TypedDict


WorkflowSeverity = Literal["error", "warning", "info"]
RepairScope = Literal["course", "chapter", "lesson", "unit", "component"]


class WorkflowIssue(TypedDict, total=False):
    code: str
    severity: WorkflowSeverity
    message: str
    path: str
    related_paths: list[str]
    objective_ids: list[str]
    learning_block_ids: list[str]
    component_ids: list[str]
    source_fact_ids: list[str]
    repairable: bool
    # Safe schema diagnostics only. These describe the contract and runtime
    # shape, never customer-authored values or source content.
    constraint: str
    expected_type: str
    actual_type: str
    validator: str
    schema_error_code: str
    # A structural schema path is server-generated validation metadata, never
    # customer/provider content. It lets repair policy distinguish an exact
    # lesson.units cardinality violation from other lesson-level arrays.
    schema_path: str
    # V5 Course Architecture repair scheduling is layer-aware. Only the
    # deterministic validator assigns this field; the provider never does.
    repair_layer: str
    # Canonical Source Map ownership metadata. These are server-generated
    # identifiers and structural counts only; they never carry source text or
    # provider-generated Blueprint content.
    concept_id: str
    concept_ids: list[str]
    section_id: str
    section_ids: list[str]
    expected_primary_owner_count: int
    actual_primary_owner_count: int
    ownership_level: str
    scope_classification: str
    coverage_state: str
    safe_reason: str
    # V5 evidence-scope diagnostics. These identifiers are server-generated
    # provenance keys; they carry no source text or provider content.
    evidence_scope_id: str
    evidence_scope_ids: list[str]
    eligible_repair_paths: list[str]
    # V5 assessment-plan compiler diagnostics are count/ID metadata only.
    # Candidate details are kept request-scoped and are never emitted to logs.
    assessment_candidate_count: int
    assessment_plan_fingerprint: str


class RepairTarget(TypedDict, total=False):
    scope: RepairScope
    path: str
    codes: list[str]
    allowed_fields: list[str]
    diagnostics: list[dict[str, str]]
    allowed_operations: list[str]
    repair_layer: str
    layer_attempt_number: int
    total_repair_provider_calls: int
    # A V5 ownership-repair target may add only these immutable Source Map
    # scope IDs to its semantic block(s). It never carries canonical facts.
    allowed_evidence_scope_ids: list[str]
    # V5 instructional-coherence repairs use typed semantic deltas rather than
    # broad learning-block replacement. These identifiers are validated against
    # the immutable baseline before a copy-on-write candidate is returned.
    semantic_operations: list[str]
    allowed_block_ids: list[str]
    allowed_objective_ids: list[str]
    # A post-allocation depth target is lesson-scoped. The provider may select
    # only one of these server-approved unit paths and an allowed anchor block
    # inside it; it never chooses evidence or fact ownership.
    allowed_unit_paths: list[str]
    # V5 evidence-semantic alignment is a concept-only typed delta. These
    # values are derived from the immutable Source Map by Python; they are
    # never accepted from a provider as authority over source ownership.
    allowed_concept_ids: list[str]
    known_concept_ids: list[str]
    required_concept_ids: list[str]
    primary_concept_options: list[list[str]]
    evidence_alignment_block_concepts: dict[str, list[str]]
    # Assessment teach→check repair is a provider-owned choice between exact
    # server-approved existing blocks. These records are never provider input
    # authority over provenance or hierarchy; they bind the one check path and
    # one possible (including cross-unit, same-lesson) teaching path.
    assessment_alignment_candidates: list[dict[str, object]]
    # Per-objective semantic selection authority for a missing check. The
    # provider can pick only one exact server-approved block per objective;
    # it never receives evidence scopes, source facts, or hierarchy authority.
    assessment_plan_candidates: list[dict[str, object]]
    assessment_plan_fingerprint: str
    assessment_plan_objectives: dict[str, str]
    deterministic_semantic_delta: bool


class WorkflowProgressEvent(TypedDict, total=False):
    code: str
    message: str
    phase: Literal["course_architecture", "lesson_generation"]
    current: int
    total: int


class WorkflowDiagnosticEvent(TypedDict, total=False):
    """Safe, machine-searchable operational metadata for one workflow stage."""

    correlation_id: str
    stage: str
    event: str
    duration_ms: int
    repair_pass_number: int
    failure_stage: str
    internal_failure_code: str
    external_failure_code: str


@dataclass(frozen=True)
class WorkflowValidationResult:
    """Deterministic validation result; warnings do not imply a repair loop."""

    issues: list[WorkflowIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[WorkflowIssue]:
        return [issue for issue in self.issues if issue.get("severity") == "error"]


@dataclass(frozen=True)
class WorkflowGenerationResult:
    value: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    # A V5 coherence pass can issue one narrow structured provider call per
    # operation type while remaining one transactional repair attempt.
    provider_call_count: int = 0


class WorkflowFailure(RuntimeError):
    """A safe workflow failure category suitable for the service boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: list[WorkflowIssue] | None = None,
        internal_code: str | None = None,
        failure_stage: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = issues or []
        # `code` remains the API-safe workflow category. More specific codes
        # are internal-only diagnostics and must never be inferred later from
        # this generic external category.
        self.internal_code = internal_code or code
        self.failure_stage = failure_stage
        self.diagnostics = diagnostics or {}


_SAFE_WORKFLOW_PATH = re.compile(
    r"^(?:course|chapter_\d+(?:\.lesson_\d+(?:\.unit_\d+(?:\.(?:block|component)_\d+)?)?)?)?$"
)
_SAFE_WORKFLOW_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_SAFE_OWNERSHIP_LEVELS = {"lesson", "unit", "learning_block"}
_SAFE_SCOPE_CLASSIFICATIONS = {
    "concept_primary_ownership",
    "section_coverage",
    "source_reference_scope",
}
_SAFE_COVERAGE_STATES = {"missing", "ambiguous", "covered"}
_SAFE_REASON_CODES = {
    "NO_PRIMARY_DESTINATION",
    "MULTIPLE_PRIMARY_DESTINATIONS",
    "NO_ELIGIBLE_PRIMARY_DESTINATION",
    "INVALID_LOCAL_ASSESSMENT_OBJECTIVE_REF",
    "NO_LESSON_UNITS",
    "NO_SAFE_TEACHING_ANCHOR",
    "SELECTION_OUTSIDE_SERVER_CANDIDATE_SET",
    "MULTIPLE_ASSESSMENT_INTERACTIONS_IN_ONE_UNIT",
    "AMBIGUOUS_OR_SEMANTIC_TEACHING_ANCHOR",
}


def safe_workflow_path(value: Any) -> str:
    """Return only structural workflow paths, never customer-generated text."""

    path = str(value or "course").strip()
    return path if _SAFE_WORKFLOW_PATH.fullmatch(path) else "course"


def safe_workflow_issue_summary(issue: WorkflowIssue, *, repairable: bool) -> dict[str, Any]:
    """Strip a validation finding down to non-content operational metadata."""

    summary = {
        "finding_code": str(issue.get("code") or "UNKNOWN"),
        "severity": str(issue.get("severity") or "error"),
        "repairable": repairable,
        "target_type": "component" if ".component_" in safe_workflow_path(issue.get("path")) else (
            "unit" if ".unit_" in safe_workflow_path(issue.get("path")) else (
                "lesson" if ".lesson_" in safe_workflow_path(issue.get("path")) else (
                    "chapter" if safe_workflow_path(issue.get("path")).startswith("chapter_") else "course"
                )
            )
        ),
        "target_path": safe_workflow_path(issue.get("path")),
    }
    for key in ("validator", "schema_error_code", "constraint", "expected_type", "actual_type"):
        value = issue.get(key)
        if isinstance(value, str) and value:
            summary[key] = value
    for key in ("concept_id", "section_id"):
        value = issue.get(key)
        if isinstance(value, str) and _SAFE_WORKFLOW_IDENTIFIER.fullmatch(value):
            summary[key] = value
    for key in ("concept_ids", "section_ids"):
        values = issue.get(key)
        if isinstance(values, list):
            safe_values = [
                value
                for value in values
                if isinstance(value, str) and _SAFE_WORKFLOW_IDENTIFIER.fullmatch(value)
            ][:12]
            if safe_values:
                summary[key] = safe_values
    for key in ("evidence_scope_id",):
        value = issue.get(key)
        if isinstance(value, str) and _SAFE_WORKFLOW_IDENTIFIER.fullmatch(value):
            summary[key] = value
    for key in ("evidence_scope_ids",):
        values = issue.get(key)
        if isinstance(values, list):
            safe_values = [
                value
                for value in values
                if isinstance(value, str) and _SAFE_WORKFLOW_IDENTIFIER.fullmatch(value)
            ][:12]
            if safe_values:
                summary[key] = safe_values
    for key in ("expected_primary_owner_count", "actual_primary_owner_count"):
        value = issue.get(key)
        if isinstance(value, int) and 0 <= value <= 1_000:
            summary[key] = value
    candidate_count = issue.get("assessment_candidate_count")
    if isinstance(candidate_count, int) and 0 <= candidate_count <= 1_000:
        summary["assessment_candidate_count"] = candidate_count
    ownership_level = issue.get("ownership_level")
    if isinstance(ownership_level, str) and ownership_level in _SAFE_OWNERSHIP_LEVELS:
        summary["ownership_level"] = ownership_level
    scope_classification = issue.get("scope_classification")
    if isinstance(scope_classification, str) and scope_classification in _SAFE_SCOPE_CLASSIFICATIONS:
        summary["scope_classification"] = scope_classification
    coverage_state = issue.get("coverage_state")
    if isinstance(coverage_state, str) and coverage_state in _SAFE_COVERAGE_STATES:
        summary["coverage_state"] = coverage_state
    safe_reason = issue.get("safe_reason")
    if isinstance(safe_reason, str) and safe_reason in _SAFE_REASON_CODES:
        summary["safe_reason"] = safe_reason
    related_paths = issue.get("related_paths")
    if isinstance(related_paths, list):
        paths = sorted({
            safe_workflow_path(path)
            for path in related_paths
            if isinstance(path, str) and safe_workflow_path(path) != "course"
        })[:12]
        if paths:
            summary["related_paths"] = paths
    eligible_paths = issue.get("eligible_repair_paths")
    if isinstance(eligible_paths, list):
        paths = sorted({
            safe_workflow_path(path)
            for path in eligible_paths
            if isinstance(path, str) and safe_workflow_path(path) != "course"
        })[:12]
        if paths:
            summary["eligible_repair_paths"] = paths
    return summary


def progress_event(
    code: str,
    message: str,
    phase: Literal["course_architecture", "lesson_generation"],
    *,
    current: int | None = None,
    total: int | None = None,
) -> WorkflowProgressEvent:
    event: WorkflowProgressEvent = {"code": code, "message": message, "phase": phase}
    if current is not None:
        event["current"] = current
    if total is not None:
        event["total"] = total
    return event


def append_progress(
    state: dict[str, Any],
    event: WorkflowProgressEvent,
) -> list[WorkflowProgressEvent]:
    return [*(state.get("progress") or []), event]


def node_duration_update(state: dict[str, Any], node: str, started_at: float) -> dict[str, int]:
    durations = dict(state.get("node_durations_ms") or {})
    durations[node] = max(0, round((perf_counter() - started_at) * 1000))
    return durations


def sanitized_workflow_diagnostics(
    state: dict[str, Any],
    *,
    workflow: Literal["course_architecture", "lesson_generation"],
    started_at: float,
) -> dict[str, Any]:
    """Expose operational state only; no prompts, source content, or secrets."""

    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    return {
        "workflow": workflow,
        "workflow_version": "langgraph-v1",
        "status": state.get("status", "failed"),
        "duration_ms": max(0, round((perf_counter() - started_at) * 1000)),
        "node_durations_ms": dict(state.get("node_durations_ms") or {}),
        "repair_count": int(state.get("repair_attempts") or 0),
        "repair_provider_calls": int(state.get("repair_provider_calls") or 0),
        "repair_layer_attempts": dict(state.get("repair_layer_attempts") or {}),
        "repair_layer_statuses": dict(state.get("repair_layer_statuses") or {}),
        "validation_codes": sorted({str(issue.get("code")) for issue in (state.get("validation_issues") or []) if issue.get("code")}),
        "source_coverage": metrics.get("source_coverage"),
        "source_fact_coverage": metrics.get("source_fact_coverage"),
        "concept_coverage": metrics.get("concept_coverage"),
        "objective_coverage": metrics.get("objective_coverage"),
        "assessment_alignment": metrics.get("assessment_alignment"),
        "instructional_depth": metrics.get("instructional_depth"),
        "component_purpose": metrics.get("component_purpose"),
        "duplicate_count": metrics.get("duplicate_count", 0),
        "pedagogical_warning_count": metrics.get("pedagogical_warning_count", 0),
        "component_counts": metrics.get("component_counts", {}),
        "progress": list(state.get("progress") or []),
    }
